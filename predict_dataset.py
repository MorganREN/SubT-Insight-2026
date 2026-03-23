"""
predict_dataset.py
批量推理入口：对 train / valid / test 三个 split 全量推理。

用法
----
1) 修改下方 RUN 配置。
2) 直接运行：python predict_dataset.py

输出规则
--------
- 每个 split 单独存入 output_dir/<split>/ 子目录。
- 面板文件名含 mIoU 信息，例如 C100_iou82.3_panel.png。
- 面板标题（Error 格中）同步显示 mIoU。
- 所有面板按 mIoU 由高到低重命名为 rank001_…、rank002_… 以便文件管理器直接排序。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from PIL import Image

from criteria import SegEvaluator
from dataload import CLASS_COLORS, CLASS_NAMES, NUM_CLASSES
from utils.runtime import load_checkpoint_compat, resolve_device, setup_logger
from utils.segmentor_loader import (
    build_segmentor_from_checkpoint,
    get_class_names_from_checkpoint,
    get_input_size_from_checkpoint,
    is_quantized_checkpoint,
    resolve_checkpoint_path,
)
from utils.segmentation_vis import blend_overlay, colorize_mask
from predictor.visuals import (
    _blank_like,
    _build_panel_2x3,
    build_error_overlay,
    build_legend_bar,
    infer_mask_path,
    load_gt_mask,
    postprocess_mask,
    preprocess_image,
)


@dataclass
class BatchPredictConfig:
    img_root: str = "dataset/tongji_data/img_dir"
    splits: list[str] = field(default_factory=lambda: ["train", "valid", "test"])
    ckpt: str = "outputs_2203/tmds_run/best.pth"
    device: str = "auto"
    output_dir: str = "outputs_2203/tmds_run1903/predict_dataset"
    input_size: int | None = None


RUN = BatchPredictConfig()


# ---------------------------------------------------------------------------
# 面板构建
# ---------------------------------------------------------------------------

def _panel_with_gt(
    image: np.ndarray,
    gt_overlay: np.ndarray,
    gt_color: np.ndarray,
    pred_overlay: np.ndarray,
    pred_color: np.ndarray,
    error_overlay: np.ndarray,
    miou: float,
    per_class_iou: np.ndarray,
    present_mask: np.ndarray,
) -> np.ndarray:
    grid = _build_panel_2x3(
        images=[image, gt_overlay, gt_color, error_overlay, pred_overlay, pred_color],
        titles=[
            "Original",
            "GT Overlay",
            "GT Segmented",
            f"Error  mIoU={miou * 100:.1f}%",
            "Pred Overlay",
            "Pred Segmented",
        ],
    )
    legend = build_legend_bar(
        CLASS_NAMES, CLASS_COLORS,
        width=grid.shape[1],
        per_class_iou=per_class_iou,
        present_mask=present_mask,
    )
    return np.concatenate([grid, legend], axis=0)


def _panel_no_gt(
    image: np.ndarray,
    pred_overlay: np.ndarray,
    pred_color: np.ndarray,
) -> np.ndarray:
    blank = _blank_like(image)
    grid = _build_panel_2x3(
        images=[image, blank, blank, blank, pred_overlay, pred_color],
        titles=[
            "Original",
            "GT Overlay (N/A)",
            "GT Segmented (N/A)",
            "Error (N/A)",
            "Pred Overlay",
            "Pred Segmented",
        ],
    )
    legend = build_legend_bar(CLASS_NAMES, CLASS_COLORS, width=grid.shape[1])
    return np.concatenate([grid, legend], axis=0)


# ---------------------------------------------------------------------------
# 主推理流程
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_batch(cfg: BatchPredictConfig) -> None:
    out_root = Path(cfg.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    setup_logger(str(out_root / "predict.log"))

    logger.info("=" * 70)
    logger.info("批量推理启动")
    logger.info("=" * 70)

    device = resolve_device(cfg.device)
    ckpt_path = resolve_checkpoint_path(cfg.ckpt, hint_script="predict_dataset.py")
    logger.info(f"使用 checkpoint: {ckpt_path}")

    ckpt = load_checkpoint_compat(ckpt_path, map_location="cpu")
    if is_quantized_checkpoint(ckpt) and device.type != "cpu":
        logger.warning("检测到量化 checkpoint，自动切换到 CPU 推理。")
        device = torch.device("cpu")

    model, model_cfg = build_segmentor_from_checkpoint(
        ckpt,
        device,
        default_num_classes=NUM_CLASSES,
        use_backbone_weight_from_cfg=False,
        default_backbone_weight_path=None,
        use_frozen_stages_from_cfg=False,
        default_frozen_stages=-1,
    )
    input_size = (
        cfg.input_size
        if cfg.input_size is not None
        else get_input_size_from_checkpoint(ckpt, default=512)
    )
    class_names = get_class_names_from_checkpoint(ckpt, default=CLASS_NAMES)
    num_classes = int(model_cfg.get("num_classes", NUM_CLASSES))
    logger.info(f"推理输入尺寸: {input_size}")

    img_root = Path(cfg.img_root)
    for split in cfg.splits:
        split_dir = img_root / split
        if not split_dir.exists():
            logger.warning(f"split 目录不存在，跳过: {split_dir}")
            continue

        out_dir = out_root / split
        out_dir.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(split_dir.glob("*.jpg")) + sorted(split_dir.glob("*.png"))
        if not image_paths:
            logger.warning(f"[{split}] 未找到图片，跳过")
            continue

        logger.info(f"[{split}] 共 {len(image_paths)} 张图片 -> {out_dir}")

        # (miou_or_None, panel_path)
        results: list[tuple[float | None, Path]] = []

        for image_path in image_paths:
            image_np, input_tensor = preprocess_image(image_path, input_size=input_size)
            logits = model(input_tensor.unsqueeze(0).to(device))
            pred = logits.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
            pred = postprocess_mask(pred, original_hw=image_np.shape[:2])

            pred_color = colorize_mask(pred, CLASS_COLORS)
            pred_overlay = blend_overlay(image_np, pred_color, alpha=0.45)

            mask_path = infer_mask_path(image_path)
            if mask_path is not None and mask_path.exists():
                gt_mask = load_gt_mask(mask_path, hw=image_np.shape[:2])
                gt_color = colorize_mask(gt_mask, CLASS_COLORS)
                gt_overlay = blend_overlay(image_np, gt_color, alpha=0.45)
                error_overlay = build_error_overlay(image_np, pred, gt_mask)

                evaluator = SegEvaluator(
                    num_classes=num_classes,
                    class_names=class_names,
                    ignore_index=255,
                )
                evaluator.update(pred[np.newaxis], gt_mask[np.newaxis])
                metrics = evaluator.compute()
                miou = float(metrics["mIoU"])
                present_mask = np.bincount(
                    gt_mask[gt_mask != 255].ravel(), minlength=num_classes
                ).astype(bool)

                panel = _panel_with_gt(
                    image=image_np,
                    gt_overlay=gt_overlay,
                    gt_color=gt_color,
                    pred_overlay=pred_overlay,
                    pred_color=pred_color,
                    error_overlay=error_overlay,
                    miou=miou,
                    per_class_iou=metrics["IoU"],
                    present_mask=present_mask,
                )
                panel_path = out_dir / f"{image_path.stem}_iou{miou * 100:.1f}_panel.png"
                Image.fromarray(panel).save(panel_path)
                results.append((miou, panel_path))
                logger.info(f"  {image_path.name:<20} mIoU={miou * 100:.2f}%")
            else:
                panel = _panel_no_gt(image_np, pred_overlay, pred_color)
                panel_path = out_dir / f"{image_path.stem}_panel.png"
                Image.fromarray(panel).save(panel_path)
                results.append((None, panel_path))
                logger.warning(f"  {image_path.name:<20} 未找到 GT mask，跳过 IoU")

        # --- 按 mIoU 由高到低重命名，加 rank 前缀 ---
        ranked = sorted(
            [(iou, p) for iou, p in results if iou is not None],
            key=lambda x: x[0],
            reverse=True,
        )
        for rank, (_, panel_path) in enumerate(ranked, start=1):
            if panel_path.exists():
                new_path = panel_path.parent / f"rank{rank:03d}_{panel_path.name}"
                panel_path.rename(new_path)

        logger.success(f"[{split}] 完成，已按 mIoU 排序，结果: {out_dir.resolve()}")

    logger.success("=" * 70)
    logger.success("批量推理全部完成")
    logger.success(f"输出根目录: {out_root.resolve()}")
    logger.success("=" * 70)


def main(cfg: BatchPredictConfig | None = None) -> None:
    run_batch(RUN if cfg is None else cfg)


if __name__ == "__main__":
    main()
