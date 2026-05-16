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

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import json
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
from predictor.tiling import tiled_predict
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
    img_root: str = "dataset/tongji_data_raw/img_dir"
    splits: list[str] = field(default_factory=lambda: ["valid"])  # 默认仅推理 valid split，修改为 ["train", "valid", "test"] 可全量推理
    ckpt: str = "outputs/tmds_run_awesome/best.pth"
    device: str = "auto"
    output_dir: str = "outputs/tmds_run_awesome/predict_dataset"
    input_size: int | None = None
    use_tiling: bool = True   # True = 对原图按群落参数做滑动窗口推理

    # ── labelme JSON 输出（data_tools/mask_to_labelme.py） ──
    save_labelme: bool = False             # True = 每张图同步导出 {stem}.json
    labelme_epsilon: float = 1.0           # cv2.approxPolyDP 简化阈值；0 = 不简化
    labelme_embed_image: bool = False      # True = 把原图 base64 嵌入 imageData


RUN = BatchPredictConfig()


def _to_serializable(value):
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, list):
        return [_to_serializable(v) for v in value]
    return value


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
        split_evaluator = SegEvaluator(
            num_classes=num_classes,
            class_names=class_names,
            ignore_index=255,
        )

        # (miou_or_None, panel_path, mask_path)
        results: list[tuple[float | None, Path, Path]] = []

        for image_path in image_paths:
            image_np = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
            if cfg.use_tiling:
                pred = tiled_predict(model, image_np, device, num_classes, input_size)
            else:
                _, input_tensor = preprocess_image(image_path, input_size=input_size)
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
                split_evaluator.update(pred[np.newaxis], gt_mask[np.newaxis])
                miou = float(metrics["mIoU"])
                present_mask = np.bincount(
                    gt_mask[gt_mask != 255].ravel(), minlength=num_classes
                ).astype(bool)

                _grid = _build_panel_2x3(
                    images=[image_np, gt_overlay, gt_color, error_overlay, pred_overlay, pred_color],
                    titles=[
                        "Original", "GT Overlay", "GT Segmented",
                        f"Error  mIoU={miou * 100:.1f}%",
                        "Pred Overlay", "Pred Segmented",
                    ],
                )
                _legend = build_legend_bar(
                    CLASS_NAMES, CLASS_COLORS, width=_grid.shape[1],
                    per_class_iou=metrics["IoU"], present_mask=present_mask,
                )
                panel = np.concatenate([_grid, _legend], axis=0)
                stem_iou = f"{image_path.stem}_iou{miou * 100:.1f}"
                panel_path = out_dir / f"{stem_iou}_panel.png"
                mask_save_path = out_dir / f"{stem_iou}_pred_mask.png"
                Image.fromarray(panel).save(panel_path)
                Image.fromarray(pred, mode="L").save(mask_save_path)
                metrics_path = out_dir / f"{stem_iou}_metrics.json"
                with open(metrics_path, "w", encoding="utf-8") as f:
                    json.dump(_to_serializable(metrics), f, ensure_ascii=False, indent=2)
                if cfg.save_labelme:
                    from data_tools.mask_to_labelme import convert as _to_labelme
                    _to_labelme(
                        pred, image_path, out_dir / f"{image_path.stem}.json",
                        epsilon=cfg.labelme_epsilon,
                        embed_image_data=cfg.labelme_embed_image,
                    )
                results.append((miou, panel_path, mask_save_path))
                logger.info(f"  {image_path.name:<20} {evaluator.summary(metrics)}")
            else:
                _grid = _build_panel_2x3(
                    images=[image_np, _blank_like(image_np), _blank_like(image_np),
                            _blank_like(image_np), pred_overlay, pred_color],
                    titles=[
                        "Original", "GT Overlay (N/A)", "GT Segmented (N/A)",
                        "Error (N/A)", "Pred Overlay", "Pred Segmented",
                    ],
                )
                _legend = build_legend_bar(CLASS_NAMES, CLASS_COLORS, width=_grid.shape[1])
                panel = np.concatenate([_grid, _legend], axis=0)
                panel_path = out_dir / f"{image_path.stem}_panel.png"
                mask_save_path = out_dir / f"{image_path.stem}_pred_mask.png"
                Image.fromarray(panel).save(panel_path)
                Image.fromarray(pred, mode="L").save(mask_save_path)
                if cfg.save_labelme:
                    from data_tools.mask_to_labelme import convert as _to_labelme
                    _to_labelme(
                        pred, image_path, out_dir / f"{image_path.stem}.json",
                        epsilon=cfg.labelme_epsilon,
                        embed_image_data=cfg.labelme_embed_image,
                    )
                results.append((None, panel_path, mask_save_path))
                logger.warning(f"  {image_path.name:<20} 未找到 GT mask，跳过 IoU")

        # --- 按 mIoU 由高到低重命名，加 rank 前缀 ---
        ranked = sorted(
            [(iou, p, m) for iou, p, m in results if iou is not None],
            key=lambda x: x[0],
            reverse=True,
        )
        for rank, (_, panel_path, mask_path) in enumerate(ranked, start=1):
            if panel_path.exists():
                panel_path.rename(panel_path.parent / f"rank{rank:03d}_{panel_path.name}")
            if mask_path.exists():
                mask_path.rename(mask_path.parent / f"rank{rank:03d}_{mask_path.name}")

        if split_evaluator.confusion_matrix.sum() > 0:
            split_metrics = split_evaluator.compute()
            split_evaluator.print_table(split_metrics)
            split_evaluator.print_task_report(split_metrics)
            split_metrics_path = out_dir / "split_metrics.json"
            with open(split_metrics_path, "w", encoding="utf-8") as f:
                json.dump(_to_serializable(split_metrics), f, ensure_ascii=False, indent=2)
            logger.info(f"[{split}] 聚合 metrics 已保存: {split_metrics_path}")

        logger.success(f"[{split}] 完成，已按 mIoU 排序，结果: {out_dir.resolve()}")

    logger.success("=" * 70)
    logger.success("批量推理全部完成")
    logger.success(f"输出根目录: {out_root.resolve()}")
    logger.success("=" * 70)


def main(cfg: BatchPredictConfig | None = None) -> None:
    run_batch(RUN if cfg is None else cfg)


if __name__ == "__main__":
    main()
