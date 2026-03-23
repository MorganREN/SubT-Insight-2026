from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from loguru import logger

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

from .config import PredictConfig
from .visuals import (
    build_error_overlay,
    infer_mask_path,
    load_gt_mask,
    postprocess_mask,
    preprocess_image,
    save_outputs_basic,
    save_outputs_with_gt,
)
from dataload import CLASS_COLORS as _DEFAULT_CLASS_COLORS


class ImagePredictor:
    def __init__(self, cfg: PredictConfig):
        self.cfg = cfg

    @staticmethod
    def _compute_single_image_metrics(
        pred: np.ndarray,
        target: np.ndarray,
        num_classes: int,
        class_names: tuple[str, ...],
        ignore_index: int = 255,
    ) -> dict:
        evaluator = SegEvaluator(
            num_classes=num_classes,
            class_names=class_names,
            ignore_index=ignore_index,
        )
        evaluator.update(pred[np.newaxis, ...], target[np.newaxis, ...])
        metrics = evaluator.compute()
        return {
            "mIoU": float(metrics["mIoU"]),
            "aAcc": float(metrics["aAcc"]),
            "IoU": metrics["IoU"],
        }

    @torch.no_grad()
    def run(self):
        cfg = self.cfg

        image_path = Path(cfg.image)
        if not image_path.exists():
            raise FileNotFoundError(f"输入图片不存在: {image_path}。请在 RUN.image 中修改为有效路径。")

        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        setup_logger(str(out_dir / "predict.log"))

        logger.info("=" * 70)
        logger.info("单图推理启动")
        logger.info("=" * 70)
        logger.info(f"输入图片: {image_path}")

        device = resolve_device(cfg.device)
        ckpt_path = resolve_checkpoint_path(cfg.ckpt, hint_script="predict_image.py")
        logger.info(f"使用 checkpoint: {ckpt_path}")

        ckpt = load_checkpoint_compat(ckpt_path, map_location="cpu")
        if is_quantized_checkpoint(ckpt) and device.type != "cpu":
            logger.warning("检测到量化 checkpoint，predict_image 自动切换到 CPU 推理。")
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

        input_size = cfg.input_size if cfg.input_size is not None else get_input_size_from_checkpoint(ckpt, default=512)
        class_names = get_class_names_from_checkpoint(ckpt, default=CLASS_NAMES)
        logger.info(f"推理输入尺寸: {input_size}")

        image_np, input_tensor = preprocess_image(image_path, input_size=input_size)
        logits = model(input_tensor.unsqueeze(0).to(device))
        pred = logits.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
        pred = postprocess_mask(pred, original_hw=image_np.shape[:2])

        pred_color = colorize_mask(pred, CLASS_COLORS)
        pred_overlay = blend_overlay(image_np, pred_color, alpha=0.45)

        mask_path = Path(cfg.mask) if cfg.mask else infer_mask_path(image_path)
        if mask_path is not None and mask_path.exists():
            gt_mask = load_gt_mask(mask_path, hw=image_np.shape[:2])
            gt_color = colorize_mask(gt_mask, CLASS_COLORS)
            gt_overlay = blend_overlay(image_np, gt_color, alpha=0.45)
            error_overlay = build_error_overlay(image_np, pred, gt_mask)

            metric = self._compute_single_image_metrics(
                pred,
                gt_mask,
                num_classes=int(model_cfg.get("num_classes", NUM_CLASSES)),
                class_names=class_names,
            )
            present_mask = np.bincount(
                gt_mask[gt_mask != 255].ravel(),
                minlength=int(model_cfg.get("num_classes", NUM_CLASSES)),
            ).astype(bool)

            save_outputs_with_gt(
                image_path=image_path,
                out_dir=out_dir,
                image=image_np,
                gt_color_mask=gt_color,
                gt_overlay=gt_overlay,
                pred_color_mask=pred_color,
                pred_overlay=pred_overlay,
                error_overlay=error_overlay,
                class_names=class_names,
                class_colors=_DEFAULT_CLASS_COLORS,
                per_class_iou=metric["IoU"],
                present_mask=present_mask,
            )
            logger.info(
                f"单图指标: mIoU={metric['mIoU']*100:.2f}%  Accuracy={metric['aAcc']*100:.2f}%"
            )
        else:
            save_outputs_basic(
                image_path=image_path,
                out_dir=out_dir,
                image=image_np,
                pred_color_mask=pred_color,
                pred_overlay=pred_overlay,
                class_names=class_names,
                class_colors=_DEFAULT_CLASS_COLORS,
            )
            logger.warning("未找到对应 GT mask，跳过 IoU/Accuracy 计算（可在 RUN.mask 显式指定）")

        logger.success("=" * 70)
        logger.success("单图推理完成")
        logger.success(f"输出目录: {out_dir.resolve()}")
        logger.success("=" * 70)
