from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from loguru import logger

from criteria import SegEvaluator
from dataload import CLASS_COLORS, CLASS_NAMES, NUM_CLASSES, build_dataloaders
from utils.runtime import load_checkpoint_compat, resolve_device, setup_logger
from utils.segmentor_loader import (
    build_segmentor_from_checkpoint,
    get_class_names_from_checkpoint,
    get_input_size_from_checkpoint,
    resolve_checkpoint_path,
)
from utils.segmentation_vis import blend_overlay, colorize_mask, denormalize_image_tensor

from .config import InferConfig


class SegmentationInferencer:
    def __init__(self, cfg: InferConfig):
        self.cfg = cfg

    @staticmethod
    def _build_panel(image: np.ndarray, gt_rgb: np.ndarray, pred_rgb: np.ndarray) -> Image.Image:
        gt_overlay = blend_overlay(image, gt_rgb)
        pred_overlay = blend_overlay(image, pred_rgb)
        panel = np.concatenate([image, gt_rgb, pred_rgb, pred_overlay, gt_overlay], axis=1)
        return Image.fromarray(panel)

    @staticmethod
    @torch.no_grad()
    def _evaluate(model, loader, device: torch.device, class_names: tuple[str, ...]):
        evaluator = SegEvaluator(num_classes=len(class_names), class_names=class_names)
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model(images)
            evaluator.update(logits, masks)

        metrics = evaluator.compute()
        evaluator.print_table(metrics)
        logger.info(
            f"评估摘要: aAcc={metrics['aAcc']*100:.2f}%  "
            f"mIoU={metrics['mIoU']*100:.2f}%  "
            f"mDice={metrics['mDice']*100:.2f}%"
        )
        return metrics

    @classmethod
    @torch.no_grad()
    def _save_visualizations(cls, model, dataset, device: torch.device, output_dir: Path, count: int):
        vis_dir = output_dir / "vis"
        vis_dir.mkdir(parents=True, exist_ok=True)

        total = min(count, len(dataset))
        logger.info(f"开始保存可视化: {total} 张 → {vis_dir}")

        for idx in range(total):
            image, gt_mask = dataset[idx]
            image_b = image.unsqueeze(0).to(device)
            logits = model(image_b)
            pred_mask = logits.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
            gt_mask_np = gt_mask.detach().cpu().numpy().astype(np.uint8)

            image_np = denormalize_image_tensor(image)
            gt_rgb = colorize_mask(gt_mask_np, CLASS_COLORS)
            pred_rgb = colorize_mask(pred_mask, CLASS_COLORS)
            panel = cls._build_panel(image_np, gt_rgb, pred_rgb)

            stem = dataset.pairs[idx][0].stem if hasattr(dataset, "pairs") else f"sample_{idx:03d}"
            panel.save(vis_dir / f"{stem}.png")

        logger.success(f"可视化保存完成: {vis_dir}")

    @staticmethod
    def _save_metrics(metrics: dict, output_dir: Path):
        serializable = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                serializable[key] = value.tolist()
            else:
                serializable[key] = float(value)

        with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
        logger.info(f"metrics 已保存: {output_dir / 'metrics.json'}")

    def run(self) -> dict:
        cfg = self.cfg

        ckpt_path = resolve_checkpoint_path(cfg.ckpt, hint_script="infer.py")
        if cfg.ckpt:
            ckpt_stem = ckpt_path.parent.name
            out_dir = Path(f"{cfg.output_dir}_{ckpt_stem}_{cfg.split}")
        else:
            out_dir = Path(f"{cfg.output_dir}_{cfg.split}")

        out_dir.mkdir(parents=True, exist_ok=True)
        setup_logger(str(out_dir / "infer.log"))

        logger.info("=" * 70)
        logger.info("推理 / 评估启动")
        logger.info("=" * 70)
        logger.info(f"配置: {asdict(cfg)}")
        logger.info(f"使用 checkpoint: {ckpt_path}")

        device = resolve_device(cfg.device, allow_mps=True, warn_mps_on_auto=True)
        ckpt = load_checkpoint_compat(ckpt_path, map_location=device)
        input_size = get_input_size_from_checkpoint(ckpt, default=512)
        class_names = get_class_names_from_checkpoint(ckpt, default=CLASS_NAMES)

        loaders = build_dataloaders(
            data_roots=[cfg.data_root],
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            input_size=input_size,
            splits=[cfg.split],
        )
        loader = loaders[cfg.split]

        logger.info(
            f"评估集: split={cfg.split}, samples={len(loader.dataset)}, "
            f"input_size={input_size}, batch_size={cfg.batch_size}"
        )

        model, _ = build_segmentor_from_checkpoint(
            ckpt,
            device,
            default_num_classes=NUM_CLASSES,
            use_backbone_weight_from_cfg=True,
            use_frozen_stages_from_cfg=True,
        )
        metrics = self._evaluate(model, loader, device, class_names=class_names)
        self._save_metrics(metrics, out_dir)

        if cfg.save_vis:
            self._save_visualizations(model, loader.dataset, device, out_dir, cfg.vis_count)

        logger.success("=" * 70)
        logger.success("推理 / 评估完成")
        logger.success(f"输出目录: {out_dir.resolve()}")
        logger.success("=" * 70)

        return metrics
