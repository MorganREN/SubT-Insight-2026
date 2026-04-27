from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from loguru import logger

from criteria import SegEvaluator
from dataload import CLASS_COLORS, CLASS_NAMES, NUM_CLASSES, build_dataloaders
from predictor.tiling import tiled_predict
from utils.runtime import load_checkpoint_compat, resolve_device, setup_logger
from utils.segmentor_loader import (
    build_segmentor_from_checkpoint,
    get_class_names_from_checkpoint,
    get_input_size_from_checkpoint,
    is_quantized_checkpoint,
    resolve_checkpoint_path,
)
from utils.segmentation_vis import blend_overlay, colorize_mask, denormalize_image_tensor

from .config import InferConfig


_TILED_WORKER_MODEL = None
_TILED_WORKER_DEVICE = torch.device("cpu")
_TILED_WORKER_NUM_CLASSES = NUM_CLASSES
_TILED_WORKER_INPUT_SIZE = 512
_TILED_WORKER_CLASS_NAMES = CLASS_NAMES
_TILED_WORKER_USE_TTA = False


def _set_torch_cpu_threads(num_threads: int) -> None:
    num_threads = max(1, int(num_threads))
    torch.set_num_threads(num_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch only allows this before inter-op parallel work starts.
        pass


def _init_tiled_worker(
    ckpt_path: str,
    num_classes: int,
    input_size: int,
    class_names: tuple[str, ...],
    use_tta: bool,
    threads_per_worker: int,
) -> None:
    global _TILED_WORKER_MODEL
    global _TILED_WORKER_NUM_CLASSES
    global _TILED_WORKER_INPUT_SIZE
    global _TILED_WORKER_CLASS_NAMES
    global _TILED_WORKER_USE_TTA

    _set_torch_cpu_threads(threads_per_worker)
    ckpt = load_checkpoint_compat(ckpt_path, map_location="cpu")
    model, _ = build_segmentor_from_checkpoint(
        ckpt,
        _TILED_WORKER_DEVICE,
        default_num_classes=NUM_CLASSES,
        use_backbone_weight_from_cfg=True,
        use_frozen_stages_from_cfg=True,
    )
    _TILED_WORKER_MODEL = model
    _TILED_WORKER_NUM_CLASSES = num_classes
    _TILED_WORKER_INPUT_SIZE = input_size
    _TILED_WORKER_CLASS_NAMES = class_names
    _TILED_WORKER_USE_TTA = use_tta


def _extract_evaluator_state(evaluator: SegEvaluator) -> dict:
    return {
        "confusion_matrix": evaluator.confusion_matrix,
        "image_tp": evaluator._image_tp.copy(),
        "image_fp": evaluator._image_fp.copy(),
        "image_fn": evaluator._image_fn.copy(),
        "num_images": evaluator._num_images,
        "crack_component_gt_total": evaluator._crack_component_gt_total,
        "crack_component_gt_hit": evaluator._crack_component_gt_hit,
        "crack_component_pred_total": evaluator._crack_component_pred_total,
        "crack_component_pred_hit": evaluator._crack_component_pred_hit,
        "crack_boundary_pred_hit": evaluator._crack_boundary_pred_hit,
        "crack_boundary_pred_total": evaluator._crack_boundary_pred_total,
        "crack_boundary_gt_hit": evaluator._crack_boundary_gt_hit,
        "crack_boundary_gt_total": evaluator._crack_boundary_gt_total,
        "crack_skeleton_pred_hit": evaluator._crack_skeleton_pred_hit,
        "crack_skeleton_pred_total": evaluator._crack_skeleton_pred_total,
        "crack_skeleton_gt_hit": evaluator._crack_skeleton_gt_hit,
        "crack_skeleton_gt_total": evaluator._crack_skeleton_gt_total,
    }


def _evaluate_tiled_image_worker(task: tuple[str, str]) -> dict:
    img_path_str, ann_dir_str = task
    img_path = Path(img_path_str)
    ann_path = Path(ann_dir_str) / f"{img_path.stem}.png"
    if not ann_path.exists():
        return {"ok": False, "image": img_path.name, "reason": "missing_gt"}

    image_np = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
    gt_mask = np.array(Image.open(ann_path).convert("L"), dtype=np.uint8)
    pred_mask = tiled_predict(
        _TILED_WORKER_MODEL,
        image_np,
        _TILED_WORKER_DEVICE,
        _TILED_WORKER_NUM_CLASSES,
        _TILED_WORKER_INPUT_SIZE,
        use_tta=_TILED_WORKER_USE_TTA,
    )

    evaluator = SegEvaluator(
        num_classes=_TILED_WORKER_NUM_CLASSES,
        class_names=_TILED_WORKER_CLASS_NAMES,
    )
    evaluator.update(pred_mask[np.newaxis], gt_mask[np.newaxis])
    return {"ok": True, "image": img_path.name, "state": _extract_evaluator_state(evaluator)}


class SegmentationInferencer:
    def __init__(self, cfg: InferConfig):
        self.cfg = cfg

    @staticmethod
    def _to_serializable(value):
        if isinstance(value, dict):
            return {k: SegmentationInferencer._to_serializable(v) for k, v in value.items()}
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, list):
            return [SegmentationInferencer._to_serializable(v) for v in value]
        return value

    @staticmethod
    def _build_panel(image: np.ndarray, gt_rgb: np.ndarray, pred_rgb: np.ndarray) -> Image.Image:
        gt_overlay = blend_overlay(image, gt_rgb)
        pred_overlay = blend_overlay(image, pred_rgb)
        panel = np.concatenate([image, gt_rgb, pred_rgb, pred_overlay, gt_overlay], axis=1)
        return Image.fromarray(panel)

    @staticmethod
    def _merge_evaluator_state(evaluator: SegEvaluator, state: dict) -> None:
        evaluator._confusion_matrix += state["confusion_matrix"]
        evaluator._image_tp += state["image_tp"]
        evaluator._image_fp += state["image_fp"]
        evaluator._image_fn += state["image_fn"]
        evaluator._num_images += state["num_images"]

        evaluator._crack_component_gt_total += state["crack_component_gt_total"]
        evaluator._crack_component_gt_hit += state["crack_component_gt_hit"]
        evaluator._crack_component_pred_total += state["crack_component_pred_total"]
        evaluator._crack_component_pred_hit += state["crack_component_pred_hit"]

        evaluator._crack_boundary_pred_hit += state["crack_boundary_pred_hit"]
        evaluator._crack_boundary_pred_total += state["crack_boundary_pred_total"]
        evaluator._crack_boundary_gt_hit += state["crack_boundary_gt_hit"]
        evaluator._crack_boundary_gt_total += state["crack_boundary_gt_total"]

        evaluator._crack_skeleton_pred_hit += state["crack_skeleton_pred_hit"]
        evaluator._crack_skeleton_pred_total += state["crack_skeleton_pred_total"]
        evaluator._crack_skeleton_gt_hit += state["crack_skeleton_gt_hit"]
        evaluator._crack_skeleton_gt_total += state["crack_skeleton_gt_total"]

    @torch.no_grad()
    def _evaluate_tiled(
        self,
        model,
        img_dir: Path,
        ann_dir: Path,
        device: torch.device,
        num_classes: int,
        input_size: int,
        class_names: tuple[str, ...],
    ) -> dict:
        """原图 tiling 推理 + 评估，pred 和 GT 均在原始分辨率下对比。"""
        cfg = self.cfg
        evaluator  = SegEvaluator(num_classes=num_classes, class_names=class_names)
        img_paths  = sorted(img_dir.glob("*.jpg"))
        total      = len(img_paths)
        tta_info = " + TTA" if cfg.use_tta else ""
        logger.info(f"tiling 评估: {total} 张原图 (input_size={input_size}{tta_info})")
        for i, img_path in enumerate(img_paths, 1):
            image_np = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
            ann_path = ann_dir / f"{img_path.stem}.png"
            if not ann_path.exists():
                logger.warning(f"缺少 GT mask，跳过: {img_path.name}")
                continue
            gt_mask  = np.array(Image.open(ann_path).convert("L"), dtype=np.uint8)
            pred_mask = tiled_predict(
                model,
                image_np,
                device,
                num_classes,
                input_size,
                use_tta=cfg.use_tta,
            )
            evaluator.update(pred_mask[np.newaxis], gt_mask[np.newaxis])
            if i % 50 == 0 or i == total:
                logger.info(f"  {i}/{total}")
        metrics = evaluator.compute()
        evaluator.print_table(metrics)
        evaluator.print_task_report(metrics)
        logger.info(f"评估摘要: {evaluator.summary(metrics)}")
        return metrics

    def _evaluate_tiled_parallel(
        self,
        ckpt_path: Path,
        img_dir: Path,
        ann_dir: Path,
        num_classes: int,
        input_size: int,
        class_names: tuple[str, ...],
    ) -> dict:
        """CPU 原图 tiling 并行评估：按原图分发到多个 worker 进程。"""
        cfg = self.cfg
        evaluator = SegEvaluator(num_classes=num_classes, class_names=class_names)
        img_paths = sorted(img_dir.glob("*.jpg"))
        total = len(img_paths)
        workers = min(max(1, int(cfg.num_workers)), max(1, total))
        cpu_count = os.cpu_count() or 1
        threads_per_worker = max(1, cpu_count // workers)
        tta_info = " + TTA" if cfg.use_tta else ""

        logger.info(
            f"CPU 并行 tiling 评估: {total} 张原图, workers={workers}, "
            f"threads/worker={threads_per_worker}, input_size={input_size}{tta_info}"
        )

        tasks = [(str(p), str(ann_dir)) for p in img_paths]
        done = 0
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_tiled_worker,
            initargs=(
                str(ckpt_path),
                num_classes,
                input_size,
                class_names,
                cfg.use_tta,
                threads_per_worker,
            ),
        ) as executor:
            futures = [executor.submit(_evaluate_tiled_image_worker, task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                done += 1
                if result.get("ok"):
                    self._merge_evaluator_state(evaluator, result["state"])
                elif result.get("reason") == "missing_gt":
                    logger.warning(f"缺少 GT mask，跳过: {result.get('image')}")

                if done % 50 == 0 or done == total:
                    logger.info(f"  {done}/{total}")

        metrics = evaluator.compute()
        evaluator.print_table(metrics)
        evaluator.print_task_report(metrics)
        logger.info(f"评估摘要: {evaluator.summary(metrics)}")
        return metrics

    @staticmethod
    @torch.no_grad()
    def _evaluate(model, loader, device: torch.device, class_names: tuple[str, ...]):
        evaluator = SegEvaluator(num_classes=len(class_names), class_names=class_names)
        for batch in loader:
            images = batch[0].to(device, non_blocking=True)
            masks  = batch[1].to(device, non_blocking=True)
            # TMDSSegmentor 在 eval 模式返回单张量，与 TunnelSegmentor 接口一致
            logits = model(images)
            evaluator.update(logits, masks)

        metrics = evaluator.compute()
        evaluator.print_table(metrics)
        evaluator.print_task_report(metrics)
        logger.info(f"评估摘要: {evaluator.summary(metrics)}")
        return metrics

    @classmethod
    @torch.no_grad()
    def _save_visualizations(cls, model, dataset, device: torch.device, output_dir: Path, count: int):
        vis_dir = output_dir / "vis"
        vis_dir.mkdir(parents=True, exist_ok=True)

        total = min(count, len(dataset))
        logger.info(f"开始保存可视化: {total} 张 → {vis_dir}")

        for idx in range(total):
            sample = dataset[idx]
            image, gt_mask = sample[0], sample[1]   # 兼容 2-tuple 和 3-tuple（含 skel_mask）
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
        with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(
                SegmentationInferencer._to_serializable(metrics),
                f,
                ensure_ascii=False,
                indent=2,
            )
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
        ckpt = load_checkpoint_compat(ckpt_path, map_location="cpu")
        if is_quantized_checkpoint(ckpt) and device.type != "cpu":
            logger.warning("检测到量化 checkpoint，infer 自动切换到 CPU 推理。")
            device = torch.device("cpu")

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

        if cfg.use_tiling:
            # 原图 tiling 推理：不经过 DataLoader，直接逐张读取原始分辨率图像
            split_key = "valid" if cfg.split in ("val", "valid") else cfg.split
            img_dir   = Path(cfg.data_root) / "img_dir" / split_key
            ann_dir   = Path(cfg.data_root) / "ann_dir" / split_key
            if device.type == "cpu" and cfg.num_workers > 1:
                metrics = self._evaluate_tiled_parallel(
                    ckpt_path, img_dir, ann_dir,
                    num_classes=NUM_CLASSES,
                    input_size=input_size,
                    class_names=class_names,
                )
            else:
                model, _ = build_segmentor_from_checkpoint(
                    ckpt,
                    device,
                    default_num_classes=NUM_CLASSES,
                    use_backbone_weight_from_cfg=True,
                    use_frozen_stages_from_cfg=True,
                )
                metrics = self._evaluate_tiled(
                    model, img_dir, ann_dir, device,
                    num_classes=NUM_CLASSES,
                    input_size=input_size,
                    class_names=class_names,
                )
        else:
            model, _ = build_segmentor_from_checkpoint(
                ckpt,
                device,
                default_num_classes=NUM_CLASSES,
                use_backbone_weight_from_cfg=True,
                use_frozen_stages_from_cfg=True,
            )
            metrics = self._evaluate(model, loader, device, class_names=class_names)

        self._save_metrics(metrics, out_dir)

        if cfg.save_vis and not cfg.use_tiling:
            self._save_visualizations(model, loader.dataset, device, out_dir, cfg.vis_count)

        logger.success("=" * 70)
        logger.success("推理 / 评估完成")
        logger.success(f"输出目录: {out_dir.resolve()}")
        logger.success("=" * 70)

        return metrics
