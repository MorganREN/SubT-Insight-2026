"""
Evaluate the four small best checkpoints on the Roboflow tunnel-crack partial
label dataset.

The converted dataset uses:
    1   = annotated crack pixels
    255 = ignore / unlabeled pixels

Because all non-crack pixels are ignore, this script reports crack coverage on
annotated crack pixels. False positives outside the annotated crack regions are
not counted and precision is therefore intentionally omitted.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from PIL import Image

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from dataload import CLASS_NAMES, NUM_CLASSES
from predictor.tiling import tiled_predict
from utils.runtime import load_checkpoint_compat, resolve_device, setup_logger
from utils.segmentor_loader import (
    build_segmentor_from_checkpoint,
    get_input_size_from_checkpoint,
)


SMALL_MODELS = {
    "benchmark_small": Path("outputs/ablation_benchmark_small/best.pth"),
    "tmds_full_small": Path("outputs/ablation_tmds_full_small/best.pth"),
    "tmds_no_cmim_small": Path("outputs/ablation_tmds_no_cmim_small/best.pth"),
    "tmds_no_routing_small": Path("outputs/ablation_tmds_no_routing_small/best.pth"),
}


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


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


@torch.no_grad()
def _evaluate_one_model(
    model_name: str,
    ckpt_path: Path,
    data_root: Path,
    split: str,
    output_root: Path,
    device: torch.device,
    patch_batch_size: int,
    use_tta: bool,
) -> dict:
    out_dir = output_root / f"infer_crack_partial_{model_name}_{split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(str(out_dir / "infer.log"))

    logger.info("=" * 70)
    logger.info(f"Roboflow crack partial inference: {model_name}")
    logger.info(f"checkpoint: {ckpt_path}")
    logger.info(f"data_root: {data_root}, split={split}, device={device}")

    ckpt = load_checkpoint_compat(ckpt_path, map_location="cpu")
    input_size = get_input_size_from_checkpoint(ckpt, default=512)
    model, _cfg = build_segmentor_from_checkpoint(
        ckpt,
        device,
        default_num_classes=NUM_CLASSES,
        use_backbone_weight_from_cfg=True,
        use_frozen_stages_from_cfg=True,
    )

    img_dir = data_root / "img_dir" / split
    ann_dir = data_root / "ann_dir" / split
    img_paths = sorted(img_dir.glob("*.jpg"))
    if not img_paths:
        raise FileNotFoundError(f"No images found: {img_dir}")

    total_crack_pixels = 0
    hit_crack_pixels = 0
    images_with_crack = 0
    images_with_crack_hit = 0
    pred_on_crack_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    per_image_rows: list[dict] = []

    for idx, img_path in enumerate(img_paths, 1):
        ann_path = ann_dir / f"{img_path.stem}.png"
        if not ann_path.exists():
            logger.warning(f"missing mask, skip: {ann_path}")
            continue

        image_np = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        gt = np.array(Image.open(ann_path).convert("L"), dtype=np.uint8)
        crack_gt = gt == 1
        gt_pixels = int(crack_gt.sum())
        if gt_pixels == 0:
            logger.warning(f"no annotated crack pixels, skip metrics for: {img_path.name}")
            continue

        pred = tiled_predict(
            model,
            image_np,
            device,
            num_classes=NUM_CLASSES,
            input_size=input_size,
            patch_batch_size=patch_batch_size,
            use_tta=use_tta,
        )

        pred_on_crack = pred[crack_gt]
        pred_hist = np.bincount(pred_on_crack.astype(np.int64), minlength=NUM_CLASSES)[:NUM_CLASSES]
        crack_hits = int(pred_hist[1])

        total_crack_pixels += gt_pixels
        hit_crack_pixels += crack_hits
        images_with_crack += 1
        if crack_hits > 0:
            images_with_crack_hit += 1
        pred_on_crack_counts += pred_hist

        per_image_rows.append(
            {
                "image": img_path.name,
                "gt_crack_pixels": gt_pixels,
                "pred_crack_on_gt_pixels": crack_hits,
                "crack_pixel_recall": _safe_div(crack_hits, gt_pixels),
            }
        )

        if idx % 50 == 0 or idx == len(img_paths):
            logger.info(f"  {idx}/{len(img_paths)}")

    class_distribution_on_gt_crack = {
        CLASS_NAMES[i]: _safe_div(int(pred_on_crack_counts[i]), total_crack_pixels)
        for i in range(NUM_CLASSES)
    }
    class_pixels_on_gt_crack = {
        CLASS_NAMES[i]: int(pred_on_crack_counts[i])
        for i in range(NUM_CLASSES)
    }

    crack_pixel_recall = _safe_div(hit_crack_pixels, total_crack_pixels)
    metrics = {
        "model": model_name,
        "checkpoint": str(ckpt_path),
        "data_root": str(data_root),
        "split": split,
        "num_images": len(img_paths),
        "images_with_annotated_crack": images_with_crack,
        "gt_crack_pixels": int(total_crack_pixels),
        "pred_crack_on_gt_pixels": int(hit_crack_pixels),
        "crack_pixel_recall": crack_pixel_recall,
        "crack_iou_on_annotated_region": crack_pixel_recall,
        "image_level_crack_hit_rate": _safe_div(images_with_crack_hit, images_with_crack),
        "pred_class_pixels_on_gt_crack": class_pixels_on_gt_crack,
        "pred_class_distribution_on_gt_crack": class_distribution_on_gt_crack,
        "note": "Non-crack pixels are 255 ignore, so false positives and precision are not measurable on this partial-label dataset.",
    }

    with open(out_dir / "metrics_crack_partial.json", "w", encoding="utf-8") as f:
        json.dump(_to_serializable(metrics), f, ensure_ascii=False, indent=2)

    with open(out_dir / "per_image_crack_partial.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_image_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_image_rows)

    logger.info(
        "summary: "
        f"crack_pixel_recall={crack_pixel_recall*100:.2f}% "
        f"image_hit_rate={metrics['image_level_crack_hit_rate']*100:.2f}% "
        f"gt_crack_pixels={total_crack_pixels}"
    )
    logger.success(f"output: {out_dir.resolve()}")
    return metrics


def _summary_row(metrics: dict, out_dir: Path) -> dict:
    dist = metrics["pred_class_distribution_on_gt_crack"]
    return {
        "model": metrics["model"],
        "output_dir": str(out_dir / f"infer_crack_partial_{metrics['model']}_{metrics['split']}"),
        "num_images": metrics["num_images"],
        "gt_crack_pixels": metrics["gt_crack_pixels"],
        "crack_pixel_recall": metrics["crack_pixel_recall"],
        "crack_iou_on_annotated_region": metrics["crack_iou_on_annotated_region"],
        "image_level_crack_hit_rate": metrics["image_level_crack_hit_rate"],
        "gt_crack_pred_as_background": dist["background"],
        "gt_crack_pred_as_crack": dist["crack"],
        "gt_crack_pred_as_leakage_b": dist["leakage_b"],
        "gt_crack_pred_as_leakage_w": dist["leakage_w"],
        "gt_crack_pred_as_leakage_g": dist["leakage_g"],
        "gt_crack_pred_as_lining_falling_off": dist["lining_falling_off"],
        "gt_crack_pred_as_segment_damage": dist["segment_damage"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate four small best checkpoints on Roboflow crack partial labels.")
    parser.add_argument("--data_root", type=Path, default=Path("dataset/roboflow_tunnel_crack_partial_raw"))
    parser.add_argument("--split", default="train", choices=["train", "valid", "test"])
    parser.add_argument("--output_root", type=Path, default=Path("outputs/roboflow_tunnel_crack_small"))
    parser.add_argument("--ckpt", type=Path, default=None, help="Evaluate a single checkpoint instead of the built-in four small models.")
    parser.add_argument("--model_name", default=None, help="Name for --ckpt results.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--patch_batch_size", type=int, default=2)
    parser.add_argument("--use_tta", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.device, allow_mps=False, warn_mps_on_auto=False)
    args.output_root.mkdir(parents=True, exist_ok=True)
    model_paths = (
        {args.model_name or args.ckpt.parent.name: args.ckpt}
        if args.ckpt is not None
        else SMALL_MODELS
    )

    metrics_all = []
    rows = []
    for model_name, ckpt_path in model_paths.items():
        if not ckpt_path.exists():
            raise FileNotFoundError(f"missing checkpoint: {ckpt_path}")
        metrics = _evaluate_one_model(
            model_name,
            ckpt_path,
            args.data_root,
            args.split,
            args.output_root,
            device,
            args.patch_batch_size,
            args.use_tta,
        )
        metrics_all.append(metrics)
        rows.append(_summary_row(metrics, args.output_root))

    summary_json = args.output_root / f"roboflow_crack_{args.split}_small_summary.json"
    summary_csv = args.output_root / f"roboflow_crack_{args.split}_small_summary.csv"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(metrics_all), f, ensure_ascii=False, indent=2)
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Summary JSON: {summary_json}")
    print(f"Summary CSV:  {summary_csv}")


if __name__ == "__main__":
    main()
