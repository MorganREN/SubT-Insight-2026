"""
Run GPU/CPU inference for all *_small best checkpoints on the Roboflow Tongji
raw dataset, with all leakage predictions merged into one leakage class before
metric computation.

Evaluation label space:
    0 background
    1 crack
    2 leakage              <- model labels 2/3/4 merged here
    3 lining_falling_off   <- not present in Roboflow data (0 GT pixels)
    4 segment_damage       <- Roboflow spalling

Run:
    conda run -n subt-2026 python scripts/infer_roboflow_small_models_merged.py --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from loguru import logger

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

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

MERGED_CLASS_NAMES = (
    "background",
    "crack",
    "leakage",
    "lining_falling_off",
    "segment_damage",
)


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


def _remap_to_merged(mask: np.ndarray) -> np.ndarray:
    out = np.full_like(mask, 255, dtype=np.uint8)
    out[mask == 0] = 0
    out[mask == 1] = 1
    out[(mask == 2) | (mask == 3) | (mask == 4)] = 2
    out[mask == 5] = 3
    out[mask == 6] = 4
    return out


def _update_confusion(cm: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> None:
    valid = gt != 255
    pred = np.clip(pred[valid].astype(np.int64), 0, cm.shape[0] - 1)
    gt = np.clip(gt[valid].astype(np.int64), 0, cm.shape[0] - 1)
    if gt.size == 0:
        return
    idx = cm.shape[0] * gt + pred
    cm += np.bincount(idx, minlength=cm.size).reshape(cm.shape)


def _compute_metrics(cm: np.ndarray) -> dict:
    cm_f = cm.astype(np.float64)
    tp = np.diag(cm_f)
    gt_sum = cm_f.sum(axis=1)
    pred_sum = cm_f.sum(axis=0)
    union = gt_sum + pred_sum - tp

    iou = tp / np.maximum(union, 1.0)
    recall = tp / np.maximum(gt_sum, 1.0)
    precision = tp / np.maximum(pred_sum, 1.0)
    dice = 2.0 * tp / np.maximum(gt_sum + pred_sum, 1.0)

    present = gt_sum > 0
    fg_present = present.copy()
    fg_present[0] = False

    per_class = {
        name: {
            "IoU": float(iou[idx]),
            "Recall": float(recall[idx]),
            "Precision": float(precision[idx]),
            "Dice": float(dice[idx]),
            "gt_pixels": int(gt_sum[idx]),
            "pred_pixels": int(pred_sum[idx]),
        }
        for idx, name in enumerate(MERGED_CLASS_NAMES)
    }

    return {
        "class_names": MERGED_CLASS_NAMES,
        "confusion_matrix": cm,
        "per_class": per_class,
        "mIoU": float(np.mean(iou[present])) if present.any() else 0.0,
        "mDice": float(np.mean(dice[present])) if present.any() else 0.0,
        "mRecall": float(np.mean(recall[present])) if present.any() else 0.0,
        "mPrecision": float(np.mean(precision[present])) if present.any() else 0.0,
        "mIoU_fg": float(np.mean(iou[fg_present])) if fg_present.any() else 0.0,
        "mDice_fg": float(np.mean(dice[fg_present])) if fg_present.any() else 0.0,
        "aAcc": float(tp.sum() / max(gt_sum.sum(), 1.0)),
    }


def _summary_row(model_name: str, metrics: dict, out_dir: Path) -> dict:
    pc = metrics["per_class"]
    return {
        "model": model_name,
        "output_dir": str(out_dir),
        "mIoU": metrics["mIoU"],
        "mIoU_fg": metrics["mIoU_fg"],
        "mDice": metrics["mDice"],
        "mDice_fg": metrics["mDice_fg"],
        "aAcc": metrics["aAcc"],
        "background_IoU": pc["background"]["IoU"],
        "crack_IoU": pc["crack"]["IoU"],
        "leakage_IoU": pc["leakage"]["IoU"],
        "lining_falling_off_IoU": pc["lining_falling_off"]["IoU"],
        "segment_damage_IoU": pc["segment_damage"]["IoU"],
        "crack_recall": pc["crack"]["Recall"],
        "leakage_recall": pc["leakage"]["Recall"],
        "lining_falling_off_recall": pc["lining_falling_off"]["Recall"],
        "segment_damage_recall": pc["segment_damage"]["Recall"],
        "crack_precision": pc["crack"]["Precision"],
        "leakage_precision": pc["leakage"]["Precision"],
        "lining_falling_off_precision": pc["lining_falling_off"]["Precision"],
        "segment_damage_precision": pc["segment_damage"]["Precision"],
    }


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
    out_dir = output_root / f"infer_merged_{model_name}_{split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(str(out_dir / "infer.log"))

    logger.info("=" * 70)
    logger.info(f"Roboflow merged leakage inference: {model_name}")
    logger.info(f"checkpoint: {ckpt_path}")
    logger.info(f"data_root: {data_root}, split={split}, device={device}")

    ckpt = load_checkpoint_compat(ckpt_path, map_location="cpu")
    input_size = get_input_size_from_checkpoint(ckpt, default=512)
    model, _cfg = build_segmentor_from_checkpoint(
        ckpt,
        device,
        default_num_classes=7,
        use_backbone_weight_from_cfg=True,
        use_frozen_stages_from_cfg=True,
    )

    img_dir = data_root / "img_dir" / split
    ann_dir = data_root / "ann_dir" / split
    img_paths = sorted(img_dir.glob("*.jpg"))
    if not img_paths:
        raise FileNotFoundError(f"No images found: {img_dir}")

    cm = np.zeros((len(MERGED_CLASS_NAMES), len(MERGED_CLASS_NAMES)), dtype=np.int64)
    for idx, img_path in enumerate(img_paths, 1):
        ann_path = ann_dir / f"{img_path.stem}.png"
        if not ann_path.exists():
            logger.warning(f"missing mask, skip: {ann_path}")
            continue
        image_np = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        gt = np.array(Image.open(ann_path).convert("L"), dtype=np.uint8)

        pred_7 = tiled_predict(
            model,
            image_np,
            device,
            num_classes=7,
            input_size=input_size,
            patch_batch_size=patch_batch_size,
            use_tta=use_tta,
        )
        pred = _remap_to_merged(pred_7)
        gt_merged = _remap_to_merged(gt)
        _update_confusion(cm, pred, gt_merged)

        if idx % 50 == 0 or idx == len(img_paths):
            logger.info(f"  {idx}/{len(img_paths)}")

    metrics = _compute_metrics(cm)
    with open(out_dir / "metrics_merged_leakage.json", "w", encoding="utf-8") as f:
        json.dump(_to_serializable(metrics), f, ensure_ascii=False, indent=2)

    logger.info(
        "summary: "
        f"aAcc={metrics['aAcc']*100:.2f}% "
        f"mIoU={metrics['mIoU']*100:.2f}% "
        f"fg_mIoU={metrics['mIoU_fg']*100:.2f}% "
        f"crack_IoU={metrics['per_class']['crack']['IoU']*100:.2f}% "
        f"leakage_IoU={metrics['per_class']['leakage']['IoU']*100:.2f}% "
        f"spalling_IoU={metrics['per_class']['segment_damage']['IoU']*100:.2f}%"
    )
    logger.success(f"output: {out_dir.resolve()}")
    return _summary_row(model_name, metrics, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all small checkpoints on Roboflow with merged leakage metrics.")
    parser.add_argument("--data_root", type=Path, default=Path("dataset/roboflow_tongji_tunnel_raw_spalling_as_segment_damage"))
    parser.add_argument("--split", default="train", choices=["train", "valid", "test"])
    parser.add_argument("--output_root", type=Path, default=Path("outputs/roboflow_tongji_small_spalling_as_segment_damage"))
    parser.add_argument("--ckpt", type=Path, default=None, help="Evaluate a single checkpoint instead of the built-in four small models.")
    parser.add_argument("--model_name", default=None, help="Name for --ckpt results.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--patch_batch_size", type=int, default=4)
    parser.add_argument("--use_tta", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.device, allow_mps=False, warn_mps_on_auto=False)
    model_paths = (
        {args.model_name or args.ckpt.parent.name: args.ckpt}
        if args.ckpt is not None
        else SMALL_MODELS
    )
    rows = []
    for model_name, ckpt_path in model_paths.items():
        if not ckpt_path.exists():
            raise FileNotFoundError(f"missing checkpoint: {ckpt_path}")
        rows.append(
            _evaluate_one_model(
                model_name,
                ckpt_path,
                args.data_root,
                args.split,
                args.output_root,
                device,
                args.patch_batch_size,
                args.use_tta,
            )
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_json = args.output_root / f"roboflow_{args.split}_small_merged_summary.json"
    summary_csv = args.output_root / f"roboflow_{args.split}_small_merged_summary.csv"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Summary JSON: {summary_json}")
    print(f"Summary CSV:  {summary_csv}")


if __name__ == "__main__":
    main()
