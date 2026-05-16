"""
Find and visualize a Roboflow crack partial-label sample where annotated crack
pixels are predicted as leakage_b.

The output panel shows:
    original | GT crack overlay | prediction overlay | GT-crack pixels predicted leakage_b
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from dataload import CLASS_COLORS, NUM_CLASSES
from predictor.tiling import tiled_predict
from utils.runtime import load_checkpoint_compat, resolve_device
from utils.segmentor_loader import (
    build_segmentor_from_checkpoint,
    get_input_size_from_checkpoint,
)


def _overlay(image: np.ndarray, mask_rgb: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    active = mask_rgb.sum(axis=2) > 0
    out = image.copy()
    out[active] = (image[active] * (1.0 - alpha) + mask_rgb[active] * alpha).astype(np.uint8)
    return out


def _color_mask(mask: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for idx, color in enumerate(CLASS_COLORS):
        rgb[mask == idx] = color
    return rgb


def _label_panel(panel: Image.Image, labels: list[str], tile_w: int) -> None:
    draw = ImageDraw.Draw(panel)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    for i, label in enumerate(labels):
        x = i * tile_w + 10
        draw.rectangle((x - 4, 8, x + 330, 36), fill=(0, 0, 0))
        draw.text((x, 12), label, fill=(255, 255, 255), font=font)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Find crack pixels predicted as leakage_b and save a visual example.")
    parser.add_argument("--ckpt", type=Path, default=Path("outputs/ablation_benchmark_small/best.pth"))
    parser.add_argument("--data_root", type=Path, default=Path("dataset/roboflow_tunnel_crack_partial_raw"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/roboflow_tunnel_crack_small/examples"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--patch_batch_size", type=int, default=2)
    parser.add_argument("--model_name", default="benchmark_small")
    args = parser.parse_args()

    device = resolve_device(args.device, allow_mps=False, warn_mps_on_auto=False)
    ckpt = load_checkpoint_compat(args.ckpt, map_location="cpu")
    input_size = get_input_size_from_checkpoint(ckpt, default=512)
    model, _cfg = build_segmentor_from_checkpoint(
        ckpt,
        device,
        default_num_classes=NUM_CLASSES,
        use_backbone_weight_from_cfg=True,
        use_frozen_stages_from_cfg=True,
    )

    img_dir = args.data_root / "img_dir" / args.split
    ann_dir = args.data_root / "ann_dir" / args.split
    img_paths = sorted(img_dir.glob("*.jpg"))
    if not img_paths:
        raise FileNotFoundError(img_dir)

    best = None
    rows = []
    for idx, img_path in enumerate(img_paths, 1):
        ann_path = ann_dir / f"{img_path.stem}.png"
        if not ann_path.exists():
            continue
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        gt = np.array(Image.open(ann_path).convert("L"), dtype=np.uint8)
        crack_gt = gt == 1
        gt_pixels = int(crack_gt.sum())
        if gt_pixels == 0:
            continue
        pred = tiled_predict(
            model,
            image,
            device,
            num_classes=NUM_CLASSES,
            input_size=input_size,
            patch_batch_size=args.patch_batch_size,
        )
        leakage_b_pixels = int(((pred == 2) & crack_gt).sum())
        crack_pixels = int(((pred == 1) & crack_gt).sum())
        bg_pixels = int(((pred == 0) & crack_gt).sum())
        leakage_b_ratio = leakage_b_pixels / gt_pixels
        row = {
            "image": img_path.name,
            "gt_crack_pixels": gt_pixels,
            "gt_crack_pred_as_leakage_b_pixels": leakage_b_pixels,
            "gt_crack_pred_as_leakage_b_ratio": leakage_b_ratio,
            "gt_crack_pred_as_crack_pixels": crack_pixels,
            "gt_crack_pred_as_crack_ratio": crack_pixels / gt_pixels,
            "gt_crack_pred_as_background_pixels": bg_pixels,
            "gt_crack_pred_as_background_ratio": bg_pixels / gt_pixels,
        }
        rows.append(row)
        if best is None or leakage_b_ratio > best["row"]["gt_crack_pred_as_leakage_b_ratio"]:
            best = {"row": row, "image": image, "gt": gt, "pred": pred, "path": img_path}
        if idx % 50 == 0 or idx == len(img_paths):
            print(f"{idx}/{len(img_paths)} best_leakage_b={best['row']['gt_crack_pred_as_leakage_b_ratio']*100:.2f}%")

    if best is None:
        raise RuntimeError("No annotated crack sample found.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{args.model_name}_crack_to_leakage_b_candidates.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["gt_crack_pred_as_leakage_b_ratio"], reverse=True))

    image = best["image"]
    gt = best["gt"]
    pred = best["pred"]
    crack_gt = gt == 1

    gt_rgb = np.zeros_like(image)
    gt_rgb[crack_gt] = CLASS_COLORS[1]
    pred_rgb = _color_mask(pred)
    leakage_b_on_crack = np.zeros_like(image)
    leakage_b_on_crack[(pred == 2) & crack_gt] = CLASS_COLORS[2]

    panels = [
        image,
        _overlay(image, gt_rgb),
        _overlay(image, pred_rgb),
        _overlay(image, leakage_b_on_crack, alpha=0.75),
    ]
    panel_arr = np.concatenate(panels, axis=1)
    panel = Image.fromarray(panel_arr)
    _label_panel(
        panel,
        [
            "Original",
            "GT crack",
            f"Pred {args.model_name}",
            "GT crack -> leakage_b",
        ],
        image.shape[1],
    )
    out_path = args.output_dir / f"{args.model_name}_crack_to_leakage_b_example.png"
    panel.save(out_path)

    print("best_image", best["path"])
    print("panel", out_path)
    print("candidates_csv", csv_path)
    print(best["row"])


if __name__ == "__main__":
    main()
