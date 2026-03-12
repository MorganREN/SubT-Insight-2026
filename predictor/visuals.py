from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from loguru import logger

from utils.segmentation_vis import normalize_image


def preprocess_image(image_path: Path, input_size: int) -> tuple[np.ndarray, torch.Tensor]:
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image, dtype=np.uint8)

    resized = image.resize((input_size, input_size), Image.BILINEAR)
    resized_np = np.array(resized, dtype=np.uint8)
    normalized = normalize_image(resized_np)
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).float()

    return image_np, tensor


def postprocess_mask(pred_mask: np.ndarray, original_hw: tuple[int, int]) -> np.ndarray:
    h, w = original_hw
    mask_img = Image.fromarray(pred_mask.astype(np.uint8), mode="L")
    mask_img = mask_img.resize((w, h), Image.NEAREST)
    return np.array(mask_img, dtype=np.uint8)


def _get_title_font(font_size: int) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    for path in candidates:
        p = Path(path)
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size=font_size)
            except Exception:
                continue
    return ImageFont.load_default()


def _resize_tile(image: np.ndarray, target_w: int = 360) -> np.ndarray:
    h, w = image.shape[:2]
    if w <= target_w:
        return image
    scale = target_w / float(w)
    new_h = max(1, int(round(h * scale)))
    resized = Image.fromarray(image).resize((target_w, new_h), Image.BILINEAR)
    return np.array(resized, dtype=np.uint8)


def _add_title_bar(image: np.ndarray, title: str, title_h: int = 40, font_size: int = 22) -> np.ndarray:
    h, w = image.shape[:2]
    canvas = Image.new("RGB", (w, h + title_h), (255, 255, 255))
    canvas.paste(Image.fromarray(image), (0, title_h))

    draw = ImageDraw.Draw(canvas)
    font = _get_title_font(font_size)
    text_bbox = draw.textbbox((0, 0), title, font=font)
    tw = text_bbox[2] - text_bbox[0]
    th = text_bbox[3] - text_bbox[1]
    tx = max((w - tw) // 2, 2)
    ty = max((title_h - th) // 2, 1)
    draw.text((tx, ty), title, fill=(20, 20, 20), font=font)

    return np.array(canvas)


def _blank_like(image: np.ndarray, value: int = 235) -> np.ndarray:
    return np.full_like(image, fill_value=value, dtype=np.uint8)


def _build_panel_2x3(images: list[np.ndarray], titles: list[str]) -> np.ndarray:
    assert len(images) == 6 and len(titles) == 6

    resized = [_resize_tile(img, target_w=360) for img in images]
    min_h = min(img.shape[0] for img in resized)
    resized = [img[:min_h, :, :] if img.shape[0] != min_h else img for img in resized]

    titled = [_add_title_bar(img, title, title_h=40, font_size=22) for img, title in zip(resized, titles)]
    row1 = np.concatenate(titled[:3], axis=1)
    row2 = np.concatenate(titled[3:], axis=1)
    return np.concatenate([row1, row2], axis=0)


def build_error_overlay(
    image: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    ignore_index: int = 255,
    alpha: float = 0.55,
) -> np.ndarray:
    valid = gt_mask != ignore_index
    correct = (pred_mask == gt_mask) & valid
    wrong = (pred_mask != gt_mask) & valid

    out = image.astype(np.float32).copy()
    out[correct] = out[correct] * (1.0 - alpha) + np.array([0, 255, 0], dtype=np.float32) * alpha
    out[wrong] = out[wrong] * (1.0 - alpha) + np.array([255, 0, 0], dtype=np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def save_outputs_basic(
    image_path: Path,
    out_dir: Path,
    image: np.ndarray,
    pred_color_mask: np.ndarray,
    pred_overlay: np.ndarray,
):
    stem = image_path.stem

    mask_path = out_dir / f"{stem}_pred_mask.png"
    overlay_path = out_dir / f"{stem}_pred_overlay.png"
    panel_path = out_dir / f"{stem}_panel.png"

    Image.fromarray(pred_color_mask).save(mask_path)
    Image.fromarray(pred_overlay).save(overlay_path)

    blank = _blank_like(image)
    panel = _build_panel_2x3(
        images=[
            image,
            blank,
            blank,
            blank,
            pred_overlay,
            pred_color_mask,
        ],
        titles=[
            "Original",
            "GT Overlay (N/A)",
            "GT Segmented (N/A)",
            "Error (N/A)",
            "Pred Overlay",
            "Pred Segmented",
        ],
    )
    Image.fromarray(panel).save(panel_path)

    logger.success(f"已保存 mask: {mask_path}")
    logger.success(f"已保存 overlay: {overlay_path}")
    logger.success(f"已保存 panel: {panel_path}")


def save_outputs_with_gt(
    image_path: Path,
    out_dir: Path,
    image: np.ndarray,
    gt_color_mask: np.ndarray,
    gt_overlay: np.ndarray,
    pred_color_mask: np.ndarray,
    pred_overlay: np.ndarray,
    error_overlay: np.ndarray,
):
    stem = image_path.stem

    gt_mask_path = out_dir / f"{stem}_gt_mask.png"
    gt_overlay_path = out_dir / f"{stem}_gt_overlay.png"
    pred_mask_path = out_dir / f"{stem}_pred_mask.png"
    pred_overlay_path = out_dir / f"{stem}_pred_overlay.png"
    error_path = out_dir / f"{stem}_error_overlay.png"
    panel_path = out_dir / f"{stem}_panel.png"

    Image.fromarray(gt_color_mask).save(gt_mask_path)
    Image.fromarray(gt_overlay).save(gt_overlay_path)
    Image.fromarray(pred_color_mask).save(pred_mask_path)
    Image.fromarray(pred_overlay).save(pred_overlay_path)
    Image.fromarray(error_overlay).save(error_path)

    panel = _build_panel_2x3(
        images=[
            image,
            gt_overlay,
            gt_color_mask,
            error_overlay,
            pred_overlay,
            pred_color_mask,
        ],
        titles=[
            "Original",
            "GT Overlay",
            "GT Segmented",
            "Error Map (G=OK, R=Err)",
            "Pred Overlay",
            "Pred Segmented",
        ],
    )
    Image.fromarray(panel).save(panel_path)

    logger.success(f"已保存 GT mask: {gt_mask_path}")
    logger.success(f"已保存 GT overlay: {gt_overlay_path}")
    logger.success(f"已保存 Pred mask: {pred_mask_path}")
    logger.success(f"已保存 Pred overlay: {pred_overlay_path}")
    logger.success(f"已保存 Error overlay: {error_path}")
    logger.success(f"已保存 6联 panel: {panel_path}")


def infer_mask_path(image_path: Path) -> Path | None:
    parts = list(image_path.parts)
    if "img_dir" not in parts:
        return None
    idx = parts.index("img_dir")
    mask_parts = parts.copy()
    mask_parts[idx] = "ann_dir"
    mask_path = Path(*mask_parts).with_suffix(".png")
    return mask_path if mask_path.exists() else None


def load_gt_mask(mask_path: Path, hw: tuple[int, int]) -> np.ndarray:
    h, w = hw
    mask = Image.open(mask_path).convert("L")
    if mask.size != (w, h):
        mask = mask.resize((w, h), Image.NEAREST)
    return np.array(mask, dtype=np.uint8)
