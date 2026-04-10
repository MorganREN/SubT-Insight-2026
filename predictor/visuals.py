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
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
        # macOS
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


def build_legend_bar(
    class_names: tuple[str, ...],
    class_colors: tuple[tuple[int, int, int], ...],
    width: int,
    per_class_iou: np.ndarray | None = None,
    present_mask: np.ndarray | None = None,
    items_per_row: int = 4,
    row_h: int = 50,
    swatch_size: int = 28,
    font_size: int = 17,
) -> np.ndarray:
    """底部图例条：色块 + 类别名 + 每类 IoU（可选）。

    Parameters
    ----------
    class_names / class_colors : 类别定义
    width : 与上方面板等宽
    per_class_iou : shape (num_classes,) float [0,1]，传入时显示 IoU
    present_mask  : shape (num_classes,) bool，True 表示该类在 GT 中出现；
                    absent 类显示 "N/A"
    """
    n = len(class_names)
    n_rows = (n + items_per_row - 1) // items_per_row
    pad_v = 8
    legend_h = n_rows * row_h + pad_v * 2

    canvas = Image.new("RGB", (width, legend_h), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    font = _get_title_font(font_size)
    item_w = width // items_per_row

    for i, (name, color) in enumerate(zip(class_names, class_colors)):
        row = i // items_per_row
        col = i % items_per_row
        x0 = col * item_w + 10
        y0 = pad_v + row * row_h + (row_h - swatch_size) // 2

        # 色块
        draw.rectangle(
            [x0, y0, x0 + swatch_size, y0 + swatch_size],
            fill=color,
            outline=(80, 80, 80),
        )

        # 文字
        if per_class_iou is not None:
            absent = present_mask is not None and not present_mask[i]
            iou_str = "N/A" if absent else f"{per_class_iou[i] * 100:.1f}%"
            label = f"{name}: {iou_str}"
        else:
            label = name

        tx = x0 + swatch_size + 6
        ty = pad_v + row * row_h + (row_h - font_size) // 2
        draw.text((tx, ty), label, fill=(20, 20, 20), font=font)

    return np.array(canvas, dtype=np.uint8)


def save_outputs_basic(
    image_path: Path,
    out_dir: Path,
    image: np.ndarray,
    pred_color_mask: np.ndarray,
    pred_overlay: np.ndarray,
    class_names: tuple[str, ...] | None = None,
    class_colors: tuple[tuple[int, int, int], ...] | None = None,
):
    stem = image_path.stem

    mask_path = out_dir / f"{stem}_pred_mask.png"
    overlay_path = out_dir / f"{stem}_pred_overlay.png"
    panel_path = out_dir / f"{stem}_panel.png"

    Image.fromarray(pred_color_mask).save(mask_path)
    Image.fromarray(pred_overlay).save(overlay_path)

    blank = _blank_like(image)
    panel = _build_panel_2x3(
        images=[image, blank, blank, blank, pred_overlay, pred_color_mask],
        titles=[
            "Original",
            "GT Overlay (N/A)",
            "GT Segmented (N/A)",
            "Error (N/A)",
            "Pred Overlay",
            "Pred Segmented",
        ],
    )

    if class_names is not None and class_colors is not None:
        legend = build_legend_bar(class_names, class_colors, width=panel.shape[1])
        panel = np.concatenate([panel, legend], axis=0)

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
    class_names: tuple[str, ...] | None = None,
    class_colors: tuple[tuple[int, int, int], ...] | None = None,
    per_class_iou: np.ndarray | None = None,
    present_mask: np.ndarray | None = None,
    pred_raw_mask: np.ndarray | None = None,
):
    stem = image_path.stem
    panel_path = out_dir / f"{stem}_panel.png"

    if pred_raw_mask is not None:
        Image.fromarray(pred_raw_mask, mode="L").save(out_dir / f"{stem}_pred_mask.png")
        logger.success(f"已保存 mask: {out_dir / f'{stem}_pred_mask.png'}")

    panel = _build_panel_2x3(
        images=[image, gt_overlay, gt_color_mask, error_overlay, pred_overlay, pred_color_mask],
        titles=[
            "Original",
            "GT Overlay",
            "GT Segmented",
            "Error Map (G=OK, R=Err)",
            "Pred Overlay",
            "Pred Segmented",
        ],
    )

    if class_names is not None and class_colors is not None:
        legend = build_legend_bar(
            class_names,
            class_colors,
            width=panel.shape[1],
            per_class_iou=per_class_iou,
            present_mask=present_mask,
        )
        panel = np.concatenate([panel, legend], axis=0)

    Image.fromarray(panel).save(panel_path)
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
