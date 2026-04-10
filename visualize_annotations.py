"""
visualize_annotations.py

在原始分辨率下，将 dataset/tongji 的 LabelMe 标注叠加到原图上，
输出 mask overlay 到 dataset/tongji/visualization/。

运行方式:
    python visualize_annotations.py [--workers N]
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


SOURCE_DIR = Path("dataset/tongji")
VIS_DIR    = SOURCE_DIR / "visualization"
OVERLAY_INTENSITY = 115   # 与 dataset_convert.py 一致


CLASS_DEFECTS = {
    1: "crack",
    2: "leakage_b",
    3: "leakage_w",
    4: "leakage_g",
    5: "lining_falling_off",
    6: "segment_damage",
}

CLASS_COLORS_RGB: dict[int, tuple[int, int, int]] = {
    1: (255,   0,   0),
    2: (255, 128,   0),
    3: (  0,   0, 255),
    4: (  0, 255, 255),
    5: (255, 255,   0),
    6: (255,   0, 255),
}

LABEL_TO_CLASS: dict[str, int] = {
    "crack": 1,
    "leakageb": 2,
    "leakagew": 3,
    "leakageg": 4,
    "liningfallingoff": 5,
    "segmentdamage": 6,
    "cracka": 1, "crackb": 1, "cracksmkjm": 1,
    "leakage": 3, "lf": 3,
    "spalling": 5, "ss": 5,
    "repair": 6, "repairs": 6, "other": 6,
}


# ──────────────────────────────────────────────────────────────────────────────
# 标签映射
# ──────────────────────────────────────────────────────────────────────────────

def _normalize(label: str) -> str:
    norm = label.strip().lower()
    for ch in (" ", "_", "-", "/"):
        norm = norm.replace(ch, "")
    return norm


def map_label(label_raw: str) -> int | None:
    label = _normalize(label_raw)
    if label in LABEL_TO_CLASS:
        return LABEL_TO_CLASS[label]
    if "crack"   in label: return 1
    if label.startswith("leakage"):
        if label.endswith("b"): return 2
        if label.endswith("g"): return 4
        return 3
    if "lining" in label and "off" in label: return 5
    if "segment" in label and "damage" in label: return 6
    return None


# ──────────────────────────────────────────────────────────────────────────────
# 图像加载
# ──────────────────────────────────────────────────────────────────────────────

def _find_image(stem: str, data: dict) -> Path | None:
    for ext in (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"):
        p = SOURCE_DIR / f"{stem}{ext}"
        if p.exists():
            return p
    img_path = data.get("imagePath", "")
    if img_path:
        cand = SOURCE_DIR / Path(img_path).name
        if cand.exists():
            return cand
    return None


def _load_image(stem: str, data: dict) -> Image.Image:
    img_path = _find_image(stem, data)
    if img_path is not None:
        img = Image.open(img_path).convert("RGB")
        return ImageOps.exif_transpose(img)
    b64 = data.get("imageData")
    if b64:
        img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        return ImageOps.exif_transpose(img)
    raise FileNotFoundError(f"无法找到图像: {stem}")


# ──────────────────────────────────────────────────────────────────────────────
# Mask 绘制
# ──────────────────────────────────────────────────────────────────────────────

def _draw_shape(draw: ImageDraw.ImageDraw, shape: dict, cls_id: int) -> None:
    points     = shape.get("points", [])
    shape_type = str(shape.get("shape_type", "polygon")).lower()
    if not points:
        return

    if shape_type == "rectangle" and len(points) >= 2:
        (x1, y1), (x2, y2) = points[:2]
        draw.rectangle([x1, y1, x2, y2], fill=cls_id)
        return

    if shape_type == "circle" and len(points) >= 2:
        (cx, cy), (px, py) = points[:2]
        r = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=cls_id)
        return

    if len(points) >= 3:
        draw.polygon([(p[0], p[1]) for p in points], fill=cls_id)
    elif len(points) == 2:
        draw.line(
            [(points[0][0], points[0][1]), (points[1][0], points[1][1])],
            fill=cls_id, width=3,
        )


def _build_mask(shapes: list[dict], width: int, height: int) -> np.ndarray:
    mask_pil = Image.new("L", (width, height), 0)
    draw     = ImageDraw.Draw(mask_pil)
    for shape in shapes:
        cls_id = map_label(str(shape.get("label", "")))
        if cls_id is not None:
            _draw_shape(draw, shape, cls_id)
    return np.array(mask_pil, dtype=np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# Overlay 合成
# ──────────────────────────────────────────────────────────────────────────────

def _make_overlay(image_arr: np.ndarray, mask_arr: np.ndarray) -> np.ndarray:
    """将彩色 mask 半透明叠加到原图，缺陷区域颜色突出。"""
    color_mask = np.zeros_like(image_arr, dtype=np.uint8)
    for cls_id, color in CLASS_COLORS_RGB.items():
        color_mask[mask_arr == cls_id] = color

    overlay = image_arr.astype(np.int16).copy()
    hit = mask_arr > 0
    overlay[hit] = np.clip(
        image_arr[hit].astype(np.int16) - OVERLAY_INTENSITY
        + color_mask[hit].astype(np.int16),
        0, 255,
    )
    return overlay.astype(np.uint8)


def _add_legend(overlay_arr: np.ndarray, present_classes: frozenset[int]) -> np.ndarray:
    """在图像底部追加只含当前样本病害类别的图例条。"""
    items = [
        (cls_id, CLASS_DEFECTS[cls_id], CLASS_COLORS_RGB[cls_id])
        for cls_id in sorted(present_classes)
        if cls_id in CLASS_DEFECTS
    ]
    if not items:
        return overlay_arr

    overlay_img = Image.fromarray(overlay_arr)
    w, h = overlay_img.size

    cols   = min(len(items), 3)
    rows   = math.ceil(len(items) / cols)
    pad    = 10
    row_h  = 28
    swatch = 18
    legend_h = pad * 2 + rows * row_h

    canvas = Image.new("RGB", (w, h + legend_h), (245, 245, 245))
    canvas.paste(overlay_img, (0, 0))

    draw = ImageDraw.Draw(canvas)
    draw.line([(0, h), (w, h)], fill=(180, 180, 180), width=1)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    col_w = max(1, w // cols)
    for idx, (cls_id, name, color) in enumerate(items):
        row = idx // cols
        col = idx % cols
        x0  = col * col_w + pad
        y0  = h + pad + row * row_h
        draw.rectangle([x0, y0, x0 + swatch, y0 + swatch],
                        fill=color, outline=(0, 0, 0), width=1)
        draw.text((x0 + swatch + 6, y0 + 2), f"{cls_id}: {name}",
                  fill=(30, 30, 30), font=font)

    return np.array(canvas, dtype=np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# 单文件处理（供多进程调用）
# ──────────────────────────────────────────────────────────────────────────────

def _process_one(json_path: Path) -> tuple[str, bool, str]:
    """返回 (stem, success, message)。"""
    stem = json_path.stem
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))

        image = _load_image(stem, data)
        w, h  = image.size

        # 若标注坐标与图像尺寸不一致，对齐后再绘制
        ann_w = int(data.get("imageWidth")  or w)
        ann_h = int(data.get("imageHeight") or h)
        if (ann_w, ann_h) != (w, h):
            image = image.resize((ann_w, ann_h), Image.LANCZOS)
            w, h  = ann_w, ann_h

        shapes = data.get("shapes", [])
        mask   = _build_mask(shapes, w, h)

        present = frozenset(
            cls_id for shape in shapes
            if (cls_id := map_label(str(shape.get("label", "")))) is not None
        )

        image_arr = np.array(image, dtype=np.uint8)
        overlay   = _make_overlay(image_arr, mask)
        overlay   = _add_legend(overlay, present)

        out_path = VIS_DIR / f"{stem}.jpg"
        Image.fromarray(overlay).save(out_path, quality=92)
        return stem, True, ""

    except Exception as exc:
        return stem, False, str(exc)


# ──────────────────────────────────────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────────────────────────────────────

def main(workers: int) -> None:
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    json_files = sorted(SOURCE_DIR.glob("*.json"))
    if not json_files:
        print(f"[错误] {SOURCE_DIR} 下未找到 JSON 文件")
        return

    total   = len(json_files)
    done    = 0
    failed  = []

    print(f"共 {total} 个样本，输出目录: {VIS_DIR}  (workers={workers})\n")

    if workers == 1:
        for jf in json_files:
            stem, ok, msg = _process_one(jf)
            done += 1
            if not ok:
                failed.append((stem, msg))
                print(f"  [失败] {stem}: {msg}")
            if done % 100 == 0 or done == total:
                print(f"  进度: {done}/{total}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_process_one, jf): jf for jf in json_files}
            for fut in as_completed(futures):
                stem, ok, msg = fut.result()
                done += 1
                if not ok:
                    failed.append((stem, msg))
                    print(f"  [失败] {stem}: {msg}")
                if done % 100 == 0 or done == total:
                    print(f"  进度: {done}/{total}")

    print(f"\n完成  成功: {total - len(failed)}  失败: {len(failed)}")
    if failed:
        print("失败列表:")
        for stem, msg in failed:
            print(f"  {stem}: {msg}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="原图尺度病害标注可视化")
    parser.add_argument("--workers", type=int,
                        default=max(1, (os.cpu_count() or 4) // 2),
                        help="并行进程数（默认 CPU 核心数的一半）")
    args = parser.parse_args()
    main(args.workers)
