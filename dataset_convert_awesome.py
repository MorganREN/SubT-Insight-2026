"""
dataset_convert_awesome.py

基于分层 Tiling 的自适应数据预处理，输出 dataset/tongji_data_awesome。
目录结构与 dataset/tongji_data 一致（img_dir / ann_dir / visualization）。

运行方式:
    python dataset_convert_awesome.py [--workers N]

流程:
    1. 数据质量审查（尺寸/模糊/标注对齐/标注完整性）
    2. 按分辨率分群（Tiny / Mobile / DSLR / HighRes）
    3. 源图像层面 80/20 划分（各群落独立分层）
    4. 自适应 Tiling（各群落不同 patch_size + stride + 镜像填充）
    5. Patch 质量筛选（纯背景保留 20%）
    6. 保存 img_dir / ann_dir / visualization
    7. 输出审查报告 CSV
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import math
import os
import random
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


# ──────────────────────────────────────────────────────────────────────────────
# 全局配置
# ──────────────────────────────────────────────────────────────────────────────

SOURCE_DIR  = Path("dataset/tongji")
TARGET_DIR  = Path("dataset/tongji_data_awesome")
RANDOM_SEED = 42
SPLIT_RATIO = (0.80, 0.20)       # (train, valid)，不单独留 test
BG_KEEP_RATE = 1.0               # tiling 阶段保留全部背景 patch，后续由 filter_background_patches.py 按纹理筛选
BLUR_THRESHOLD = 20.0            # Laplacian 方差低于此值视为模糊
# 注：隧道图像对比度天然低于普通场景，全数据集中位数约 58，p10 约 11。
# 设 20 约剔除最差 12%（极度模糊/抖动），保留绝大多数有效样本。
OVERLAY_INTENSITY = 115

# 尺寸异常剔除阈值
MIN_PIXELS    = 192 * 192        # 低于此像素量剔除
MAX_ASPECT    = 10.0             # 宽高比超过此值剔除

# 分群像素阈值
TINY_MAX   = 200_000             # < 200K px → Tiny
MOBILE_MAX = 2_000_000           # 200K ~ 2M px → Mobile
DSLR_MAX   = 8_000_000           # 2M ~ 8M px → DSLR
# > 8M px → HighRes

# 各群落 Tiling 参数 (patch_size, stride)
TILING_PARAMS = {
    "tiny":    None,                  # 不 Tiling，放大到 512×512
    "mobile":  (512, 384),            # 25% 重叠
    "dslr":    (640, 400),            # 37.5% 重叠
    "highres": (768, 384),            # 50% 重叠
}

NUM_CLASSES = 7   # 0=background, 1-6=defects

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
    "crack": 1, "leakageb": 2, "leakagew": 3, "leakageg": 4,
    "liningfallingoff": 5, "segmentdamage": 6,
    "cracka": 1, "crackb": 1, "cracksmkjm": 1,
    "leakage": 3, "lf": 3,
    "spalling": 5, "ss": 5,
    "repair": 6, "repairs": 6, "other": 6,
}


# ──────────────────────────────────────────────────────────────────────────────
# 数据类
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SourceSample:
    stem: str
    json_path: Path
    # 审查结果
    passed: bool = False
    reject_reason: str = ""
    width: int = 0
    height: int = 0
    pixels: int = 0
    blur_var: float = 0.0
    group: str = ""          # tiny / mobile / dslr / highres
    split: str = ""          # train / valid
    has_defect: bool = False


# ──────────────────────────────────────────────────────────────────────────────
# 标签映射
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_label(label: str) -> str:
    norm = label.strip().lower()
    for ch in (" ", "_", "-", "/"):
        norm = norm.replace(ch, "")
    return norm


def map_label(raw: str) -> Optional[int]:
    label = _normalize_label(raw)
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

def _find_image_path(stem: str, data: dict) -> Optional[Path]:
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
    p = _find_image_path(stem, data)
    if p is not None:
        return ImageOps.exif_transpose(Image.open(p).convert("RGB"))
    b64 = data.get("imageData")
    if b64:
        return ImageOps.exif_transpose(
            Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        )
    raise FileNotFoundError(f"无图像: {stem}")


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
        r = math.hypot(cx - px, cy - py)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=cls_id)
        return
    if len(points) >= 3:
        draw.polygon([(p[0], p[1]) for p in points], fill=cls_id)
    elif len(points) == 2:
        draw.line([(points[0][0], points[0][1]), (points[1][0], points[1][1])],
                  fill=cls_id, width=3)


def _build_mask(shapes: list, width: int, height: int) -> np.ndarray:
    pil = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(pil)
    for shape in shapes:
        cls_id = map_label(str(shape.get("label", "")))
        if cls_id is not None:
            _draw_shape(draw, shape, cls_id)
    return np.array(pil, dtype=np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# 步骤 1：数据质量审查
# ──────────────────────────────────────────────────────────────────────────────

def _blur_variance(image: Image.Image) -> float:
    """Laplacian 梯度方差，值越低越模糊。"""
    gray = np.array(image.convert("L"), dtype=np.float32)
    # 3×3 Laplacian kernel
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    from scipy.signal import convolve2d
    lap = convolve2d(gray, kernel, mode="valid")
    return float(lap.var())


def audit_sample(json_path: Path) -> SourceSample:
    stem   = json_path.stem
    sample = SourceSample(stem=stem, json_path=json_path)

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        sample.reject_reason = f"JSON解析失败: {e}"
        return sample

    # 加载图像
    try:
        image = _load_image(stem, data)
    except Exception as e:
        sample.reject_reason = f"图像加载失败: {e}"
        return sample

    w, h = image.size
    sample.width  = w
    sample.height = h
    sample.pixels = w * h

    # 尺寸检查
    if w * h < MIN_PIXELS:
        sample.reject_reason = f"像素量过小({w}×{h}={w*h}<{MIN_PIXELS})"
        return sample
    aspect = max(w, h) / min(w, h)
    if aspect > MAX_ASPECT:
        sample.reject_reason = f"宽高比异常({aspect:.1f}>{MAX_ASPECT})"
        return sample

    # 模糊检查
    try:
        blur = _blur_variance(image)
        sample.blur_var = blur
        if blur < BLUR_THRESHOLD:
            sample.reject_reason = f"图像模糊(Laplacian方差={blur:.1f}<{BLUR_THRESHOLD})"
            return sample
    except Exception:
        # scipy 不可用时跳过模糊检查
        sample.blur_var = -1.0

    # 标注对齐检查
    ann_w = int(data.get("imageWidth") or w)
    ann_h = int(data.get("imageHeight") or h)
    if (ann_w, ann_h) != (w, h):
        # 尝试缩放对齐，非致命；记录但不拒绝
        pass  # 在后续处理中处理

    # 标注完整性：是否有病害
    shapes = data.get("shapes", [])
    cls_ids = {map_label(str(s.get("label", ""))) for s in shapes}
    cls_ids.discard(None)
    sample.has_defect = bool(cls_ids)

    sample.passed = True
    return sample


# ──────────────────────────────────────────────────────────────────────────────
# 步骤 2：分群
# ──────────────────────────────────────────────────────────────────────────────

def assign_group(sample: SourceSample) -> str:
    px = sample.pixels
    if px < TINY_MAX:
        return "tiny"
    if px < MOBILE_MAX:
        return "mobile"
    if px < DSLR_MAX:
        return "dslr"
    return "highres"


# ──────────────────────────────────────────────────────────────────────────────
# 步骤 3：源图像层面划分
# ──────────────────────────────────────────────────────────────────────────────

def split_by_source(samples: list[SourceSample], seed: int) -> None:
    """在各群落内分别按 SPLIT_RATIO 划分，结果写入 sample.split。"""
    rng = random.Random(seed)
    groups: dict[str, list[SourceSample]] = {}
    for s in samples:
        groups.setdefault(s.group, []).append(s)

    for group_samples in groups.values():
        rng.shuffle(group_samples)
        n       = len(group_samples)
        n_train = max(1, round(n * SPLIT_RATIO[0]))
        for i, s in enumerate(group_samples):
            s.split = "train" if i < n_train else "valid"


# ──────────────────────────────────────────────────────────────────────────────
# 步骤 4：自适应 Tiling
# ──────────────────────────────────────────────────────────────────────────────

def _reflect_pad(arr: np.ndarray, pad_h: int, pad_w: int) -> np.ndarray:
    """对 HW 或 HWC 数组做镜像填充（右下方向）。"""
    if arr.ndim == 2:
        return np.pad(arr, ((0, pad_h), (0, pad_w)), mode="reflect")
    return np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")


def tile_image_and_mask(
    image_arr: np.ndarray,   # H×W×3 uint8
    mask_arr:  np.ndarray,   # H×W uint8
    patch_size: int,
    stride: int,
) -> List[Tuple[np.ndarray, np.ndarray, int, int]]:
    """
    滑动窗口裁取 patch，边界镜像填充。
    返回 [(img_patch, mask_patch, row_idx, col_idx), ...]
    """
    H, W = image_arr.shape[:2]

    # 镜像填充到能被 stride 整除（右侧和下侧）
    pad_h = (stride - (H - patch_size) % stride) % stride if H > patch_size else max(0, patch_size - H)
    pad_w = (stride - (W - patch_size) % stride) % stride if W > patch_size else max(0, patch_size - W)

    if pad_h > 0 or pad_w > 0:
        image_arr = _reflect_pad(image_arr, pad_h, pad_w)
        mask_arr  = _reflect_pad(mask_arr,  pad_h, pad_w)

    PH, PW = image_arr.shape[:2]
    patches = []
    row_idx = 0
    for y in range(0, PH - patch_size + 1, stride):
        col_idx = 0
        for x in range(0, PW - patch_size + 1, stride):
            img_p  = image_arr[y:y + patch_size, x:x + patch_size]
            mask_p = mask_arr [y:y + patch_size, x:x + patch_size]
            patches.append((img_p, mask_p, row_idx, col_idx))
            col_idx += 1
        row_idx += 1
    return patches


# ──────────────────────────────────────────────────────────────────────────────
# 步骤 5：Patch 质量筛选
# ──────────────────────────────────────────────────────────────────────────────

def _is_background(mask_patch: np.ndarray) -> bool:
    return int(mask_patch.max()) == 0


# ──────────────────────────────────────────────────────────────────────────────
# Overlay 可视化（复用 dataset_convert.py 风格）
# ──────────────────────────────────────────────────────────────────────────────

def _make_overlay(image_arr: np.ndarray, mask_arr: np.ndarray) -> np.ndarray:
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


def _add_legend(overlay_arr: np.ndarray, present: set[int]) -> np.ndarray:
    items = [(cid, CLASS_DEFECTS[cid], CLASS_COLORS_RGB[cid])
             for cid in sorted(present) if cid in CLASS_DEFECTS]
    if not items:
        return overlay_arr
    img = Image.fromarray(overlay_arr)
    w, h = img.size
    cols = min(len(items), 3)
    rows = math.ceil(len(items) / cols)
    pad, row_h, swatch = 8, 24, 14
    legend_h = pad * 2 + rows * row_h
    canvas = Image.new("RGB", (w, h + legend_h), (245, 245, 245))
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.line([(0, h), (w, h)], fill=(180, 180, 180), width=1)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    col_w = max(1, w // cols)
    for idx, (cid, name, color) in enumerate(items):
        x0 = (idx % cols) * col_w + pad
        y0 = h + pad + (idx // cols) * row_h
        draw.rectangle([x0, y0, x0 + swatch, y0 + swatch], fill=color, outline=(0, 0, 0))
        draw.text((x0 + swatch + 4, y0 + 1), f"{cid}:{name}", fill=(30, 30, 30), font=font)
    return np.array(canvas, dtype=np.uint8)


def _make_segmented(image_arr: np.ndarray, mask_arr: np.ndarray) -> np.ndarray:
    seg = np.zeros_like(image_arr)
    hit = mask_arr > 0
    seg[hit] = image_arr[hit]
    return seg


# ──────────────────────────────────────────────────────────────────────────────
# 单样本处理（可并行）
# ──────────────────────────────────────────────────────────────────────────────

def _process_sample(
    sample: SourceSample,
    dirs: dict,
    bg_rng: random.Random,
) -> list[str]:
    """
    返回该样本生成的所有 patch stem 列表（用于统计）。
    dirs 包含各 split 的输出路径。
    """
    data   = json.loads(sample.json_path.read_text(encoding="utf-8"))
    image  = _load_image(sample.stem, data)

    # 对齐图像与标注尺寸
    ann_w = int(data.get("imageWidth")  or image.size[0])
    ann_h = int(data.get("imageHeight") or image.size[1])
    if image.size != (ann_w, ann_h):
        image = image.resize((ann_w, ann_h), Image.LANCZOS)

    W, H = image.size
    shapes = data.get("shapes", [])
    mask_arr  = _build_mask(shapes, W, H)
    image_arr = np.array(image, dtype=np.uint8)

    split = sample.split

    generated = []

    # ── Tiny 群：放大到 512×512 ──────────────────────────────────────────────
    if sample.group == "tiny":
        img_resized  = image.resize((512, 512), Image.LANCZOS)
        mask_resized = np.array(
            Image.fromarray(mask_arr).resize((512, 512), Image.NEAREST),
            dtype=np.uint8,
        )
        img_arr_r = np.array(img_resized, dtype=np.uint8)
        stem = sample.stem

        # 纯背景筛选
        if _is_background(mask_resized) and bg_rng.random() > BG_KEEP_RATE:
            return []

        _save_patch(stem, img_arr_r, mask_resized, split, dirs)
        generated.append(stem)
        return generated

    # ── 其他群：Tiling ───────────────────────────────────────────────────────
    patch_size, stride = TILING_PARAMS[sample.group]
    patches = tile_image_and_mask(image_arr, mask_arr, patch_size, stride)

    # 先统计哪些是背景，用于按比例采样
    bg_indices   = [i for i, (_, mp, _, _) in enumerate(patches) if _is_background(mp)]
    keep_bg_n    = max(1, round(len(bg_indices) * BG_KEEP_RATE))
    bg_rng.shuffle(bg_indices)
    keep_bg_set  = set(bg_indices[:keep_bg_n])

    for i, (img_p, mask_p, row, col) in enumerate(patches):
        if _is_background(mask_p) and i not in keep_bg_set:
            continue

        stem = f"{sample.stem}_r{row:03d}_c{col:03d}"
        _save_patch(stem, img_p, mask_p, split, dirs)
        generated.append(stem)

    return generated


def _save_patch(
    stem: str,
    img_arr: np.ndarray,
    mask_arr: np.ndarray,
    split: str,
    dirs: dict,
) -> None:
    """保存图像、mask 及可视化。"""
    # img
    Image.fromarray(img_arr).save(
        dirs["img"][split] / f"{stem}.jpg", quality=95
    )
    # ann
    Image.fromarray(mask_arr, mode="L").save(
        dirs["ann"][split] / f"{stem}.png"
    )
    # overlay
    present = set(int(v) for v in np.unique(mask_arr) if v > 0)
    overlay = _make_overlay(img_arr, mask_arr)
    overlay = _add_legend(overlay, present)
    Image.fromarray(overlay).save(
        dirs["overlay"][split] / f"{stem}.png"
    )
    # segmented
    seg = _make_segmented(img_arr, mask_arr)
    Image.fromarray(seg).save(
        dirs["segmented"][split] / f"{stem}.png"
    )


def _worker_process(sample: SourceSample, dirs: dict) -> list[str]:
    """多进程入口（模块级，可 pickle）。"""
    rng = random.Random(RANDOM_SEED + hash(sample.stem) % 100_000)
    return _process_sample(sample, dirs, rng)


# ──────────────────────────────────────────────────────────────────────────────
# 目录初始化
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_dirs() -> dict:
    if TARGET_DIR.exists():
        shutil.rmtree(TARGET_DIR)
    dirs: dict = {"img": {}, "ann": {}, "overlay": {}, "segmented": {}}
    for split in ("train", "valid"):
        dirs["img"][split]       = TARGET_DIR / "img_dir"     / split
        dirs["ann"][split]       = TARGET_DIR / "ann_dir"     / split
        dirs["overlay"][split]   = TARGET_DIR / "visualization" / split / "mask_overlay"
        dirs["segmented"][split] = TARGET_DIR / "visualization" / split / "segmented"
        for key in dirs:
            dirs[key][split].mkdir(parents=True, exist_ok=True)
    return dirs


# ──────────────────────────────────────────────────────────────────────────────
# 审查报告
# ──────────────────────────────────────────────────────────────────────────────

def _write_audit_report(samples: list[SourceSample], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "stem", "width", "height", "pixels", "blur_var",
            "passed", "reject_reason", "group", "split", "has_defect",
        ])
        writer.writeheader()
        for s in samples:
            writer.writerow({
                "stem": s.stem, "width": s.width, "height": s.height,
                "pixels": s.pixels, "blur_var": f"{s.blur_var:.1f}",
                "passed": s.passed, "reject_reason": s.reject_reason,
                "group": s.group, "split": s.split, "has_defect": s.has_defect,
            })
    print(f"审查报告已保存: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────────────────────────────────────

def main(workers: int) -> None:
    random.seed(RANDOM_SEED)

    json_files = sorted(SOURCE_DIR.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"SOURCE_DIR 下无 JSON: {SOURCE_DIR}")

    print(f"共 {len(json_files)} 个 JSON，开始审查…")

    # ── 步骤 1：审查（并行） ──────────────────────────────────────────────────
    samples: list[SourceSample] = []
    if workers == 1:
        for jf in json_files:
            samples.append(audit_sample(jf))
    else:
        with ProcessPoolExecutor(max_workers=workers) as exe:
            futures = {exe.submit(audit_sample, jf): jf for jf in json_files}
            for fut in as_completed(futures):
                samples.append(fut.result())

    passed  = [s for s in samples if s.passed]
    rejected = [s for s in samples if not s.passed]
    print(f"审查完成：通过 {len(passed)}，剔除 {len(rejected)}")

    # 打印剔除摘要
    reason_count: dict[str, int] = {}
    for s in rejected:
        key = s.reject_reason.split("(")[0].split(":")[0]
        reason_count[key] = reason_count.get(key, 0) + 1
    for reason, cnt in sorted(reason_count.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {cnt} 张")

    # ── 步骤 2：分群 ─────────────────────────────────────────────────────────
    for s in passed:
        s.group = assign_group(s)
    group_counts = {}
    for s in passed:
        group_counts[s.group] = group_counts.get(s.group, 0) + 1
    print("\n分群结果:")
    for g in ("tiny", "mobile", "dslr", "highres"):
        print(f"  {g:<8}: {group_counts.get(g, 0):4d} 张  "
              f"(tiling={TILING_PARAMS[g]})")

    # ── 步骤 3：源图像层面划分 ───────────────────────────────────────────────
    split_by_source(passed, RANDOM_SEED)
    train_n = sum(1 for s in passed if s.split == "train")
    valid_n = sum(1 for s in passed if s.split == "valid")
    print(f"\n划分：train={train_n}  valid={valid_n}")

    # ── 初始化输出目录 ───────────────────────────────────────────────────────
    dirs = _ensure_dirs()

    # ── 步骤 4-5：Tiling + 筛选 + 保存 ──────────────────────────────────────
    print(f"\n生成 patch（workers={workers}）…")
    total_patches = 0
    done = 0
    # 每个样本用独立的 bg_rng（基于 stem + seed），保证可复现且不跨进程共享状态
    bg_rng = random.Random(RANDOM_SEED + 1)

    if workers == 1:
        for s in passed:
            stems = _process_sample(s, dirs, bg_rng)
            total_patches += len(stems)
            done += 1
            if done % 100 == 0 or done == len(passed):
                print(f"  {done}/{len(passed)}  patches={total_patches}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as exe:
            futures = {exe.submit(_worker_process, s, dirs): s for s in passed}
            for fut in as_completed(futures):
                stems = fut.result()
                total_patches += len(stems)
                done += 1
                if done % 100 == 0 or done == len(passed):
                    print(f"  {done}/{len(passed)}  patches={total_patches}")

    # ── 统计各 split 输出 ────────────────────────────────────────────────────
    for split in ("train", "valid"):
        n = len(list(dirs["img"][split].glob("*.jpg")))
        print(f"  {split}: {n} 个 patch")

    # ── 审查报告 ─────────────────────────────────────────────────────────────
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    _write_audit_report(samples, TARGET_DIR / "audit_report.csv")

    print(f"\n完成  总 patch={total_patches}  输出目录={TARGET_DIR.resolve()}")
    print("类别定义: 0=background, "
          + ", ".join(f"{k}={v}" for k, v in CLASS_DEFECTS.items()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分层 Tiling 数据预处理")
    parser.add_argument("--workers", type=int,
                        default=max(1, (os.cpu_count() or 4) // 2),
                        help="并行进程数（默认 CPU 核数的一半）")
    args = parser.parse_args()
    main(args.workers)
