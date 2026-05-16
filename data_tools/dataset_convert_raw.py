"""
dataset_convert_raw.py

将 dataset/tongji 的原始图像（不做任何缩放/Tiling）按与
dataset_convert_awesome.py 完全相同的审查 + 分层划分规则，
输出到 dataset/tongji_data_raw，目录结构与 dataset/tongji_data 一致：

    dataset/tongji_data_raw/
        img_dir/{train,valid}/          ← 原始分辨率图像
        ann_dir/{train,valid}/          ← 原始分辨率 mask
        visualization/{train,valid}/
            mask_overlay/               ← 彩色叠加可视化
            segmented/                  ← 仅保留病害区域

用途：对原始图像做单张/批量推理时，可通过此目录获得 GT mask 和可视化，
      用于计算 IoU、对比预测结果与标注。

运行方式:
    python dataset_convert_raw.py [--workers N]
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
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


# ──────────────────────────────────────────────────────────────────────────────
# 配置（审查 / 划分参数与 dataset_convert_awesome.py 保持一致）
# ──────────────────────────────────────────────────────────────────────────────

SOURCE_DIR     = Path("dataset/tongji")
TARGET_DIR     = Path("dataset/tongji_data_raw")
RANDOM_SEED    = 42
SPLIT_RATIO    = (0.80, 0.20)
BLUR_THRESHOLD = 20.0
MIN_PIXELS     = 192 * 192
MAX_ASPECT     = 10.0
OVERLAY_INTENSITY = 115

# 分群阈值（仅用于分层划分，不影响图像处理）
TINY_MAX   = 200_000
MOBILE_MAX = 2_000_000
DSLR_MAX   = 8_000_000

CLASS_DEFECTS: dict[int, str] = {
    1: "crack", 2: "leakage_b", 3: "leakage_w",
    4: "leakage_g", 5: "lining_falling_off", 6: "segment_damage",
}
CLASS_COLORS_RGB: dict[int, tuple[int, int, int]] = {
    1: (255,   0,   0), 2: (255, 128,   0), 3: (  0,   0, 255),
    4: (  0, 255, 255), 5: (255, 255,   0), 6: (255,   0, 255),
}
LABEL_TO_CLASS: dict[str, int] = {
    "crack": 1, "leakageb": 2, "leakagew": 3, "leakageg": 4,
    "liningfallingoff": 5, "segmentdamage": 6,
    "cracka": 1, "crackb": 1, "cracksmkjm": 1,
    "leakage": 3, "lf": 3, "spalling": 5, "ss": 5,
    "repair": 6, "repairs": 6, "other": 6,
}


# ──────────────────────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_label(raw: str) -> str:
    s = raw.strip().lower()
    for ch in (" ", "_", "-", "/"):
        s = s.replace(ch, "")
    return s


def _map_label(raw: str) -> Optional[int]:
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


def _find_image(stem: str, data: dict) -> Optional[Path]:
    for ext in (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"):
        p = SOURCE_DIR / f"{stem}{ext}"
        if p.exists():
            return p
    path = data.get("imagePath", "")
    if path:
        cand = SOURCE_DIR / Path(path).name
        if cand.exists():
            return cand
    return None


def _load_image(stem: str, data: dict) -> Image.Image:
    p = _find_image(stem, data)
    if p:
        return ImageOps.exif_transpose(Image.open(p).convert("RGB"))
    b64 = data.get("imageData")
    if b64:
        return ImageOps.exif_transpose(
            Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        )
    raise FileNotFoundError(f"无图像: {stem}")


def _blur_var(image: Image.Image) -> float:
    gray = np.array(image.convert("L"), dtype=np.float32)
    lap = (np.roll(gray,-1,0)+np.roll(gray,1,0)
           +np.roll(gray,-1,1)+np.roll(gray,1,1)-4*gray)
    return float(lap.var())


def _draw_shape(draw: ImageDraw.ImageDraw, shape: dict, cls_id: int) -> None:
    pts       = shape.get("points", [])
    shp_type  = str(shape.get("shape_type", "polygon")).lower()
    if not pts:
        return
    if shp_type == "rectangle" and len(pts) >= 2:
        (x1,y1),(x2,y2) = pts[:2]
        draw.rectangle([x1,y1,x2,y2], fill=cls_id)
        return
    if shp_type == "circle" and len(pts) >= 2:
        (cx,cy),(px,py) = pts[:2]
        r = math.hypot(cx-px, cy-py)
        draw.ellipse([cx-r,cy-r,cx+r,cy+r], fill=cls_id)
        return
    if len(pts) >= 3:
        draw.polygon([(p[0],p[1]) for p in pts], fill=cls_id)
    elif len(pts) == 2:
        draw.line([(pts[0][0],pts[0][1]),(pts[1][0],pts[1][1])],
                  fill=cls_id, width=3)


def _build_mask(shapes: list, w: int, h: int) -> np.ndarray:
    pil  = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(pil)
    for shape in shapes:
        cid = _map_label(str(shape.get("label", "")))
        if cid is not None:
            _draw_shape(draw, shape, cid)
    return np.array(pil, dtype=np.uint8)


def _make_overlay(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    cmap = np.zeros_like(img, dtype=np.uint8)
    for cid, color in CLASS_COLORS_RGB.items():
        cmap[mask == cid] = color
    out = img.astype(np.int16).copy()
    hit = mask > 0
    out[hit] = np.clip(img[hit].astype(np.int16) - OVERLAY_INTENSITY
                       + cmap[hit].astype(np.int16), 0, 255)
    return out.astype(np.uint8)


def _add_legend(arr: np.ndarray, present: set[int]) -> np.ndarray:
    items = [(cid, CLASS_DEFECTS[cid], CLASS_COLORS_RGB[cid])
             for cid in sorted(present) if cid in CLASS_DEFECTS]
    if not items:
        return arr
    base = Image.fromarray(arr)
    w, h = base.size
    cols = min(len(items), 3)
    rows = math.ceil(len(items) / cols)
    pad, row_h, swatch = 8, 24, 14
    canvas = Image.new("RGB", (w, h + pad*2 + rows*row_h), (245,245,245))
    canvas.paste(base, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.line([(0,h),(w,h)], fill=(180,180,180), width=1)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    col_w = max(1, w // cols)
    for idx, (cid, name, color) in enumerate(items):
        x0 = (idx % cols)*col_w + pad
        y0 = h + pad + (idx // cols)*row_h
        draw.rectangle([x0,y0,x0+swatch,y0+swatch], fill=color, outline=(0,0,0))
        draw.text((x0+swatch+4, y0+1), f"{cid}:{name}", fill=(30,30,30), font=font)
    return np.array(canvas, dtype=np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# 审查（与 dataset_convert_awesome.py 相同逻辑）
# ──────────────────────────────────────────────────────────────────────────────

def _audit(json_path: Path) -> dict:
    stem   = json_path.stem
    result = dict(stem=stem, passed=False, reject_reason="",
                  width=0, height=0, pixels=0, blur_var=0.0,
                  group="", split="", has_defect=False)
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        result["reject_reason"] = f"JSON解析失败:{e}"; return result

    try:
        image = _load_image(stem, data)
    except Exception as e:
        result["reject_reason"] = f"图像加载失败:{e}"; return result

    w, h = image.size
    result.update(width=w, height=h, pixels=w*h)

    if w * h < MIN_PIXELS:
        result["reject_reason"] = f"像素量过小({w}×{h})"; return result
    if max(w,h) / min(w,h) > MAX_ASPECT:
        result["reject_reason"] = f"宽高比异常({max(w,h)/min(w,h):.1f})"; return result

    try:
        bv = _blur_var(image)
        result["blur_var"] = bv
        if bv < BLUR_THRESHOLD:
            result["reject_reason"] = f"图像模糊(var={bv:.1f})"; return result
    except Exception:
        result["blur_var"] = -1.0

    shapes  = data.get("shapes", [])
    cls_ids = {_map_label(str(s.get("label",""))) for s in shapes}
    cls_ids.discard(None)
    result["has_defect"] = bool(cls_ids)
    result["passed"]     = True
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 分群 + 分层划分
# ──────────────────────────────────────────────────────────────────────────────

def _assign_group(px: int) -> str:
    if px < TINY_MAX:   return "tiny"
    if px < MOBILE_MAX: return "mobile"
    if px < DSLR_MAX:   return "dslr"
    return "highres"


def _split_samples(samples: list[dict], seed: int) -> None:
    rng    = random.Random(seed)
    groups: dict[str, list[dict]] = {}
    for s in samples:
        groups.setdefault(s["group"], []).append(s)
    for grp in groups.values():
        rng.shuffle(grp)
        n_train = max(1, round(len(grp) * SPLIT_RATIO[0]))
        for i, s in enumerate(grp):
            s["split"] = "train" if i < n_train else "valid"


# ──────────────────────────────────────────────────────────────────────────────
# 单样本处理（可并行）
# ──────────────────────────────────────────────────────────────────────────────

def _process(sample: dict, dirs: dict) -> tuple[str, bool, str]:
    stem  = sample["stem"]
    split = sample["split"]
    try:
        data  = json.loads((SOURCE_DIR / f"{stem}.json").read_text(encoding="utf-8"))
        image = _load_image(stem, data)

        # 对齐标注坐标与图像尺寸
        ann_w = int(data.get("imageWidth")  or image.size[0])
        ann_h = int(data.get("imageHeight") or image.size[1])
        if image.size != (ann_w, ann_h):
            image = image.resize((ann_w, ann_h), Image.LANCZOS)

        img_arr = np.array(image, dtype=np.uint8)
        mask    = _build_mask(data.get("shapes", []), ann_w, ann_h)

        # img
        Image.fromarray(img_arr).save(
            dirs["img"][split] / f"{stem}.jpg", quality=95)
        # ann
        Image.fromarray(mask, mode="L").save(
            dirs["ann"][split] / f"{stem}.png")
        # overlay
        present = set(int(v) for v in np.unique(mask) if v > 0)
        overlay = _add_legend(_make_overlay(img_arr, mask), present)
        Image.fromarray(overlay).save(
            dirs["overlay"][split] / f"{stem}.png")
        # segmented
        seg = np.zeros_like(img_arr)
        seg[mask > 0] = img_arr[mask > 0]
        Image.fromarray(seg).save(
            dirs["segmented"][split] / f"{stem}.png")

        return stem, True, ""
    except Exception as e:
        return stem, False, str(e)


# ──────────────────────────────────────────────────────────────────────────────
# 目录初始化
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_dirs() -> dict:
    if TARGET_DIR.exists():
        shutil.rmtree(TARGET_DIR)
    dirs: dict = {"img": {}, "ann": {}, "overlay": {}, "segmented": {}}
    for split in ("train", "valid"):
        dirs["img"][split]       = TARGET_DIR / "img_dir"       / split
        dirs["ann"][split]       = TARGET_DIR / "ann_dir"       / split
        dirs["overlay"][split]   = TARGET_DIR / "visualization" / split / "mask_overlay"
        dirs["segmented"][split] = TARGET_DIR / "visualization" / split / "segmented"
        for key in dirs:
            dirs[key][split].mkdir(parents=True, exist_ok=True)
    return dirs


# ──────────────────────────────────────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────────────────────────────────────

def main(workers: int) -> None:
    json_files = sorted(SOURCE_DIR.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"未找到 JSON: {SOURCE_DIR}")

    print(f"共 {len(json_files)} 个样本，开始审查…")

    # ── 审查（并行） ──────────────────────────────────────────────────────────
    if workers == 1:
        samples = [_audit(jf) for jf in json_files]
    else:
        with ProcessPoolExecutor(max_workers=workers) as exe:
            samples = list(exe.map(_audit, json_files))

    passed   = [s for s in samples if s["passed"]]
    rejected = [s for s in samples if not s["passed"]]
    print(f"通过 {len(passed)}  剔除 {len(rejected)}")
    reason_cnt: dict[str, int] = {}
    for s in rejected:
        k = s["reject_reason"].split("(")[0].split(":")[0]
        reason_cnt[k] = reason_cnt.get(k, 0) + 1
    for r, n in sorted(reason_cnt.items(), key=lambda x: -x[1]):
        print(f"  {r}: {n} 张")

    # ── 分群 + 划分 ───────────────────────────────────────────────────────────
    for s in passed:
        s["group"] = _assign_group(s["pixels"])
    passed.sort(key=lambda s: s["stem"])   # 确保顺序与 awesome 脚本一致
    _split_samples(passed, RANDOM_SEED)

    group_cnt = {}
    for s in passed:
        group_cnt[s["group"]] = group_cnt.get(s["group"], 0) + 1
    print("\n分群:")
    for g in ("tiny","mobile","dslr","highres"):
        print(f"  {g:<8}: {group_cnt.get(g,0):4d} 张")
    train_n = sum(1 for s in passed if s["split"]=="train")
    valid_n = sum(1 for s in passed if s["split"]=="valid")
    print(f"\n划分: train={train_n}  valid={valid_n}")

    # ── 处理（并行） ──────────────────────────────────────────────────────────
    dirs   = _ensure_dirs()
    done   = 0
    failed = []
    print(f"\n生成（workers={workers}）…")

    if workers == 1:
        for s in passed:
            _, ok, msg = _process(s, dirs)
            if not ok: failed.append((s["stem"], msg))
            done += 1
            if done % 100 == 0 or done == len(passed):
                print(f"  {done}/{len(passed)}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as exe:
            futs = {exe.submit(_process, s, dirs): s for s in passed}
            for fut in as_completed(futs):
                _, ok, msg = fut.result()
                if not ok: failed.append((futs[fut]["stem"], msg))
                done += 1
                if done % 100 == 0 or done == len(passed):
                    print(f"  {done}/{len(passed)}")

    # ── 审查报告 ──────────────────────────────────────────────────────────────
    report_path = TARGET_DIR / "audit_report.csv"
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "stem","width","height","pixels","blur_var",
            "passed","reject_reason","group","split","has_defect"])
        writer.writeheader()
        writer.writerows(samples)
    print(f"\n审查报告: {report_path}")

    for split in ("train","valid"):
        n = len(list(dirs["img"][split].glob("*.jpg")))
        print(f"  {split}: {n} 张")

    if failed:
        print(f"\n失败 {len(failed)} 张:")
        for stem, msg in failed[:10]:
            print(f"  {stem}: {msg}")

    print(f"\n完成  输出: {TARGET_DIR.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="原始分辨率数据集生成")
    parser.add_argument("--workers", type=int,
                        default=max(1, (os.cpu_count() or 4) // 2))
    args = parser.parse_args()
    main(args.workers)
