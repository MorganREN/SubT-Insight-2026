"""
analyze_dataset_resolution.py

分析 dataset/tongji 原始图像分辨率分布，
并量化 dataset_convert.py 的 640×640 缩放策略对信息的影响。

运行方式:
    python analyze_dataset_resolution.py
"""

from __future__ import annotations

import io
import base64
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

# ── 与 dataset_convert.py 保持一致 ────────────────────────────────────────────
SOURCE_DIR  = Path("dataset/tongji")
TARGET_SIZE = (640, 640)          # dataset_convert 的输出尺寸
TRAIN_SIZE  = 512                 # augmentation 最终模型输入尺寸


# ──────────────────────────────────────────────────────────────────────────────
# 图像读取（复用 dataset_convert 逻辑，优先外部文件，其次 JSON 内嵌）
# ──────────────────────────────────────────────────────────────────────────────

def _find_image_path(stem: str, data: dict) -> Path | None:
    for ext in (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"):
        p = SOURCE_DIR / f"{stem}{ext}"
        if p.exists():
            return p
    image_path = data.get("imagePath", "")
    if image_path:
        cand = SOURCE_DIR / Path(image_path).name
        if cand.exists():
            return cand
    return None


def _load_orig_size(json_path: Path) -> tuple[int, int] | None:
    """返回 (width, height)，失败返回 None。"""
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    # 优先从外部图像文件获取真实尺寸
    img_path = _find_image_path(json_path.stem, data)
    if img_path is not None:
        try:
            with Image.open(img_path) as img:
                img = ImageOps.exif_transpose(img)
                return img.size  # (w, h)
        except Exception:
            pass

    # 退而求其次：从 JSON 内嵌 imageData 解码
    b64 = data.get("imageData")
    if b64:
        try:
            img = Image.open(io.BytesIO(base64.b64decode(b64)))
            img = ImageOps.exif_transpose(img)
            return img.size
        except Exception:
            pass

    # 最后：直接用 JSON 记录的宽高
    w = data.get("imageWidth")
    h = data.get("imageHeight")
    if w and h:
        return int(w), int(h)

    return None


# ──────────────────────────────────────────────────────────────────────────────
# 核心指标计算
# ──────────────────────────────────────────────────────────────────────────────

def _scale_ratio(orig_w: int, orig_h: int, target_w: int, target_h: int) -> float:
    """dataset_convert 缩放后像素总量 / 原始像素总量（<1 为缩小即丢信息）。"""
    return (target_w * target_h) / (orig_w * orig_h)


def _min_side(w: int, h: int) -> int:
    return min(w, h)


def _max_side(w: int, h: int) -> int:
    return max(w, h)


def _aspect_ratio(w: int, h: int) -> float:
    return max(w, h) / min(w, h)


# ──────────────────────────────────────────────────────────────────────────────
# 分析主函数
# ──────────────────────────────────────────────────────────────────────────────

def analyze() -> None:
    json_files = sorted(SOURCE_DIR.glob("*.json"))
    if not json_files:
        print(f"[错误] 未找到 JSON 文件: {SOURCE_DIR}")
        return

    print(f"扫描 {SOURCE_DIR} …（共 {len(json_files)} 个 JSON）\n")

    sizes: list[tuple[int, int]] = []   # (w, h)
    failed: list[str] = []

    for jf in json_files:
        size = _load_orig_size(jf)
        if size is None:
            failed.append(jf.name)
        else:
            sizes.append(size)

    print(f"成功读取: {len(sizes)}  失败: {len(failed)}")
    if failed:
        print(f"  失败文件: {failed[:10]}{'...' if len(failed)>10 else ''}")
    print()

    widths  = np.array([s[0] for s in sizes])
    heights = np.array([s[1] for s in sizes])
    pixels  = widths * heights
    scales  = np.array([_scale_ratio(w, h, *TARGET_SIZE) for w, h in sizes])
    # val 管线：LongestMaxSize(512)
    val_scales = np.array(
        [TRAIN_SIZE / max(w, h) for w, h in sizes]
    )
    # val 实际到模型的总缩放（相对原图）
    val_total = np.array(
        [(TRAIN_SIZE / max(w, h)) * (TRAIN_SIZE / max(w, h)) * (min(w, h) / max(w, h))
         if max(w, h) > 0 else 0.0
         for w, h in sizes]
    )

    # ── 1. 分辨率基本统计 ──────────────────────────────────────────────────────
    print("=" * 60)
    print("【1】原始图像分辨率统计")
    print("=" * 60)
    for label, arr in [("宽(px)", widths), ("高(px)", heights), ("总像素(万)", pixels/1e4)]:
        print(f"  {label:<12}  min={arr.min():.0f}  max={arr.max():.0f}"
              f"  mean={arr.mean():.0f}  median={np.median(arr):.0f}")
    print()

    # ── 2. 高频尺寸 Top-10 ────────────────────────────────────────────────────
    print("=" * 60)
    print("【2】最常见原始尺寸 Top-10（宽×高）")
    print("=" * 60)
    size_counter = Counter(sizes)
    for (w, h), cnt in size_counter.most_common(10):
        bar = "█" * min(cnt, 40)
        print(f"  {w:5d}×{h:<5d}  {cnt:4d} 张  {bar}")
    print()

    # ── 3. 尺寸区间分布 ──────────────────────────────────────────────────────
    print("=" * 60)
    print("【3】最长边分布区间")
    print("=" * 60)
    long_sides = np.maximum(widths, heights)
    bins = [0, 512, 640, 1024, 1920, 3000, 99999]
    labels = ["≤512", "513-640", "641-1024", "1025-1920", "1921-3000", ">3000"]
    for lo, hi, lab in zip(bins, bins[1:], labels):
        cnt = int(((long_sides > lo) & (long_sides <= hi)).sum())
        pct = cnt / len(sizes) * 100
        bar = "█" * int(pct / 2)
        print(f"  {lab:<12}  {cnt:4d} 张  {pct:5.1f}%  {bar}")
    print()

    # ── 4. 宽高比分布 ────────────────────────────────────────────────────────
    print("=" * 60)
    print("【4】宽高比分布（最长/最短）")
    print("=" * 60)
    ars = np.array([_aspect_ratio(w, h) for w, h in sizes])
    ar_bins = [1.0, 1.1, 1.33, 1.5, 1.78, 2.0, 99.0]
    ar_labels = ["1.0-1.1(≈正方)", "1.1-1.33", "1.33-1.5", "1.5-1.78(≈16:9)", "1.78-2.0", ">2.0"]
    for lo, hi, lab in zip(ar_bins, ar_bins[1:], ar_labels):
        cnt = int(((ars >= lo) & (ars < hi)).sum())
        pct = cnt / len(sizes) * 100
        bar = "█" * int(pct / 2)
        print(f"  {lab:<20}  {cnt:4d} 张  {pct:5.1f}%  {bar}")
    print()

    # ── 5. dataset_convert 缩放影响（原图 → 640×640）────────────────────────
    print("=" * 60)
    print(f"【5】dataset_convert 缩放影响（原图 → {TARGET_SIZE[0]}×{TARGET_SIZE[1]}）")
    print("=" * 60)

    enlarged  = int((scales > 1.0).sum())   # 原图比640×640小，被放大
    same      = int((scales == 1.0).sum())
    shrunk    = int((scales < 1.0).sum())   # 原图比640×640大，被压缩 → 丢信息
    print(f"  被放大（原图<目标，插值填充）: {enlarged:4d} 张  ({enlarged/len(sizes)*100:.1f}%)")
    print(f"  无变化（原图=目标）          : {same:4d} 张  ({same/len(sizes)*100:.1f}%)")
    print(f"  被压缩（原图>目标，丢信息）  : {shrunk:4d} 张  ({shrunk/len(sizes)*100:.1f}%)")
    print()

    shrunk_scales = scales[scales < 1.0]
    if len(shrunk_scales):
        print(f"  压缩图像的像素保留率:")
        print(f"    最少保留: {shrunk_scales.min()*100:.1f}%  "
              f"最多保留: {shrunk_scales.max()*100:.1f}%  "
              f"平均保留: {shrunk_scales.mean()*100:.1f}%")
        # 损失 > 50%
        heavy = int((shrunk_scales < 0.5).sum())
        print(f"    像素损失 >50% 的图像: {heavy} 张 ({heavy/len(sizes)*100:.1f}%)")
    print()

    # ── 6. 宽高比失真分析 ────────────────────────────────────────────────────
    print("=" * 60)
    print("【6】强制缩放为正方形导致的宽高比失真")
    print("=" * 60)
    # dataset_convert 将图像强制 resize 到正方形，宽高比失真 = |原AR - 1| 代表拉伸程度
    distortions = np.abs(ars - 1.0)  # 偏离正方形的程度
    no_distort   = int((distortions < 0.05).sum())
    mild_distort = int(((distortions >= 0.05) & (distortions < 0.33)).sum())
    heavy_distort= int((distortions >= 0.33).sum())
    print(f"  无明显失真（原图接近正方，AR<1.05） : {no_distort:4d} 张  ({no_distort/len(sizes)*100:.1f}%)")
    print(f"  轻微失真（AR 1.05~1.33）            : {mild_distort:4d} 张  ({mild_distort/len(sizes)*100:.1f}%)")
    print(f"  明显失真（AR>1.33，强制拉伸/压缩）  : {heavy_distort:4d} 张  ({heavy_distort/len(sizes)*100:.1f}%)")
    print(f"  平均宽高比: {ars.mean():.3f}  最大宽高比: {ars.max():.3f}")
    print()

    # ── 7. 对比：若改用 val 管线策略（保持宽高比+padding）────────────────────
    print("=" * 60)
    print(f"【7】对比：若 convert 改用「保持宽高比+padding 到 {TARGET_SIZE[0]}」")
    print("=" * 60)
    # 等比缩放后最长边=640，短边 padding，像素保留率 = (缩放后真实内容像素) / 原始像素
    keep_ar_pixels = np.array([
        (TARGET_SIZE[0] / max(w, h)) ** 2 * w * h
        for w, h in sizes
    ])
    orig_pixels = pixels.astype(float)
    keep_ar_scales = keep_ar_pixels / orig_pixels

    shrunk_keep = keep_ar_scales[keep_ar_scales < 1.0]
    print(f"  被压缩的图像: {len(shrunk_keep)} 张")
    if len(shrunk_keep):
        print(f"    平均像素保留率: {shrunk_keep.mean()*100:.1f}%"
              f"  （对比强制拉伸方案: {shrunk_scales.mean()*100:.1f}% 若有压缩）")
    # padding 浪费的面积比例
    padding_waste = np.array([
        1.0 - min(w, h) / max(w, h)   # padding 占 640×640 的比例
        for w, h in sizes
    ])
    print(f"  padding 平均浪费面积比: {padding_waste.mean()*100:.1f}%  "
          f"最大: {padding_waste.max()*100:.1f}%")
    print()

    # ── 8. 综合建议 ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("【8】综合结论")
    print("=" * 60)
    print(f"  总样本: {len(sizes)} 张")
    print(f"  原图中位分辨率: {int(np.median(widths))}×{int(np.median(heights))}")
    print(f"  原图中位像素量: {np.median(pixels)/1e6:.2f} MP")
    print(f"  640×640 像素量: {640*640/1e6:.2f} MP")
    print(f"  512×512 像素量: {512*512/1e6:.2f} MP  （模型最终输入）")
    print()
    print(f"  当前方案（强制拉伸到 640×640）问题:")
    print(f"    · {shrunk} 张图像被压缩（丢失像素信息）")
    print(f"    · {heavy_distort} 张图像宽高比失真 >1.33（几何形变影响裂缝/渗漏形态）")
    print(f"    · 原始 LabelMe 标注坐标在原图尺寸下绘制，")
    print(f"      若图像与 JSON 记录尺寸不符会先对齐，再统一压缩")
    print()


if __name__ == "__main__":
    analyze()
