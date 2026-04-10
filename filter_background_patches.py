"""
filter_background_patches.py

对 dataset/tongji_data_awesome 中的纯背景 patch 做质量筛选：
  1. 区分含病害 patch 与纯背景 patch（mask 全为 0）
  2. 对纯背景 patch 计算 Laplacian 梯度方差（纹理越丰富值越高）
  3. 按方差从高到低排序，保留前 N 张（N ≤ 含病害 patch 数 × keep_ratio）
  4. 删除剩余低纹理纯背景 patch（img / ann / visualization 同步删除）

运行方式:
    python filter_background_patches.py [--ratio 0.25] [--dry_run]

参数:
    --ratio    保留背景 patch 数量上限 = 含病害 patch 数 × ratio，默认 0.25
    --dry_run  仅打印统计，不实际删除文件
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image


DATASET_DIR = Path("dataset/tongji_data_awesome")
SPLITS = ["train", "valid"]


# ──────────────────────────────────────────────────────────────────────────────
# Laplacian 梯度方差（衡量纹理丰富程度）
# ──────────────────────────────────────────────────────────────────────────────

_LAP_KERNEL = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)


def _texture_score(img_path: Path) -> float:
    """用 Laplacian 方差衡量图像纹理复杂度，值越高越丰富。"""
    gray = np.array(Image.open(img_path).convert("L"), dtype=np.float32)
    # 手动卷积（避免 scipy 依赖）：用差分近似 Laplacian
    lap = (
        np.roll(gray, -1, axis=0) + np.roll(gray, 1, axis=0)
        + np.roll(gray, -1, axis=1) + np.roll(gray, 1, axis=1)
        - 4 * gray
    )
    return float(lap.var())


# ──────────────────────────────────────────────────────────────────────────────
# 单 split 处理
# ──────────────────────────────────────────────────────────────────────────────

def _related_paths(stem: str, split: str) -> list[Path]:
    """返回一个 patch 对应的所有文件路径（img + ann + vis）。"""
    return [
        DATASET_DIR / "img_dir"  / split / f"{stem}.jpg",
        DATASET_DIR / "ann_dir"  / split / f"{stem}.png",
        DATASET_DIR / "visualization" / split / "mask_overlay" / f"{stem}.png",
        DATASET_DIR / "visualization" / split / "segmented"    / f"{stem}.png",
    ]


def process_split(split: str, keep_ratio: float, dry_run: bool) -> dict:
    ann_dir = DATASET_DIR / "ann_dir" / split
    img_dir = DATASET_DIR / "img_dir" / split

    all_stems = [p.stem for p in sorted(ann_dir.glob("*.png"))]

    # ── 分类：含病害 vs 纯背景 ──────────────────────────────────────────────
    defect_stems: list[str] = []
    bg_stems:     list[str] = []

    print(f"[{split}] 扫描 {len(all_stems)} 个 patch 的 mask …")
    for stem in all_stems:
        mask = np.array(Image.open(ann_dir / f"{stem}.png"), dtype=np.uint8)
        if mask.max() == 0:
            bg_stems.append(stem)
        else:
            defect_stems.append(stem)

    n_defect = len(defect_stems)
    n_bg     = len(bg_stems)
    keep_n   = min(n_bg, round(n_defect * keep_ratio))

    print(f"[{split}] 含病害={n_defect}  纯背景={n_bg}  "
          f"保留上限={keep_n}（ratio={keep_ratio}）  待删除={n_bg - keep_n}")

    if n_bg <= keep_n:
        print(f"[{split}] 背景数量未超上限，无需删除。")
        return {"split": split, "defect": n_defect, "bg_total": n_bg,
                "bg_kept": n_bg, "bg_removed": 0}

    # ── 对背景 patch 计算纹理分数 ──────────────────────────────────────────
    print(f"[{split}] 计算 {n_bg} 个背景 patch 的纹理分数 …")
    scores: list[tuple[float, str]] = []
    for i, stem in enumerate(bg_stems):
        score = _texture_score(img_dir / f"{stem}.jpg")
        scores.append((score, stem))
        if (i + 1) % 500 == 0 or (i + 1) == n_bg:
            print(f"  {i + 1}/{n_bg}")

    # 按分数降序排列，保留前 keep_n
    scores.sort(key=lambda x: x[0], reverse=True)
    to_keep   = {stem for _, stem in scores[:keep_n]}
    to_remove = [stem for _, stem in scores[keep_n:]]

    # 打印分数边界，方便判断阈值合理性
    cutoff_score = scores[keep_n - 1][0] if keep_n > 0 else 0.0
    min_score    = scores[-1][0]
    max_score    = scores[0][0]
    print(f"[{split}] 纹理分数范围: {min_score:.1f} ~ {max_score:.1f}  "
          f"截断分数: {cutoff_score:.1f}")

    # ── 删除低纹理背景 patch ──────────────────────────────────────────────
    removed = 0
    for stem in to_remove:
        for path in _related_paths(stem, split):
            if path.exists():
                if not dry_run:
                    path.unlink()
                removed += 1

    action = "（dry_run，未实际删除）" if dry_run else ""
    print(f"[{split}] 已删除 {len(to_remove)} 个背景 patch（{removed} 个文件）{action}")

    return {
        "split":      split,
        "defect":     n_defect,
        "bg_total":   n_bg,
        "bg_kept":    keep_n,
        "bg_removed": len(to_remove),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="纯背景 patch 纹理筛选")
    parser.add_argument("--ratio",   type=float, default=0.25,
                        help="保留背景数量上限 = 含病害数 × ratio（默认 0.25）")
    parser.add_argument("--dry_run", action="store_true",
                        help="仅统计，不删除文件")
    args = parser.parse_args()

    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"数据集目录不存在: {DATASET_DIR}")

    print(f"数据集: {DATASET_DIR}")
    print(f"keep_ratio={args.ratio}  dry_run={args.dry_run}\n")

    results = []
    for split in SPLITS:
        result = process_split(split, args.ratio, args.dry_run)
        results.append(result)
        print()

    # ── 汇总 ────────────────────────────────────────────────────────────────
    print("=" * 50)
    print("汇总")
    print("=" * 50)
    total_before = total_after = 0
    for r in results:
        before = r["defect"] + r["bg_total"]
        after  = r["defect"] + r["bg_kept"]
        total_before += before
        total_after  += after
        bg_ratio_after = r["bg_kept"] / r["defect"] if r["defect"] > 0 else 0
        print(f"  [{r['split']}]  处理前={before}  处理后={after}  "
              f"删除={r['bg_removed']}  "
              f"背景/病害={bg_ratio_after:.1%}")
    print(f"  总计  处理前={total_before}  处理后={total_after}  "
          f"删除={total_before - total_after}")


if __name__ == "__main__":
    main()
