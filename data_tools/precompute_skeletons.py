"""
tools/precompute_skeletons.py

离线预计算裂缝 GT 骨架掩码，供 SkeletonLoss 使用。

使用方法
--------
    python tools/precompute_skeletons.py \
        --data_root dataset/tongji_data \
        --splits train \
        --crack_class_idx 1

输出目录结构
-----------
    {data_root}/skel_dir/{split}/{name}.png
    （像素值：255 = 骨架，0 = 非骨架，与 GT mask 同名）

依赖
----
    scikit-image   （pip install scikit-image）
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from loguru import logger


def precompute_split(
    data_root: Path,
    split: str,
    crack_class_idx: int,
) -> int:
    """处理单个 split，返回成功处理的文件数。"""
    try:
        from skimage.morphology import skeletonize
    except ImportError as e:
        raise ImportError(
            "precompute_skeletons 需要 scikit-image：pip install scikit-image"
        ) from e

    ann_dir  = data_root / "ann_dir" / split
    skel_dir = data_root / "skel_dir" / split
    skel_dir.mkdir(parents=True, exist_ok=True)

    mask_paths = sorted(ann_dir.glob("*.png"))
    if not mask_paths:
        logger.warning(f"[{split}] ann_dir 未找到任何 .png 文件: {ann_dir}")
        return 0

    count = 0
    for mask_path in mask_paths:
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
        crack_binary = (mask == crack_class_idx).astype(np.uint8)   # {0,1}

        if crack_binary.sum() == 0:
            # 无裂缝像素：骨架全零
            skel = np.zeros_like(crack_binary, dtype=np.uint8)
        else:
            skel = skeletonize(crack_binary).astype(np.uint8)       # {0,1}

        # 保存为 {0, 255} 的灰度 PNG
        out_path = skel_dir / mask_path.name
        Image.fromarray(skel * 255).save(out_path)
        count += 1

    logger.success(f"[{split}] 骨架掩码生成完成: {count} 张 → {skel_dir}")
    return count


def main():
    parser = argparse.ArgumentParser(
        description="离线预计算裂缝骨架掩码"
    )
    parser.add_argument("--data_root", required=True, help="数据集根目录")
    parser.add_argument(
        "--splits", nargs="+", default=["train"],
        help="要处理的 split 列表，默认 train"
    )
    parser.add_argument(
        "--crack_class_idx", type=int, default=1,
        help="裂缝类别索引，默认 1"
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"data_root 不存在: {data_root}")

    total = 0
    for split in args.splits:
        total += precompute_split(data_root, split, args.crack_class_idx)

    logger.success(f"全部完成：共处理 {total} 张掩码")


if __name__ == "__main__":
    main()
