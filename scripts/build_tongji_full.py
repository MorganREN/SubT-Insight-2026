"""
build_tongji_full.py

把 ``dataset/tongji_data_awesome`` 的 ``train`` + ``valid`` 两个 split 合并成
单一的 ``train`` split，落地到 ``dataset/tongji_data_awesome_full/``。
全程使用 **符号链接**，不复制原图与原 mask；幂等，重复运行无副作用。

用法
----
    conda run -n subt-2026 python scripts/build_tongji_full.py
    conda run -n subt-2026 python scripts/build_tongji_full.py \\
        --src dataset/tongji_data_awesome \\
        --dst dataset/tongji_data_awesome_full
"""

from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
from pathlib import Path

from loguru import logger


def ensure_symlink(src: Path, dst: Path) -> None:
    """确保 ``dst`` 是指向 ``src`` 的符号链接；若已存在则不动。"""
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src.resolve())


def merge_tongji_full(src: Path, dst: Path) -> int:
    """
    把 ``src/img_dir/{train,valid}`` 与 ``src/ann_dir/{train,valid}`` 下的所有
    (image, mask) 对，符号链接到 ``dst/img_dir/train`` 与 ``dst/ann_dir/train``。

    Returns
    -------
    int
        成功落地的 (image, mask) 对数。
    """
    img_dst = dst / "img_dir" / "train"
    ann_dst = dst / "ann_dir" / "train"
    img_dst.mkdir(parents=True, exist_ok=True)
    ann_dst.mkdir(parents=True, exist_ok=True)

    n = 0
    for split in ("train", "valid"):
        src_imgs = src / "img_dir" / split
        src_anns = src / "ann_dir" / split
        if not src_imgs.exists():
            logger.warning(f"源目录缺失，跳过: {src_imgs}")
            continue
        for img in sorted(src_imgs.glob("*.jpg")):
            stem = img.stem
            mask = src_anns / f"{stem}.png"
            if not mask.exists():
                logger.warning(f"缺对应 mask，跳过: {img.name}")
                continue
            ensure_symlink(img, img_dst / f"{stem}.jpg")
            ensure_symlink(mask, ann_dst / f"{stem}.png")
            n += 1

    logger.info(f"{dst} 合并完成: {n} 张样本（全部归入 train split）")
    return n


def main():
    parser = argparse.ArgumentParser(
        description="把 tongji_data_awesome 的 train+valid 合并成单一 train split（符号链接）。"
    )
    parser.add_argument("--src", type=Path, default=Path("dataset/tongji_data_awesome"),
                        help="源数据集根目录（含 img_dir/{train,valid} 与 ann_dir/{train,valid}）")
    parser.add_argument("--dst", type=Path, default=Path("dataset/tongji_data_awesome_full"),
                        help="目标数据集根目录")
    args = parser.parse_args()
    merge_tongji_full(args.src, args.dst)


if __name__ == "__main__":
    main()
