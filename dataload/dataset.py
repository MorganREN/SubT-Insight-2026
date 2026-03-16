"""
dataload/dataset.py
隧道缺陷语义分割 Dataset。

目录约定
--------
    {data_root}/
        img_dir/{split}/{name}.jpg      ← RGB 图像
        ann_dir/{split}/{name}.png      ← 灰度掩码（像素值 = 类别 ID）

类别映射（与 tongji_data 一致）
-------------------------------------
    0  background（背景）
    1  crack（裂缝）
    2  leakage_b（渗漏 B）
    3  leakage_w（渗漏 W）
    4  leakage_g（渗漏 G）
    5  lining_falling_off（衬砌脱落）
    6  segment_damage（管片损伤）
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from loguru import logger


#: 类别名称（索引即类别 ID）
CLASS_NAMES: Tuple[str, ...] = (
    "background",
    "crack",
    "leakage_b",
    "leakage_w",
    "leakage_g",
    "lining_falling_off",
    "segment_damage",
)

#: 可视化 RGB 颜色（每类一个颜色）
CLASS_COLORS: Tuple[Tuple[int, int, int], ...] = (
    (0,   0,   0),    # 0 background  黑
    (255, 0,   0),    # 1 crack       红
    (255, 128, 0),    # 2 leakage_b   橙
    (0,   0,   255),  # 3 leakage_w   蓝
    (0,   255, 255),  # 4 leakage_g   青
    (255, 255, 0),    # 5 lining_off  黄
    (255, 0,   255),  # 6 seg_damage  洋红
)

#: 类别数量
NUM_CLASSES: int = len(CLASS_NAMES)


# ──────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────────────────────────────────────

def _collect_pairs(
    data_root: str | Path,
    split: str,
) -> List[Tuple[Path, Path]]:
    """扫描单个 data_root，收集 (img_path, mask_path) 配对。"""
    root    = Path(data_root)
    img_dir = root / "img_dir" / split
    ann_dir = root / "ann_dir" / split

    if not img_dir.exists():
        logger.warning(f"img_dir 不存在，跳过: {img_dir}")
        return []

    pairs: List[Tuple[Path, Path]] = []
    for img_path in sorted(img_dir.glob("*.jpg")):
        stem     = img_path.stem
        ann_path = ann_dir / f"{stem}.png"
        if ann_path.exists():
            pairs.append((img_path, ann_path))
        else:
            logger.warning(f"缺少对应 mask，跳过图像: {img_path.name}")

    return pairs


class TunnelDefectDataset(Dataset):
    """
    隧道缺陷语义分割数据集。

    支持同时加载多个 data_root，方便合并 aug_data1 / aug_data_extra 等子集。

    Parameters
    ----------
    data_roots : str | Sequence[str]
        一个或多个数据集根目录路径。每个目录需包含
        ``img_dir/{split}/`` 和 ``ann_dir/{split}/`` 子目录。
    split : str
        数据集划分："train" / "valid" / "test"。
    augmentation : callable, optional
        接受关键字参数 ``image`` (np.ndarray uint8 RGB) 和
        ``mask`` (np.ndarray uint8) 的增强对象，返回含
        ``"image"`` (Tensor float32) 和 ``"mask"`` (Tensor int64) 的 dict。
        传 ``None`` 则仅读取图像和掩码，不做任何变换（返回原始 ndarray）。
    image_suffix : str
        图像文件后缀，默认 ".jpg"。
    mask_suffix : str
        掩码文件后缀，默认 ".png"。

    Attributes
    ----------
    num_classes : int
        类别总数（7）。
    class_names : tuple[str]
        各类别名称。
    class_colors : tuple[tuple[int,int,int]]
        各类别 RGB 可视化颜色。
    pairs : list[tuple[Path, Path]]
        所有 (img_path, mask_path) 配对。
    """

    # 类别元信息（直接引用模块级常量，方便外部访问）
    num_classes:  int                            = NUM_CLASSES
    class_names:  Tuple[str, ...]                = CLASS_NAMES
    class_colors: Tuple[Tuple[int, int, int], ...] = CLASS_COLORS

    def __init__(
        self,
        data_roots: str | Sequence[str | Path],
        split: str,
        augmentation: Optional[Callable] = None,
        *,
        image_suffix:  str = ".jpg",
        mask_suffix:   str = ".png",
        ignore_index:  int = 255,
    ):
        # 统一转 list
        if isinstance(data_roots, (str, Path)):
            data_roots = [data_roots]

        self.data_roots   = [Path(r) for r in data_roots]
        # "val" 是 "valid" 的别名，目录都用 "valid"命名
        _split = split.lower()
        self.split        = "valid" if _split == "val" else _split
        self.augmentation = augmentation
        self.image_suffix = image_suffix
        self.mask_suffix  = mask_suffix
        self.ignore_index = ignore_index

        # 收集所有根目录下的 (img, mask) 配对
        self.pairs: List[Tuple[Path, Path]] = []
        for root in self.data_roots:
            new_pairs = _collect_pairs(root, self.split)
            logger.info(
                f"[{split}] {root.name}: {len(new_pairs)} 个样本"
            )
            self.pairs.extend(new_pairs)

        if len(self.pairs) == 0:
            logger.warning(
                f"split='{split}' 没有找到任何样本，"
                f"请检查 data_roots={[str(r) for r in self.data_roots]}"
            )
        else:
            logger.success(
                f"[{split}] 共加载 {len(self.pairs)} 个样本"
                f"（来自 {len(self.data_roots)} 个数据集）"
            )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        """加载并返回第 idx 个样本。"""
        img_path, ann_path = self.pairs[idx]

        # ── 读取图像（RGB）──
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)

        # ── 读取掩码（灰度，uint8 类别索引）──
        mask = np.array(Image.open(ann_path).convert("L"), dtype=np.uint8)

        # ── 安全检查：将越界且非 ignore_index 的像素置为背景（0）──
        # 注意：不能用 clip(0, num_classes-1)，那会把 ignore_index=255 压成最后一类
        out_of_range = (mask >= self.num_classes) & (mask != self.ignore_index)
        if out_of_range.any():
            mask = mask.copy()
            mask[out_of_range] = 0

        # ── 数据增强 ──
        if self.augmentation is not None:
            output = self.augmentation(image=image, mask=mask)
            image = output["image"]   # Tensor float32 (3,H,W)
            mask  = output["mask"]    # Tensor int64   (H,W)
        else:
            # 无增强时，仍转为 Tensor（方便 DataLoader 批处理）
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask  = torch.from_numpy(mask).long()

        return image, mask

    def get_class_weights(self) -> torch.Tensor:
        """统计像素频率并返回类别权重。"""
        counts = np.zeros(self.num_classes, dtype=np.int64)
        for _, ann_path in self.pairs:
            mask = np.array(Image.open(ann_path).convert("L"), dtype=np.uint8)
            # 同 __getitem__：保留 ignore_index，仅修正其他越界 ID
            out_of_range = (mask >= self.num_classes) & (mask != self.ignore_index)
            if out_of_range.any():
                mask = mask.copy()
                mask[out_of_range] = 0
            for c in range(self.num_classes):
                counts[c] += (mask == c).sum()

        freq = counts / counts.sum()
        # 平滑避免零频类别导致除零
        freq = np.where(freq == 0, 1e-6, freq)
        weights = 1.0 / freq
        weights /= weights.sum()           # 归一化到总和=1

        logger.info("类别像素频率:")
        for i, (name, f, w) in enumerate(
            zip(self.class_names, freq, weights)
        ):
            logger.info(f"  [{i}] {name:<12s}  freq={f:.4f}  weight={w:.4f}")

        return torch.tensor(weights, dtype=torch.float32)

    def __repr__(self) -> str:
        roots_str = ", ".join(r.name for r in self.data_roots)
        return (
            f"TunnelDefectDataset("
            f"split='{self.split}', "
            f"n={len(self.pairs)}, "
            f"roots=[{roots_str}])"
        )
