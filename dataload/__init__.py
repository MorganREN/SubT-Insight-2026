"""
dataload/
隧道缺陷语义分割数据加载模块。

公开接口
--------
    SegmentationAugmentation      数据增强（train/val/test 三模式）
    TunnelDefectDataset           PyTorch Dataset（支持多数据集根目录合并）
    SegmentationDataLoaderFactory DataLoader 工厂类
    build_dataloaders             便捷工厂函数，一行获取所有 split 的 DataLoader
    NUM_CLASSES                   类别数量（5）
    CLASS_NAMES                   类别名称元组
    CLASS_COLORS                  类别可视化颜色元组

快速上手
--------
    from dataload import build_dataloaders

    loaders = build_dataloaders(
        data_roots=["dataset/aug_data1"],
        batch_size=8,
        num_workers=4,
        input_size=512,
    )
    for images, masks in loaders["train"]:
        # images: Tensor float32 (B,3,512,512)
        # masks:  Tensor int64   (B,512,512)
        ...
"""

from .augmentation import SegmentationAugmentation
from .dataset import (
    TunnelDefectDataset,
    NUM_CLASSES,
    CLASS_NAMES,
    CLASS_COLORS,
)
from .dataloader import SegmentationDataLoaderFactory, build_dataloaders

__all__ = [
    # 增强
    "SegmentationAugmentation",
    # 数据集
    "TunnelDefectDataset",
    "NUM_CLASSES",
    "CLASS_NAMES",
    "CLASS_COLORS",
    # 数据加载
    "SegmentationDataLoaderFactory",
    "build_dataloaders",
]
