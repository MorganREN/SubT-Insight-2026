"""语义分割 DataLoader 工厂。"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import torch
from torch.utils.data import DataLoader

from .augmentation import SegmentationAugmentation
from .dataset import TunnelDefectDataset
from loguru import logger


class SegmentationDataLoaderFactory:
    """
    语义分割 DataLoader 工厂类。

    负责为每个 split（train / val / test）创建对应的
    TunnelDefectDataset + SegmentationAugmentation + DataLoader 三元组。

    Parameters
    ----------
    data_roots : str | Sequence[str]
        数据集根目录，可传多个以合并数据。
    batch_size : int
        每个 batch 的样本数。默认 8。
    num_workers : int
        DataLoader 并行读取进程数。默认 4（设为 0 可关闭多进程，便于调试）。
    input_size : int
        增强后图像的正方形边长（像素）。默认 512。
    pin_memory : bool
        是否将数据固定到 CUDA 页锁定内存（有 GPU 时建议 True）。
        默认自动检测 CUDA 是否可用。
    persistent_workers : bool
        是否保持 worker 进程不销毁（num_workers>0 时有效）。默认 True。
    aug_kwargs : dict, optional
        额外传递给 SegmentationAugmentation 的关键字参数
        （如 hflip_p、elastic_p 等），仅对 train 增强生效。
    splits : list[str]
        要创建 DataLoader 的 split 列表。
        默认 ["train", "val", "test"]。

    Attributes
    ----------
    datasets : dict[str, TunnelDefectDataset]
        各 split 的 Dataset 对象，可用于访问类别权重等信息。
    """

    def __init__(
        self,
        data_roots: Union[str, Sequence[str]],
        *,
        batch_size: int = 8,
        num_workers: int = 4,
        input_size: int = 512,
        pin_memory: Optional[bool] = None,
        persistent_workers: bool = True,
        aug_kwargs: Optional[dict] = None,
        splits: Optional[List[str]] = None,
    ):
        if isinstance(data_roots, str):
            data_roots = [data_roots]
        self.data_roots  = list(data_roots)
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.input_size  = input_size
        self.aug_kwargs  = aug_kwargs or {}
        self.splits      = splits or ["train", "val", "test"]

        # 自动检测 pin_memory
        if pin_memory is None:
            pin_memory = torch.cuda.is_available()
        self.pin_memory = pin_memory

        # persistent_workers 只在 num_workers>0 时有意义
        self.persistent_workers = persistent_workers and (num_workers > 0)

        # 构建各 split 的 Dataset
        self.datasets: Dict[str, TunnelDefectDataset] = {}
        self._loaders: Dict[str, DataLoader] = {}

        for split in self.splits:
            # 内部 split 名称规范化：val 和 valid 都映射到 "valid" 目录
            aug_mode = "train" if split == "train" else "val"
            aug = SegmentationAugmentation(
                mode=aug_mode,
                input_size=input_size,
                **self.aug_kwargs,
            )
            ds = TunnelDefectDataset(
                data_roots=self.data_roots,
                split=split,
                augmentation=aug,
            )
            self.datasets[split] = ds
            self._loaders[split] = self._make_loader(ds, split)

        logger.info(
            f"SegmentationDataLoaderFactory 构建完成:\n"
            + "\n".join(
                f"  [{s}] {len(self.datasets[s])} 样本, "
                f"batch_size={batch_size}, "
                f"iter/epoch≈{len(self.datasets[s]) // batch_size + 1}"
                for s in self.splits
                if len(self.datasets.get(s, [])) > 0
            )
        )

    def _make_loader(self, dataset: TunnelDefectDataset, split: str) -> DataLoader:
        """为给定 split 创建 DataLoader。"""
        is_train = (split == "train")
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=is_train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=is_train,         # train 丢弃不足一批的尾部样本
        )

    def get(self, split: str) -> DataLoader:
        """获取指定 split 的 DataLoader。"""
        if split not in self._loaders:
            raise KeyError(
                f"split='{split}' 不存在，可用: {list(self._loaders.keys())}"
            )
        return self._loaders[split]

    def __getitem__(self, split: str) -> DataLoader:
        """支持 factory['train'] 语法。"""
        return self.get(split)

    def get_class_weights(self, split: str = "train") -> torch.Tensor:
        """
        代理调用 Dataset.get_class_weights()。

        Returns shape=(num_classes,) float32 权重张量，文档见 TunnelDefectDataset。
        """
        return self.datasets[split].get_class_weights()

    def __repr__(self) -> str:
        lines = [
            f"SegmentationDataLoaderFactory(",
            f"  data_roots={[r for r in self.data_roots]},",
            f"  batch_size={self.batch_size},",
            f"  input_size={self.input_size},",
            f"  num_workers={self.num_workers},",
        ]
        for s in self.splits:
            ds = self.datasets.get(s)
            n = len(ds) if ds else 0
            lines.append(f"  [{s}] {n} samples,")
        lines.append(")")
        return "\n".join(lines)


def build_dataloaders(
    data_roots: Union[str, Sequence[str]],
    *,
    batch_size: int = 8,
    num_workers: int = 4,
    input_size: int = 512,
    pin_memory: Optional[bool] = None,
    splits: Optional[List[str]] = None,
    aug_kwargs: Optional[dict] = None,
) -> Dict[str, DataLoader]:
    """一行代码获取所有 split 的 DataLoader 字典。"""
    factory = SegmentationDataLoaderFactory(
        data_roots=data_roots,
        batch_size=batch_size,
        num_workers=num_workers,
        input_size=input_size,
        pin_memory=pin_memory,
        aug_kwargs=aug_kwargs,
        splits=splits or ["train", "val", "test"],
    )
    return {split: factory.get(split) for split in factory.splits}
