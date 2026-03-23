"""
criteria/
损失函数与评估指标模块。

Losses
------
    WeightedCrossEntropyLoss  — 带类别权重 + label smoothing 的交叉熵
    DiceLoss                  — Soft Dice Loss（支持 ignore_index / class weights）
    FocalLoss                 — Focal Loss（Lin et al. 2017）
    CombinedLoss              — 多损失加权组合，.last_components 可用于日志
    TopologyLoss              — 基于持续同调的裂缝拓扑连续性损失（需 gudhi）
    SkeletonLoss              — 裂缝骨架一致性损失（需预计算骨架掩码）

Metrics
-------
    SegEvaluator              — 基于混淆矩阵的语义分割评估器
"""

from criteria.losses import (
    CombinedLoss,
    DiceLoss,
    FocalLoss,
    WeightedCrossEntropyLoss,
)
from criteria.metrics import SegEvaluator
from criteria.topo_loss import TopologyLoss
from criteria.skeleton_loss import SkeletonLoss

__all__ = [
    # Losses
    "WeightedCrossEntropyLoss",
    "DiceLoss",
    "FocalLoss",
    "CombinedLoss",
    "TopologyLoss",
    "SkeletonLoss",
    # Metrics
    "SegEvaluator",
]
