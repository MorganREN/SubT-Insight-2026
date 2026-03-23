"""
criteria/skeleton_loss.py

骨架一致性损失（Skeleton Consistency Loss）。

原理
----
将预测概率图仅在裂缝 GT 骨架像素处与目标（全 1）计算 BCE，
专门惩罚裂缝中心线的预测准确性，而非整体区域像素。

骨架掩码由离线脚本 tools/precompute_skeletons.py 预先生成：
  {data_root}/skel_dir/{split}/{name}.png  （像素值：0 or 255）

前向接口
--------
    logits    : [B, C, H, W]    预测 logits
    skel_mask : [B, H, W]       骨架掩码（1 = 骨架像素，0 = 非骨架）
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SkeletonLoss(nn.Module):
    """
    骨架一致性损失。

    Parameters
    ----------
    crack_class_idx : int
        裂缝类别索引，默认 1。
    """

    def __init__(self, crack_class_idx: int = 1):
        super().__init__()
        self.crack_class_idx = crack_class_idx

    def forward(
        self,
        logits: torch.Tensor,
        skel_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        logits    : [B, C, H, W]
        skel_mask : [B, H, W] int64 或 bool，1 = 骨架像素

        Returns
        -------
        scalar Tensor（在骨架像素处的 BCE，无骨架像素时返回 0）
        """
        probs = F.softmax(logits, dim=1)
        crack_prob = probs[:, self.crack_class_idx]   # [B, H, W]

        valid = skel_mask.bool()
        if not valid.any():
            return crack_prob.sum() * 0.0

        pred   = crack_prob[valid].clamp(1e-4, 1 - 1e-4)  # 防止 log(0) 在 FP16 下溢出
        target = torch.ones_like(pred)                    # 骨架处目标概率=1
        return F.binary_cross_entropy(pred, target)
