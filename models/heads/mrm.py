from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class MorphologicalRoutingModule(nn.Module):
    """
    形态路由模块（MRM）。

    以 C3（H/16）特征为输入，通过非对称卷积和方形卷积分别感知
    线型结构（裂缝）和面型结构（渗漏/剥落），输出软路由权重 alpha。

    alpha ∈ [0,1]：
        → 1：当前位置更可能是线型病害（裂缝）
        → 0：当前位置更可能是面型病害

    Args:
        in_channels: C3 的通道数（ConvNeXt-Tiny 为 384）
        mid_channels: 中间特征维度
    """
    def __init__(self, in_channels: int = 384, mid_channels: int = 64):
        super().__init__()
        # 线型分支：交叉非对称卷积（模拟细长条纹感知）
        self.linear_branch = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=(1, 15), padding=(0, 7), bias=False),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=(15, 1), padding=(7, 0), bias=False),
            nn.GroupNorm(max(1, mid_channels // 8), mid_channels),  # GN 不依赖 batch 统计，bs=2 也稳定
            nn.GELU(),
        )
        # 面型分支：方形卷积（感知块状区域）
        self.areal_branch = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=7, padding=3, bias=False),
            nn.GroupNorm(max(1, mid_channels // 8), mid_channels),
            nn.GELU(),
        )
        # 路由头：输出 alpha
        self.router = nn.Sequential(
            nn.Conv2d(mid_channels * 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, c3: torch.Tensor) -> torch.Tensor:
        """
        Args:
            c3: [B, in_channels, H/16, W/16]
        Returns:
            alpha: [B, 1, H/16, W/16]  路由权重图
        """
        l = self.linear_branch(c3)
        a = self.areal_branch(c3)
        return self.router(torch.cat([l, a], dim=1))
