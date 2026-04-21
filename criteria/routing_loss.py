"""
criteria/routing_loss.py

GT 监督路由损失（Routing Loss）。

通过 GT mask 直接监督 MRM 输出的 α 路由权重，打破路由塌缩：
  - 裂缝像素（class 1）      → 期望 α → 1.0（送入线型流）
  - 面型病害像素（class 2-6）→ 期望 α → 0.0（送入面型流）
  - 背景像素（class 0）      → 忽略，不参与 loss 计算

NaN 安全保证
-----------
1. alpha 在 TMDSSegmentor.forward 中已 clamp 到 [0.1, 0.9]，
   BCE 的 log 输入始终在 (0,1) 开区间内，不会产生 log(0)=-inf。
2. 当 batch 内无任何病害像素时，返回 alpha.sum() * 0.0，
   保留梯度图的连通性同时贡献零损失，不会出现 0/0。
3. mask 下采样使用最近邻插值，不引入非整数类别值。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RoutingLoss(nn.Module):
    """
    GT 监督路由损失。

    Args:
        crack_class_idx:  裂缝类别 ID（default=1）
        areal_class_idxs: 面型病害类别 ID 元组（default=(2,3,4,5,6)）
    """

    def __init__(
        self,
        crack_class_idx: int = 1,
        areal_class_idxs: tuple[int, ...] = (2, 3, 4, 5, 6),
    ):
        super().__init__()
        self.crack_idx   = crack_class_idx
        self.areal_idxs  = areal_class_idxs

    def forward(self, alpha: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            alpha: [B, 1, Ha, Wa]  MRM 路由权重，已 clamp 到 [0.1, 0.9]
            masks: [B, H,  W ]     GT 整数类别标签

        Returns:
            标量 loss（当无病害像素时返回零）
        """
        B, _, Ha, Wa = alpha.shape

        # ── 1. 将 GT mask 下采样到 alpha 分辨率（最近邻，保留类别整数值）──────
        masks_down = F.interpolate(
            masks.float().unsqueeze(1),   # [B,1,H,W]
            size=(Ha, Wa),
            mode="nearest",
        ).squeeze(1).long()              # [B, Ha, Wa]

        # ── 2. 构建监督 mask 和 target ────────────────────────────────────────
        crack_mask = masks_down == self.crack_idx              # [B, Ha, Wa] bool
        areal_mask = torch.zeros_like(crack_mask)
        for c in self.areal_idxs:
            areal_mask = areal_mask | (masks_down == c)

        valid_mask = crack_mask | areal_mask                   # 有病害的位置

        # 无病害像素时保留梯度连通性，贡献零 loss
        if not valid_mask.any():
            return alpha.sum() * 0.0

        # target: crack→1.0, areal→0.0，背景不参与
        target = crack_mask.float()                            # [B, Ha, Wa]

        # ── 3. 计算 BCE（alpha 已在 [0.1,0.9]，log 不会溢出）────────────────
        # F.binary_cross_entropy 在 AMP autocast 上下文中不安全（PyTorch 禁止
        # autocast 将其输入降精度），此处强制 float32 执行。
        alpha_2d = alpha.squeeze(1)                            # [B, Ha, Wa]
        with torch.amp.autocast(alpha.device.type, enabled=False):
            bce = F.binary_cross_entropy(
                alpha_2d.float(), target.float(), reduction="none",
            )                                                  # [B, Ha, Wa]

        # ── 4. 按类别均衡求均值（消除 crack/areal 像素数量不均衡导致的梯度偏置）──
        # 直接对所有 valid 像素平均时，areal 像素（85~95%）会压制 crack 像素（5~15%），
        # 导致路由器学到"全面型"的捷径解。改为对两类分别求均值后再平均，
        # 使 crack 和 areal 各贡献 50% 梯度，与各自像素数量无关。
        crack_loss = (
            bce[crack_mask].mean() if crack_mask.any() else alpha.sum() * 0.0
        )
        areal_loss = (
            bce[areal_mask].mean() if areal_mask.any() else alpha.sum() * 0.0
        )
        # 若某类在本 batch 中完全缺席，只用出现的那一类，避免 0/2 稀释
        n_types = (crack_mask.any().float() + areal_mask.any().float()).clamp(min=1.0)
        loss = (crack_loss + areal_loss) / n_types
        return loss
