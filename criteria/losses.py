"""
criteria/losses.py

包含以下类:
    WeightedCrossEntropyLoss  — 带类别权重的交叉熵（应对类别不均衡）
    DiceLoss                  — Soft Dice Loss（对小目标友好）
    FocalLoss                 — Focal Loss（聚焦难样本，Lin et al. 2017）
    CombinedLoss              — 可配置的多损失加权组合
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


class WeightedCrossEntropyLoss(nn.Module):
    """带类别权重的交叉熵损失。"""

    def __init__(
        self,
        num_classes: int,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = 255,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.num_classes     = num_classes
        self.ignore_index    = ignore_index
        self.label_smoothing = label_smoothing

        if class_weights is not None:
            assert class_weights.shape == (num_classes,), (
                f"class_weights 形状应为 ({num_classes},)，实际: {class_weights.shape}"
            )
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.register_buffer("class_weights", None)

        logger.debug(
            f"WeightedCrossEntropyLoss: num_classes={num_classes}, "
            f"label_smoothing={label_smoothing}, "
            f"class_weights={'自动' if class_weights is None else '已设置'}"
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.contiguous()
        targets = targets.contiguous()
        return F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )

    def extra_repr(self) -> str:
        return (
            f"num_classes={self.num_classes}, "
            f"ignore_index={self.ignore_index}, "
            f"label_smoothing={self.label_smoothing}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# DiceLoss
# ──────────────────────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """Soft Dice Loss。"""

    def __init__(
        self,
        num_classes: int,
        smooth: float = 1e-5,
        ignore_index: int = 255,
        reduction: str = "mean",
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        assert reduction in ("mean", "sum", "none"), \
            f"reduction 必须是 'mean'/'sum'/'none'，得到: '{reduction}'"
        self.num_classes  = num_classes
        self.smooth       = smooth
        self.ignore_index = ignore_index
        self.reduction    = reduction

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.register_buffer("class_weights", None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.contiguous()
        targets = targets.contiguous()
        B, C, H, W = logits.shape

        # Softmax 转为概率
        probs = logits.softmax(dim=1)  # (B, C, H, W)

        # One-hot 编码目标：将 ignore_index 像素暂时置 0（会被 valid_mask 排除）
        valid_mask = (targets != self.ignore_index)           # (B, H, W) bool

        # 用于 one-hot 的目标副本，将 ignore 像素映射到 0（不影响结果，后续 mask 掉）
        targets_clean = targets.clone()
        targets_clean[~valid_mask] = 0

        # one-hot: (B, C, H, W)
        targets_onehot = F.one_hot(targets_clean, num_classes=C)  # (B,H,W,C)
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).contiguous().float()  # (B,C,H,W)

        # 将 ignore 区域在 probs 和 targets_onehot 中都置零（排除计算）
        valid_mask_4d = valid_mask.unsqueeze(1).float()  # (B,1,H,W)
        probs          = probs          * valid_mask_4d
        targets_onehot = targets_onehot * valid_mask_4d

        # 展平空间维度: (B, C, H*W)
        probs_flat   = probs.reshape(B, C, -1)           # (B, C, H*W)
        targets_flat = targets_onehot.reshape(B, C, -1)  # (B, C, H*W)

        # 计算各类别 Dice: mean over batch
        # intersection: (B, C)  →  mean over B  →  (C,)
        intersection = (probs_flat * targets_flat).sum(dim=2)  # (B, C)
        denominator  = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)  # (B, C)

        dice_per_class = (2.0 * intersection + self.smooth) / \
                         (denominator + self.smooth)         # (B, C)
        dice_per_class = dice_per_class.mean(dim=0)          # (C,) — batch mean

        dice_loss_per_class = 1.0 - dice_per_class           # (C,)

        # 可选加权
        if self.class_weights is not None:
            dice_loss_per_class = dice_loss_per_class * self.class_weights

        if self.reduction == "mean":
            return dice_loss_per_class.mean()
        elif self.reduction == "sum":
            return dice_loss_per_class.sum()
        else:
            return dice_loss_per_class

    def extra_repr(self) -> str:
        return (
            f"num_classes={self.num_classes}, "
            f"smooth={self.smooth}, "
            f"reduction='{self.reduction}'"
        )


# ──────────────────────────────────────────────────────────────────────────────
# FocalLoss
# ──────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal Loss。"""

    def __init__(
        self,
        num_classes: int,
        alpha: float | Sequence[float] = 1.0,
        gamma: float = 2.0,
        ignore_index: int = 255,
        reduction: str = "mean",
    ):
        super().__init__()
        self.num_classes  = num_classes
        self.gamma        = gamma
        self.ignore_index = ignore_index
        self.reduction    = reduction

        # alpha 统一转为 buffer
        if isinstance(alpha, (float, int)):
            alpha_tensor = torch.full((num_classes,), float(alpha))
        else:
            alpha_tensor = torch.as_tensor(alpha, dtype=torch.float32)
            assert alpha_tensor.shape == (num_classes,), \
                f"alpha 长度应为 {num_classes}，实际: {alpha_tensor.shape}"
        self.register_buffer("alpha", alpha_tensor)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.contiguous()
        targets = targets.contiguous()
        B, C, H, W = logits.shape

        # 计算标准 CE（逐像素，不降维）→ (B, H, W)
        ce_loss = F.cross_entropy(
            logits, targets,
            reduction="none",
            ignore_index=self.ignore_index,
        )

        # 计算每个像素属于其真实类别的概率 p_t
        with torch.no_grad():
            # 对 ignore 像素，用 target 的 clamp 避免越界
            targets_safe = targets.clamp(0, C - 1)
            probs = logits.softmax(dim=1)                         # (B, C, H, W)
            p_t   = probs.gather(1, targets_safe.unsqueeze(1))    # (B, 1, H, W)
            p_t   = p_t.squeeze(1)                                 # (B, H, W)

            # 调制因子: (1 - p_t)^gamma
            modulating_factor = (1.0 - p_t) ** self.gamma         # (B, H, W)

            # alpha 权重: 取每个像素对应类别的 alpha
            alpha_t = self.alpha[targets_safe]                     # (B, H, W)

        # 对 ignore 像素，ce_loss 已为 0（F.cross_entropy 的 ignore_index 机制）
        focal_loss = alpha_t * modulating_factor * ce_loss         # (B, H, W)

        # 过滤 ignore 像素后做 reduction
        valid_mask = (targets != self.ignore_index)                # (B, H, W)
        focal_loss = focal_loss[valid_mask]

        if self.reduction == "mean":
            return focal_loss.mean() if focal_loss.numel() > 0 else logits.sum() * 0
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss

    def extra_repr(self) -> str:
        return (
            f"num_classes={self.num_classes}, "
            f"gamma={self.gamma}, "
            f"reduction='{self.reduction}'"
        )


class CombinedLoss(nn.Module):
    """多损失加权组合。"""

    SUPPORTED = {"ce", "dice", "focal"}

    def __init__(
        self,
        losses: List[str],
        weights: Optional[List[float]] = None,
        num_classes: int = 5,
        ignore_index: int = 255,
        ce_kwargs:    Optional[dict] = None,
        dice_kwargs:  Optional[dict] = None,
        focal_kwargs: Optional[dict] = None,
    ):
        super().__init__()

        # 校验
        for name in losses:
            assert name in self.SUPPORTED, \
                f"不支持的损失: '{name}'，可选: {self.SUPPORTED}"
        if weights is None:
            weights = [1.0] * len(losses)
        assert len(weights) == len(losses), \
            f"losses 和 weights 长度不一致: {len(losses)} vs {len(weights)}"

        self.loss_names = losses
        self.loss_weights = weights

        # 构建各子损失
        self._loss_modules = nn.ModuleDict()

        if "ce" in losses:
            kw = dict(num_classes=num_classes, ignore_index=ignore_index)
            kw.update(ce_kwargs or {})
            self._loss_modules["ce"] = WeightedCrossEntropyLoss(**kw)

        if "dice" in losses:
            kw = dict(num_classes=num_classes, ignore_index=ignore_index)
            kw.update(dice_kwargs or {})
            self._loss_modules["dice"] = DiceLoss(**kw)

        if "focal" in losses:
            kw = dict(num_classes=num_classes, ignore_index=ignore_index)
            kw.update(focal_kwargs or {})
            self._loss_modules["focal"] = FocalLoss(**kw)

        # 保存最近一次各子损失值（用于日志）
        self.last_components: Dict[str, float] = {}

        logger.debug(
            f"CombinedLoss: {' + '.join(f'{w}×{n}' for n, w in zip(losses, weights))}"
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        total = torch.zeros((), dtype=torch.float32, device=logits.device)  # float32 累加器
        self.last_components.clear()

        for name, weight in zip(self.loss_names, self.loss_weights):
            component = self._loss_modules[name](logits, targets)
            self.last_components[name] = float(component.detach().item())
            total = total + weight * component

        return total

    def extra_repr(self) -> str:
        parts = " + ".join(
            f"{w}×{n}" for n, w in zip(self.loss_names, self.loss_weights)
        )
        return f"formula='{parts}'"
