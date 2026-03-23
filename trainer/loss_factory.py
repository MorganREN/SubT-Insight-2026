"""
trainer/loss_factory.py

损失函数工厂。

公开接口
--------
    build_loss(cfg, class_weights, device)
        标准训练损失（TunnelSegmentor 使用）。

    build_tmds_criterion(cfg, stage, class_weights, device)
        TMDS 三阶段训练损失（TMDSSegmentor 使用）。
        返回 TMDSCriterion，其 forward(outputs, masks, skel_masks) 接口统一
        处理 dict 或 Tensor 形式的 outputs。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from criteria import CombinedLoss, TopologyLoss, SkeletonLoss

from .config import TrainConfig


# ──────────────────────────────────────────────────────────────────────────────
# 内部表：loss_name → (loss 列表, 默认权重列表)
# ──────────────────────────────────────────────────────────────────────────────

_LOSS_TABLE: dict[str, tuple[list[str], list[float]]] = {
    "ce":            (["ce"],                    [1.0]),
    "dice":          (["dice"],                  [1.0]),
    "focal":         (["focal"],                 [1.0]),
    "ce+dice":       (["ce", "dice"],            [1.0, 1.0]),
    "ce+focal":      (["ce", "focal"],           [1.0, 1.0]),
    "dice+focal":    (["dice", "focal"],         [2.0, 1.0]),
    "ce+dice+focal": (["ce", "dice", "focal"],   [1.0, 1.0, 2.0]),
}


def _build_combined_loss(
    loss_name: str,
    num_classes: int,
    class_weights: Optional[torch.Tensor],
    loss_weights: Optional[tuple[float, ...]],
    device: Optional[torch.device],
) -> CombinedLoss:
    """根据 loss_name 构建 CombinedLoss 并移至 device。"""
    if loss_name not in _LOSS_TABLE:
        raise ValueError(
            f"不支持的 loss_name: '{loss_name}'，"
            f"可选: {list(_LOSS_TABLE.keys())}"
        )
    losses, default_weights = _LOSS_TABLE[loss_name]
    weights = list(loss_weights) if loss_weights is not None else default_weights
    if len(weights) != len(losses):
        raise ValueError(
            f"loss_weights 长度 ({len(weights)}) 与 loss_name='{loss_name}' "
            f"所含损失数 ({len(losses)}) 不一致"
        )
    if class_weights is not None and device is not None:
        class_weights = class_weights.to(device)
    cw_kw = {"class_weights": class_weights} if class_weights is not None else {}
    criterion = CombinedLoss(
        losses=losses,
        weights=weights,
        num_classes=num_classes,
        ce_kwargs=cw_kw,
        dice_kwargs=cw_kw,
        focal_kwargs={},
    )
    if device is not None:
        criterion = criterion.to(device)
    return criterion


# ──────────────────────────────────────────────────────────────────────────────
# 标准损失（TunnelSegmentor）
# ──────────────────────────────────────────────────────────────────────────────

def build_loss(
    cfg: TrainConfig,
    class_weights: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> CombinedLoss:
    """构建标准 CombinedLoss，供 TunnelSegmentor 使用。"""
    return _build_combined_loss(
        loss_name=cfg.loss_name,
        num_classes=cfg.num_classes,
        class_weights=class_weights,
        loss_weights=cfg.loss_weights,
        device=device,
    )


# ──────────────────────────────────────────────────────────────────────────────
# TMDS 联合损失（TMDSSegmentor）
# ──────────────────────────────────────────────────────────────────────────────

class TMDSCriterion(nn.Module):
    """
    TMDS 联合损失。

    forward(outputs, masks, skel_masks=None) 接口：
    - outputs 为 dict（训练模式）：分别计算 main / linear_aux / areal_aux 损失
    - outputs 为 Tensor（推理评估时调用）：仅计算 base 损失

    属性 last_components (dict) 记录上一次 forward 各项损失值，供日志使用。
    """

    def __init__(
        self,
        base_criterion: CombinedLoss,
        aux_weight: float,
        topo_loss: Optional[TopologyLoss] = None,
        skeleton_loss: Optional[SkeletonLoss] = None,
        topo_weight: float = 0.5,
        skeleton_weight: float = 1.0,
    ):
        super().__init__()
        self.base_criterion  = base_criterion
        self.aux_weight      = aux_weight
        self.topo_loss       = topo_loss
        self.skeleton_loss   = skeleton_loss
        self.topo_weight     = topo_weight
        self.skeleton_weight = skeleton_weight
        self.last_components: dict[str, float] = {}

    def forward(
        self,
        outputs: torch.Tensor | dict,
        masks: torch.Tensor,
        skel_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not isinstance(outputs, dict):
            # 推理模式或标准前向
            return self.base_criterion(outputs, masks)

        main    = outputs["main"]
        lin_aux = outputs["linear_aux"]
        are_aux = outputs["areal_aux"]

        main_loss = self.base_criterion(main, masks)
        lin_loss  = self.base_criterion(lin_aux, masks)
        are_loss  = self.base_criterion(are_aux, masks)

        loss = main_loss + self.aux_weight * (lin_loss + are_loss)
        self.last_components = {
            "main":    main_loss.item(),
            "lin_aux": lin_loss.item() * self.aux_weight,
            "are_aux": are_loss.item() * self.aux_weight,
        }

        if self.topo_loss is not None:
            tl = self.topo_loss(main, masks)
            loss = loss + self.topo_weight * tl
            self.last_components["topo"] = tl.item() * self.topo_weight

        if self.skeleton_loss is not None and skel_masks is not None:
            sl = self.skeleton_loss(main, skel_masks)
            loss = loss + self.skeleton_weight * sl
            self.last_components["skel"] = sl.item() * self.skeleton_weight

        return loss


def build_tmds_criterion(
    cfg: TrainConfig,
    stage: int,
    class_weights: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> TMDSCriterion:
    """
    构建 TMDSCriterion。

    Parameters
    ----------
    cfg   : TrainConfig（含 TMDS 专属字段）
    stage : 当前训练阶段索引（0/1/2）
    """
    loss_name = cfg.stage_loss_names[stage]
    base = _build_combined_loss(
        loss_name=loss_name,
        num_classes=cfg.num_classes,
        class_weights=class_weights,
        loss_weights=None,    # stage 损失使用内置默认权重
        device=device,
    )

    # 拓扑损失和骨架损失仅在第三阶段（stage=2）启用
    topo_loss     = TopologyLoss(crack_class_idx=1) if stage == 2 else None
    skeleton_loss = (
        SkeletonLoss(crack_class_idx=1)
        if (stage == 2 and cfg.use_skeleton_loss)
        else None
    )
    if topo_loss is not None and device is not None:
        topo_loss = topo_loss.to(device)
    if skeleton_loss is not None and device is not None:
        skeleton_loss = skeleton_loss.to(device)

    return TMDSCriterion(
        base_criterion=base,
        aux_weight=cfg.aux_loss_weight,
        topo_loss=topo_loss,
        skeleton_loss=skeleton_loss,
        topo_weight=cfg.topo_loss_weight,
        skeleton_weight=cfg.skeleton_loss_weight,
    )
