"""
utils/optimizer.py
优化器构建工具，支持 backbone / head 差分学习率。

支持的 optimizer 类型
----------------------
    "adamw"  → torch.optim.AdamW
    "sgd"    → torch.optim.SGD
    "adam"   → torch.optim.Adam
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from loguru import logger


def build_param_groups(
    model: nn.Module,
    base_lr: float,
    backbone_lr_mult: float = 0.1,
    weight_decay: float = 1e-2,
    backbone_module_name: str = "backbone",
    no_decay_keywords: tuple = ("bias", "norm", "bn"),
) -> List[Dict[str, Any]]:
    """按 backbone/head 与 decay/no_decay 划分参数组。"""
    backbone_params_decay    = []
    backbone_params_no_decay = []
    head_params_decay        = []
    head_params_no_decay     = []

    backbone = getattr(model, backbone_module_name, None)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # 冻结参数跳过

        is_backbone = (
            backbone is not None
            and name.startswith(f"{backbone_module_name}.")
        )

        is_no_decay = any(kw in name for kw in no_decay_keywords)

        if is_backbone:
            if is_no_decay:
                backbone_params_no_decay.append(param)
            else:
                backbone_params_decay.append(param)
        else:
            if is_no_decay:
                head_params_no_decay.append(param)
            else:
                head_params_decay.append(param)

    backbone_lr = base_lr * backbone_lr_mult

    param_groups = []
    if backbone_params_decay:
        param_groups.append({
            "params": backbone_params_decay,
            "lr": backbone_lr,
            "weight_decay": weight_decay,
            "name": "backbone/decay",
        })
    if backbone_params_no_decay:
        param_groups.append({
            "params": backbone_params_no_decay,
            "lr": backbone_lr,
            "weight_decay": 0.0,
            "name": "backbone/no_decay",
        })
    if head_params_decay:
        param_groups.append({
            "params": head_params_decay,
            "lr": base_lr,
            "weight_decay": weight_decay,
            "name": "head/decay",
        })
    if head_params_no_decay:
        param_groups.append({
            "params": head_params_no_decay,
            "lr": base_lr,
            "weight_decay": 0.0,
            "name": "head/no_decay",
        })

    logger.info("Optimizer param_groups:")
    total_trainable = 0
    for g in param_groups:
        n = sum(p.numel() for p in g["params"]) / 1e6
        total_trainable += n
        logger.info(
            f"  [{g['name']:>20s}]  lr={g['lr']:.2e}  "
            f"wd={g['weight_decay']:.0e}  params={n:.2f}M"
        )
    logger.info(f"  总可训练参数: {total_trainable:.2f}M")

    return param_groups


def build_optimizer(
    model: nn.Module,
    optimizer_type: str = "adamw",
    base_lr: float = 1e-4,
    backbone_lr_mult: float = 0.1,
    weight_decay: float = 1e-2,
    momentum: float = 0.9,
    backbone_module_name: str = "backbone",
    no_decay_keywords: tuple = ("bias", "norm", "bn"),
) -> torch.optim.Optimizer:
    """构建带差分学习率的 Optimizer。"""
    param_groups = build_param_groups(
        model,
        base_lr=base_lr,
        backbone_lr_mult=backbone_lr_mult,
        weight_decay=weight_decay,
        backbone_module_name=backbone_module_name,
        no_decay_keywords=no_decay_keywords,
    )

    opt_type = optimizer_type.lower()
    if opt_type == "adamw":
        optimizer = torch.optim.AdamW(param_groups, lr=base_lr)
    elif opt_type == "adam":
        optimizer = torch.optim.Adam(param_groups, lr=base_lr)
    elif opt_type == "sgd":
        optimizer = torch.optim.SGD(
            param_groups, lr=base_lr,
            momentum=momentum, nesterov=True,
        )
    else:
        raise ValueError(
            f"不支持的 optimizer_type: '{optimizer_type}'，"
            f"可选: 'adamw', 'adam', 'sgd'"
        )

    logger.info(
        f"Optimizer: {optimizer.__class__.__name__}  "
        f"base_lr={base_lr:.2e}  "
        f"backbone_lr={base_lr * backbone_lr_mult:.2e}  "
        f"wd={weight_decay:.0e}"
    )
    return optimizer
