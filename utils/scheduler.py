"""学习率调度器构建工具，支持 warmup + 主调度器组合。"""

from __future__ import annotations

from typing import List, Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    MultiStepLR,
    PolynomialLR,
    SequentialLR,
    LRScheduler,
)
from loguru import logger


# ──────────────────────────────────────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────────────────────────────────────

def build_scheduler(
    optimizer: Optimizer,
    scheduler_type: str = "cosine",
    total_epochs: int = 100,
    warmup_epochs: int = 5,
    min_lr_ratio: float = 1e-2,
    warmup_start_lr_ratio: float = 1e-3,
    poly_power: float = 0.9,
    step_milestones: Optional[List[int]] = None,
    step_gamma: float = 0.1,
) -> LRScheduler:
    """构建 [LinearWarmup → 主调度器] 组合调度器。"""
    after_warmup_epochs = max(total_epochs - warmup_epochs, 1)

    # ── 主调度器 ──
    stype = scheduler_type.lower()
    if stype == "cosine":
        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=after_warmup_epochs,
            eta_min=0,
        )
        _apply_eta_min(optimizer, main_scheduler, min_lr_ratio)

    elif stype == "poly":
        main_scheduler = PolynomialLR(
            optimizer,
            total_iters=after_warmup_epochs,
            power=poly_power,
        )

    elif stype == "step":
        if step_milestones is None:
            m1 = int(total_epochs * 2 / 3)
            m2 = int(total_epochs * 5 / 6)
            step_milestones = [m1, m2]
        main_scheduler = MultiStepLR(
            optimizer,
            milestones=step_milestones,
            gamma=step_gamma,
        )

    else:
        raise ValueError(
            f"不支持的 scheduler_type: '{scheduler_type}'，"
            f"可选: 'cosine', 'poly', 'step'"
        )

    if warmup_epochs <= 0:
        logger.info(
            f"LR Scheduler: {scheduler_type}  "
            f"total_epochs={total_epochs}  no warmup"
        )
        return main_scheduler

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=warmup_start_lr_ratio,   # 起始 LR = base_lr * warmup_start_lr_ratio
        end_factor=1.0,                        # 结束 LR = base_lr（满 warmup_epochs 后）
        total_iters=warmup_epochs,
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs],
    )

    logger.info(
        f"LR Scheduler: LinearWarmup({warmup_epochs}ep) → "
        f"{scheduler_type}({after_warmup_epochs}ep)  "
        f"total_epochs={total_epochs}  min_lr_ratio={min_lr_ratio:.0e}"
    )
    return scheduler


def _apply_eta_min(
    optimizer: Optimizer,
    scheduler: CosineAnnealingLR,
    min_lr_ratio: float,
) -> None:
    """为 CosineAnnealingLR 设置 eta_min。"""
    ref_lr     = optimizer.param_groups[0]["lr"]
    scheduler.eta_min = ref_lr * min_lr_ratio


def get_lr(optimizer: Optimizer) -> List[float]:
    """返回当前 optimizer 各 param_group 的 LR。"""
    return [pg["lr"] for pg in optimizer.param_groups]


def log_lr(optimizer: Optimizer) -> None:
    """输出各 param_group 的当前 LR。"""
    for i, pg in enumerate(optimizer.param_groups):
        name = pg.get("name", str(i))
        logger.info(f"  LR [{name:>20s}]: {pg['lr']:.4e}")
