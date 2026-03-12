from __future__ import annotations

from criteria import CombinedLoss

from .config import TrainConfig


def build_loss(cfg: TrainConfig, class_weights=None, device=None):
    if class_weights is not None and device is not None:
        class_weights = class_weights.to(device)

    ce_kwargs = {"class_weights": class_weights} if class_weights is not None else {}
    dice_kwargs = {"class_weights": class_weights} if class_weights is not None else {}
    focal_kwargs = {}

    loss_table = {
        "ce": (["ce"], [1.0]),
        "dice": (["dice"], [1.0]),
        "focal": (["focal"], [1.0]),
        "ce+dice": (["ce", "dice"], [1.0, 1.0]),
        "ce+focal": (["ce", "focal"], [1.0, 1.0]),
        "dice+focal": (["dice", "focal"], [1.0, 1.0]),
    }
    losses, weights = loss_table[cfg.loss_name]
    return CombinedLoss(
        losses=losses,
        weights=weights,
        num_classes=cfg.num_classes,
        ce_kwargs=ce_kwargs,
        dice_kwargs=dice_kwargs,
        focal_kwargs=focal_kwargs,
    )
