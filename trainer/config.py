from __future__ import annotations

from dataclasses import dataclass

from dataload import CLASS_NAMES, NUM_CLASSES


@dataclass
class TrainConfig:
    data_root: str = "dataset/tongji_data"
    output_dir: str = "outputs/train_run"
    device: str = "auto"
    resume: str = ""
    dry_run: bool = False
    seed: int = 42

    num_classes: int = NUM_CLASSES
    class_names: tuple[str, ...] = CLASS_NAMES
    backbone_weight_path: str | None = "dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth"
    head_type: str = "uper"
    head_channels: int = 160
    frozen_stages: int = 1

    input_size: int = 384
    batch_size: int = 4
    num_workers: int = 2

    loss_name: str = "ce+dice"
    loss_weights: tuple[float, ...] | None = None  # None = 使用 loss_factory 内置默认权重
    use_class_weights: bool = False
    epochs: int = 8
    base_lr: float = 2e-4
    backbone_lr_mult: float = 0.05
    weight_decay: float = 1e-2
    optimizer_type: str = "adamw"  # "adamw" / "adam" / "sgd"
    warmup_epochs: int = 1
    scheduler: str = "cosine"
    clip_grad: float = 1.0
    val_interval: int = 1
    use_amp: bool = True
