from __future__ import annotations

from dataclasses import dataclass

from dataload import NUM_CLASSES


@dataclass
class TrainConfig:
    data_root: str = "dataset/tongji_data"
    output_dir: str = "outputs/train_run"
    device: str = "auto"
    resume: str = ""
    dry_run: bool = False
    seed: int = 42

    num_classes: int = NUM_CLASSES
    backbone_type: str = "convnext_tiny"       # "convnext_tiny" | "vit_s16plus"
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

    # ── TMDS 专属配置（use_tmds=False 时以下字段全部忽略）────────────────────
    use_tmds: bool = False          # True = 使用 TMDSSegmentor 替换 TunnelSegmentor

    # DSA 解码器超参数
    dsa_num_heads:       int = 4    # 可变形条状注意力的头数
    dsa_num_strips:      int = 4    # 每头的条数（方向数）
    dsa_points_per_strip: int = 8   # 每条的采样点数

    # 三阶段训练（三元组分别对应 Stage1 / Stage2 / Stage3）
    # 总 epoch = sum(stage_epochs)，会覆盖 epochs 字段
    stage_epochs:        tuple[int, int, int] = (20, 40, 40)
    # frozen_stages 对应各阶段 backbone 冻结数（-1=全冻结, 0=不冻结, N=冻结前N个stage）
    stage_frozen_stages: tuple[int, int, int] = (-1, 2, 0)
    # 各阶段 base_lr（解码头学习率）
    stage_base_lrs:      tuple[float, float, float] = (1e-3, 5e-4, 1e-4)
    # 各阶段损失函数名（见 loss_factory 支持的 key）
    # Stage3 会额外叠加 topo_loss 和 skeleton_loss（若启用）
    stage_loss_names:    tuple[str, str, str] = ("ce+dice", "ce+dice+focal", "ce+dice+focal")

    # TMDS 辅助损失权重
    aux_loss_weight:      float = 0.4   # linear_aux 和 areal_aux 损失各自乘以该系数
    topo_loss_weight:     float = 0.5   # 拓扑损失权重（Stage3 启用）
    skeleton_loss_weight: float = 1.0   # 骨架损失权重（Stage3 且 use_skeleton_loss=True 时启用）
    use_skeleton_loss:    bool  = False  # 须先运行 tools/precompute_skeletons.py
