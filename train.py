"""
train.py
训练入口（精简版）。

用法
----
1) 修改下方 RUN 配置。
2) 直接运行：python train.py

说明
----
训练细节（训练循环、验证、checkpoint、日志等）已拆分到 `trainer/` 目录。
"""

from __future__ import annotations

from dataload import NUM_CLASSES
from trainer import SegmentationTrainer, TrainConfig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 标准训练配置（TunnelSegmentor，use_tmds=False）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RUN = TrainConfig(
    # ── 路径 ──────────────────────────────────────────────────────────────────
    data_root            = "dataset/tongji_data_awesome",
    output_dir           = "outputs/train_run_awesome",
    backbone_type        = "convnext_tiny",         # "convnext_tiny" | "vit_s16plus"
    backbone_weight_path = "dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth",

    # ── 运行控制 ─────────────────────────────────────────────────────────────
    device   = "auto",
    resume   = "",
    dry_run  = False,
    seed     = 42,

    # ── 模型结构 ─────────────────────────────────────────────────────────────
    num_classes   = NUM_CLASSES,
    head_type     = "mlp",       # "uper"（UPerHead，精度高）/ "mlp"（MLPHead，参数少）
    head_channels = 128,          # UPerHead 参考值：128≈0.6M / 160≈0.9M / 256≈2.2M

    # ── 数据 ─────────────────────────────────────────────────────────────────
    input_size  = 512,
    batch_size  = 4,
    num_workers = 2,

    # ── 损失函数 ─────────────────────────────────────────────────────────────
    use_class_weights = True,

    # ── 三阶段渐进解冻训练（总 epoch = 20+30+50 = 100）──────────────────────
    use_stages           = True,
    stage_epochs         = (20, 30, 50),
    # -1=全冻结骨干, 1=仅冻结stem+stage0, 0=全解冻
    stage_frozen_stages  = (-1, 1, 0),
    # 各阶段解码头学习率；骨干 LR = base_lr × backbone_lr_mult
    stage_base_lrs       = (1e-3, 6e-4, 2e-4),
    stage_loss_names     = ("dice+focal", "dice+focal", "dice+focal"),

    # ── 通用训练超参数 ────────────────────────────────────────────────────────
    max_steps        = 0,         # >0 = 每 epoch 最多跑多少 step，用于快速调试
    backbone_lr_mult = 0.05,      # 骨干 LR 倍率；数据充足时可适当增大
    weight_decay     = 1e-2,
    optimizer_type   = "adamw",
    scheduler        = "cosine",
    clip_grad        = 1.0,
    val_interval     = 5,
    use_amp          = True,
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TMDS 三阶段训练配置（将 RUN = TMDS_RUN 即可切换）
#
# 前置步骤（use_skeleton_loss=True 时）：
#   python tools/precompute_skeletons.py --data_root dataset/tongji_data --splits train
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TMDS_RUN = TrainConfig(
    # ── 路径 ──────────────────────────────────────────────────────────────────
    data_root            = "dataset/tongji_data_awesome",
    output_dir           = "outputs/tmds_run_awesome",
    backbone_type        = "convnext_tiny",           # 骨干类型："convnext_tiny" | "vit_s16plus"
    backbone_weight_path = "dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth",

    # ── 运行控制 ────────────────────────────────────────ww─────────────────────
    device  = "auto",
    seed    = 42,
    dry_run = False,
    resume  = "",

    # ── 模型结构 ─────────────────────────────────────────────────────────────
    num_classes   = NUM_CLASSES,
    head_channels = 128,          # TMDS 建议 128（MRM/DSA/CMIM 通道宽度）
    use_tmds      = True,         # ← 关键开关：使用 TMDSSegmentor

    # DSA 解码器超参数（通常无需调整）
    dsa_num_heads        = 4,
    dsa_num_strips       = 4,
    dsa_points_per_strip = 8,

    # ── 数据 ─────────────────────────────────────────────────────────────────
    input_size  = 512,
    batch_size  = 2,
    num_workers = 2,

    # ── 三阶段训练（总 epoch = 20+30+50 = 100）───────────────────────────────
    # stage_epochs 三元组各对应一个阶段的 epoch 数
    stage_epochs        = (20, 30, 50),
    max_steps=1000,
    # stage_frozen_stages：-1=全冻结, 1=仅冻结stem+stage0, 0=全解冻
    # 渐进解冻：头部先在稳定预训练特征上收敛，再逐步放开骨干
    stage_frozen_stages = (-1, 1, 0),
    # 各阶段 base_lr（解码头学习率）
    stage_base_lrs      = (1e-3, 6e-4, 2e-4),
    # 各阶段损失组合（第三阶段额外叠加 skeleton）
    stage_loss_names    = ("dice+focal", "dice+focal", "dice+focal"),
    routing_loss_weight = 0.5,

    # ── TMDS 辅助损失权重 ─────────────────────────────────────────────────────
    aux_loss_weight      = 0.4,   # 线型/面型辅助输出各自的损失系数
    skeleton_loss_weight = 0.5,   # 骨架损失系数
    use_skeleton_loss    = False,  # True = 启用骨架损失（需先运行 precompute_skeletons.py）

    # ── 通用训练超参数 ────────────────────────────────────────────────────────
    use_class_weights = True,
    optimizer_type    = "adamw",
    weight_decay      = 4e-4,
    backbone_lr_mult  = 0.005,
    scheduler         = "cosine",
    clip_grad         = 1.0,
    val_interval      = 5,        # 三阶段训练共 100 epoch，每 5 epoch 验证一次
    use_amp           = True,
)


def main(cfg: TrainConfig | None = None):
    cfg = RUN if cfg is None else cfg
    trainer = SegmentationTrainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main(TMDS_RUN)
