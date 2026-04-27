"""
train.py
训练入口（精简版）。

用法
----
1) 直接运行（使用默认的 TMDS_RUN 配置）：
   python train.py --use_tmds

2) 覆盖部分参数：
   python train.py --use_tmds --batch_size 8 --output_dir outputs/my_run

3) 修改下方 RUN / TMDS_RUN 配置对象可调整 tuple 类型参数
   （如 stage_epochs / stage_base_lrs）。

说明
----
训练细节（训练循环、验证、checkpoint、日志等）已拆分到 `trainer/` 目录。
"""

from __future__ import annotations

import argparse
import dataclasses

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
    use_cmim            = True,

    # ── 数据 ─────────────────────────────────────────────────────────────────
    input_size  = 512,
    batch_size  = 2,
    num_workers = 2,

    # ── 三阶段训练（总 epoch = 20+30+50 = 100）───────────────────────────────
    # stage_epochs 三元组各对应一个阶段的 epoch 数
    stage_epochs        = (20, 30, 50),
    max_steps=0,
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
    weight_decay      = 1e-2,
    backbone_lr_mult  = 0.05,
    scheduler         = "cosine",
    clip_grad         = 1.0,
    val_interval      = 5,        # 三阶段训练共 100 epoch，每 5 epoch 验证一次
    use_amp           = True,
)


def main():
    parser = argparse.ArgumentParser(description="SubT-Insight 训练入口")
    parser.add_argument("--use_tmds", action="store_true", help="使用 TMDS_RUN 配置（TMDSSegmentor）")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--backbone_type", type=str, default=None, choices=["convnext_tiny", "vit_s16plus"])
    parser.add_argument("--backbone_weight_path", type=str, default=None)
    parser.add_argument("--head_channels", type=int, default=None)
    parser.add_argument("--mrm_stage_idx", type=int, default=None, choices=[0, 1, 2, 3],
                        help="MRM 输入骨干阶段：0=C1/H4, 1=C2/H8, 2=C3/H16(默认), 3=C4/H32")
    parser.add_argument("--use_cmim", action=argparse.BooleanOptionalAction, default=None,
                        help="TMDS 是否启用跨形态交互模块 CMIM")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="单阶段训练总 epoch（use_stages=False 时生效）")
    parser.add_argument("--base_lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--routing_loss_weight", type=float, default=None)
    parser.add_argument("--aux_loss_weight", type=float, default=None)
    parser.add_argument(
        "--rare_class_weights",
        type=str,
        default=None,
        help="稀有类采样权重，格式如 '1:3.0,3:2.0'；不传则关闭增强采样",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--use_amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use_stages", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dry_run", action="store_true", default=None)
    args = parser.parse_args()

    base_cfg = TMDS_RUN if args.use_tmds else RUN

    overrides = {
        k: v for k, v in vars(args).items()
        if v is not None and k != "use_tmds" and k in base_cfg.__dataclass_fields__
    }

    if overrides.get("rare_class_weights") is not None:
        raw_weights = overrides["rare_class_weights"]
        overrides["rare_class_weights"] = {
            int(pair.split(":")[0]): float(pair.split(":")[1])
            for pair in raw_weights.split(",")
            if ":" in pair
        }

    cfg = dataclasses.replace(base_cfg, **overrides)

    SegmentationTrainer(cfg).run()


if __name__ == "__main__":
    main()
