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
# 如需切换 TMDS，将 use_tmds=True 并参考下方 TMDS_RUN 注释块。
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RUN = TrainConfig(
    # ── 路径 ──────────────────────────────────────────────────────────────────
    data_root            = "dataset/tongji_data",   # 数据集根目录，需含 img_dir/ 和 ann_dir/
    output_dir           = "outputs/train_run",     # 训练输出目录，存放 best.pth / last.pth / train.log
    backbone_type        = "vit_s16plus",         # 骨干类型："convnext_tiny" | "vit_s16plus"
    backbone_weight_path = "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",  # DINOv3 预训练权重路径；None = 随机初始化

    # ── 运行控制 ─────────────────────────────────────────────────────────────
    device   = "auto",   # 计算设备："auto"（优先 CUDA）/ "cuda" / "cpu" / "mps"
    resume   = "",       # 断点续训路径（填 last.pth 路径）；空字符串 = 从头训练
    dry_run  = False,    # True = 仅验证数据/模型/损失初始化是否正常，不执行训练循环
    seed     = 42,       # 随机种子，控制数据增强与权重初始化的可复现性

    # ── 模型结构 ─────────────────────────────────────────────────────────────
    num_classes   = NUM_CLASSES,  # 分割类别数（含背景），与数据集标注一致，通常不需要修改
    head_type     = "uper",       # 解码头类型："uper"（UPerHead，精度高）/ "mlp"（MLPHead，参数少）
    head_channels = 160,          # 解码头内部通道宽度；越大精度越高但显存/参数量增加
                                  # UPerHead 参考值：128≈0.6M / 160≈0.9M / 256≈2.2M / 512≈8.0M
    frozen_stages = 1,            # 冻结 backbone 前 N 个 stage（0=不冻结，1=冻结 stem+stage0，-1=全冻结）
                                  # 数据少时建议 1~2，数据充足时建议 0

    # ── 数据 ─────────────────────────────────────────────────────────────────
    input_size  = 384,  # 模型输入的正方形边长（像素）；训练/推理需保持一致
                        # 参考值：256（快速实验）/ 384（默认）/ 512（高精度，显存需求翻倍）
    batch_size  = 4,    # 每个训练 step 的样本数；显存不足时减小，建议保持 ≥ 2
    num_workers = 2,    # DataLoader 并行加载的进程数；一般设为 CPU 核数的一半

    # ── 损失函数 ─────────────────────────────────────────────────────────────
    loss_name     = "dice+focal",  # 损失组合："ce" / "dice" / "focal" /
                                   #           "ce+dice" / "ce+focal" / "dice+focal" /
                                   #           "ce+dice+focal"
                                   # 存在严重类别不均衡时推荐 "dice+focal" 或 "ce+dice"
    loss_weights  = None,          # 各子损失的加权系数，None = 使用内置默认值
                                   # 示例：loss_name="dice+focal" 时默认 (2.0, 1.0)
                                   #        可覆盖为 (1.0, 1.0) 等均等权重
    use_class_weights = False,     # True = 按像素频率自动计算类别权重传给 CE/Dice
                                   # 类别极度不均衡时开启；会遍历全部 mask 文件，耗时约数秒

    # ── 训练超参数 ───────────────────────────────────────────────────────────
    epochs           = 80,      # 总训练轮数（use_tmds=True 时由 stage_epochs 决定）
    base_lr          = 2e-4,    # 解码头的初始学习率；骨干 LR = base_lr × backbone_lr_mult
    backbone_lr_mult = 0.05,    # 骨干学习率相对于 base_lr 的倍率；建议 0.01~0.1
    weight_decay     = 1e-2,    # AdamW/SGD 的 L2 正则化系数；bias 和 Norm 层不受此影响
    optimizer_type   = "adamw", # 优化器："adamw"（推荐）/ "adam" / "sgd"
    warmup_epochs    = 1,       # 学习率从 0 线性升至 base_lr 所需的 epoch 数
    scheduler        = "cosine",# 学习率调度策略："cosine" / "poly" / "step"
    clip_grad        = 1.0,     # 梯度裁剪阈值（max_norm）；0 = 不裁剪
    val_interval     = 1,       # 每隔多少 epoch 做一次验证；增大可加快训练但减少 checkpoint 机会
    use_amp          = True,    # True = 启用自动混合精度（FP16）训练，仅 CUDA 生效
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TMDS 三阶段训练配置（将 RUN = TMDS_RUN 即可切换）
#
# 前置步骤（use_skeleton_loss=True 时）：
#   python tools/precompute_skeletons.py --data_root dataset/tongji_data --splits train
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TMDS_RUN = TrainConfig(
    # ── 路径 ──────────────────────────────────────────────────────────────────
    data_root            = "dataset/tongji_data",
    output_dir           = "outputs/tmds_run",
    backbone_type        = "convnext_tiny",           # 骨干类型："convnext_tiny" | "vit_s16plus"
    backbone_weight_path = "dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth",

    # ── 运行控制 ─────────────────────────────────────────────────────────────
    device  = "auto",
    seed    = 42,
    dry_run = False,
    resume  = "",

    # ── 模型结构 ─────────────────────────────────────────────────────────────
    num_classes   = NUM_CLASSES,
    head_channels = 128,          # TMDS 建议 256（MRM/DSA/CMIM 通道宽度）
    use_tmds      = True,         # ← 关键开关：使用 TMDSSegmentor

    # DSA 解码器超参数（通常无需调整）
    dsa_num_heads        = 4,
    dsa_num_strips       = 4,
    dsa_points_per_strip = 8,

    # ── 数据 ─────────────────────────────────────────────────────────────────
    input_size  = 384,
    batch_size  = 2,
    num_workers = 2,

    # ── 三阶段训练（总 epoch = 20+40+40 = 100）───────────────────────────────
    # stage_epochs 三元组各对应一个阶段的 epoch 数
    stage_epochs        = (20, 40, 40),
    # stage_frozen_stages：-1=全冻结, 2=冻结前两个stage, 1=仅冻结stem+stage0
    # Stage3 改为 frozen_stages=1（保持浅层骨干冻结）而非 0（全解冻），原因：
    #   解码器在 Stage1/2 学会了与半冻结骨干特征配合；全解冻+拓扑损失同时上线
    #   会导致特征分布突变 + 集中梯度双重冲击，crack IoU 从 40% 崩至个位数。
    #   frozen_stages=1 只放开骨干深层（stage1~3），保持 stem+stage0 稳定性。
    stage_frozen_stages = (-1, 2, 1),
    # 各阶段 base_lr（解码头学习率）
    stage_base_lrs      = (3e-4, 2e-4, 6e-5),
    # 各阶段损失组合（第三阶段额外叠加 topo/skeleton）
    stage_loss_names    = ("ce+dice", "ce+dice+focal", "ce+dice+focal"),

    # ── TMDS 辅助损失权重 ─────────────────────────────────────────────────────
    aux_loss_weight      = 0.4,   # 线型/面型辅助输出各自的损失系数
    topo_loss_weight     = 0.1,   # 拓扑损失系数（已归一化，保守起点；可逐步调至 0.2）
    skeleton_loss_weight = 0.5,   # 骨架损失系数
    use_skeleton_loss    = False,  # True = 启用骨架损失（需先运行 precompute_skeletons.py）

    # ── 通用训练超参数 ────────────────────────────────────────────────────────
    use_class_weights = False,
    optimizer_type    = "adamw",
    weight_decay      = 5e-3,
    backbone_lr_mult  = 0.01,
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
