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


RUN = TrainConfig(
    # ── 路径 ──────────────────────────────────────────────────────────────────
    data_root       = "dataset/tongji_data",  # 数据集根目录，需含 img_dir/ 和 ann_dir/
    output_dir      = "outputs/train_run",    # 训练输出目录，存放 best.pth / last.pth / train.log
    backbone_weight_path = "dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth",  # DINOv3 预训练权重路径；None = 随机初始化

    # ── 运行控制 ─────────────────────────────────────────────────────────────
    device   = "auto",   # 计算设备："auto"（优先 CUDA）/ "cuda" / "cpu" / "mps"
    resume   = "",       # 断点续训路径（填 last.pth 路径）；空字符串 = 从头训练
    dry_run  = False,    # True = 仅验证数据/模型/损失初始化是否正常，不执行训练循环
    seed     = 42,       # 随机种子，控制数据增强与权重初始化的可复现性

    # ── 模型结构 ─────────────────────────────────────────────────────────────
    num_classes  = NUM_CLASSES,  # 分割类别数（含背景），与数据集标注一致，通常不需要修改
    head_type    = "uper",       # 解码头类型："uper"（UPerHead，精度高）/ "mlp"（MLPHead，参数少）
    head_channels = 160,         # 解码头内部通道宽度；越大精度越高但显存/参数量增加
                                 # UPerHead 参考值：128≈0.6M / 160≈0.9M / 256≈2.2M / 512≈8.0M
    frozen_stages = 1,           # 冻结 backbone 前 N 个 stage（0=不冻结，1=冻结 stem+stage0，-1=全冻结）
                                 # 数据少时建议 1~2，数据充足时建议 0

    # ── 数据 ─────────────────────────────────────────────────────────────────
    input_size  = 384,  # 模型输入的正方形边长（像素）；训练/推理需保持一致
                        # 参考值：256（快速实验）/ 384（默认）/ 512（高精度，显存需求翻倍）
    batch_size  = 4,    # 每个训练 step 的样本数；显存不足时减小，建议保持 ≥ 2
    num_workers = 2,    # DataLoader 并行加载的进程数；一般设为 CPU 核数的一半

    # ── 损失函数 ─────────────────────────────────────────────────────────────
    loss_name     = "dice+focal",  # 损失组合："ce" / "dice" / "focal" /
                                   #           "ce+dice" / "ce+focal" / "dice+focal"
                                   # 存在严重类别不均衡时推荐 "dice+focal" 或 "ce+dice"
    loss_weights  = None,          # 各子损失的加权系数，None = 使用内置默认值
                                   # 示例：loss_name="dice+focal" 时默认 (2.0, 1.0)
                                   #        可覆盖为 (1.0, 1.0) 等均等权重
    use_class_weights = False,     # True = 按像素频率自动计算类别权重传给 CE/Dice
                                   # 类别极度不均衡时开启；会遍历全部 mask 文件，耗时约数秒

    # ── 训练超参数 ───────────────────────────────────────────────────────────
    epochs           = 80,    # 总训练轮数
    base_lr          = 2e-4,  # 解码头的初始学习率；骨干 LR = base_lr × backbone_lr_mult
    backbone_lr_mult = 0.05,  # 骨干学习率相对于 base_lr 的倍率；建议 0.01~0.1
                              # 越小则骨干变化越慢，适合预训练权重质量高的场景
    weight_decay     = 1e-2,  # AdamW/SGD 的 L2 正则化系数；bias 和 Norm 层不受此影响
    optimizer_type   = "adamw",  # 优化器："adamw"（推荐）/ "adam" / "sgd"
    warmup_epochs    = 1,     # 学习率从 0 线性升至 base_lr 所需的 epoch 数
    scheduler        = "cosine",  # 学习率调度策略："cosine" / "poly" / "step"
    clip_grad        = 1.0,   # 梯度裁剪阈值（max_norm）；0 = 不裁剪
    val_interval     = 1,     # 每隔多少 epoch 做一次验证；增大可加快训练但减少 checkpoint 机会
    use_amp          = True,  # True = 启用自动混合精度（FP16）训练，仅 CUDA 生效；可减少显存、加速训练
)


def main(cfg: TrainConfig | None = None):
    cfg = RUN if cfg is None else cfg
    trainer = SegmentationTrainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
