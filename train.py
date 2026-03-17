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
    data_root="dataset/tongji_data",
    output_dir="outputs/train_run",
    device="auto",
    resume="",
    dry_run=False,
    seed=42,
    num_classes=NUM_CLASSES,
    head_type="uper",          # mlp / uper
    head_channels=48,           # UPer 通道宽度（48≈0.74M 参数）
    frozen_stages=-1,
    input_size=384,
    batch_size=4,
    num_workers=2,
    loss_name="dice+focal",
    use_class_weights=False,
    epochs=80,
    base_lr=5e-4,
    backbone_lr_mult=0.1,
    weight_decay=1e-2,
    warmup_epochs=1,
    scheduler="cosine",
    clip_grad=1.0,
    val_interval=1,
    use_amp=True,
)


def main(cfg: TrainConfig | None = None):
    cfg = RUN if cfg is None else cfg
    trainer = SegmentationTrainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
