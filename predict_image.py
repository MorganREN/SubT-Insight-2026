"""
predict_image.py
单图推理入口（精简版）。

用法
----
1) 修改下方 RUN 配置。
2) 直接运行：python predict_image.py

说明
----
推理、可视化、单图指标计算逻辑已拆分到 `predictor/` 目录。
"""

from __future__ import annotations

from predictor import ImagePredictor, PredictConfig


RUN = PredictConfig(
    image="dataset/aug_data1/img_dir/train/1_1_036_31_jpg.rf.a33c41b4952f8685c269f91f8f3d47fe.jpg",
    mask="",
    ckpt="",            # 留空 = 自动选 outputs/ 下最新 checkpoint
    device="auto",      # auto / cuda / cpu
    output_dir="outputs/predict",
    input_size=None,
)


def main(cfg: PredictConfig | None = None):
    cfg = RUN if cfg is None else cfg
    predictor = ImagePredictor(cfg)
    predictor.run()


if __name__ == "__main__":
    main()
