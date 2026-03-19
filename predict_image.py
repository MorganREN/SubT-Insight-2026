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
    image="dataset/tongji_data/img_dir/test/C175.jpg",
    mask="",
    ckpt="outputs/train_run/best.pth",            # 留空 = 自动选 outputs/ 下最新 checkpoint
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
