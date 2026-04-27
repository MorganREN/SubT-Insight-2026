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
    image="/home/uqmren2/projects/SubT-Insight-2026/dataset/tongji_data_awesome/img_dir/valid/ES6_r004_c003.jpg",
    mask="",
    ckpt="outputs/quantized_tmds_run/model_int8.pth",            # 留空 = 自动选 outputs/ 下最新 checkpoint
    device="auto",      # auto / cuda / cpu
    output_dir="outputs/quantized_tmds_run",
    input_size=None,
    use_tiling=False,   # 是否使用 tiling 推理（适用于超大图，输出与 GT 在原始分辨率对齐）
)


def main(cfg: PredictConfig | None = None):
    cfg = RUN if cfg is None else cfg
    predictor = ImagePredictor(cfg)
    predictor.run()


if __name__ == "__main__":
    main()
