"""
infer.py
推理/评估入口（精简版）。

用法
----
1) 修改下方 RUN 配置。
2) 直接运行：python infer.py

说明
----
推理细节（模型恢复、评估、可视化、metrics 保存）已拆分到 `inference/` 目录。
"""

from __future__ import annotations

from inference import InferConfig, SegmentationInferencer


RUN = InferConfig(
    ckpt="",                 # 留空 = 自动选择 outputs/ 下最新 checkpoint
    data_root="dataset/tongji_data",
    split="test",            # val / test
    device="auto",          # auto / cuda / cpu / mps
    batch_size=4,
    num_workers=2,
    save_vis=False,
    vis_count=5,
    output_dir="outputs/infer",
)


def main(cfg: InferConfig | None = None):
    cfg = RUN if cfg is None else cfg
    inferencer = SegmentationInferencer(cfg)
    inferencer.run()


if __name__ == "__main__":
    main()
