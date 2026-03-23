"""
infer.py
推理/评估入口（精简版）。

用法
----
1) 修改下方 RUN 配置（标准模型）或 TMDS_RUN（TMDS 模型）。
2) 直接运行：python infer.py

说明
----
推理细节（模型恢复、评估、可视化、metrics 保存）已拆分到 `inference/` 目录。
checkpoint 中的 use_tmds 字段会自动决定重建 TunnelSegmentor 还是 TMDSSegmentor。
"""

from __future__ import annotations

from inference import InferConfig, SegmentationInferencer


# ── 标准模型推理配置 ──────────────────────────────────────────────────────────
RUN = InferConfig(
    ckpt="",                 # 留空 = 自动选择 outputs/ 下最新 checkpoint
    data_root="dataset/tongji_data",
    split="val",             # val / test
    device="auto",           # auto / cuda / cpu / mps
    batch_size=4,
    num_workers=2,
    save_vis=False,
    vis_count=5,
    output_dir="outputs/infer",
)

# ── TMDS 模型推理配置（将 main() 参数改为 TMDS_RUN 即可）────────────────────
TMDS_RUN = InferConfig(
    ckpt="outputs/quantized/model_int8.pth",
    data_root="dataset/tongji_data",
    split="test",
    device="auto",
    batch_size=2,            # TMDS 推理显存需求约为标准模型 2×，建议 batch=2
    num_workers=2,
    save_vis=True,
    vis_count=10,
    output_dir="outputs/infer_tmds",
)


def main(cfg: InferConfig | None = None):
    cfg = RUN if cfg is None else cfg
    inferencer = SegmentationInferencer(cfg)
    inferencer.run()


if __name__ == "__main__":
    main(TMDS_RUN)
