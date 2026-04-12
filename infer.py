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
    data_root="dataset/tongji_data_awesome",
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
    ckpt="outputs/tmds_run_awesome/best.pth",
    data_root="dataset/tongji_data_awesome",
    split="valid",
    device="auto",
    batch_size=2,            # TMDS 推理显存需求约为标准模型 2×，建议 batch=2
    num_workers=2,
    save_vis=True,
    vis_count=10,
    output_dir="outputs/tmds_run/infer_tmds",
)


# ── 原始分辨率数据集评估（tongji_data_raw，tiling 推理）────────────────────────
RAW_RUN = InferConfig(
    ckpt="outputs/quantized_tmds_run/model_int8.pth",
    data_root="dataset/tongji_data_raw",
    split="valid",
    device="auto",
    batch_size=1,        # tiling 模式下 batch_size 无效，保留字段兼容性
    num_workers=0,
    save_vis=False,
    vis_count=0,
    output_dir="outputs/quantized_infer_raw",
    use_tiling=True,     # 原图 tiling 推理，pred 与 GT 在原始分辨率下对比
)


def main(cfg: InferConfig | None = None):
    cfg = RUN if cfg is None else cfg
    inferencer = SegmentationInferencer(cfg)
    inferencer.run()


if __name__ == "__main__":
    main(RAW_RUN)
