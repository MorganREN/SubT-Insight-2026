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

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import dataclasses

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
    ckpt="outputs/ablation_tmds_full/best.pth",
    data_root="dataset/tongji_data_raw",
    split="valid",
    device="auto",
    batch_size=1,        # tiling 模式下 batch_size 无效，保留字段兼容性
    num_workers=0,
    save_vis=False,
    vis_count=0,
    output_dir="outputs/ablation_tmds_full",
    use_tiling=True,     # 原图 tiling 推理，pred 与 GT 在原始分辨率下对比
    use_tta=False,
)

_BASE_CONFIGS = {"raw": RAW_RUN, "tmds": TMDS_RUN, "standard": RUN}


def main():
    parser = argparse.ArgumentParser(description="SubT-Insight 推理/评估入口")
    parser.add_argument("--run", choices=list(_BASE_CONFIGS), default="raw")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--split", type=str, default=None, choices=["train", "val", "valid", "test"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--use_tiling", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save_vis", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--vis_count", type=int, default=None)
    parser.add_argument("--use_tta", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()

    base_cfg = _BASE_CONFIGS[args.run]
    overrides = {
        k: v for k, v in vars(args).items()
        if v is not None and k != "run" and k in base_cfg.__dataclass_fields__
    }
    cfg = dataclasses.replace(base_cfg, **overrides)
    SegmentationInferencer(cfg).run()


if __name__ == "__main__":
    main()
