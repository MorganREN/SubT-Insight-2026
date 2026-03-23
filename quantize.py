"""
quantize.py
模型 Post-Training Quantization (PTQ) 入口脚本。

用法
----
1) 按需修改下方 RUN 配置。
2) 直接运行：python quantize.py

说明
----
- ckpt       : checkpoint 路径；留空则自动搜索 outputs/ 下最新的 best.pth / last.pth
- mode       : 量化模式
    "dynamic" : 动态量化（无需校准数据，推荐快速验证）
    "static"  : 静态 FX PTQ（需要校准数据，压缩率更好）
- backend    : 量化后端，x86/服务器用 "fbgemm"，ARM/移动端用 "qnnpack"
- output_dir : 量化结果保存目录（模型文件 + 大小对比报告）
- data_root  : static 模式需要的校准数据集根目录
- calib_batches : static 模式校准所用 batch 数

量化结果
--------
    outputs/quantized/
        model_int8.pth          量化 checkpoint（state_dict + 元信息，推荐）
        quantize_summary.json   大小 / 压缩比摘要
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from loguru import logger

from dataload import NUM_CLASSES, CLASS_NAMES, build_dataloaders
from utils.runtime import (
    find_latest_checkpoint,
    load_checkpoint_compat,
    resolve_device,
    setup_logger,
)
from utils.segmentor_loader import (
    build_segmentor_from_checkpoint,
    get_input_size_from_checkpoint,
    resolve_checkpoint_path,
    QUANTIZED_CKPT_FORMAT,
)
from utils.quantizer import ModelQuantizer, QuantizerConfig


# 运行配置（按需修改）
@dataclass
class QuantizeRunConfig:
    ckpt: str = "outputs_2203/train_run/best.pth"                        # 留空 = 自动搜索 outputs/ 下最新 best.pth
    mode: str = "static"                 # "dynamic" | "static"
    backend: str = "fbgemm"              # "fbgemm" (x86) | "qnnpack" (ARM)
    calib_batches: int = 64              # 静态量化校准 batch 数
    data_root: str = "dataset/tongji_data"  # 校准数据集根目录
    split: str = "val"
    batch_size: int = 8
    num_workers: int = 2
    output_dir: str = "outputs_2203/quantized_uper"


RUN = QuantizeRunConfig()

def main(cfg: QuantizeRunConfig | None = None) -> None:
    cfg = RUN if cfg is None else cfg

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(str(out_dir / "quantize.log"))

    logger.info("=" * 70)
    logger.info("PTQ 量化任务启动")
    logger.info("=" * 70)

    # ── 1. 解析 checkpoint 路径 ──────────────────────────────────────────────
    ckpt_path = resolve_checkpoint_path(cfg.ckpt, hint_script="quantize.py")
    logger.info(f"使用 checkpoint: {ckpt_path}")

    # ── 2. 加载浮点模型（强制 CPU，PTQ 在 CPU 上运行）─────────────────────────
    device = torch.device("cpu")
    ckpt = load_checkpoint_compat(ckpt_path, map_location=device)
    float_model, model_cfg = build_segmentor_from_checkpoint(
        ckpt,
        device,
        default_num_classes=NUM_CLASSES,
        use_backbone_weight_from_cfg=False,
        default_backbone_weight_path=None,
        use_frozen_stages_from_cfg=False,
        default_frozen_stages=-1,
    )
    float_model.eval()

    # ── 3. 准备校准数据（仅 static 模式需要）──────────────────────────────────
    calib_loader = None
    if cfg.mode == "static":
        input_size = get_input_size_from_checkpoint(ckpt, default=512)
        logger.info(
            f"静态量化：加载校准数据集  data_root={cfg.data_root}  "
            f"split={cfg.split}  input_size={input_size}"
        )
        loaders = build_dataloaders(
            data_roots=cfg.data_root,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            input_size=input_size,
            splits=[cfg.split],
        )
        calib_loader = loaders[cfg.split]
        logger.info(
            f"校准数据集加载完成：{len(calib_loader.dataset)} 张图像，"
            f"最多校准 {cfg.calib_batches} 个 batch。"
        )

    # ── 4. 执行 PTQ ──────────────────────────────────────────────────────────
    q_cfg = QuantizerConfig(
        mode=cfg.mode,
        backend=cfg.backend,
        calib_batches=cfg.calib_batches,
    )
    quantizer = ModelQuantizer(q_cfg)
    q_model = quantizer.quantize(float_model, calibration_loader=calib_loader)

    # ── 5. 保存量化 checkpoint（安全格式）────────────────────────────────────
    out_path = out_dir / "model_int8.pth"
    quant_ckpt = {
        "format": QUANTIZED_CKPT_FORMAT,
        "quantized": True,
        "quant_mode": cfg.mode,
        "backend": cfg.backend,
        "dynamic_layers": ["Linear"],
        "config": ckpt.get("config", ckpt.get("args", {})),
        "model": q_model.state_dict(),
        "source_checkpoint": str(ckpt_path),
    }
    torch.save(quant_ckpt, out_path)
    logger.info(f"量化 checkpoint 已保存: {out_path}")

    # ── 6. 保存摘要 JSON ─────────────────────────────────────────────────────
    import io

    def _size_mb(model):
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        return round(buf.tell() / 1024 / 1024, 2)

    orig_size = _size_mb(float_model)
    q_size = round(out_path.stat().st_size / 1024 / 1024, 2)
    summary = {
        "source_checkpoint": str(ckpt_path),
        "quantize_mode": cfg.mode,
        "backend": cfg.backend,
        "original_size_mb": orig_size,
        "quantized_file_size_mb": q_size,
        "compression_ratio": round(orig_size / q_size, 3) if q_size > 0 else None,
        "output_model": str(out_path),
    }
    summary_path = out_dir / "quantize_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.success("=" * 70)
    logger.success("量化任务完成！")
    logger.success(f"  浮点模型大小:   ~{orig_size} MB")
    logger.success(f"  量化模型大小:   ~{q_size} MB")
    logger.success(
        f"  压缩比:        {summary['compression_ratio']}x"
        if summary["compression_ratio"] else "  压缩比:        N/A"
    )
    logger.success(f"  保存路径:       {out_path}")
    logger.success(f"  摘要报告:       {summary_path}")
    logger.success("=" * 70)


if __name__ == "__main__":
    main()
