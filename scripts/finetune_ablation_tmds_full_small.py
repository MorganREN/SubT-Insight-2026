"""
finetune_ablation_tmds_full_small.py

Continue-train outputs/ablation_tmds_full_small/best.pth with:
- Train: tongji_data_awesome (train + valid combined into a single train split)
- Valid: 全部 324 张 roboflow_tongji_tunnel_source + 150 张随机抽样的
        roboflow_tunnel_crack_partial_raw（seed=42）
- Warm-start：仅加载模型权重，optimizer/scheduler/epoch 全部重置
- 评估时合并 leakage 子类：pred/gt 中 {3,4} → 2 后再累积混淆矩阵

Output: outputs/ablation_tmds_full_small_ft_v1/

用法
----
    conda run -n subt-2026 python scripts/finetune_ablation_tmds_full_small.py
    conda run -n subt-2026 python scripts/finetune_ablation_tmds_full_small.py --dry_run
    conda run -n subt-2026 python scripts/finetune_ablation_tmds_full_small.py --max_steps 4
"""

from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import dataclasses
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from loguru import logger
from torch.amp import autocast

from criteria import SegEvaluator
from trainer import SegmentationTrainer

from scripts.train import TMDS_RUN


# ── 常量 ─────────────────────────────────────────────────────────────────────
TONGJI_SRC      = Path("dataset/tongji_data_awesome")
TONGJI_FULL     = Path("dataset/tongji_data_awesome_full")
RFSRC           = Path("dataset/roboflow_tongji_tunnel_source")
RFCRACK         = Path("dataset/roboflow_tunnel_crack_partial_raw")
FT_VALID        = Path("dataset/finetune_valid")

ORIG_BEST       = Path("outputs/ablation_tmds_full_small/best.pth")
OUT_DIR         = Path("outputs/ablation_tmds_full_small_ft_v1")
WARMSTART_CKPT  = OUT_DIR / "_warmstart_init.pth"

# Roboflow tongji_source 类别 → 项目类别（详见计划）
#   bg(0)→bg(0), cracks(1)→crack(1), leakage(2)→leakage_b 占位(2),
#   spalling(3)→segment_damage(6)。其他像素 → 255 (ignore)
RFSRC_PIXEL_MAP = {0: 0, 1: 1, 2: 2, 3: 6}
N_CRACK_PARTIAL = 150
SAMPLE_SEED     = 42


# ── 数据准备：幂等 ──────────────────────────────────────────────────────────

def _ensure_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src.resolve())


def prepare_tongji_full_train() -> int:
    """合并 tongji_data_awesome 的 train+valid 为一个 train split（符号链接）。"""
    img_dst = TONGJI_FULL / "img_dir" / "train"
    ann_dst = TONGJI_FULL / "ann_dir" / "train"
    img_dst.mkdir(parents=True, exist_ok=True)
    ann_dst.mkdir(parents=True, exist_ok=True)

    n = 0
    for split in ("train", "valid"):
        src_imgs = TONGJI_SRC / "img_dir" / split
        src_anns = TONGJI_SRC / "ann_dir" / split
        if not src_imgs.exists():
            logger.warning(f"源目录缺失: {src_imgs}")
            continue
        for img in sorted(src_imgs.glob("*.jpg")):
            stem = img.stem
            mask = src_anns / f"{stem}.png"
            if not mask.exists():
                continue
            _ensure_symlink(img, img_dst / f"{stem}.jpg")
            _ensure_symlink(mask, ann_dst / f"{stem}.png")
            n += 1
    logger.info(f"tongji_data_awesome_full 准备完成: {n} 张样本（仅 train split）")
    return n


def _convert_rfsrc(out_img_dir: Path, out_ann_dir: Path) -> list[str]:
    """
    把 roboflow_tongji_tunnel_source/train 下的 (image, _mask.png) 对：
      - image .jpg：符号链接，前缀 rfsrc_
      - mask .png：按 RFSRC_PIXEL_MAP 重映射后写出新文件
    """
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_ann_dir.mkdir(parents=True, exist_ok=True)

    src_dir = RFSRC / "train"
    if not src_dir.exists():
        logger.error(f"源目录缺失: {src_dir}")
        return []

    names: list[str] = []
    for img in sorted(src_dir.glob("*.jpg")):
        stem = img.stem
        if stem.endswith("_mask"):  # 防御：mask 文件理论上是 .png
            continue
        mask_src = src_dir / f"{stem}_mask.png"
        if not mask_src.exists():
            logger.warning(f"rfsrc 缺 mask，跳过: {stem}")
            continue

        new_stem = f"rfsrc_{stem}"
        out_img = out_img_dir / f"{new_stem}.jpg"
        out_ann = out_ann_dir / f"{new_stem}.png"

        _ensure_symlink(img, out_img)

        if not out_ann.exists():
            arr = np.array(Image.open(mask_src).convert("L"), dtype=np.uint8)
            remapped = np.full_like(arr, 255)  # 未列出的像素 → ignore
            for src_id, dst_id in RFSRC_PIXEL_MAP.items():
                remapped[arr == src_id] = dst_id
            Image.fromarray(remapped, mode="L").save(out_ann)

        names.append(new_stem)

    return names


def _sample_crack_partial(out_img_dir: Path, out_ann_dir: Path,
                          n: int, seed: int) -> list[str]:
    """从 roboflow_tunnel_crack_partial_raw/img_dir/train 用固定种子抽样。"""
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_ann_dir.mkdir(parents=True, exist_ok=True)

    src_imgs = sorted((RFCRACK / "img_dir" / "train").glob("*.jpg"))
    if not src_imgs:
        logger.error(f"crack_partial 源为空: {RFCRACK}")
        return []

    chosen = random.Random(seed).sample(src_imgs, min(n, len(src_imgs)))

    names: list[str] = []
    for img in chosen:
        stem = img.stem
        mask = RFCRACK / "ann_dir" / "train" / f"{stem}.png"
        if not mask.exists():
            logger.warning(f"crack_partial 缺 mask，跳过: {stem}")
            continue
        new_stem = f"rfcrack_{stem}"
        _ensure_symlink(img, out_img_dir / f"{new_stem}.jpg")
        _ensure_symlink(mask, out_ann_dir / f"{new_stem}.png")
        names.append(new_stem)

    return names


def prepare_finetune_valid() -> dict:
    """构建 dataset/finetune_valid/img_dir/valid + ann_dir/valid。"""
    img_dir = FT_VALID / "img_dir" / "valid"
    ann_dir = FT_VALID / "ann_dir" / "valid"

    rfsrc_names   = _convert_rfsrc(img_dir, ann_dir)
    rfcrack_names = _sample_crack_partial(img_dir, ann_dir,
                                          n=N_CRACK_PARTIAL, seed=SAMPLE_SEED)

    manifest = {
        "seed": SAMPLE_SEED,
        "rfsrc_count": len(rfsrc_names),
        "rfcrack_count": len(rfcrack_names),
        "total": len(rfsrc_names) + len(rfcrack_names),
        "rfsrc_pixel_map": {str(k): v for k, v in RFSRC_PIXEL_MAP.items()},
        "rfsrc_names": rfsrc_names,
        "rfcrack_names": rfcrack_names,
    }
    (FT_VALID / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info(
        f"finetune_valid 准备完成: rfsrc={len(rfsrc_names)} + "
        f"rfcrack={len(rfcrack_names)} = {manifest['total']} 张"
    )
    return manifest


def prepare_warmstart_ckpt() -> None:
    """从 best.pth 剥离出只含 'model' 的检查点，供 cfg.resume 使用。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if WARMSTART_CKPT.exists():
        return
    if not ORIG_BEST.exists():
        raise FileNotFoundError(f"找不到原模型: {ORIG_BEST}")
    ckpt = torch.load(ORIG_BEST, map_location="cpu", weights_only=False)
    if "model" not in ckpt:
        raise KeyError(f"{ORIG_BEST} 中不含 'model' 键，无法 warm-start")
    torch.save({"model": ckpt["model"]}, WARMSTART_CKPT)
    logger.info(f"Warm-start checkpoint 写入: {WARMSTART_CKPT}")


# ── 自定义 Trainer：评估时合并 leakage 桶 ──────────────────────────────────

class FineTuneTrainer(SegmentationTrainer):
    """覆写 _validate：对 pred/gt 中的 {3,4} → 2 后再累积混淆矩阵。

    Roboflow tongji_source 的 leakage 单类在数据准备阶段被映射到 class 2
    (leakage_b 位)。模型仍输出 7 类，所以这里把 pred 的 leakage_w/leakage_g
    都视作与 leakage_b 等价；GT 端的合并是防御性写法（数据准备已处理）。
    """

    @staticmethod
    @torch.no_grad()
    def _validate(model, loader, criterion, evaluator: SegEvaluator,
                  device, use_amp: bool, max_steps: int = 0):
        model.eval()
        evaluator.reset()
        total_loss = 0.0
        valid_batches = 0

        for step, batch in enumerate(loader, start=1):
            if max_steps > 0 and step > max_steps:
                break
            images = batch[0].to(device, non_blocking=True)
            masks  = batch[1].to(device, non_blocking=True)

            with autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss   = criterion(logits, masks)

            total_loss += loss.item()
            valid_batches += 1

            pred = logits.argmax(dim=1)
            pred_e = torch.where(
                (pred == 3) | (pred == 4),
                torch.full_like(pred, 2),
                pred,
            )
            masks_e = torch.where(
                (masks == 3) | (masks == 4),
                torch.full_like(masks, 2),
                masks,
            )
            evaluator.update(pred_e, masks_e)

        metrics = evaluator.compute()
        avg_loss = total_loss / max(valid_batches, 1)
        return avg_loss, metrics


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune ablation_tmds_full_small on tongji-full + Roboflow valid."
    )
    parser.add_argument("--dry_run", action="store_true",
                        help="只跑数据/模型/损失/优化器/调度器初始化，不进入训练循环")
    parser.add_argument("--max_steps", type=int, default=0,
                        help=">0 时每 epoch 最多 N 个 step（冒烟测试）")
    args = parser.parse_args()

    # 1. 幂等数据准备
    prepare_tongji_full_train()
    prepare_finetune_valid()
    prepare_warmstart_ckpt()

    # 2. 配置覆盖
    overrides = dict(
        data_root            = str(TONGJI_FULL),
        extra_data_roots     = (str(FT_VALID),),
        output_dir           = str(OUT_DIR),
        resume               = str(WARMSTART_CKPT),

        # 对齐原 ablation_tmds_full_small 关键超参
        head_channels        = 32,
        batch_size           = 32,
        num_workers          = 6,
        routing_loss_weight  = 0.2,
        use_cmim             = True,
        rare_class_weights   = {1: 3.0, 3: 2.0},
        stage_loss_names     = ("dice+focal", "dice+focal", "dice+focal"),
        use_class_weights    = True,
        use_skeleton_loss    = False,

        # 缩短的 3 阶段（共 40 ep）
        stage_epochs         = (8, 12, 20),
        stage_base_lrs       = (1e-3, 6e-4, 2e-4),
        stage_frozen_stages  = (-1, 1, 0),
        val_interval         = 2,

        dry_run              = args.dry_run,
        max_steps            = args.max_steps,
    )
    cfg = dataclasses.replace(TMDS_RUN, **overrides)

    FineTuneTrainer(cfg).run()


if __name__ == "__main__":
    main()
