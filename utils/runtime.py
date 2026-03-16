from __future__ import annotations

import pickle
import sys
from pathlib import Path

import torch
from loguru import logger


def setup_logger(log_file: str):
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | {message}",
    )
    logger.add(
        log_file,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <7} | {message}",
        encoding="utf-8",
    )


def resolve_device(
    device_name: str,
    *,
    allow_mps: bool = False,
    warn_mps_on_auto: bool = False,
) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"设备: CUDA  ({torch.cuda.get_device_name(0)})")
            return device

        logger.info("设备: CPU")
        if warn_mps_on_auto and torch.backends.mps.is_available() and not allow_mps:
            logger.warning("检测到 Apple MPS，但默认不启用；如需尝试请显式传 --device mps")
        return torch.device("cpu")

    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda 已指定，但当前环境不可用 CUDA")
        device = torch.device("cuda")
        logger.info(f"设备: CUDA  ({torch.cuda.get_device_name(0)})")
        return device

    if device_name == "mps":
        if not allow_mps:
            raise RuntimeError("当前脚本未开启 MPS 选项")
        if not torch.backends.mps.is_available():
            raise RuntimeError("--device mps 已指定，但当前环境不可用 MPS")
        logger.warning("设备: Apple MPS (实验性)；当前项目仅保证 CUDA 路径")
        return torch.device("mps")

    logger.info("设备: CPU")
    return torch.device("cpu")


def load_checkpoint_compat(path: str | Path, map_location: torch.device | str | None = None) -> dict:
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt_path}")

    logger.info(f"加载 checkpoint: {ckpt_path}")
    try:
        return torch.load(ckpt_path, map_location=map_location)
    except pickle.UnpicklingError as e:
        if "Weights only load failed" not in str(e):
            raise
        logger.warning(
            "检测到 PyTorch>=2.6 的 weights_only 限制，"
            "回退到 weights_only=False 重新加载（仅对受信任 checkpoint 使用）"
        )
        try:
            return torch.load(ckpt_path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(ckpt_path, map_location=map_location)


def find_latest_checkpoint(outputs_root: str | Path = "outputs") -> Path:
    root = Path(outputs_root)
    if not root.exists():
        raise FileNotFoundError(f"未找到输出目录: {root}")

    candidates = list(root.rglob("best.pth")) + list(root.rglob("last.pth"))
    if not candidates:
        raise FileNotFoundError("未在 outputs/ 下找到 best.pth 或 last.pth")

    return max(candidates, key=lambda p: p.stat().st_mtime)


def restore_training_checkpoint(path: str, model, optimizer=None, scheduler=None, device=None) -> tuple[int, float]:
    map_location = device or "cpu"
    ckpt = load_checkpoint_compat(path, map_location=map_location)

    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])

    start_epoch = ckpt.get("epoch", 0) + 1
    best_miou = ckpt.get("best_miou", -1.0)
    logger.info(f"续训自 epoch={start_epoch - 1}，当前 best_mIoU={best_miou:.4f}")
    return start_epoch, best_miou
