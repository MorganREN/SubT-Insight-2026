from __future__ import annotations

from pathlib import Path

import torch
from loguru import logger

from dataload import NUM_CLASSES
from models.segmentor import TunnelSegmentor
from utils.runtime import find_latest_checkpoint


def resolve_checkpoint_path(
    ckpt: str | Path | None,
    *,
    outputs_root: str | Path = "outputs",
    hint_script: str = "当前脚本",
) -> Path:
    if ckpt:
        ckpt_path = Path(ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"RUN.ckpt 指向的文件不存在: {ckpt_path}")
        return ckpt_path

    try:
        return find_latest_checkpoint(outputs_root)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"未找到可用 checkpoint。请先运行 train.py 生成 best.pth/last.pth，"
            f"或在 {hint_script} 的 RUN.ckpt 中填写 checkpoint 路径。"
        ) from e


def get_input_size_from_checkpoint(ckpt: dict, default: int = 512) -> int:
    cfg = ckpt.get("config", ckpt.get("args", {}))
    return int(cfg.get("input_size", default))


def get_class_names_from_checkpoint(ckpt: dict, default: tuple[str, ...]):
    cfg = ckpt.get("config", ckpt.get("args", {}))
    return tuple(cfg.get("class_names", default))


def build_segmentor_from_checkpoint(
    ckpt: dict,
    device: torch.device,
    *,
    default_num_classes: int = NUM_CLASSES,
    use_backbone_weight_from_cfg: bool = True,
    default_backbone_weight_path: str | None = None,
    use_frozen_stages_from_cfg: bool = True,
    default_frozen_stages: int = -1,
) -> tuple[TunnelSegmentor, dict]:
    cfg = ckpt.get("config", ckpt.get("args", {}))

    num_classes = int(cfg.get("num_classes", default_num_classes))
    head_type = cfg.get("head_type", "mlp")

    if use_backbone_weight_from_cfg:
        backbone_weight_path = cfg.get("backbone_weight_path", default_backbone_weight_path)
    else:
        backbone_weight_path = default_backbone_weight_path

    if use_frozen_stages_from_cfg:
        frozen_stages = int(cfg.get("frozen_stages", default_frozen_stages))
    else:
        frozen_stages = default_frozen_stages

    model = TunnelSegmentor(
        num_classes=num_classes,
        backbone_weight_path=backbone_weight_path,
        head_type=head_type,
        frozen_stages=frozen_stages,
    ).to(device)

    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    logger.info(
        f"模型恢复完成: head_type={head_type}, "
        f"num_classes={num_classes}, frozen_stages={frozen_stages}"
    )
    return model, cfg
