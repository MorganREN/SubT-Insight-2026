from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger

from dataload import NUM_CLASSES
from models.segmentor import TunnelSegmentor
from models.segmentor_tmds import TMDSSegmentor
from utils.runtime import find_latest_checkpoint


QUANTIZED_CKPT_FORMAT = "subt_quantized_v1"


def is_quantized_checkpoint(ckpt: object) -> bool:
    return isinstance(ckpt, dict) and bool(ckpt.get("quantized", False))


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
) -> tuple[nn.Module, dict]:
    if is_quantized_checkpoint(ckpt):
        return _build_quantized_segmentor_from_checkpoint(
            ckpt,
            device,
            default_num_classes=default_num_classes,
            use_backbone_weight_from_cfg=use_backbone_weight_from_cfg,
            default_backbone_weight_path=default_backbone_weight_path,
            use_frozen_stages_from_cfg=use_frozen_stages_from_cfg,
            default_frozen_stages=default_frozen_stages,
        )

    cfg = ckpt.get("config", ckpt.get("args", {}))

    # TMDS checkpoint 由 use_tmds=True 标识
    if cfg.get("use_tmds", False):
        return _build_tmds_segmentor_from_checkpoint(
            ckpt, cfg, device,
            default_num_classes=default_num_classes,
            use_backbone_weight_from_cfg=use_backbone_weight_from_cfg,
            default_backbone_weight_path=default_backbone_weight_path,
        )

    num_classes = int(cfg.get("num_classes", default_num_classes))
    head_type = cfg.get("head_type", "mlp")
    head_channels = int(cfg.get("head_channels", 512))

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
        head_channels=head_channels,
        frozen_stages=frozen_stages,
    ).to(device)

    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    logger.info(
        f"模型恢复完成: head_type={head_type}, "
        f"num_classes={num_classes}, head_channels={head_channels}, "
        f"frozen_stages={frozen_stages}"
    )
    return model, cfg


def _build_tmds_segmentor_from_checkpoint(
    ckpt: dict,
    cfg: dict,
    device: torch.device,
    *,
    default_num_classes: int,
    use_backbone_weight_from_cfg: bool,
    default_backbone_weight_path: str | None,
) -> tuple[TMDSSegmentor, dict]:
    """从 TMDS checkpoint 重建 TMDSSegmentor 并加载权重。"""
    num_classes   = int(cfg.get("num_classes",   default_num_classes))
    head_channels = int(cfg.get("head_channels", 256))
    backbone_type        = cfg.get("backbone_type",        "convnext_tiny")
    dsa_num_heads        = int(cfg.get("dsa_num_heads",        4))
    dsa_num_strips       = int(cfg.get("dsa_num_strips",       4))
    dsa_points_per_strip = int(cfg.get("dsa_points_per_strip", 8))

    # 推理时不加载骨干预训练权重（直接从 checkpoint 的 state_dict 恢复）
    backbone_weight_path = (
        cfg.get("backbone_weight_path", default_backbone_weight_path)
        if use_backbone_weight_from_cfg
        else default_backbone_weight_path
    )

    model = TMDSSegmentor(
        num_classes=num_classes,
        backbone_type=backbone_type,
        backbone_weight_path=backbone_weight_path,
        frozen_stages=-1,             # 推理时无需冻结，eval() 已禁用梯度
        head_channels=head_channels,
        dsa_num_heads=dsa_num_heads,
        dsa_num_strips=dsa_num_strips,
        dsa_points_per_strip=dsa_points_per_strip,
    ).to(device)

    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    logger.info(
        f"TMDS 模型恢复完成: backbone={backbone_type}, num_classes={num_classes}, "
        f"head_channels={head_channels}, "
        f"dsa_heads={dsa_num_heads}, strips={dsa_num_strips}, "
        f"points={dsa_points_per_strip}"
    )
    return model, cfg


def _build_quantized_segmentor_from_checkpoint(
    ckpt: dict,
    device: torch.device,
    *,
    default_num_classes: int,
    use_backbone_weight_from_cfg: bool,
    default_backbone_weight_path: str | None,
    use_frozen_stages_from_cfg: bool,
    default_frozen_stages: int,
) -> tuple[nn.Module, dict]:
    fmt = ckpt.get("format", "")
    if fmt and fmt != QUANTIZED_CKPT_FORMAT:
        raise ValueError(f"不支持的量化 checkpoint 格式: {fmt}")

    cfg = ckpt.get("config", ckpt.get("args", {}))
    quant_mode = str(ckpt.get("quant_mode", "dynamic")).lower()
    backend = str(ckpt.get("backend", "fbgemm"))

    if quant_mode != "dynamic":
        raise NotImplementedError(
            "当前仅支持 dynamic 量化 checkpoint 的重建加载。"
        )

    num_classes = int(cfg.get("num_classes", default_num_classes))
    head_type = cfg.get("head_type", "mlp")
    head_channels = int(cfg.get("head_channels", 512))

    if use_backbone_weight_from_cfg:
        backbone_weight_path = cfg.get("backbone_weight_path", default_backbone_weight_path)
    else:
        backbone_weight_path = default_backbone_weight_path

    if use_frozen_stages_from_cfg:
        frozen_stages = int(cfg.get("frozen_stages", default_frozen_stages))
    else:
        frozen_stages = default_frozen_stages

    # Dynamic quantized modules are CPU-only in PyTorch.
    if device.type != "cpu":
        logger.warning("检测到量化模型，推理设备自动切换为 CPU。")

    float_model = TunnelSegmentor(
        num_classes=num_classes,
        backbone_weight_path=backbone_weight_path,
        head_type=head_type,
        head_channels=head_channels,
        frozen_stages=frozen_stages,
    ).cpu().eval()

    torch.backends.quantized.engine = backend
    layer_names = ckpt.get("dynamic_layers", ["Linear"])
    q_layers = set()
    if "Linear" in layer_names:
        q_layers.add(nn.Linear)
    if "Conv2d" in layer_names:
        q_layers.add(nn.Conv2d)
    if not q_layers:
        q_layers = {nn.Linear}

    q_model = torch.quantization.quantize_dynamic(
        float_model,
        q_layers,
        dtype=torch.qint8,
    )

    state_dict = ckpt.get("model", ckpt.get("model_state_dict"))
    if state_dict is None:
        raise KeyError("量化 checkpoint 缺少 model/model_state_dict")
    q_model.load_state_dict(state_dict)
    q_model.eval()

    logger.info(
        f"量化模型恢复完成: mode={quant_mode}, backend={backend}, "
        f"head_type={head_type}, num_classes={num_classes}, "
        f"head_channels={head_channels}"
    )
    return q_model, cfg
