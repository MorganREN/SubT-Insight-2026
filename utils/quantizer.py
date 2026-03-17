"""
utils/quantizer.py
Post-Training Quantization (PTQ) 工具类。

支持两种量化模式：
    dynamic  : 动态量化，对 Linear/Conv2d 权重量化为 INT8，无需校准数据，
               适合快速验证。
    static   : 静态量化（FX Graph Mode），同时量化激活值，需要校准数据集，
               压缩率和推理加速更好。
"""

from __future__ import annotations

import copy
import io
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.nn as nn
from loguru import logger

@dataclass
class QuantizerConfig:
    """量化配置。

    Attributes
    ----------
    mode : str
        量化模式：``"dynamic"`` （动态 PTQ）或 ``"static"``（静态 PTQ）。
    backend : str
        量化后端：``"fbgemm"``（x86 服务器，默认）或 ``"qnnpack"``（ARM/移动端）。
    dtype : torch.dtype
        权重量化目标精度，默认 ``torch.qint8``。
    calib_batches : int
        静态量化时用于校准的 batch 数量，默认 64。
    dynamic_layers : set[type]
        动态量化时要量化的层类型集合，默认 {nn.Linear}。
    """
    mode: str = "dynamic"
    backend: str = "fbgemm"
    dtype: torch.dtype = torch.qint8
    calib_batches: int = 64
    dynamic_layers: set = field(
        default_factory=lambda: {nn.Linear}
    )


def _model_size_mb(model: nn.Module) -> float:
    """估算模型参数 + buffer 占用的内存大小（MB）。"""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1024 / 1024


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class ModelQuantizer:
    """Post-Training Quantization (PTQ) 执行器。

    Parameters
    ----------
    cfg : QuantizerConfig
        量化配置对象。如不传，则使用默认动态量化配置。
    """

    def __init__(self, cfg: Optional[QuantizerConfig] = None):
        self.cfg = cfg or QuantizerConfig()

    def quantize(
        self,
        model: nn.Module,
        calibration_loader: Optional[Iterable] = None,
        example_inputs: Optional[torch.Tensor] = None,
    ) -> nn.Module:
        """对浮点模型执行 PTQ，返回量化后的模型（CPU 上）。

        Parameters
        ----------
        model : nn.Module
            已 eval() 的浮点模型。
        calibration_loader : Iterable, optional
            静态量化所需的校准 DataLoader；动态量化时忽略。
        example_inputs : Tensor, optional
            静态量化所需的示例输入张量（用于 FX tracing）；
            若不传则自动从 calibration_loader 中取第一个 batch。

        Returns
        -------
        nn.Module
            INT8 量化后的模型（在 CPU 上）。
        """
        cfg = self.cfg
        mode = cfg.mode.lower()

        model = copy.deepcopy(model).cpu().eval()
        orig_size = _model_size_mb(model)
        orig_params = _count_params(model)

        logger.info("=" * 60)
        logger.info(f"PTQ 启动  |  mode={mode}  backend={cfg.backend}")
        logger.info(f"原始模型:  {orig_params / 1e6:.2f}M 参数,  ~{orig_size:.1f} MB")
        logger.info("=" * 60)

        if mode == "dynamic":
            q_model = self._quantize_dynamic(model)
        elif mode == "static":
            q_model = self._quantize_static(
                model,
                calibration_loader=calibration_loader,
                example_inputs=example_inputs,
            )
        else:
            raise ValueError(
                f"不支持的量化模式: '{mode}'。可选: 'dynamic', 'static'。"
            )

        q_size = _model_size_mb(q_model)
        compression = orig_size / q_size if q_size > 0 else float("inf")
        logger.success(
            f"量化完成！  量化后大小: ~{q_size:.1f} MB  "
            f"（压缩比 {compression:.2f}x）"
        )
        return q_model

    def save(
        self,
        q_model: nn.Module,
        path: str | Path,
        *,
        also_torchscript: bool = False,
    ) -> Path:
        """将量化模型保存到磁盘。

        Parameters
        ----------
        q_model : nn.Module
            ``quantize()`` 返回的量化模型。
        path : str | Path
            保存路径（例如 ``"outputs/quantized/model_int8.pth"``）。
        also_torchscript : bool
            是否同时尝试导出 TorchScript（``.torchscript.pt``）格式；
            部分架构可能不支持，失败时仅打印警告而不中断。

        Returns
        -------
        Path
            实际保存的文件路径。
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(q_model, out_path)
        logger.info(f"量化模型已保存: {out_path}  ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")

        if also_torchscript:
            ts_path = out_path.with_suffix(".torchscript.pt")
            try:
                scripted = torch.jit.script(q_model)
                torch.jit.save(scripted, str(ts_path))
                logger.info(f"TorchScript 已保存: {ts_path}")
            except Exception as exc:
                logger.warning(f"TorchScript 导出失败（跳过）: {exc}")

        return out_path

    def _quantize_dynamic(self, model: nn.Module) -> nn.Module:
        """动态量化：对权重执行 INT8 量化，激活值在推理时动态量化。

        无需校准数据，兼容几乎所有模型结构。
        """
        torch.backends.quantized.engine = self.cfg.backend
        logger.info(
            f"动态量化目标层: "
            f"{[cls.__name__ for cls in self.cfg.dynamic_layers]}"
        )
        q_model = torch.quantization.quantize_dynamic(
            model,
            self.cfg.dynamic_layers,
            dtype=self.cfg.dtype,
        )
        logger.info("动态量化完成。")
        return q_model

    def _quantize_static(
        self,
        model: nn.Module,
        calibration_loader: Optional[Iterable],
        example_inputs: Optional[torch.Tensor],
    ) -> nn.Module:
        """静态量化（FX Graph Mode PTQ）：同时量化权重与激活值。

        步骤:
            1. prepare_fx：插入 Observer
            2. 校准：用真实数据跑若干 batch，采集激活统计量
            3. convert_fx：将 Observer 替换为真正的量化算子
        """
        try:
            from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
            from torch.ao.quantization import get_default_qconfig_mapping
        except ImportError as exc:
            raise ImportError(
                "静态 FX PTQ 需要 PyTorch >= 1.13 且含 torch.ao.quantization。"
            ) from exc

        if calibration_loader is None:
            raise ValueError(
                "静态量化 (mode='static') 需要传入 calibration_loader。"
            )

        torch.backends.quantized.engine = self.cfg.backend

        # ── 取示例输入用于 FX tracing ──────────────────────────────────────
        if example_inputs is None:
            first_batch = next(iter(calibration_loader))
            if isinstance(first_batch, (list, tuple)):
                example_inputs = first_batch[0]
            else:
                example_inputs = first_batch
        example_inputs = example_inputs[:1].cpu()

        # ── Prepare (插入 Observer) ────────────────────────────────────────
        qconfig_mapping = get_default_qconfig_mapping(self.cfg.backend)
        logger.info("静态量化 prepare_fx 开始（FX Graph Mode）...")
        try:
            model_prepared = prepare_fx(
                model,
                qconfig_mapping,
                example_inputs=(example_inputs,),
            )
        except Exception as exc:
            logger.warning(
                f"FX tracing 失败: {exc}\n"
                "回退到动态量化 (dynamic PTQ)。"
            )
            return self._quantize_dynamic(model)

        logger.info(
            f"校准中（最多 {self.cfg.calib_batches} 个 batch）..."
        )
        model_prepared.eval()
        calib_count = 0
        with torch.no_grad():
            for batch in calibration_loader:
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch
                model_prepared(images.cpu())
                calib_count += 1
                if calib_count >= self.cfg.calib_batches:
                    break
        logger.info(f"校准完成，共使用 {calib_count} 个 batch。")

        logger.info("convert_fx 转换中...")
        q_model = convert_fx(model_prepared)
        logger.info("静态量化完成。")
        return q_model
