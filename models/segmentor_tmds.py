"""
models/segmentor_tmds.py

TMDS（Topology-aware Morphological Decoupled Segmentation）分割器。

相比 TunnelSegmentor 的改进：
  - MRM 将骨干特征软路由至线型流（裂缝）和面型流（其他病害）
  - 线型流使用 DSADecoder（可变形条状注意力）捕获裂缝长程连续性
  - 面型流使用 ArealDecoder（FPN+PPM 变体）捕获块状区域
  - CMIM 实现两流之间的双向上下文交互
  - 训练时返回 dict（含辅助输出），推理时返回单张量
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from .backbones.dinov3_convnext import DINOv3ConvNeXt
from .backbones.dinov3_vits16plus import DINOv3ViTS16Plus
from .heads.mrm import MorphologicalRoutingModule
from .heads.dsa_decoder import DSADecoder
from .heads.cmim import CrossMorphologyInteractionModule


class _PPM(nn.Module):
    """Pyramid Pooling Module（面型流专用）。"""
    def __init__(self, in_channels: int, out_channels: int, pool_scales: tuple = (1, 2, 4, 8)):
        super().__init__()
        _ng = max(1, out_channels // 8)
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.GroupNorm(_ng, out_channels),
                nn.ReLU(inplace=True),
            )
            for s in pool_scales
        ])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + len(pool_scales) * out_channels, out_channels,
                      3, padding=1, bias=False),
            nn.GroupNorm(_ng, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[2:]
        parts = [x] + [
            F.interpolate(stage(x), size=size, mode='bilinear', align_corners=False)
            for stage in self.stages
        ]
        return self.bottleneck(torch.cat(parts, dim=1))


class ArealDecoder(nn.Module):
    """
    面型流解码器（FPN + PPM）。
    输出特征图，不含分类层。

    Args:
        in_channels_list: 骨干各阶段通道数
        channels:         统一通道宽度
        pool_scales:      PPM 池化尺度
    Returns:
        [B, channels, H/4, W/4]
    """
    def __init__(
        self,
        in_channels_list: tuple,
        channels: int = 256,
        pool_scales: tuple = (1, 2, 4, 8),
    ):
        super().__init__()
        self.ppm = _PPM(in_channels_list[-1], channels, pool_scales)

        _ng = max(1, channels // 8)
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, channels, 1, bias=False),
                nn.GroupNorm(_ng, channels),
                nn.ReLU(inplace=True),
            )
            for c in in_channels_list[:-1]
        ])
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.GroupNorm(_ng, channels),
                nn.ReLU(inplace=True),
            )
            for _ in in_channels_list[:-1]
        ])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(len(in_channels_list) * channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_ng, channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        # PPM on deepest feature
        ppm_out = self.ppm(features[-1])
        fpn_outs = [ppm_out]

        # Top-down FPN
        for i in range(len(features) - 2, -1, -1):
            lat = self.lateral_convs[i](features[i])
            top = F.interpolate(fpn_outs[-1], size=lat.shape[2:],
                                mode='bilinear', align_corners=False)
            fpn_outs.append(self.fpn_convs[i](lat + top))

        fpn_outs = fpn_outs[::-1]  # [C1, C2, C3, C4] 顺序

        # 上采样到 H/4 并拼接
        target = fpn_outs[0].shape[2:]
        merged = [fpn_outs[0]] + [
            F.interpolate(f, size=target, mode='bilinear', align_corners=False)
            for f in fpn_outs[1:]
        ]
        return self.bottleneck(torch.cat(merged, dim=1))


class TMDSSegmentor(nn.Module):
    """
    TMDS 端到端分割器。

    Args:
        num_classes:            分割类别数（含背景）
        backbone_weight_path:   DINOv3 预训练权重路径
        frozen_stages:          初始冻结阶段数（训练中会动态变化）
        head_channels:          解码头统一通道宽度
        pool_scales:            面型流 PPM 尺度
        dsa_num_heads:          DSA 注意力头数
        dsa_num_strips:         DSA 每头条数
        dsa_points_per_strip:   DSA 每条采样点数

    训练模式输出（dict）：
        "main":        [B, num_classes, H, W]  主输出（两流融合）
        "linear_aux":  [B, num_classes, H, W]  线型流辅助输出
        "areal_aux":   [B, num_classes, H, W]  面型流辅助输出

    推理模式输出：
        [B, num_classes, H, W]  主输出
    """

    def __init__(
        self,
        num_classes: int = 7,
        backbone_type: str = "convnext_tiny",
        backbone_weight_path: str | None = None,
        frozen_stages: int = 1,
        head_channels: int = 256,
        pool_scales: tuple = (1, 2, 4, 8),
        dsa_num_heads: int = 4,
        dsa_num_strips: int = 4,
        dsa_points_per_strip: int = 8,
    ):
        super().__init__()

        if backbone_type == "vit_s16plus":
            self.backbone = DINOv3ViTS16Plus(
                weight_path=backbone_weight_path,
                frozen_stages=frozen_stages,
            )
        else:  # "convnext_tiny"（默认）
            self.backbone = DINOv3ConvNeXt(
                weight_path=backbone_weight_path,
                frozen_stages=frozen_stages,
            )
        in_ch = self.backbone.out_channels   # [96, 192, 384, 768]

        # MRM 使用 C3（384 通道）
        self.mrm = MorphologicalRoutingModule(in_channels=in_ch[2])

        # 双流解码器
        self.linear_decoder = DSADecoder(
            in_channels_list=in_ch,
            channels=head_channels,
            dsa_num_heads=dsa_num_heads,
            dsa_num_strips=dsa_num_strips,
            dsa_points_per_strip=dsa_points_per_strip,
        )
        self.areal_decoder = ArealDecoder(
            in_channels_list=in_ch,
            channels=head_channels,
            pool_scales=pool_scales,
        )

        # CMIM：head 数须整除 head_channels，取 head_channels // 32，最小为 1
        cmim_heads = max(1, head_channels // 32)
        self.cmim = CrossMorphologyInteractionModule(
            channels=head_channels,
            num_heads=cmim_heads,
        )

        # 分类头
        self.main_cls = nn.Conv2d(head_channels * 2, num_classes, 1)
        self.linear_aux_cls = nn.Conv2d(head_channels, num_classes, 1)
        self.areal_aux_cls = nn.Conv2d(head_channels, num_classes, 1)

        self.num_classes = num_classes
        self._log_params()

    def _log_params(self):
        bb = sum(p.numel() for p in self.backbone.parameters()) / 1e6
        dec = sum(
            p.numel() for p in list(self.linear_decoder.parameters())
            + list(self.areal_decoder.parameters())
            + list(self.cmim.parameters())
            + list(self.mrm.parameters())
        ) / 1e6
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        bb_name = (
            "DINOv3 ViT-S+/16"
            if isinstance(self.backbone, DINOv3ViTS16Plus)
            else "DINOv3 ConvNeXt-Tiny"
        )
        logger.info(
            f"TMDSSegmentor 初始化完成:\n"
            f"  Backbone:  {bb_name} ({bb:.1f}M)\n"
            f"  Decoders:  MRM + DSA + Areal + CMIM ({dec:.1f}M)\n"
            f"  总参数:    {bb + dec:.1f}M  (可训练: {trainable:.1f}M)"
        )

    def set_frozen_stages(self, frozen_stages: int):
        """动态更新 backbone 冻结阶段数（三阶段训练时调用）。"""
        self.backbone.set_frozen_stages(frozen_stages)
        logger.info(f"Backbone frozen_stages 更新为: {frozen_stages}")

    def forward(self, x: torch.Tensor) -> torch.Tensor | dict:
        input_size = x.shape[2:]

        # ── 骨干特征提取 ───────────────────────────────────────────────────────
        features = self.backbone(x)   # [C1@/4, C2@/8, C3@/16, C4@/32]

        # ── MRM 路由 ──────────────────────────────────────────────────────────
        alpha = self.mrm(features[2])   # [B, 1, H/16, W/16]

        # clamp 到 [0.1, 0.9]：防止极端路由将某条流的特征归零。
        # 归零后 GroupNorm 的方差≈0，输出被放大数百倍，FP16 下直接 overflow/NaN。
        # [0.1, 0.9] 保留了足够的路由分辨力，同时确保两条流始终有有效信号。
        alpha = alpha.clamp(0.1, 0.9)

        linear_feats, areal_feats = [], []
        for feat in features:
            a = F.interpolate(alpha, size=feat.shape[2:],
                              mode='bilinear', align_corners=False)
            linear_feats.append(feat * a)
            areal_feats.append(feat * (1.0 - a))

        # ── 双流解码 ──────────────────────────────────────────────────────────
        F_L = self.linear_decoder(linear_feats)   # [B, ch, H/4, W/4]
        F_A = self.areal_decoder(areal_feats)     # [B, ch, H/4, W/4]

        # ── 跨形态交互 ────────────────────────────────────────────────────────
        F_L, F_A = self.cmim(F_L, F_A)

        # ── 分类 ──────────────────────────────────────────────────────────────
        main    = self.main_cls(torch.cat([F_L, F_A], dim=1))
        lin_aux = self.linear_aux_cls(F_L)
        are_aux = self.areal_aux_cls(F_A)

        # 上采样到输入分辨率
        def _up(t):
            return F.interpolate(t, size=input_size, mode='bilinear', align_corners=False)

        main    = _up(main)
        lin_aux = _up(lin_aux)
        are_aux = _up(are_aux)

        if self.training:
            return {"main": main, "linear_aux": lin_aux, "areal_aux": are_aux}
        return main
