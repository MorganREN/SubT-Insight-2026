from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossMorphologyInteractionModule(nn.Module):
    """
    跨形态交互模块（CMIM）。

    线型流（F_L，裂缝主导）向面型流（F_A，渗漏/剥落主导）提供
    单向门控交叉注意力上下文。F_L 保持原样，避免面型流反向污染
    稀疏裂缝特征。

    内存优化：交叉注意力在 attn_stride 倍降采样的低分辨率上计算，
    注意力上下文上采样回原分辨率后再做门控融合。

    显存对比（input=384, head_channels=256, B=2）：
        attn_stride=1（原始）：注意力矩阵 9216²×8heads×2 = 2.7GB  → OOM on 4050
        attn_stride=4（默认）：注意力矩阵 576²×8heads×2  = 21MB   ✓

    Args:
        channels:    输入/输出特征通道数
        num_heads:   交叉注意力头数
        attn_stride: 注意力计算的降采样倍数（默认 4，即在 H/16 分辨率做注意力）
    """

    def __init__(self, channels: int = 256, num_heads: int = 8, attn_stride: int = 4):
        super().__init__()
        self.channels    = channels
        self.attn_stride = attn_stride

        # 降采样到低分辨率（仅用于注意力计算）
        self.pool = nn.AvgPool2d(attn_stride, attn_stride) if attn_stride > 1 else nn.Identity()

        # Legacy inactive modules kept for strict checkpoint compatibility.
        # The one-way CMIM no longer lets F_L attend to F_A.
        self.L_to_A = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        # F_A 关注 F_L（在低分辨率计算）
        self.A_to_L = nn.MultiheadAttention(channels, num_heads, batch_first=True)

        # 门控融合在全分辨率执行
        # gate_L/norm_L are retained but inactive for checkpoint compatibility.
        self.gate_L = nn.Sequential(nn.Linear(channels * 2, channels), nn.Sigmoid())
        self.gate_A = nn.Sequential(nn.Linear(channels * 2, channels), nn.Sigmoid())

        self.norm_L = nn.LayerNorm(channels)
        self.norm_A = nn.LayerNorm(channels)

    def forward(
        self, F_L: torch.Tensor, F_A: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            F_L: [B, channels, H, W]  线型特征
            F_A: [B, channels, H, W]  面型特征
        Returns:
            F_L_out, F_A_enh: F_L 原样返回，F_A 由 F_L 上下文增强，形状不变
        """
        B, C, H, W = F_L.shape

        # ── 低分辨率交叉注意力 ─────────────────────────────────────────────────
        FL_s = self.pool(F_L)                                    # [B, C, H/s, W/s]
        FA_s = self.pool(F_A)
        _, _, Hs, Ws = FL_s.shape

        fl = FL_s.view(B, C, Hs * Ws).permute(0, 2, 1)          # [B, HW_s, C]
        fa = FA_s.view(B, C, Hs * Ws).permute(0, 2, 1)

        # MultiheadAttention 在 fp16 下 Q·K^T 随特征幅值增大可超过 65504 → inf → NaN
        # 强制 float32 计算后还原为原始 dtype（参考 DSADecoder 的同类处理）。
        # 单向交互：仅 F_A 关注 F_L，F_L 不再被 F_A 反向改写。
        _dtype = fl.dtype
        a_ctx_s, _ = self.A_to_L(fa.float(), fl.float(), fl.float())
        a_ctx_s = a_ctx_s.to(_dtype)

        # ── 将注意力上下文上采样到全分辨率 ────────────────────────────────────
        A_ctx = a_ctx_s.permute(0, 2, 1).view(B, C, Hs, Ws)
        if self.attn_stride > 1:
            A_ctx = F.interpolate(A_ctx, size=(H, W), mode='bilinear', align_corners=False)

        # ── 全分辨率门控融合（Linear 不随 HW 二次增长）────────────────────────
        fa_full = F_A.view(B, C, H * W).permute(0, 2, 1)
        a_ctx_f = A_ctx.view(B, C, H * W).permute(0, 2, 1)

        gate_a  = self.gate_A(torch.cat([fa_full, a_ctx_f], dim=-1))
        fa_enh  = self.norm_A(fa_full.float() + gate_a.float() * a_ctx_f.float()).to(_dtype)

        F_L_out = F_L
        F_A_enh = fa_enh.permute(0, 2, 1).view(B, C, H, W)
        return F_L_out, F_A_enh
