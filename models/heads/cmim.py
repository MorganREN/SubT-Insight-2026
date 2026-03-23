from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossMorphologyInteractionModule(nn.Module):
    """
    跨形态交互模块（CMIM）。

    线型流（F_L，裂缝主导）与面型流（F_A，渗漏/剥落主导）之间的
    双向门控交叉注意力，允许两类病害特征互为上下文。

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

        # F_L 关注 F_A（在低分辨率计算）
        self.L_to_A = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        # F_A 关注 F_L（在低分辨率计算）
        self.A_to_L = nn.MultiheadAttention(channels, num_heads, batch_first=True)

        # 门控融合在全分辨率执行
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
            F_L_enh, F_A_enh: 增强后特征，形状不变
        """
        B, C, H, W = F_L.shape

        # ── 低分辨率交叉注意力 ─────────────────────────────────────────────────
        FL_s = self.pool(F_L)                                    # [B, C, H/s, W/s]
        FA_s = self.pool(F_A)
        _, _, Hs, Ws = FL_s.shape

        fl = FL_s.view(B, C, Hs * Ws).permute(0, 2, 1)          # [B, HW_s, C]
        fa = FA_s.view(B, C, Hs * Ws).permute(0, 2, 1)

        l_ctx_s, _ = self.L_to_A(fl, fa, fa)                    # [B, HW_s, C]
        a_ctx_s, _ = self.A_to_L(fa, fl, fl)

        # ── 将注意力上下文上采样到全分辨率 ────────────────────────────────────
        L_ctx = l_ctx_s.permute(0, 2, 1).view(B, C, Hs, Ws)
        A_ctx = a_ctx_s.permute(0, 2, 1).view(B, C, Hs, Ws)
        if self.attn_stride > 1:
            L_ctx = F.interpolate(L_ctx, size=(H, W), mode='bilinear', align_corners=False)
            A_ctx = F.interpolate(A_ctx, size=(H, W), mode='bilinear', align_corners=False)

        # ── 全分辨率门控融合（Linear 不随 HW 二次增长）────────────────────────
        fl_full = F_L.view(B, C, H * W).permute(0, 2, 1)        # [B, HW, C]
        fa_full = F_A.view(B, C, H * W).permute(0, 2, 1)
        l_ctx_f = L_ctx.view(B, C, H * W).permute(0, 2, 1)
        a_ctx_f = A_ctx.view(B, C, H * W).permute(0, 2, 1)

        gate_l  = self.gate_L(torch.cat([fl_full, l_ctx_f], dim=-1))
        fl_enh  = self.norm_L(fl_full + gate_l * l_ctx_f)

        gate_a  = self.gate_A(torch.cat([fa_full, a_ctx_f], dim=-1))
        fa_enh  = self.norm_A(fa_full + gate_a * a_ctx_f)

        F_L_enh = fl_enh.permute(0, 2, 1).view(B, C, H, W)
        F_A_enh = fa_enh.permute(0, 2, 1).view(B, C, H, W)
        return F_L_enh, F_A_enh
