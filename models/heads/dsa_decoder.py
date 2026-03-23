from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeformableStripAttention(nn.Module):
    """
    可变形条状注意力（DSA）。

    为每个注意力头学习 num_strips 个方向角，沿每个方向采样
    points_per_strip 个点（F.grid_sample）。方向在全局共享（不随位置变化），
    对每个空间位置执行各向异性注意力聚合。

    Args:
        dim:               输入/输出通道数
        num_heads:         注意力头数
        num_strips:        每个头的条方向数
        points_per_strip:  每条的采样点数
        max_offset:        最大采样偏移（归一化坐标，[-1,1] 空间）
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        num_strips: int = 4,
        points_per_strip: int = 8,
        max_offset: float = 0.5,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_strips = num_strips
        self.M = points_per_strip
        self.max_offset = max_offset
        self.scale = self.head_dim ** -0.5

        # 全局方向预测：全局均值池化 → 线性 → (cos θ, sin θ) 对
        self.direction_pred = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, num_heads * num_strips * 2),
        )
        self.q_proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.k_proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.v_proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.out_proj = nn.Conv2d(dim, dim, 1)
        self.norm = nn.GroupNorm(num_groups=num_heads, num_channels=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, dim, H, W]
        Returns:
            out: [B, dim, H, W]
        """
        B, C, H, W = x.shape
        device = x.device

        # ── 1. 预测条方向 ──────────────────────────────────────────────────────
        dirs = self.direction_pred(x)                           # [B, nh*ns*2]
        dirs = dirs.view(B, self.num_heads, self.num_strips, 2)
        # FP16 下 eps=1e-12 会被截断为 0，导致 0/0=NaN；强制 float32 归一化后还原
        dirs = F.normalize(dirs.float(), dim=-1, eps=1e-6).to(x.dtype)

        # ── 2. Q / K / V 投影 ─────────────────────────────────────────────────
        Q = self.q_proj(x)   # [B, C, H, W]
        K = self.k_proj(x)
        V = self.v_proj(x)

        # ── 3. 构建采样坐标 ────────────────────────────────────────────────────
        # 沿条方向的 M 个采样点
        t = torch.linspace(-self.max_offset, self.max_offset, self.M, device=device)

        # strip_offsets: [B, nh, ns, M, 2]
        strip_offsets = dirs.unsqueeze(-2) * t.view(1, 1, 1, -1, 1)

        # 归一化基础网格 (x 轴对应 W，y 轴对应 H)
        ys = torch.linspace(-1, 1, H, device=device)
        xs = torch.linspace(-1, 1, W, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        base = torch.stack([grid_x, grid_y], dim=-1)            # [H, W, 2]
        base_flat = base.view(1, H * W, 1, 2)                   # [1, H*W, 1, 2]

        # ── 4. 逐注意力头执行条状注意力 ────────────────────────────────────────
        head_outputs = []
        for h in range(self.num_heads):
            hd = self.head_dim
            K_h = K[:, h * hd:(h + 1) * hd]   # [B, hd, H, W]
            V_h = V[:, h * hd:(h + 1) * hd]
            Q_h = Q[:, h * hd:(h + 1) * hd]

            # offsets for this head: [B, ns, M, 2]
            off_h = strip_offsets[:, h]         # [B, ns, M, 2]
            off_h = off_h.unsqueeze(2)          # [B, ns, 1, M, 2]

            # sample_grid: [B, ns, H*W, M, 2]
            sg = (base_flat + off_h).clamp(-1, 1)              # [B, ns, H*W, M, 2]
            # reshape → [B*ns, H*W, M, 2]
            sg_flat = sg.reshape(B * self.num_strips, H * W, self.M, 2)

            # Expand K_h and V_h for all strips
            K_h_exp = K_h.unsqueeze(1).expand(-1, self.num_strips, -1, -1, -1)
            K_h_flat = K_h_exp.reshape(B * self.num_strips, hd, H, W)
            V_h_exp = V_h.unsqueeze(1).expand(-1, self.num_strips, -1, -1, -1)
            V_h_flat = V_h_exp.reshape(B * self.num_strips, hd, H, W)

            # grid_sample → [B*ns, hd, H*W, M]
            sK = F.grid_sample(K_h_flat, sg_flat, mode='bilinear',
                               align_corners=True, padding_mode='border')
            sV = F.grid_sample(V_h_flat, sg_flat, mode='bilinear',
                               align_corners=True, padding_mode='border')

            # Reshape → [B, H*W, ns*M, hd]
            sK = sK.reshape(B, self.num_strips, hd, H * W, self.M)
            sK = sK.permute(0, 3, 1, 4, 2).reshape(B, H * W, self.num_strips * self.M, hd)
            sV = sV.reshape(B, self.num_strips, hd, H * W, self.M)
            sV = sV.permute(0, 3, 1, 4, 2).reshape(B, H * W, self.num_strips * self.M, hd)

            # Q: [B, H*W, 1, hd]
            Q_flat = Q_h.view(B, hd, H * W).permute(0, 2, 1).unsqueeze(2)

            # 注意力权重在 float32 下计算，防止 Q@K 随权重增大后在 FP16 下 overflow
            # FP16 max=65504，head_dim=64 的点积在值域扩张后可轻易超过此阈值
            attn = torch.matmul(Q_flat.float(), sK.float().transpose(-1, -2)) * self.scale
            attn = F.softmax(attn, dim=-1).to(Q_flat.dtype)

            # 加权求和: [B, H*W, hd]
            out_h = torch.matmul(attn, sV).squeeze(2)          # [B, H*W, hd]
            out_h = out_h.permute(0, 2, 1).view(B, hd, H, W)   # [B, hd, H, W]
            head_outputs.append(out_h)

        # ── 5. 拼接各头 + 输出投影 ────────────────────────────────────────────
        out = torch.cat(head_outputs, dim=1)    # [B, C, H, W]
        out = self.out_proj(self.norm(out))
        return out + x                           # 残差连接


class DSADecoder(nn.Module):
    """
    线型流解码器（Deformable Strip Attention Decoder）。

    FPN 融合多尺度特征后，在最高分辨率（H/4）执行可变形条状注意力，
    用于捕获裂缝的长程线型依赖。

    Args:
        in_channels_list: 骨干各阶段通道数，如 (96, 192, 384, 768)
        channels:         统一通道宽度
        dsa_num_heads:    DSA 注意力头数
        dsa_num_strips:   每头的条方向数
        dsa_points_per_strip: 每条的采样点数
    Returns:
        [B, channels, H/4, W/4]  线型特征图（不含分类头）
    """
    def __init__(
        self,
        in_channels_list: tuple,
        channels: int = 256,
        dsa_num_heads: int = 4,
        dsa_num_strips: int = 4,
        dsa_points_per_strip: int = 8,
    ):
        super().__init__()
        _ng = max(1, channels // 8)   # GroupNorm 组数，channels=256 → 32 组，8ch/组
        # 侧边连接：将各尺度特征映射到 channels
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, channels, 1, bias=False),
                nn.GroupNorm(_ng, channels),
                nn.GELU(),
            )
            for c in in_channels_list
        ])
        # FPN 融合卷积
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.GroupNorm(_ng, channels),
                nn.GELU(),
            )
            for _ in in_channels_list
        ])
        # DSA 在降采样 2× 后的分辨率（H/8）执行，显著降低显存
        # H/4=96×96=9216 tokens → H/8=48×48=2304 tokens，中间张量减少 16×
        self.dsa = DeformableStripAttention(
            dim=channels,
            num_heads=dsa_num_heads,
            num_strips=dsa_num_strips,
            points_per_strip=dsa_points_per_strip,
        )
        # 输出层归一化
        self.out_norm = nn.GroupNorm(_ng, channels)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: [C1@/4, C2@/8, C3@/16, C4@/32]，已由 MRM 路由加权
        Returns:
            [B, channels, H/4, W/4]
        """
        # 侧边连接
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # 自顶向下 FPN 融合
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:],
                mode='bilinear', align_corners=False,
            )

        # FPN 卷积
        fpn_outs = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]

        # 上采样到 H/4，拼接所有尺度
        target_size = fpn_outs[0].shape[2:]
        fused = fpn_outs[0]
        for feat in fpn_outs[1:]:
            fused = fused + F.interpolate(feat, size=target_size,
                                          mode='bilinear', align_corners=False)

        # 在 H/8 分辨率执行 DSA，减少 16× 中间张量（H/8=2304 tokens vs H/4=9216 tokens）
        # DSA.forward 内部已含残差（out + x），直接上采样回 H/4 即可
        fused_half = F.avg_pool2d(fused, kernel_size=2, stride=2)   # [B, C, H/8, W/8]
        attended   = self.dsa(fused_half)                            # [B, C, H/8, W/8]（含残差）
        out        = F.interpolate(attended, size=target_size,
                                   mode='bilinear', align_corners=False)  # [B, C, H/4, W/4]
        return self.out_norm(out)
