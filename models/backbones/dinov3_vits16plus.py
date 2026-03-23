"""
models/backbones/dinov3_vits16plus.py

DINOv3 ViT-S+/16 Distilled Backbone，封装为 nn.Module。

从 ViT-S/16（depth=12）的四个中间 block 抽取特征，经通道投影 + 空间缩放后
输出与 ConvNeXt-Tiny 同尺度的四级特征：
    Stage 0: [B,  96, H/4,  W/4 ]  (×4 双线性上采样)
    Stage 1: [B, 192, H/8,  W/8 ]  (×2 双线性上采样)
    Stage 2: [B, 384, H/16, W/16]  (直接输出)
    Stage 3: [B, 768, H/32, W/32]  (×2 平均池化下采样)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from loguru import logger


_EMBED_DIM = 384   # ViT-Small 嵌入维度
_DEPTH = 12        # ViT-Small Transformer block 数量

# 每组 3 个 block（共 4 组），各取最后一个 block 的输出作为多尺度特征
_EXTRACT_BLOCKS = [2, 5, 8, 11]   # 0-indexed


# ═══════════════════════════════════════════════════════════════════════════════
# DINOv3 ViT-S+/16 Backbone
# ═══════════════════════════════════════════════════════════════════════════════

class DINOv3ViTS16Plus(nn.Module):
    """DINOv3 ViT-S+/16 Distilled Backbone 封装。"""

    OUT_CHANNELS = (96, 192, 384, 768)

    def __init__(
        self,
        weight_path: str = None,
        frozen_stages: int = -1,
    ):
        super().__init__()

        # timm ViT-S/16，关闭分类头，返回全部 token 序列
        self.vit = timm.create_model(
            "vit_small_patch16_224",
            pretrained=False,
            num_classes=0,
            global_pool="",
            dynamic_img_size=True,
        )

        # 通道投影：ViT-S embed_dim=384 → 各 stage 目标通道数
        self.proj0 = nn.Conv2d(_EMBED_DIM,  96, 1, bias=False)
        self.proj1 = nn.Conv2d(_EMBED_DIM, 192, 1, bias=False)
        self.proj2 = nn.Conv2d(_EMBED_DIM, 384, 1, bias=False)
        self.proj3 = nn.Conv2d(_EMBED_DIM, 768, 1, bias=False)

        # 前向钩子存储中间特征
        self._hook_feats: dict[int, torch.Tensor] = {}
        self._hook_handles = []
        self._register_hooks()

        if weight_path is not None:
            self._load_pretrained(weight_path)

        self.frozen_stages = frozen_stages
        if frozen_stages != 0:
            self._freeze_stages(frozen_stages)

    # ── 钩子 ──────────────────────────────────────────────────────────────────

    def _register_hooks(self):
        for slot, blk_idx in enumerate(_EXTRACT_BLOCKS):
            def _make_hook(s):
                def _hook(m, inp, out):
                    self._hook_feats[s] = out
                return _hook
            h = self.vit.blocks[blk_idx].register_forward_hook(_make_hook(slot))
            self._hook_handles.append(h)

    # ── 权重加载 ───────────────────────────────────────────────────────────────

    def _load_pretrained(self, weight_path: str):
        checkpoint = torch.load(weight_path, map_location="cpu")

        # DINOv2/v3 checkpoint 可能将权重包在 "model"/"teacher"/"student" 下
        for wrap_key in ("model", "teacher", "student"):
            if wrap_key in checkpoint:
                checkpoint = checkpoint[wrap_key]
                break

        # 去掉常见前缀
        checkpoint = {
            k.replace("backbone.", "", 1).replace("module.", "", 1): v
            for k, v in checkpoint.items()
        }

        # pos_embed 形状适配（ViT-S+ 含 register tokens 时 pos_embed 可能不同）
        if "pos_embed" in checkpoint:
            src = checkpoint["pos_embed"]
            tgt = self.vit.pos_embed
            if src.shape != tgt.shape:
                logger.info(
                    f"pos_embed 形状不匹配: checkpoint={src.shape} model={tgt.shape}，"
                    f"自动插值适配。"
                )
                checkpoint["pos_embed"] = _interpolate_pos_embed(src, tgt)

        missing, unexpected = self.vit.load_state_dict(checkpoint, strict=False)
        logger.info(
            f"ViT-S 权重加载: {len(checkpoint) - len(unexpected)} 参数已加载, "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
        if missing:
            logger.debug(f"Missing keys (前5): {missing[:5]}")
        logger.success("✅ DINOv3 ViT-S+/16 预训练权重加载成功！")

    # ── 冻结 ───────────────────────────────────────────────────────────────────

    def _freeze_stages(self, frozen_stages: int):
        """
        冻结策略（与 DINOv3ConvNeXt 语义对齐）：
          frozen_stages = -1: 冻结全部 ViT 参数
          frozen_stages = 0:  不冻结
          frozen_stages = N:  冻结 patch_embed / cls_token / pos_embed
                              + 前 N 组 blocks（每组 3 个，共 4 组）
        """
        if frozen_stages == -1:
            for p in self.vit.parameters():
                p.requires_grad = False
            logger.info("❄️ ViT Backbone 全部冻结")
            return

        # 冻结嵌入参数
        if frozen_stages >= 1:
            for name, p in self.vit.named_parameters():
                if name.startswith(("patch_embed", "cls_token", "pos_embed")):
                    p.requires_grad = False

        # 冻结前 N 组 block（每组 3 个，最多 12 个）
        n_frozen_blocks = min(frozen_stages * 3, _DEPTH)
        for idx in range(n_frozen_blocks):
            for p in self.vit.blocks[idx].parameters():
                p.requires_grad = False

        frozen = sum(1 for p in self.vit.parameters() if not p.requires_grad)
        total  = sum(1 for p in self.vit.parameters())
        logger.info(f"❄️ 冻结了 {frozen_stages} 个 stage ({frozen}/{total} 参数被冻结)")

    def set_frozen_stages(self, frozen_stages: int):
        """动态更新冻结阶段数（与 DINOv3ConvNeXt 接口一致）。"""
        for p in self.vit.parameters():
            p.requires_grad = True
        self.frozen_stages = frozen_stages
        if frozen_stages != 0:
            self._freeze_stages(frozen_stages)

    # ── 属性 ───────────────────────────────────────────────────────────────────

    @property
    def out_channels(self):
        return list(self.OUT_CHANNELS)

    # ── 前向传播 ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> list:
        """
        Args:
            x: [B, 3, H, W]，H 和 W 须为 16 的整数倍

        Returns:
            List[Tensor]: 4 个 stage 的特征图（尺度与 ConvNeXt-Tiny 一致）
        """
        B, _, H, W = x.shape
        ph, pw = H // 16, W // 16

        self._hook_feats.clear()
        self.vit(x)   # 前向传播触发钩子，各中间输出存入 _hook_feats

        projs = [self.proj0, self.proj1, self.proj2, self.proj3]
        target_sizes = [
            (H // 4,  W // 4),    # stage 0: /4
            (H // 8,  W // 8),    # stage 1: /8
            (H // 16, W // 16),   # stage 2: /16
            (H // 32, W // 32),   # stage 3: /32
        ]

        out = []
        for slot, (proj, tgt) in enumerate(zip(projs, target_sizes)):
            tokens = self._hook_feats[slot]          # [B, 1+N_reg+N_patch, C]
            # 取最后 ph*pw 个 token 作为 patch tokens（兼容有/无 register tokens）
            patch_tokens = tokens[:, -ph * pw:, :]  # [B, ph*pw, C]
            feat = patch_tokens.transpose(1, 2).reshape(B, _EMBED_DIM, ph, pw)
            feat = proj(feat)
            if feat.shape[2:] != tgt:
                feat = F.interpolate(feat, size=tgt, mode="bilinear", align_corners=False)
            out.append(feat)

        return out


# ── 辅助函数 ──────────────────────────────────────────────────────────────────

def _interpolate_pos_embed(
    src: torch.Tensor, tgt: torch.Tensor
) -> torch.Tensor:
    """
    将 src pos_embed 插值到 tgt 形状，保留 cls token。
    src: [1, 1+N_src, C]（N_src 可能含 register token 位置）
    tgt: [1, 1+N_tgt, C]
    """
    src_cls = src[:, :1, :]
    N_tgt = tgt.shape[1] - 1

    # 如果 src 含额外 token（如 register），取最后 N_src_patch 个 patch 位置
    N_src_all = src.shape[1] - 1
    if N_src_all == N_tgt:
        return src
    patch_src = src[:, -N_tgt:, :] if N_src_all > N_tgt else src[:, 1:, :]

    if patch_src.shape[1] != N_tgt:
        gs_src = int(round(patch_src.shape[1] ** 0.5))
        gs_tgt = int(round(N_tgt ** 0.5))
        patch_src = patch_src.transpose(1, 2).reshape(1, -1, gs_src, gs_src)
        patch_src = F.interpolate(
            patch_src.float(), size=(gs_tgt, gs_tgt),
            mode="bicubic", align_corners=False,
        ).flatten(2).transpose(1, 2).to(src.dtype)

    return torch.cat([src_cls, patch_src], dim=1)


# ═══════════════════════════════════════════════════════════════════════════════
# __main__: 单元测试
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("DINOv3ViTS16Plus Backbone 单元测试")
    logger.info("=" * 60)

    WEIGHT_PATH = "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"

    backbone = DINOv3ViTS16Plus(weight_path=WEIGHT_PATH, frozen_stages=0)
    backbone.eval()

    dummy = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        features = backbone(dummy)

    logger.info(f"out_channels: {backbone.out_channels}")
    for i, f in enumerate(features):
        logger.info(f"  Stage {i}: {f.shape}")
    assert len(features) == 4
    assert features[0].shape == (2,  96, 128, 128)
    assert features[1].shape == (2, 192,  64,  64)
    assert features[2].shape == (2, 384,  32,  32)
    assert features[3].shape == (2, 768,  16,  16)
    logger.success("✅ 前向传播测试通过")

    backbone_frozen = DINOv3ViTS16Plus(weight_path=WEIGHT_PATH, frozen_stages=2)
    frozen_count = sum(1 for p in backbone_frozen.parameters() if not p.requires_grad)
    trainable_count = sum(1 for p in backbone_frozen.parameters() if p.requires_grad)
    logger.info(f"冻结={frozen_count}, 可训练={trainable_count}")
    assert frozen_count > 0 and trainable_count > 0
    logger.success("✅ 冻结测试通过")

    logger.success("✅ DINOv3ViTS16Plus 全部测试通过！")
