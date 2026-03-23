"""
models/backbones/dinov3_convnext.py
DINOv3 ConvNeXt-Tiny Backbone，封装为 nn.Module。

输出 4 个 stage 的多尺度特征 (features_only 模式)：
    Stage 0: [B,  96, 128, 128]  (1/4)
    Stage 1: [B, 192,  64,  64]  (1/8)
    Stage 2: [B, 384,  32,  32]  (1/16)
    Stage 3: [B, 768,  16,  16]  (1/32)
"""

import re
import torch
import torch.nn as nn
import timm
from loguru import logger


# ═══════════════════════════════════════════════════════════════════════════════
# 权重映射: DINOv3 官方 checkpoint → timm convnext_tiny
# ═══════════════════════════════════════════════════════════════════════════════

def _convert_dinov3_to_timm(dinov3_state_dict: dict, features_only: bool = True) -> dict:
    """
    将 DINOv3 ConvNeXt-Tiny 官方预训练权重的 key 映射为 timm 格式。
    """
    timm_state_dict = {}
    skipped = []

    for k, v in dinov3_state_dict.items():
        new_key = None

        # ── downsample_layers.0 → stem ──
        m = re.match(r"downsample_layers\.0\.(\d+)\.(.*)", k)
        if m:
            new_key = f"stem.{m.group(1)}.{m.group(2)}"

        # ── downsample_layers.{1,2,3} → stages.{1,2,3}.downsample ──
        if new_key is None:
            m = re.match(r"downsample_layers\.(\d+)\.(\d+)\.(.*)", k)
            if m:
                stage_idx = int(m.group(1))
                sub_idx = m.group(2)
                rest = m.group(3)
                new_key = f"stages.{stage_idx}.downsample.{sub_idx}.{rest}"

        # ── stages.X.Y.dwconv → stages.X.blocks.Y.conv_dw ──
        if new_key is None:
            m = re.match(r"stages\.(\d+)\.(\d+)\.dwconv\.(.*)", k)
            if m:
                new_key = f"stages.{m.group(1)}.blocks.{m.group(2)}.conv_dw.{m.group(3)}"

        # ── stages.X.Y.pwconv1 → stages.X.blocks.Y.mlp.fc1 ──
        if new_key is None:
            m = re.match(r"stages\.(\d+)\.(\d+)\.pwconv1\.(.*)", k)
            if m:
                new_key = f"stages.{m.group(1)}.blocks.{m.group(2)}.mlp.fc1.{m.group(3)}"

        # ── stages.X.Y.pwconv2 → stages.X.blocks.Y.mlp.fc2 ──
        if new_key is None:
            m = re.match(r"stages\.(\d+)\.(\d+)\.pwconv2\.(.*)", k)
            if m:
                new_key = f"stages.{m.group(1)}.blocks.{m.group(2)}.mlp.fc2.{m.group(3)}"

        # ── stages.X.Y.norm → stages.X.blocks.Y.norm ──
        if new_key is None:
            m = re.match(r"stages\.(\d+)\.(\d+)\.norm\.(.*)", k)
            if m:
                new_key = f"stages.{m.group(1)}.blocks.{m.group(2)}.norm.{m.group(3)}"

        # ── stages.X.Y.gamma → stages.X.blocks.Y.gamma ──
        if new_key is None:
            m = re.match(r"stages\.(\d+)\.(\d+)\.gamma", k)
            if m:
                new_key = f"stages.{m.group(1)}.blocks.{m.group(2)}.gamma"

        # ── norm.* / norms.* → 跳过 (分类头参数) ──
        if new_key is None:
            if k.startswith("norm.") or k.startswith("norms."):
                skipped.append(k)
                continue

        # features_only 模式: stem.0 → stem_0, stages.0 → stages_0
        if new_key is not None and features_only:
            m2 = re.match(r"(stem)\.(\d+)\.(.*)", new_key)
            if m2:
                new_key = f"{m2.group(1)}_{m2.group(2)}.{m2.group(3)}"
            m2 = re.match(r"(stages)\.(\d+)\.(.*)", new_key)
            if m2:
                new_key = f"{m2.group(1)}_{m2.group(2)}.{m2.group(3)}"

        if new_key is not None:
            timm_state_dict[new_key] = v
        else:
            skipped.append(k)

    logger.info(
        f"权重映射完成: {len(timm_state_dict)} 个参数已映射, "
        f"{len(skipped)} 个参数已跳过"
    )
    if skipped:
        logger.debug(f"跳过的 key: {skipped}")

    return timm_state_dict


def convert_dinov3_to_timm(dinov3_state_dict: dict, features_only: bool = True) -> dict:
    """公开的权重 key 映射函数。"""
    return _convert_dinov3_to_timm(dinov3_state_dict, features_only=features_only)


def load_dinov3_convnext_tiny(
    weight_path: str = "dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth",
    features_only: bool = True,
) -> nn.Module:
    """公开的模型加载函数。"""
    model = timm.create_model(
        "convnext_tiny",
        pretrained=False,
        features_only=features_only,
    )

    checkpoint = torch.load(weight_path, map_location="cpu")
    timm_weights = convert_dinov3_to_timm(checkpoint, features_only=features_only)

    missing, unexpected = model.load_state_dict(timm_weights, strict=False)
    if missing:
        logger.warning(f"缺失的 key ({len(missing)}): {missing[:5]}...")
    if unexpected:
        logger.warning(f"多余的 key ({len(unexpected)}): {unexpected[:5]}...")
    logger.success("✅ DINOv3 ConvNeXt-Tiny 预训练权重加载成功！")

    return model


# ═══════════════════════════════════════════════════════════════════════════════
# DINOv3 ConvNeXt Backbone
# ═══════════════════════════════════════════════════════════════════════════════

class DINOv3ConvNeXt(nn.Module):
    """DINOv3 ConvNeXt-Tiny Backbone 封装。"""

    # ConvNeXt-Tiny 各 stage 输出通道数
    OUT_CHANNELS = (96, 192, 384, 768)

    def __init__(
        self,
        weight_path: str = None,
        frozen_stages: int = -1,
    ):
        super().__init__()

        # 创建 timm convnext_tiny (features_only 模式)
        self.backbone = timm.create_model(
            "convnext_tiny",
            pretrained=False,
            features_only=True,
        )

        # 加载 DINOv3 预训练权重
        if weight_path is not None:
            self._load_pretrained(weight_path)

        # 冻结指定 stage
        self.frozen_stages = frozen_stages
        if frozen_stages != 0:
            self._freeze_stages(frozen_stages)

    def _load_pretrained(self, weight_path: str):
        """加载并映射 DINOv3 预训练权重。"""
        checkpoint = torch.load(weight_path, map_location="cpu")
        timm_weights = _convert_dinov3_to_timm(checkpoint, features_only=True)

        missing, unexpected = self.backbone.load_state_dict(timm_weights, strict=False)
        if missing:
            logger.warning(f"缺失的 key ({len(missing)}): {missing[:5]}...")
        if unexpected:
            logger.warning(f"多余的 key ({len(unexpected)}): {unexpected[:5]}...")
        logger.success("✅ DINOv3 ConvNeXt-Tiny 预训练权重加载成功！")

    def _freeze_stages(self, frozen_stages: int):
        """
        冻结前 N 个 stage 的参数。

        frozen_stages = -1: 冻结全部
        frozen_stages = 0:  不冻结
        frozen_stages = N:  冻结 stem + stage_0 ~ stage_{N-1}
        """
        if frozen_stages == -1:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("❄️ Backbone 全部冻结")
            return

        # 冻结 stem
        if frozen_stages >= 1:
            for name, param in self.backbone.named_parameters():
                if name.startswith("stem"):
                    param.requires_grad = False

        # 冻结 stage 0 ~ stage_{frozen_stages - 1}
        for stage_idx in range(frozen_stages):
            prefix = f"stages_{stage_idx}."
            for name, param in self.backbone.named_parameters():
                if name.startswith(prefix):
                    param.requires_grad = False

        frozen_params = sum(1 for p in self.backbone.parameters() if not p.requires_grad)
        total_params = sum(1 for p in self.backbone.parameters())
        logger.info(
            f"❄️ 冻结了 {frozen_stages} 个 stage "
            f"({frozen_params}/{total_params} 参数被冻结)"
        )

    def set_frozen_stages(self, frozen_stages: int):
        """
        动态更新冻结阶段数。

        先解冻全部参数，再按新的 frozen_stages 重新冻结，
        以确保状态与 __init__ 时的行为一致。

        frozen_stages = -1: 冻结全部
        frozen_stages = 0:  不冻结
        frozen_stages = N:  冻结 stem + stage_0 ~ stage_{N-1}
        """
        # 先全部解冻
        for param in self.backbone.parameters():
            param.requires_grad = True

        self.frozen_stages = frozen_stages
        if frozen_stages != 0:
            self._freeze_stages(frozen_stages)

    @property
    def out_channels(self):
        """各 stage 输出通道数的列表。"""
        return list(self.OUT_CHANNELS)

    def forward(self, x: torch.Tensor) -> list:
        """
        Args:
            x: [B, 3, H, W] 输入图像

        Returns:
            List[Tensor]: 4 个 stage 的特征图
        """
        return self.backbone(x)


# ═══════════════════════════════════════════════════════════════════════════════
# __main__: 单元测试
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("DINOv3ConvNeXt Backbone 单元测试")
    logger.info("=" * 60)

    WEIGHT_PATH = "dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth"

    # 测试 1: 基本前向传播
    backbone = DINOv3ConvNeXt(weight_path=WEIGHT_PATH, frozen_stages=0)
    backbone.eval()

    dummy = torch.randn(2, 3, 512, 512)
    features = backbone(dummy)

    logger.info(f"out_channels: {backbone.out_channels}")
    for i, f in enumerate(features):
        logger.info(f"  Stage {i}: {f.shape}")
    assert len(features) == 4
    assert features[0].shape == (2, 96, 128, 128)
    assert features[3].shape == (2, 768, 16, 16)
    logger.success("✅ 前向传播测试通过")

    # 测试 2: 冻结 stage
    backbone_frozen = DINOv3ConvNeXt(weight_path=WEIGHT_PATH, frozen_stages=2)
    frozen_count = sum(1 for p in backbone_frozen.parameters() if not p.requires_grad)
    trainable_count = sum(1 for p in backbone_frozen.parameters() if p.requires_grad)
    logger.info(f"冻结={frozen_count}, 可训练={trainable_count}")
    assert frozen_count > 0 and trainable_count > 0
    logger.success("✅ 冻结测试通过")

    # 测试 3: 全冻结
    backbone_all_frozen = DINOv3ConvNeXt(weight_path=WEIGHT_PATH, frozen_stages=-1)
    trainable = sum(1 for p in backbone_all_frozen.parameters() if p.requires_grad)
    assert trainable == 0
    logger.success("✅ 全冻结测试通过")

    logger.success("✅ DINOv3ConvNeXt 全部测试通过！")
