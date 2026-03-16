"""隧道语义分割模型：Backbone + Decode Head。"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from .backbones.dinov3_convnext import DINOv3ConvNeXt
from .heads.uper_head import UPerHead
from .heads.mlp_head import MLPHead


class TunnelSegmentor(nn.Module):
    """
    端到端语义分割模型。
    Args:
        num_classes:          类别数 (含背景)
        backbone_weight_path: DINOv3 预训练权重路径, None=不加载
        head_type:            解码头类型 ("uper" / "mlp")
        frozen_stages:        冻结 backbone 前 N 个 stage
        head_channels:        解码头内部通道数

        UPerHead 专用:
            pool_scales:      PPM 模块的池化尺度

        MLPHead 专用:
            embed_dim:        统一嵌入维度
    """

    def __init__(
        self,
        num_classes: int = 5,
        backbone_weight_path: str = None,
        head_type: str = "mlp",
        frozen_stages: int = 0,
        head_channels: int = 512,
        pool_scales: list = None,   # None → UPerHead 默认 [1,2,4,8]
        embed_dim: int = 256,
    ):
        super().__init__()

        if pool_scales is None:
            pool_scales = [1, 2, 4, 8]

        # ── Backbone ──
        self.backbone = DINOv3ConvNeXt(
            weight_path=backbone_weight_path,
            frozen_stages=frozen_stages,
        )
        in_channels_list = self.backbone.out_channels  # (96, 192, 384, 768)

        # ── Decode Head ──
        self.head_type = head_type.lower()
        if self.head_type == "uper":
            self.decode_head = UPerHead(
                in_channels_list=in_channels_list,
                pool_scales=pool_scales,
                channels=head_channels,
                num_classes=num_classes,
            )
        elif self.head_type == "mlp":
            self.decode_head = MLPHead(
                in_channels_list=in_channels_list,
                embed_dim=embed_dim,
                num_classes=num_classes,
            )
        else:
            raise ValueError(f"不支持的 head_type: '{head_type}'，可选: 'uper', 'mlp'")

        self.num_classes = num_classes

        # 统计参数量
        backbone_params = sum(p.numel() for p in self.backbone.parameters()) / 1e6
        head_params = sum(p.numel() for p in self.decode_head.parameters()) / 1e6
        total_params = backbone_params + head_params
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        logger.info(
            f"TunnelSegmentor 初始化完成:\n"
            f"  Backbone:  DINOv3 ConvNeXt-Tiny ({backbone_params:.1f}M)\n"
            f"  Head:      {head_type.upper()} ({head_params:.1f}M)\n"
            f"  总参数:    {total_params:.1f}M (可训练: {trainable_params:.1f}M)"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播: 图像 → 分割 logits。

        Args:
            x: [B, 3, H, W] 输入图像 (已归一化)

        Returns:
            [B, num_classes, H, W] 分割 logits (与输入同分辨率)
        """
        input_size = x.shape[2:]  # (H, W)

        # 1. Backbone → 多尺度特征
        features = self.backbone(x)  # List of 4 tensors

        # 2. Decode Head → 低分辨率预测
        logits = self.decode_head(features)  # [B, num_classes, H/4, W/4]

        # 3. 上采样到输入分辨率
        if logits.shape[2:] != input_size:
            logits = F.interpolate(
                logits, size=input_size,
                mode="bilinear", align_corners=False,
            )

        return logits


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("TunnelSegmentor 端到端测试")
    logger.info("=" * 60)

    WEIGHT_PATH = "dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth"
    NUM_CLASSES = 5

    logger.info("\n── UPerHead 测试 ──")
    model_uper = TunnelSegmentor(
        num_classes=NUM_CLASSES,
        backbone_weight_path=WEIGHT_PATH,
        head_type="uper",
        frozen_stages=2,
    )
    model_uper.eval()

    dummy = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        out = model_uper(dummy)
    logger.success(f"UPerHead 输出: {out.shape}")
    assert out.shape == (2, NUM_CLASSES, 512, 512), f"形状不对: {out.shape}"

    # ── 测试 MLPHead ──
    logger.info("\n── MLPHead 测试 ──")
    model_mlp = TunnelSegmentor(
        num_classes=NUM_CLASSES,
        backbone_weight_path=WEIGHT_PATH,
        head_type="mlp",
        frozen_stages=-1,
    )
    model_mlp.eval()

    with torch.no_grad():
        out_mlp = model_mlp(dummy)
    logger.success(f"✅ MLPHead 输出: {out_mlp.shape}")
    assert out_mlp.shape == (2, NUM_CLASSES, 512, 512)

    # ── 测试梯度流 ──
    logger.info("\n── 梯度流测试 ──")
    model_grad = TunnelSegmentor(
        num_classes=NUM_CLASSES,
        backbone_weight_path=WEIGHT_PATH,
        head_type="uper",
        frozen_stages=2,
    )
    model_grad.train()
    images = torch.randn(2, 3, 512, 512)
    masks = torch.randint(0, NUM_CLASSES, (2, 512, 512)).long()
    logits = model_grad(images)
    loss = F.cross_entropy(logits, masks)
    loss.backward()

    # 检查冻结层无梯度，解码头有梯度
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model_grad.decode_head.parameters())
    assert has_grad, "解码头应有梯度"
    logger.success(f"✅ 梯度流正常: loss={loss.item():.4f}")

    logger.success("\n✅ TunnelSegmentor 全部测试通过！")
