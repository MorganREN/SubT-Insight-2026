"""
dataload/augmentation.py
语义分割数据增强管线。

设计原则
--------
1. 空间变换（翻转/旋转/裁剪）同时作用于 image 和 mask，保证空间一致性。
2. 颜色/光照变换仅作用于 image，不修改 mask 像素值。
3. 分三种模式（train / val / test），val 与 test 不含随机增强。
4. 归一化使用 ImageNet 均值方差（与 DINOv3 预训练一致）。
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils.constants import IMAGENET_MEAN as _IMAGENET_MEAN, IMAGENET_STD as _IMAGENET_STD


class SegmentationAugmentation:
    """
    隧道缺陷语义分割数据增强。

    Parameters
    ----------
    mode : str
        "train"：随机空间 + 颜色增强，含随机裁剪
        "val"   / "valid"：仅 Resize + 归一化
        "test"  ：同 val，不含随机性
    input_size : int
        模型输入的正方形边长（像素），默认 512。
    hflip_p : float
        水平翻转概率，仅 train 生效。默认 0.5。
    vflip_p : float
        垂直翻转概率，仅 train 生效。默认 0.3。
    rotate90_p : float
        90° 旋转（随机 k 次）概率，仅 train 生效。默认 0.3。
    color_jitter_p : float
        亮度/对比度/gamma 颜色扰动概率，仅 train 生效。默认 0.5。
    gauss_noise_p : float
        高斯噪声概率，仅 train 生效。默认 0.2。
    elastic_p : float
        弹性变形概率，仅 train 生效（模拟裂缝/渗漏的几何变化）。默认 0.2。
    """

    MODES = {"train", "val", "valid", "test"}

    def __init__(
        self,
        mode: str = "train",
        input_size: int = 512,
        *,
        hflip_p: float = 0.5,
        vflip_p: float = 0.3,
        rotate90_p: float = 0.3,
        color_jitter_p: float = 0.5,
        gauss_noise_p: float = 0.2,
        elastic_p: float = 0.2,
    ):
        mode = mode.lower()
        if mode not in self.MODES:
            raise ValueError(f"mode 必须是 {self.MODES}，得到: '{mode}'")

        self.mode = mode
        self.input_size = input_size

        if mode == "train":
            self._transform = self._build_train_transform(
                input_size,
                hflip_p=hflip_p,
                vflip_p=vflip_p,
                rotate90_p=rotate90_p,
                color_jitter_p=color_jitter_p,
                gauss_noise_p=gauss_noise_p,
                elastic_p=elastic_p,
            )
        else:
            # val / valid / test：确定性处理
            self._transform = self._build_eval_transform(input_size)

    # ──────────────────────────────────────────────────────────────────────────
    # 内部构建方法
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_train_transform(
        input_size: int,
        *,
        hflip_p: float,
        vflip_p: float,
        rotate90_p: float,
        color_jitter_p: float,
        gauss_noise_p: float,
        elastic_p: float,
    ) -> A.Compose:
        """
        训练增强管线（albumentations Compose）。

        管线顺序:
            1. 随机放缩后裁剪 → 得到 input_size × input_size 区域
               (RandomResizedCrop 保留缺陷的相对完整性)
            2. 空间翻转 / 旋转 → 扩充视角多样性
            3. 弹性变形 → 模拟裂缝、渗漏的真实形变
            4. 颜色扰动 → 适应隧道不同光照/污渍
            5. 高斯噪声 → 提升对传感器噪声的鲁棒性
            6. 归一化 + ToTensorV2
        """
        return A.Compose([
            # ── 1. 随机放缩裁剪：scale 范围保留 40%~100% 原图信息 ──
            A.RandomResizedCrop(
                height=input_size,
                width=input_size,
                scale=(0.4, 1.0),
                ratio=(0.75, 1.33),
                interpolation=1,        # cv2.INTER_LINEAR
            ),

            # ── 2. 空间翻转 / 旋转 ──
            A.HorizontalFlip(p=hflip_p),
            A.VerticalFlip(p=vflip_p),
            A.RandomRotate90(p=rotate90_p),

            # ── 3. 弹性变形（轻微，避免 mask 过度扭曲）──
            A.ElasticTransform(
                alpha=40,
                sigma=5,
                alpha_affine=5,         # 仿射扰动幅度（轻微）
                p=elastic_p,
            ),

            # ── 4. 颜色 / 光照扰动（仅作用于 image）──
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
                A.RandomGamma(gamma_limit=(70, 130)),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                ),
            ], p=color_jitter_p),

            # ── 5. 高斯噪声 ──
            A.GaussNoise(p=gauss_noise_p),

            # ── 6. 归一化 + 转 Tensor ──
            A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ToTensorV2(),   # image: C×H×W float32；mask: H×W uint8（值不变）
        ])

    @staticmethod
    def _build_eval_transform(input_size: int) -> A.Compose:
        """验证/测试增强管线。"""
        return A.Compose([
            # 先缩放：最长边 = input_size，保持宽高比
            A.LongestMaxSize(max_size=input_size, interpolation=1),
            # 再填充：短边补到 input_size，填充值 image=0，mask=0（背景）
            A.PadIfNeeded(
                min_height=input_size,
                min_width=input_size,
                border_mode=0,          # cv2.BORDER_CONSTANT
                value=0,                # image 填 0（黑色）
                mask_value=255,         # mask 填 ignore_index，padding 区域不参与评估
            ),
            A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ToTensorV2(),
        ])

    def __call__(self, image, mask):
        """
        执行增强。
        """
        result = self._transform(image=image, mask=mask)
        result["mask"] = result["mask"].squeeze(0).long()
        return result

    def __repr__(self) -> str:
        return (
            f"SegmentationAugmentation("
            f"mode='{self.mode}', input_size={self.input_size})"
        )
