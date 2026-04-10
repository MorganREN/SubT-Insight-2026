import os
from typing import Dict, Optional

import cv2
import matplotlib.pyplot as plt
import torch
from PIL import Image
from sklearn.decomposition import PCA
from torchvision import transforms
from loguru import logger

from .constants import IMAGENET_MEAN, IMAGENET_STD


def load_image(image_path: str, target_size: int = 512):
    """加载图像并进行标准化预处理。"""
    img = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    img_tensor = transform(img).unsqueeze(0)  # [1, 3, H, W]
    return img, img_tensor


def get_dinov3_convnext_features(model, img_tensor: torch.Tensor, stage_idx: int = -1):
    """
    提取 DINOv3 ConvNeXt 的多尺度特征，并返回指定 stage。

    Args:
        model:      features_only=True 的 timm 模型
        img_tensor: [B, 3, H, W]
        stage_idx:  要用于可视化的 stage 索引 (-1 表示最后一个 stage)

    Returns:
        选定 stage 的特征图 [B, C, H', W']
    """
    model.eval()
    with torch.no_grad():
        multi_scale_features = model(img_tensor)

    logger.info("多尺度特征图:")
    for index, feat in enumerate(multi_scale_features):
        logger.info(f"  Stage {index}: {feat.shape}")

    return multi_scale_features[stage_idx]


def visualize_pca(
    features: torch.Tensor,
    original_img: Image.Image,
    mask_overlay_img: Optional[Image.Image] = None,
    segmented_img: Optional[Image.Image] = None,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    将高维特征图通过 PCA 降维至 RGB 并可视化。
    可选同时展示 mask_overlay 和 segmented 可视化图。
    """
    _, channels, h, w = features.shape
    features_reshaped = features.squeeze(0).reshape(channels, -1).permute(1, 0).cpu().numpy()

    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(features_reshaped)

    for index in range(3):
        col = pca_features[:, index]
        col_min, col_max = col.min(), col.max()
        if col_max - col_min > 1e-8:
            pca_features[:, index] = (col - col_min) / (col_max - col_min)
        else:
            pca_features[:, index] = 0.0

    pca_img = pca_features.reshape(h, w, 3)
    pca_img_resized = cv2.resize(
        pca_img,
        (original_img.size[0], original_img.size[1]),
        interpolation=cv2.INTER_NEAREST,
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title("Original Tunnel Image")
    axes[0, 0].axis("off")

    if mask_overlay_img is not None:
        axes[0, 1].imshow(mask_overlay_img)
        axes[0, 1].set_title("Mask Overlay (GT)")
    else:
        axes[0, 1].set_title("Mask Overlay (N/A)")
    axes[0, 1].axis("off")

    if segmented_img is not None:
        axes[1, 0].imshow(segmented_img)
        axes[1, 0].set_title("Segmented Region (GT)")
    else:
        axes[1, 0].set_title("Segmented (N/A)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(pca_img_resized)
    axes[1, 1].set_title(f"DINOv3 Feature PCA ({channels}ch → RGB)")
    axes[1, 1].axis("off")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"可视化已保存: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def find_visualization_images(image_path: str) -> Dict[str, Optional[Image.Image]]:
    """
    根据图像路径，查找对应的 mask_overlay 和 segmented 可视化图。

    输入示例:
        dataset/aug_data1/img_dir/train/xxx.jpg

    返回:
        {'mask_overlay': PIL Image or None, 'segmented': PIL Image or None}
    """
    result = {"mask_overlay": None, "segmented": None}

    parts = image_path.replace("\\", "/").split("/")
    try:
        img_dir_idx = parts.index("img_dir")
    except ValueError:
        logger.warning(f"无法从路径中定位 img_dir: {image_path}")
        return result

    split = parts[img_dir_idx + 1]
    stem = os.path.splitext(parts[img_dir_idx + 2])[0]
    base_dir = "/".join(parts[:img_dir_idx])

    for vis_type in ["mask_overlay", "segmented"]:
        vis_path = os.path.join(base_dir, "visualization", split, vis_type, stem + ".png")
        if os.path.exists(vis_path):
            result[vis_type] = Image.open(vis_path).convert("RGB")
            logger.info(f"找到 {vis_type}: {vis_path}")
        else:
            logger.debug(f"未找到 {vis_type}: {vis_path}")

    return result
