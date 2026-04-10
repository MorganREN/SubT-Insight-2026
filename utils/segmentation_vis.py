from __future__ import annotations

import numpy as np
import torch

from .constants import IMAGENET_MEAN as _MEAN, IMAGENET_STD as _STD

IMAGENET_MEAN = np.array(_MEAN, dtype=np.float32)
IMAGENET_STD  = np.array(_STD,  dtype=np.float32)


def normalize_image(image: np.ndarray) -> np.ndarray:
    image_f = image.astype(np.float32) / 255.0
    return (image_f - IMAGENET_MEAN) / IMAGENET_STD


def denormalize_image_tensor(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    image = image * IMAGENET_STD + IMAGENET_MEAN
    image = np.clip(image, 0.0, 1.0)
    return (image * 255.0).astype(np.uint8)


def colorize_mask(mask: np.ndarray, class_colors) -> np.ndarray:
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls_id, rgb in enumerate(class_colors):
        color[mask == cls_id] = rgb
    return color


def blend_overlay(image: np.ndarray, color_mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    blended = image.astype(np.float32) * (1.0 - alpha) + color_mask.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)
