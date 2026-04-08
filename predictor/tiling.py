"""
predictor/tiling.py

基于滑动窗口 Tiling 的原图推理。

流程
----
1. 按像素数判断群落（tiny / mobile / dslr / highres），与训练时 dataset_convert_awesome.py 保持一致
2. 以群落对应参数做滑动窗口裁剪（镜像填充），每个 patch resize 到 512×512 送入模型
3. 模型输出 softmax 概率图（不 argmax），反 resize 回原始 patch 尺寸
4. 用高斯权重在原图尺寸上做加权累加，最后 argmax 得到最终分割结果

公开接口
--------
    tiled_predict(model, image_np, device, num_classes, input_size=512) -> np.ndarray
        image_np : H×W×3 uint8 原图
        返回    : H×W uint8 预测 mask（类别索引）
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from utils.segmentation_vis import normalize_image

# ── 与 dataset_convert_awesome.py 严格对齐的群落参数 ──────────────────────────
_TINY_MAX   = 200_000
_MOBILE_MAX = 2_000_000
_DSLR_MAX   = 8_000_000

# (patch_size, stride)；tiny 群不 tiling，直接 resize
_TILING_PARAMS: dict[str, tuple[int, int] | None] = {
    "tiny":    None,
    "mobile":  (512, 384),
    "dslr":    (640, 400),
    "highres": (768, 384),
}


def _assign_group(h: int, w: int) -> str:
    px = h * w
    if px < _TINY_MAX:
        return "tiny"
    if px < _MOBILE_MAX:
        return "mobile"
    if px < _DSLR_MAX:
        return "dslr"
    return "highres"


def _reflect_pad(arr: np.ndarray, pad_h: int, pad_w: int) -> np.ndarray:
    if arr.ndim == 2:
        return np.pad(arr, ((0, pad_h), (0, pad_w)), mode="reflect")
    return np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")


def _make_gaussian_weight(patch_size: int, sigma_ratio: float = 0.25) -> np.ndarray:
    """生成 patch_size×patch_size 的二维高斯权重（中心高、边缘低）。"""
    sigma = patch_size * sigma_ratio
    ax = np.arange(patch_size) - (patch_size - 1) / 2.0
    gauss_1d = np.exp(-0.5 * (ax / sigma) ** 2)
    gauss_2d = np.outer(gauss_1d, gauss_1d)
    return (gauss_2d / gauss_2d.max()).astype(np.float32)


def _patch_to_tensor(patch_arr: np.ndarray, input_size: int) -> torch.Tensor:
    """HWC uint8 -> resized -> normalized -> CHW float Tensor。"""
    resized = np.array(
        Image.fromarray(patch_arr).resize((input_size, input_size), Image.BILINEAR),
        dtype=np.uint8,
    )
    norm = normalize_image(resized)            # HWC float32
    return torch.from_numpy(norm.transpose(2, 0, 1)).float()  # CHW


@torch.no_grad()
def tiled_predict(
    model: torch.nn.Module,
    image_np: np.ndarray,
    device: torch.device,
    num_classes: int,
    input_size: int = 512,
) -> np.ndarray:
    """
    对原图做 tiling 推理，返回原图尺寸的预测 mask。

    Parameters
    ----------
    model       : 已 eval() 且在 device 上的模型
    image_np    : H×W×3 uint8 原始图像
    device      : 推理设备
    num_classes : 类别数
    input_size  : 模型输入边长（默认 512）

    Returns
    -------
    pred_mask : H×W uint8 语义分割结果（类别索引）
    """
    H, W = image_np.shape[:2]
    group = _assign_group(H, W)
    params = _TILING_PARAMS[group]

    # ── Tiny 群：整图 resize 后单次推理 ────────────────────────────────────────
    if params is None:
        tensor = _patch_to_tensor(image_np, input_size).unsqueeze(0).to(device)
        logits = model(tensor)
        prob = F.softmax(logits.float(), dim=1).squeeze(0)          # (C, input_size, input_size)
        prob_np = prob.cpu().numpy().transpose(1, 2, 0)              # (input_size, input_size, C)
        # 反 resize 到原图尺寸
        prob_full = np.array(
            Image.fromarray(
                prob_np.reshape(input_size * input_size, num_classes).astype(np.float32)
            ).resize((W, H), Image.BILINEAR)
        )  # 此路径改用 F.interpolate 更干净
        # 用 torch interpolate 准确反 resize
        prob_t = torch.from_numpy(prob_np.transpose(2, 0, 1)).unsqueeze(0)  # (1, C, h, w)
        prob_full_t = F.interpolate(prob_t, size=(H, W), mode="bilinear", align_corners=False)
        return prob_full_t.squeeze(0).argmax(dim=0).numpy().astype(np.uint8)

    patch_size, stride = params

    # ── 镜像填充使图像能被完整 tiling ─────────────────────────────────────────
    pad_h = (stride - (H - patch_size) % stride) % stride if H > patch_size else max(0, patch_size - H)
    pad_w = (stride - (W - patch_size) % stride) % stride if W > patch_size else max(0, patch_size - W)
    if pad_h > 0 or pad_w > 0:
        image_pad = _reflect_pad(image_np, pad_h, pad_w)
    else:
        image_pad = image_np
    PH, PW = image_pad.shape[:2]

    # ── 概率累加矩阵（保存在 CPU，节省显存）────────────────────────────────────
    prob_sum   = np.zeros((num_classes, PH, PW), dtype=np.float64)
    weight_sum = np.zeros((PH, PW), dtype=np.float64)
    gauss      = _make_gaussian_weight(patch_size).astype(np.float64)  # (ps, ps)

    # ── 逐 patch 推理 ─────────────────────────────────────────────────────────
    for y in range(0, PH - patch_size + 1, stride):
        for x in range(0, PW - patch_size + 1, stride):
            patch = image_pad[y:y + patch_size, x:x + patch_size]  # (ps, ps, 3)

            tensor = _patch_to_tensor(patch, input_size).unsqueeze(0).to(device)
            logits = model(tensor)                                   # (1, C, input_size, input_size)

            # softmax 概率，反 resize 回 patch_size
            prob_t = F.softmax(logits.float(), dim=1)               # (1, C, input_size, input_size)
            prob_t = F.interpolate(
                prob_t, size=(patch_size, patch_size),
                mode="bilinear", align_corners=False,
            )                                                        # (1, C, ps, ps)
            prob_np = prob_t.squeeze(0).cpu().numpy().astype(np.float64)  # (C, ps, ps)

            # 高斯加权累加
            prob_sum[:, y:y + patch_size, x:x + patch_size]   += prob_np * gauss[np.newaxis]
            weight_sum[y:y + patch_size, x:x + patch_size]    += gauss

    # ── 归一化 + argmax ────────────────────────────────────────────────────────
    # weight_sum 中每个像素 > 0（至少被一个 patch 覆盖）
    weight_sum = np.maximum(weight_sum, 1e-8)
    avg_prob = prob_sum / weight_sum[np.newaxis]                    # (C, PH, PW)

    # 裁掉填充区域，还原到原图尺寸
    avg_prob = avg_prob[:, :H, :W]                                  # (C, H, W)
    pred_mask = avg_prob.argmax(axis=0).astype(np.uint8)            # (H, W)
    return pred_mask
