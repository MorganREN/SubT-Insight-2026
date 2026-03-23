"""
criteria/topo_loss.py

基于持续同调的拓扑损失（Topology-Preserving Loss）。

参考：Hu et al., "Topology-Preserving Deep Image Segmentation", NeurIPS 2019.

原理
----
对裂缝类别的预测概率图使用 0 维持续同调（β₀，连通分量计数）：
  1. 以 (1 - prob) 为滤流函数在预测图和真值图上各自构建 Cubical Complex；
  2. 调用 gudhi.CubicalComplex.cofaces_of_persistence_pairs() 获取 0 维持续对：
       有限对：[birth_pixel_idx, death_pixel_idx]（像素级扁平索引）
       本质对：[birth_pixel_idx]（永不消亡的最大连通分量）
  3. 按持续度降序匹配预测与真值的持续对，在关键像素处施加 L2 损失：
       - 匹配对：推动预测关键像素的概率向真值关键像素的真值看齐；
       - 多余预测分量：在 birth 像素处压到 0（消灭多余连通块）；
       - 缺失真值分量：在真值 birth 像素处推到 1（创建缺失连通块）。

依赖
----
    gudhi >= 3.7   （pip install gudhi）

注意
----
- 拓扑计算在 CPU 上进行（numpy），是 O(HW log HW) 的操作；
  建议仅在第三训练阶段启用，max_pairs 控制计算量上限。
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# 底层持续同调工具
# ──────────────────────────────────────────────────────────────────────────────

def _persistence_pairs_0dim(
    f: np.ndarray,
    max_pairs: int,
) -> List[Tuple[int, int, float]]:
    """
    计算 2D 函数 f 的 0 维持续同调对。

    使用下水平集滤流：值小的像素先出现。对于 f = 1 - prob，高概率像素先出现，
    从而优先形成裂缝连通分量。

    调用 gudhi.CubicalComplex.cofaces_of_persistence_pairs()，
    其返回的单元索引与行主序的扁平像素索引一一对应。

    Returns
    -------
    List of (birth_pixel_idx, death_pixel_idx, persistence)，
    death_pixel_idx = -1 表示本质对（永不消亡）。
    按 persistence 降序排列，长度 ≤ max_pairs。
    """
    try:
        import gudhi
    except ImportError as e:
        raise ImportError(
            "TopologyLoss 需要 gudhi 库：pip install gudhi"
        ) from e

    H, W = f.shape
    f_flat = f.flatten()

    cc = gudhi.CubicalComplex(
        dimensions=[H, W],
        top_dimensional_cells=f_flat.tolist(),
    )
    cc.compute_persistence()
    pairs = cc.cofaces_of_persistence_pairs()
    # pairs[0]: 有限持续对列表（按维度索引）
    #   pairs[0][d] shape = (N, 2) → [[birth_idx, death_idx], ...]
    # pairs[1]: 本质持续对列表（按维度索引）
    #   pairs[1][d] shape = (M,)   → [birth_idx, ...]

    results: List[Tuple[int, int, float]] = []

    # 0 维有限对
    if len(pairs[0]) > 0:
        finite_0 = pairs[0][0]              # shape (N, 2) or empty
        for row in finite_0:
            b_idx, d_idx = int(row[0]), int(row[1])
            persistence = abs(float(f_flat[d_idx]) - float(f_flat[b_idx]))
            results.append((b_idx, d_idx, persistence))

    # 0 维本质对（持续度设为 inf，参与排序后置顶）
    if len(pairs[1]) > 0:
        ess_0 = pairs[1][0]                  # shape (M,) or empty
        for b_idx in ess_0:
            results.append((int(b_idx), -1, float('inf')))

    results.sort(key=lambda x: x[2], reverse=True)
    return results[:max_pairs]


# ──────────────────────────────────────────────────────────────────────────────
# 单张图的拓扑损失
# ──────────────────────────────────────────────────────────────────────────────

def _topo_loss_single(
    crack_prob: torch.Tensor,
    crack_gt: torch.Tensor,
    max_pairs: int,
) -> torch.Tensor:
    """
    计算单张图的拓扑损失。

    Parameters
    ----------
    crack_prob : [H, W] float32 Tensor，裂缝类概率（requires_grad）
    crack_gt   : [H, W] int64 Tensor，1 = 裂缝
    max_pairs  : 最大持续对数量

    Returns
    -------
    scalar Tensor（可反向传播）
    """
    zero = crack_prob.sum() * 0.0   # 保持设备、dtype 与 grad_fn

    if crack_gt.sum() == 0:
        return zero

    prob_np = crack_prob.detach().cpu().numpy().astype(np.float32)
    gt_np   = (crack_gt > 0).cpu().numpy().astype(np.float32)

    # 使用 1 - value 作滤流：高概率/真值=1 的像素先出现
    pred_pairs = _persistence_pairs_0dim(1.0 - prob_np, max_pairs)
    gt_pairs   = _persistence_pairs_0dim(1.0 - gt_np,   max_pairs)

    if not pred_pairs and not gt_pairs:
        return zero

    prob_flat = crack_prob.view(-1)                                      # [H*W]，保留 grad
    gt_flat   = torch.from_numpy(gt_np.flatten()).to(crack_prob.device)  # [H*W]，无 grad

    n_match = min(len(pred_pairs), len(gt_pairs))
    term_count = 0
    loss = zero

    # ── 匹配对：在关键像素处让预测向真值靠拢 ──
    for i in range(n_match):
        pb, pd, _ = pred_pairs[i]
        gb, gd, _ = gt_pairs[i]

        # birth 像素：裂缝种子，概率应接近真值 birth 像素的真值（1.0）
        loss = loss + (prob_flat[pb] - gt_flat[gb]) ** 2
        term_count += 1

        # death 像素：合并边界，仅在非本质对时施加（本质对 d_idx = -1）
        if pd >= 0 and gd >= 0:
            loss = loss + (prob_flat[pd] - gt_flat[gd]) ** 2
            term_count += 1
        elif pd >= 0:   # pred 有 death 但 gt 没有 → 压低 death 处概率
            loss = loss + prob_flat[pd] ** 2
            term_count += 1
        elif gd >= 0:   # gt 有 death 但 pred 没有 → 推高 birth 处概率
            loss = loss + (prob_flat[pb] - 1.0) ** 2
            term_count += 1

    # ── 多余预测分量：将 birth 像素概率压到 0（消灭多余连通块）──
    for i in range(n_match, len(pred_pairs)):
        pb, pd, _ = pred_pairs[i]
        loss = loss + prob_flat[pb] ** 2
        term_count += 1
        if pd >= 0:
            loss = loss + prob_flat[pd] ** 2
            term_count += 1

    # ── 缺失真值分量：在真值 birth 像素处推到 1（创建缺失连通块）──
    for i in range(n_match, len(gt_pairs)):
        gb, gd, _ = gt_pairs[i]
        loss = loss + (prob_flat[gb] - 1.0) ** 2
        term_count += 1
        if gd >= 0:
            loss = loss + (prob_flat[gd] - 1.0) ** 2
            term_count += 1

    # 按实际参与的像素对数归一化，使量级与 CE/Dice 同阶（避免梯度爆炸）
    return loss / max(term_count, 1)


# ──────────────────────────────────────────────────────────────────────────────
# 公开 Module
# ──────────────────────────────────────────────────────────────────────────────

class TopologyLoss(nn.Module):
    """
    0 维持续同调拓扑损失，仅作用于裂缝类。

    Parameters
    ----------
    crack_class_idx : int
        裂缝的类别索引，默认 1。
    max_pairs : int
        每张图最多参与匹配的持续对数量，用于控制计算量。
        建议范围 20~100；设得过大时细碎噪声分量会参与匹配，
        可能引入不必要的梯度噪声。
    """

    def __init__(self, crack_class_idx: int = 1, max_pairs: int = 50):
        super().__init__()
        self.crack_class_idx = crack_class_idx
        self.max_pairs = max_pairs

    def forward(
        self,
        logits: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : [B, C, H, W]
        masks  : [B, H, W] int64 类别标签
        """
        probs = F.softmax(logits, dim=1)
        crack_probs = probs[:, self.crack_class_idx]          # [B, H, W]
        crack_gt    = (masks == self.crack_class_idx).long()  # [B, H, W]

        B = logits.shape[0]
        total = logits.new_zeros(1)[0]
        for b in range(B):
            total = total + _topo_loss_single(
                crack_probs[b], crack_gt[b], self.max_pairs
            )
        return total / B
