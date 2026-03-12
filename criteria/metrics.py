"""
criteria/metrics.py
语义分割评估指标。

核心类
------
    SegEvaluator
        内部维护混淆矩阵，逐 batch 累积预测结果，
        一次性计算 IoU / Accuracy / Dice / Precision 等指标。

支持的指标
----------
    Per-class IoU       (Intersection over Union)
    Per-class Accuracy  (Recall / Sensitivity)
    Per-class Dice      (F1 Score)
    Per-class Precision
    mIoU                (mean IoU，仅对 gt 出现的类别均值)
    mAcc                (mean Accuracy)
    mDice               (mean Dice)
    aAcc                (all-pixel Accuracy，全局像素准确率)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from loguru import logger


class SegEvaluator:
    """基于混淆矩阵的语义分割评估器。"""

    def __init__(
        self,
        num_classes: int,
        class_names: Optional[Tuple[str, ...]] = None,
        ignore_index: int = 255,
        nan_to_num: float = 0.0,
    ):
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.nan_to_num   = nan_to_num

        if class_names is not None:
            assert len(class_names) == num_classes, (
                f"class_names 长度 ({len(class_names)}) != num_classes ({num_classes})"
            )
            self.class_names = list(class_names)
        else:
            self.class_names = [str(i) for i in range(num_classes)]

        # 混淆矩阵: cm[true_label][pred_label]
        self._confusion_matrix = np.zeros(
            (num_classes, num_classes), dtype=np.int64
        )

    def reset(self):
        """清空混淆矩阵，重新开始累积（通常在每个 epoch 开始时调用）。"""
        self._confusion_matrix.fill(0)

    def update(
        self,
        pred: Union[torch.Tensor, np.ndarray],
        target: Union[torch.Tensor, np.ndarray],
    ):
        """
        用一个 batch 的预测和标签更新混淆矩阵（可调用多次）。

        Parameters
        ----------
        pred : Tensor or ndarray
            可以是:
            - logits  (B, C, H, W) → 自动取 argmax 作为预测类别
            - 类别 ID (B, H, W)    → 直接使用
        target : Tensor or ndarray
            真实标签 (B, H, W)，像素值 = 类别 ID。
            等于 ignore_index 的像素会被跳过。
        """
        # ── 转 numpy ──
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()

        # ── logits (B, C, H, W) → argmax → (B, H, W) ──
        if pred.ndim == 4:
            pred = pred.argmax(axis=1)

        pred   = pred.astype(np.int64).ravel()
        target = target.astype(np.int64).ravel()

        # ── 过滤 ignore_index ──
        valid  = target != self.ignore_index
        pred   = pred[valid]
        target = target[valid]

        if pred.size == 0:
            return  # 该 batch 全为 ignore，跳过

        # ── 防御：预测超出范围时 clip 到合法区间 ──
        pred = np.clip(pred, 0, self.num_classes - 1)

        # ── 通过 bincount 高效累积混淆矩阵 ──
        indices = self.num_classes * target + pred
        cm = np.bincount(indices, minlength=self.num_classes ** 2)
        self._confusion_matrix += cm.reshape(self.num_classes, self.num_classes)

    @property
    def confusion_matrix(self) -> np.ndarray:
        """返回当前混淆矩阵的副本，shape=(num_classes, num_classes)。"""
        return self._confusion_matrix.copy()

    def compute(self) -> Dict[str, Union[float, np.ndarray]]:
        """
        根据已累积的混淆矩阵计算所有评估指标。

        Returns
        -------
        dict with keys:
            "IoU"       : ndarray (num_classes,) — 每类 IoU
            "Acc"       : ndarray (num_classes,) — 每类 Accuracy (Recall)
            "Dice"      : ndarray (num_classes,) — 每类 Dice (F1)
            "Precision" : ndarray (num_classes,) — 每类 Precision
            "mIoU"      : float   — 均值 IoU（仅对 gt 中出现的类别）
            "mAcc"      : float   — 均值 Accuracy
            "mDice"     : float   — 均值 Dice
            "aAcc"      : float   — 全局像素准确率 TP_all / Pixels_all
        """
        cm = self._confusion_matrix.astype(np.float64)

        tp       = np.diag(cm)                        # (C,) 各类 True Positive
        gt_sum   = cm.sum(axis=1)                     # (C,) 各类真实像素总数（行和）
        pred_sum = cm.sum(axis=0)                     # (C,) 各类预测像素总数（列和）
        union    = gt_sum + pred_sum - tp             # (C,) 并集

        # ── per-class IoU: TP / (TP + FP + FN) ──
        iou = tp / np.maximum(union, 1)

        # ── per-class Accuracy (Recall): TP / (TP + FN) ──
        acc = tp / np.maximum(gt_sum, 1)

        # ── per-class Precision: TP / (TP + FP) ──
        precision = tp / np.maximum(pred_sum, 1)

        # ── per-class Dice (F1): 2*TP / (2*TP + FP + FN) ──
        dice = 2 * tp / np.maximum(gt_sum + pred_sum, 1)

        # ── 处理 gt 和 pred 中都未出现的类别 ──
        absent = (gt_sum == 0) & (pred_sum == 0)
        for arr in (iou, acc, precision, dice):
            arr[absent] = self.nan_to_num

        # ── 均值：仅对 gt 中出现过的类别取均 ──
        present = gt_sum > 0
        miou  = float(np.mean(iou[present]))  if present.any() else 0.0
        macc  = float(np.mean(acc[present]))  if present.any() else 0.0
        mdice = float(np.mean(dice[present])) if present.any() else 0.0

        # ── 全局像素准确率 ──
        aacc = float(tp.sum() / max(gt_sum.sum(), 1))

        return {
            "IoU":       iou,
            "Acc":       acc,
            "Dice":      dice,
            "Precision": precision,
            "mIoU":      miou,
            "mAcc":      macc,
            "mDice":     mdice,
            "aAcc":      aacc,
        }

    def print_table(self, metrics: Optional[Dict] = None):
        """
        以表格形式打印各类指标（风格仿 mmseg 评估输出）。

        Parameters
        ----------
        metrics : dict, optional
            compute() 的返回值。传 None 则自动调用 compute()。
        """
        if metrics is None:
            metrics = self.compute()

        iou  = metrics["IoU"]
        acc  = metrics["Acc"]
        dice = metrics["Dice"]
        prec = metrics["Precision"]

        sep    = "+" + "-" * 14 + "+" + "-" * 10 + "+" + "-" * 10 + "+" + "-" * 10 + "+" + "-" * 10 + "+"
        header = f"| {'Class':^12s} | {'IoU':^8s} | {'Acc':^8s} | {'Dice':^8s} | {'Prec':^8s} |"

        lines = [sep, header, sep]
        for i in range(self.num_classes):
            name = self.class_names[i]
            lines.append(
                f"| {name:>12s} | {iou[i]*100:7.2f}% | {acc[i]*100:7.2f}% "
                f"| {dice[i]*100:7.2f}% | {prec[i]*100:7.2f}% |"
            )
        lines += [
            sep,
            f"| {'mIoU':>12s} | {metrics['mIoU']*100:7.2f}% | "
            f"{metrics['mAcc']*100:7.2f}% | {metrics['mDice']*100:7.2f}% | {'':>8s} |",
            f"| {'aAcc':>12s} | {metrics['aAcc']*100:7.2f}% | "
            f"{'':>8s} | {'':>8s} | {'':>8s} |",
            sep,
        ]
        logger.info(f"评估结果:\n" + "\n".join(lines))

    def summary(self, metrics: Optional[Dict] = None) -> str:
        """
        返回一行摘要字符串，适合写入 tqdm 进度条或 loguru 日志。

        示例输出: "aAcc: 97.53% | mIoU: 62.18% | mAcc: 71.04% | mDice: 72.31%"
        """
        if metrics is None:
            metrics = self.compute()
        return (
            f"aAcc: {metrics['aAcc']*100:.2f}% | "
            f"mIoU: {metrics['mIoU']*100:.2f}% | "
            f"mAcc: {metrics['mAcc']*100:.2f}% | "
            f"mDice: {metrics['mDice']*100:.2f}%"
        )

    def to_dict(
        self,
        metrics: Optional[Dict] = None,
        prefix: str = "val/",
    ) -> Dict[str, float]:
        """
        将指标展平为 ``{prefix}metric: value`` 字典。

        适合直接传给 ``wandb.log()`` 或 ``SummaryWriter.add_scalars()``。

        Parameters
        ----------
        metrics : dict, optional
            compute() 的返回值，None 时自动计算。
        prefix : str
            键名前缀，默认 ``"val/"``。

        Returns
        -------
        dict[str, float]
        """
        if metrics is None:
            metrics = self.compute()

        flat: Dict[str, float] = {
            f"{prefix}aAcc":  metrics["aAcc"],
            f"{prefix}mIoU":  metrics["mIoU"],
            f"{prefix}mAcc":  metrics["mAcc"],
            f"{prefix}mDice": metrics["mDice"],
        }
        for i in range(self.num_classes):
            name = self.class_names[i]
            flat[f"{prefix}IoU/{name}"]  = float(metrics["IoU"][i])
            flat[f"{prefix}Acc/{name}"]  = float(metrics["Acc"][i])
            flat[f"{prefix}Dice/{name}"] = float(metrics["Dice"][i])
        return flat

    def __repr__(self) -> str:
        return (
            f"SegEvaluator(num_classes={self.num_classes}, "
            f"class_names={self.class_names})"
        )
