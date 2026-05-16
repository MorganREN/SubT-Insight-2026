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

import cv2
import numpy as np
import torch
from loguru import logger


_BOUNDARY_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
_MATCH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
_SKIMAGE_SKELETONIZE = None


def _safe_skeletonize(mask: np.ndarray) -> np.ndarray:
    """优先使用 skimage skeletonize，失败时回退到 OpenCV 形态学骨架化。"""
    global _SKIMAGE_SKELETONIZE

    bin_mask = (mask > 0).astype(np.uint8)
    if bin_mask.sum() == 0:
        return bin_mask

    if _SKIMAGE_SKELETONIZE is None:
        try:
            from skimage.morphology import skeletonize as _skeletonize
            _SKIMAGE_SKELETONIZE = _skeletonize
        except Exception:
            _SKIMAGE_SKELETONIZE = False

    if _SKIMAGE_SKELETONIZE:
        return _SKIMAGE_SKELETONIZE(bin_mask.astype(bool)).astype(np.uint8)

    work = (bin_mask * 255).astype(np.uint8)
    skel = np.zeros_like(work)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(work, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(work, temp)
        skel = cv2.bitwise_or(skel, temp)
        work = eroded
        if cv2.countNonZero(work) == 0:
            break
    return (skel > 0).astype(np.uint8)


def _extract_boundary(mask: np.ndarray) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    if mask.sum() == 0:
        return mask
    eroded = cv2.erode(mask, _BOUNDARY_KERNEL, iterations=1)
    return (mask & (1 - eroded)).astype(np.uint8)


def _tolerant_overlap_stats(
    pred_map: np.ndarray,
    gt_map: np.ndarray,
) -> tuple[int, int, int, int]:
    pred_map = (pred_map > 0).astype(np.uint8)
    gt_map = (gt_map > 0).astype(np.uint8)

    pred_total = int(pred_map.sum())
    gt_total = int(gt_map.sum())
    if pred_total == 0 and gt_total == 0:
        return 0, 0, 0, 0

    gt_band = cv2.dilate(gt_map, _MATCH_KERNEL, iterations=1)
    pred_band = cv2.dilate(pred_map, _MATCH_KERNEL, iterations=1)
    pred_hit = int((pred_map & gt_band).sum())
    gt_hit = int((gt_map & pred_band).sum())
    return pred_hit, pred_total, gt_hit, gt_total


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

        self.background_idx = (
            self.class_names.index("background")
            if "background" in self.class_names else 0
        )
        self.crack_idx = (
            self.class_names.index("crack")
            if "crack" in self.class_names else None
        )
        self.foreground_indices = [
            i for i in range(num_classes) if i != self.background_idx
        ]
        self.leakage_indices = [
            i for i, name in enumerate(self.class_names)
            if name.startswith("leakage_")
        ]

        # 混淆矩阵: cm[true_label][pred_label]
        self._confusion_matrix = np.zeros(
            (num_classes, num_classes), dtype=np.int64
        )
        self.reset()

    def reset(self):
        """清空混淆矩阵，重新开始累积（通常在每个 epoch 开始时调用）。"""
        self._confusion_matrix.fill(0)
        self._image_tp = np.zeros(self.num_classes, dtype=np.int64)
        self._image_fp = np.zeros(self.num_classes, dtype=np.int64)
        self._image_fn = np.zeros(self.num_classes, dtype=np.int64)
        self._num_images = 0

        self._crack_component_gt_total = 0
        self._crack_component_gt_hit = 0
        self._crack_component_pred_total = 0
        self._crack_component_pred_hit = 0

        self._crack_boundary_pred_hit = 0
        self._crack_boundary_pred_total = 0
        self._crack_boundary_gt_hit = 0
        self._crack_boundary_gt_total = 0

        self._crack_skeleton_pred_hit = 0
        self._crack_skeleton_pred_total = 0
        self._crack_skeleton_gt_hit = 0
        self._crack_skeleton_gt_total = 0

    def _update_image_level_stats(self, pred_i: np.ndarray, target_i: np.ndarray):
        valid = target_i != self.ignore_index
        if not valid.any():
            return

        pred_valid = np.clip(pred_i[valid], 0, self.num_classes - 1)
        target_valid = target_i[valid]
        self._num_images += 1

        for class_idx in range(self.num_classes):
            gt_present = np.any(target_valid == class_idx)
            pred_present = np.any(pred_valid == class_idx)
            if gt_present and pred_present:
                self._image_tp[class_idx] += 1
            elif gt_present and not pred_present:
                self._image_fn[class_idx] += 1
            elif pred_present and not gt_present:
                self._image_fp[class_idx] += 1

        if self.crack_idx is None:
            return

        gt_crack = ((target_i == self.crack_idx) & valid).astype(np.uint8)
        pred_crack = ((pred_i == self.crack_idx) & valid).astype(np.uint8)

        if gt_crack.sum() > 0:
            n_gt, gt_labels = cv2.connectedComponents(gt_crack, connectivity=8)
            for label_id in range(1, n_gt):
                comp = gt_labels == label_id
                self._crack_component_gt_total += 1
                if pred_crack[comp].any():
                    self._crack_component_gt_hit += 1

        if pred_crack.sum() > 0:
            n_pred, pred_labels = cv2.connectedComponents(pred_crack, connectivity=8)
            for label_id in range(1, n_pred):
                comp = pred_labels == label_id
                self._crack_component_pred_total += 1
                if gt_crack[comp].any():
                    self._crack_component_pred_hit += 1

        pred_boundary = _extract_boundary(pred_crack)
        gt_boundary = _extract_boundary(gt_crack)
        pred_hit, pred_total, gt_hit, gt_total = _tolerant_overlap_stats(
            pred_boundary, gt_boundary
        )
        self._crack_boundary_pred_hit += pred_hit
        self._crack_boundary_pred_total += pred_total
        self._crack_boundary_gt_hit += gt_hit
        self._crack_boundary_gt_total += gt_total

        pred_skeleton = _safe_skeletonize(pred_crack)
        gt_skeleton = _safe_skeletonize(gt_crack)
        pred_hit, pred_total, gt_hit, gt_total = _tolerant_overlap_stats(
            pred_skeleton, gt_skeleton
        )
        self._crack_skeleton_pred_hit += pred_hit
        self._crack_skeleton_pred_total += pred_total
        self._crack_skeleton_gt_hit += gt_hit
        self._crack_skeleton_gt_total += gt_total

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
        if pred.ndim == 2:
            pred = pred[None, ...]
        if target.ndim == 2:
            target = target[None, ...]

        assert pred.shape == target.shape, (
            f"pred shape {pred.shape} 与 target shape {target.shape} 不一致"
        )

        pred = pred.astype(np.int64)
        target = target.astype(np.int64)

        for pred_i, target_i in zip(pred, target):
            self._update_image_level_stats(pred_i, target_i)

        pred = pred.ravel()
        target = target.ravel()

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
        mprecision = float(np.mean(precision[present])) if present.any() else 0.0

        fg_present = np.zeros_like(present, dtype=bool)
        fg_present[self.foreground_indices] = present[self.foreground_indices]
        miou_fg = float(np.mean(iou[fg_present])) if fg_present.any() else 0.0
        macc_fg = float(np.mean(acc[fg_present])) if fg_present.any() else 0.0
        mdice_fg = float(np.mean(dice[fg_present])) if fg_present.any() else 0.0
        mprecision_fg = float(np.mean(precision[fg_present])) if fg_present.any() else 0.0

        # ── 全局像素准确率 ──
        aacc = float(tp.sum() / max(gt_sum.sum(), 1))

        image_precision = self._image_tp / np.maximum(self._image_tp + self._image_fp, 1)
        image_recall = self._image_tp / np.maximum(self._image_tp + self._image_fn, 1)
        image_f1 = (
            2 * self._image_tp / np.maximum(2 * self._image_tp + self._image_fp + self._image_fn, 1)
        )
        image_absent = (self._image_tp + self._image_fp + self._image_fn) == 0
        for arr in (image_precision, image_recall, image_f1):
            arr[image_absent] = self.nan_to_num

        image_present = (self._image_tp + self._image_fn) > 0
        image_fg_present = np.zeros_like(image_present, dtype=bool)
        image_fg_present[self.foreground_indices] = image_present[self.foreground_indices]
        mimage_recall_fg = float(np.mean(image_recall[image_fg_present])) if image_fg_present.any() else 0.0
        mimage_f1_fg = float(np.mean(image_f1[image_fg_present])) if image_fg_present.any() else 0.0

        crack_iou = crack_dice = crack_recall = crack_precision = 0.0
        crack_image_recall = crack_image_precision = crack_image_f1 = 0.0
        if self.crack_idx is not None:
            crack_iou = float(iou[self.crack_idx])
            crack_dice = float(dice[self.crack_idx])
            crack_recall = float(acc[self.crack_idx])
            crack_precision = float(precision[self.crack_idx])
            crack_image_recall = float(image_recall[self.crack_idx])
            crack_image_precision = float(image_precision[self.crack_idx])
            crack_image_f1 = float(image_f1[self.crack_idx])

        crack_component_recall = (
            float(self._crack_component_gt_hit / max(self._crack_component_gt_total, 1))
            if self._crack_component_gt_total > 0 else 0.0
        )
        crack_component_precision = (
            float(self._crack_component_pred_hit / max(self._crack_component_pred_total, 1))
            if self._crack_component_pred_total > 0 else 0.0
        )
        crack_component_f1 = (
            2 * crack_component_precision * crack_component_recall
            / max(crack_component_precision + crack_component_recall, 1e-12)
            if (crack_component_precision + crack_component_recall) > 0 else 0.0
        )

        crack_boundary_precision = (
            float(self._crack_boundary_pred_hit / max(self._crack_boundary_pred_total, 1))
            if self._crack_boundary_pred_total > 0 else 0.0
        )
        crack_boundary_recall = (
            float(self._crack_boundary_gt_hit / max(self._crack_boundary_gt_total, 1))
            if self._crack_boundary_gt_total > 0 else 0.0
        )
        crack_boundary_f1 = (
            2 * crack_boundary_precision * crack_boundary_recall
            / max(crack_boundary_precision + crack_boundary_recall, 1e-12)
            if (crack_boundary_precision + crack_boundary_recall) > 0 else 0.0
        )

        crack_skeleton_precision = (
            float(self._crack_skeleton_pred_hit / max(self._crack_skeleton_pred_total, 1))
            if self._crack_skeleton_pred_total > 0 else 0.0
        )
        crack_skeleton_recall = (
            float(self._crack_skeleton_gt_hit / max(self._crack_skeleton_gt_total, 1))
            if self._crack_skeleton_gt_total > 0 else 0.0
        )
        crack_skeleton_f1 = (
            2 * crack_skeleton_precision * crack_skeleton_recall
            / max(crack_skeleton_precision + crack_skeleton_recall, 1e-12)
            if (crack_skeleton_precision + crack_skeleton_recall) > 0 else 0.0
        )

        leakage_miou = 0.0
        leakage_mdice = 0.0
        leakage_confusion = np.zeros((0, 0), dtype=np.float64)
        leakage_confusion_normalized = np.zeros((0, 0), dtype=np.float64)
        if self.leakage_indices:
            leakage_present = present[self.leakage_indices]
            if leakage_present.any():
                leakage_miou = float(np.mean(iou[self.leakage_indices][leakage_present]))
                leakage_mdice = float(np.mean(dice[self.leakage_indices][leakage_present]))
            leakage_confusion = cm[np.ix_(self.leakage_indices, self.leakage_indices)]
            leakage_confusion_normalized = (
                leakage_confusion
                / np.maximum(gt_sum[self.leakage_indices][:, None], 1.0)
            )

        return {
            "IoU":       iou,
            "Acc":       acc,
            "Dice":      dice,
            "Precision": precision,
            "ImagePrecision": image_precision,
            "ImageRecall": image_recall,
            "ImageF1": image_f1,
            "mIoU":      miou,
            "mAcc":      macc,
            "mDice":     mdice,
            "mPrecision": mprecision,
            "mIoU_fg":   miou_fg,
            "mAcc_fg":   macc_fg,
            "mDice_fg":  mdice_fg,
            "mPrecision_fg": mprecision_fg,
            "mImageRecall_fg": mimage_recall_fg,
            "mImageF1_fg": mimage_f1_fg,
            "aAcc":      aacc,
            "crack_iou": crack_iou,
            "crack_dice": crack_dice,
            "crack_recall": crack_recall,
            "crack_precision": crack_precision,
            "crack_image_recall": crack_image_recall,
            "crack_image_precision": crack_image_precision,
            "crack_image_f1": crack_image_f1,
            "crack_component_recall": crack_component_recall,
            "crack_component_precision": crack_component_precision,
            "crack_component_f1": crack_component_f1,
            "crack_boundary_precision": crack_boundary_precision,
            "crack_boundary_recall": crack_boundary_recall,
            "crack_boundary_f1": crack_boundary_f1,
            "crack_skeleton_precision": crack_skeleton_precision,
            "crack_skeleton_recall": crack_skeleton_recall,
            "crack_skeleton_f1": crack_skeleton_f1,
            "leakage_mIoU": leakage_miou,
            "leakage_mDice": leakage_mdice,
            "leakage_confusion": leakage_confusion,
            "leakage_confusion_normalized": leakage_confusion_normalized,
            "leakage_class_names": [self.class_names[i] for i in self.leakage_indices],
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
            f"mDice: {metrics['mDice']*100:.2f}% | "
            f"fg mIoU: {metrics['mIoU_fg']*100:.2f}% | "
            f"crack Dice: {metrics['crack_dice']*100:.2f}% | "
            f"crack R/P: {metrics['crack_recall']*100:.2f}%/{metrics['crack_precision']*100:.2f}%"
        )

    def task_report(self, metrics: Optional[Dict] = None) -> str:
        """返回更贴合当前隧道病害任务的多行摘要。"""
        if metrics is None:
            metrics = self.compute()

        fg_image_recall_parts = [
            f"{self.class_names[i]}={metrics['ImageRecall'][i]*100:.1f}%"
            for i in self.foreground_indices
        ]
        lines = [
            "任务导向评估摘要:",
            (
                f"  Overall: aAcc={metrics['aAcc']*100:.2f}%  "
                f"mIoU={metrics['mIoU']*100:.2f}%  "
                f"mDice={metrics['mDice']*100:.2f}%  "
                f"fg-mIoU={metrics['mIoU_fg']*100:.2f}%  "
                f"fg-mDice={metrics['mDice_fg']*100:.2f}%"
            ),
            (
                f"  Crack: IoU={metrics['crack_iou']*100:.2f}%  "
                f"Dice={metrics['crack_dice']*100:.2f}%  "
                f"Recall={metrics['crack_recall']*100:.2f}%  "
                f"Precision={metrics['crack_precision']*100:.2f}%  "
                f"ImgR={metrics['crack_image_recall']*100:.2f}%  "
                f"CompR={metrics['crack_component_recall']*100:.2f}%  "
                f"BoundaryF1={metrics['crack_boundary_f1']*100:.2f}%  "
                f"SkeletonF1={metrics['crack_skeleton_f1']*100:.2f}%"
            ),
            (
                f"  Leakage: mIoU={metrics['leakage_mIoU']*100:.2f}%  "
                f"mDice={metrics['leakage_mDice']*100:.2f}%"
            ),
            "  Image Recall (foreground): " + "  ".join(fg_image_recall_parts),
        ]

        leakage_names = metrics.get("leakage_class_names", [])
        leakage_conf = metrics.get("leakage_confusion_normalized")
        if len(leakage_names) > 0 and isinstance(leakage_conf, np.ndarray) and leakage_conf.size > 0:
            lines.append("  Leakage confusion (row=GT, col=Pred, normalized by GT pixels):")
            for row_name, row in zip(leakage_names, leakage_conf):
                row_str = "  ".join(
                    f"{col_name}={value*100:.1f}%"
                    for col_name, value in zip(leakage_names, row)
                )
                lines.append(f"    {row_name}: {row_str}")
        return "\n".join(lines)

    def print_task_report(self, metrics: Optional[Dict] = None):
        if metrics is None:
            metrics = self.compute()
        logger.info(self.task_report(metrics))

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
            f"{prefix}mPrecision": metrics["mPrecision"],
            f"{prefix}mIoU_fg": metrics["mIoU_fg"],
            f"{prefix}mDice_fg": metrics["mDice_fg"],
            f"{prefix}mImageRecall_fg": metrics["mImageRecall_fg"],
            f"{prefix}crack_iou": metrics["crack_iou"],
            f"{prefix}crack_dice": metrics["crack_dice"],
            f"{prefix}crack_recall": metrics["crack_recall"],
            f"{prefix}crack_precision": metrics["crack_precision"],
            f"{prefix}crack_image_recall": metrics["crack_image_recall"],
            f"{prefix}crack_component_recall": metrics["crack_component_recall"],
            f"{prefix}crack_boundary_f1": metrics["crack_boundary_f1"],
            f"{prefix}crack_skeleton_f1": metrics["crack_skeleton_f1"],
            f"{prefix}leakage_mIoU": metrics["leakage_mIoU"],
            f"{prefix}leakage_mDice": metrics["leakage_mDice"],
        }
        for i in range(self.num_classes):
            name = self.class_names[i]
            flat[f"{prefix}IoU/{name}"]  = float(metrics["IoU"][i])
            flat[f"{prefix}Acc/{name}"]  = float(metrics["Acc"][i])
            flat[f"{prefix}Dice/{name}"] = float(metrics["Dice"][i])
            flat[f"{prefix}Precision/{name}"] = float(metrics["Precision"][i])
            flat[f"{prefix}ImageRecall/{name}"] = float(metrics["ImageRecall"][i])
        return flat

    def __repr__(self) -> str:
        return (
            f"SegEvaluator(num_classes={self.num_classes}, "
            f"class_names={self.class_names})"
        )
