"""
mask_to_labelme.py

把推理输出的类别索引 mask（uint8，pixel value = class id ∈ {0..6}）转成
labelme 5.x 风格 JSON。命名风格对齐 dataset/tongji 原始标注：
所有类别一律 ``shape_type="polygon"``。

用法
----
单图模式::

    conda run -n subt-2026 python data_tools/mask_to_labelme.py \\
        --mask outputs/.../C100_pred_mask.png \\
        --image dataset/tongji_data_awesome/img_dir/valid/C100.jpg \\
        --out  /tmp/C100.json

批量模式::

    conda run -n subt-2026 python data_tools/mask_to_labelme.py \\
        --mask_dir outputs/ablation_tmds_full_small/predict_dataset/valid \\
        --image_dir dataset/tongji_data_awesome/img_dir/valid \\
        --out_dir  outputs/ablation_tmds_full_small/predict_dataset_labelme/valid

集成
----
``scripts/predict_image.py`` / ``scripts/predict_dataset.py`` 设
``save_labelme=True`` 时会调用本文件的 :func:`convert`。
"""

from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import base64
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from loguru import logger
from PIL import Image


# 项目内部 snake_case 类别（dataload/dataset.py:36-44） → tongji 原始 JSON 命名
# 0 (background) 不输出 polygon；255 (ignore) 同样跳过
MASK_TO_TONGJI_LABEL: dict[int, str] = {
    1: "crack",
    2: "leakageB",
    3: "leakageW",
    4: "leakageG",
    5: "lining falling off",
    6: "segment damage",
}

LABELME_VERSION_DEFAULT = "5.4.1"


# ── 核心：mask → labelme shapes ───────────────────────────────────────────

def mask_to_shapes(
    mask: np.ndarray,
    *,
    epsilon: float = 1.0,
    label_map: dict[int, str] = MASK_TO_TONGJI_LABEL,
) -> list[dict]:
    """
    逐 class 找连通域 → ``cv2.findContours`` → ``cv2.approxPolyDP`` →
    labelme shape dict（polygon）。

    Parameters
    ----------
    mask : np.ndarray
        uint8 (H, W)，pixel value = class id。0 / 255 自动跳过。
    epsilon : float
        ``cv2.approxPolyDP`` 简化阈值（像素）。0 = 不简化。
    label_map : dict[int, str]
        class id → labelme label。未列出的 id 不产生 shape。

    Returns
    -------
    list[dict]
        每项形如::

            {"label": "crack",
             "points": [[x, y], ...],
             "group_id": None,
             "shape_type": "polygon",
             "flags": {}}

        点数 < 3 的 contour 自动丢弃（labelme polygon 至少 3 点）。
    """
    if mask.ndim != 2:
        raise ValueError(f"mask 必须是 2D，实际 shape={mask.shape}")
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    shapes: list[dict] = []
    for class_id, label in label_map.items():
        binary = (mask == class_id).astype(np.uint8)
        if not binary.any():
            continue
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        for contour in contours:
            if epsilon > 0:
                contour = cv2.approxPolyDP(contour, epsilon=epsilon, closed=True)
            pts = contour.reshape(-1, 2)
            if len(pts) < 3:
                continue
            shapes.append({
                "label": label,
                "points": [[float(x), float(y)] for x, y in pts],
                "group_id": None,
                "shape_type": "polygon",
                "flags": {},
            })
    return shapes


def encode_image_b64(image_path: Path) -> str:
    """读取图片文件，base64 编码（用于 labelme imageData 字段）。"""
    return base64.b64encode(Path(image_path).read_bytes()).decode("ascii")


def write_labelme_json(
    out_path: Path,
    *,
    image_path: str,
    image_h: int,
    image_w: int,
    shapes: list[dict],
    image_data_b64: str | None = None,
    version: str = LABELME_VERSION_DEFAULT,
) -> None:
    """写出 labelme 5.x 风格 JSON（兼容老版 labelme 打开）。"""
    payload = {
        "version": version,
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path,
        "imageData": image_data_b64,
        "imageHeight": int(image_h),
        "imageWidth": int(image_w),
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def convert(
    mask: np.ndarray | str | Path,
    image_path: str | Path,
    out_json_path: str | Path,
    *,
    epsilon: float = 1.0,
    embed_image_data: bool = False,
    label_map: dict[int, str] = MASK_TO_TONGJI_LABEL,
) -> dict:
    """
    单图便利函数：(mask, image_path) → labelme JSON。

    Parameters
    ----------
    mask : np.ndarray | path-like
        若为 ndarray，直接使用；否则按 PIL ``mode='L'`` 读取。
    image_path : path-like
        原图路径。仅用来：
        - 读取 H/W（若 mask 已是 ndarray，直接用 mask 的 shape）
        - 取 imagePath 字段（写为相对于 JSON 所在目录的相对路径，便于 labelme 打开）
        - 若 ``embed_image_data=True``，base64 嵌入

    Returns
    -------
    dict
        ``{"n_shapes": int, "per_class_count": {label: int, ...}}``。
    """
    image_path = Path(image_path)
    out_json_path = Path(out_json_path)

    if isinstance(mask, np.ndarray):
        mask_arr = mask
    else:
        mask_arr = np.array(Image.open(mask).convert("L"), dtype=np.uint8)

    h, w = mask_arr.shape[:2]

    shapes = mask_to_shapes(mask_arr, epsilon=epsilon, label_map=label_map)

    # imagePath：相对于 JSON 所在目录的相对路径，labelme 打开 JSON 时能定位原图。
    # 若 embed_image_data=True 则同时把图片 base64 进 JSON，labelme 不依赖外部图片也能渲染。
    try:
        rel_image = _os.path.relpath(image_path, out_json_path.parent)
    except ValueError:
        rel_image = image_path.name

    image_data_b64 = encode_image_b64(image_path) if embed_image_data else None

    write_labelme_json(
        out_json_path,
        image_path=rel_image,
        image_h=h,
        image_w=w,
        shapes=shapes,
        image_data_b64=image_data_b64,
    )

    per_class_count: dict[str, int] = {}
    for s in shapes:
        per_class_count[s["label"]] = per_class_count.get(s["label"], 0) + 1
    return {"n_shapes": len(shapes), "per_class_count": per_class_count}


# ── 批量模式 ───────────────────────────────────────────────────────────────

def _strip_known_decorations(stem: str, mask_suffix: str) -> str:
    """
    去掉 mask 文件名上的修饰前/后缀，还原成原图 stem。

    支持：
    - 必须先去掉 ``mask_suffix`` 末尾（默认 '_pred_mask.png' 的 '_pred_mask' 部分）
    - predict_dataset.py 加的 ``rank{N}_`` 前缀与 ``_iou{F}`` 后缀（出现时一并剥掉）
    """
    suffix_no_ext = Path(mask_suffix).stem if "." in mask_suffix else mask_suffix
    if stem.endswith(suffix_no_ext):
        stem = stem[: -len(suffix_no_ext)]
    # rank001_ ... rank999_
    import re as _re
    stem = _re.sub(r"^rank\d{3,}_", "", stem)
    # _iou12.3 / _iou82.0 等浮点尾巴
    stem = _re.sub(r"_iou\d+(?:\.\d+)?$", "", stem)
    return stem


def convert_dir(
    mask_dir: Path,
    image_dir: Path,
    out_dir: Path,
    *,
    mask_suffix: str = "_pred_mask.png",
    image_suffixes: Iterable[str] = (".jpg", ".png", ".jpeg"),
    epsilon: float = 1.0,
    embed_image_data: bool = False,
) -> dict:
    """批量扫描 ``mask_dir/*{mask_suffix}``，与 ``image_dir`` 同 stem 配对后转 JSON。"""
    mask_dir = Path(mask_dir)
    image_dir = Path(image_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pattern = f"*{mask_suffix}"
    mask_paths = sorted(mask_dir.glob(pattern))
    if not mask_paths:
        logger.warning(f"未找到匹配 mask: {mask_dir / pattern}")
        return {"converted": 0, "missing_image": 0}

    converted = 0
    missing_image = 0
    for mp in mask_paths:
        clean_stem = _strip_known_decorations(mp.stem, mask_suffix)
        image_path: Path | None = None
        for sfx in image_suffixes:
            cand = image_dir / f"{clean_stem}{sfx}"
            if cand.exists():
                image_path = cand
                break
        if image_path is None:
            logger.warning(f"找不到对应原图，跳过: stem={clean_stem}")
            missing_image += 1
            continue
        out_json = out_dir / f"{clean_stem}.json"
        info = convert(
            mp, image_path, out_json,
            epsilon=epsilon, embed_image_data=embed_image_data,
        )
        logger.info(
            f"{mp.name:<60s} → {out_json.name}  "
            f"shapes={info['n_shapes']}  per_class={info['per_class_count']}"
        )
        converted += 1

    logger.success(f"批量转换完成: 成功 {converted}，缺原图 {missing_image}")
    return {"converted": converted, "missing_image": missing_image}


# ── CLI ────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="把推理输出的类别索引 mask 转成 labelme polygon JSON。"
    )
    parser.add_argument("--mask",       type=Path, default=None,
                        help="单图模式：mask PNG 路径")
    parser.add_argument("--image",      type=Path, default=None,
                        help="单图模式：原图路径")
    parser.add_argument("--out",        type=Path, default=None,
                        help="单图模式：输出 JSON 路径")

    parser.add_argument("--mask_dir",   type=Path, default=None,
                        help="批量模式：mask 目录")
    parser.add_argument("--image_dir",  type=Path, default=None,
                        help="批量模式：原图目录")
    parser.add_argument("--out_dir",    type=Path, default=None,
                        help="批量模式：输出 JSON 目录")
    parser.add_argument("--mask_suffix", type=str, default="_pred_mask.png",
                        help="批量模式：mask 文件后缀（默认 '_pred_mask.png'）")

    parser.add_argument("--epsilon",    type=float, default=1.0,
                        help="cv2.approxPolyDP 简化阈值（像素）；0 = 不简化")
    parser.add_argument("--embed_image_data", action="store_true",
                        help="把原图 base64 嵌入 JSON 的 imageData 字段")
    return parser


def main():
    args = _build_parser().parse_args()

    single_mode = args.mask and args.image and args.out
    batch_mode  = args.mask_dir and args.image_dir and args.out_dir

    if single_mode and batch_mode:
        raise SystemExit("不能同时指定单图模式与批量模式参数")
    if not (single_mode or batch_mode):
        raise SystemExit(
            "需指定 --mask/--image/--out（单图）或 --mask_dir/--image_dir/--out_dir（批量）"
        )

    if single_mode:
        info = convert(
            args.mask, args.image, args.out,
            epsilon=args.epsilon, embed_image_data=args.embed_image_data,
        )
        logger.success(
            f"已写出 {args.out}  shapes={info['n_shapes']}  "
            f"per_class={info['per_class_count']}"
        )
    else:
        convert_dir(
            args.mask_dir, args.image_dir, args.out_dir,
            mask_suffix=args.mask_suffix,
            epsilon=args.epsilon, embed_image_data=args.embed_image_data,
        )


if __name__ == "__main__":
    main()
