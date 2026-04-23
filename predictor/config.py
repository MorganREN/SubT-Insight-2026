from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PredictConfig:
    image: str = "dataset/tongji_data/img_dir/train/C0001.jpg"
    mask: str = ""
    ckpt: str = ""
    device: str = "auto"
    output_dir: str = "outputs/predict"
    input_size: int | None = None
    use_tiling: bool = True   # True = 按群落参数做滑动窗口推理（推荐用于原图）
    use_tta: bool = False     # 仅在 use_tiling=True 时生效
