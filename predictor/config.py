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
