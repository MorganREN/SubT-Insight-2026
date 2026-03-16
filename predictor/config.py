from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PredictConfig:
    image: str = "dataset/aug_data1/img_dir/train/1_1_036_31_jpg.rf.a33c41b4952f8685c269f91f8f3d47fe.jpg"
    mask: str = ""
    ckpt: str = ""
    device: str = "auto"
    output_dir: str = "outputs/predict"
    input_size: int | None = None
