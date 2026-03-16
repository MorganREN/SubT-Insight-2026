from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InferConfig:
    ckpt: str = ""
    data_root: str = "dataset/aug_data1"
    split: str = "val"
    device: str = "auto"
    batch_size: int = 4
    num_workers: int = 2
    save_vis: bool = False
    vis_count: int = 5
    output_dir: str = "outputs/infer"
