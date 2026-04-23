from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InferConfig:
    ckpt: str = ""
    data_root: str = "dataset/tongji_data"
    split: str = "val"
    device: str = "auto"
    batch_size: int = 4
    num_workers: int = 2
    save_vis: bool = False
    vis_count: int = 5
    output_dir: str = "outputs/infer"
    use_tiling: bool = False  # True = 原图 tiling 推理（用于 tongji_data_raw 等原始分辨率数据集）
    use_tta: bool = False     # 仅在 use_tiling=True 时生效
