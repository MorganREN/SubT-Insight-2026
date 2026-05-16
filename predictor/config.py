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

    # ── labelme JSON 输出（data_tools/mask_to_labelme.py） ──
    save_labelme: bool = False             # True = 推理后同步导出 {stem}.json
    labelme_epsilon: float = 1.0           # cv2.approxPolyDP 简化阈值；0 = 不简化
    labelme_embed_image: bool = False      # True = 把原图 base64 嵌入 imageData 字段
