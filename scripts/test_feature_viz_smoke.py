import os
import sys

import torch
from PIL import Image
from loguru import logger

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.feature_viz import visualize_pca


def main():
    logger.info("运行 PCA 可视化冒烟测试（不依赖 backbone）")

    dummy_features = torch.randn(1, 768, 16, 16)
    dummy_original = Image.new("RGB", (512, 512), color=(45, 45, 45))

    output_path = "outputs/feature_pca_smoke.png"
    visualize_pca(
        dummy_features,
        original_img=dummy_original,
        mask_overlay_img=None,
        segmented_img=None,
        save_path=output_path,
        show=False,
    )

    if not os.path.exists(output_path):
        raise FileNotFoundError(f"输出文件未生成: {output_path}")

    logger.success(f"冒烟测试通过，输出文件: {output_path}")


if __name__ == "__main__":
    main()
