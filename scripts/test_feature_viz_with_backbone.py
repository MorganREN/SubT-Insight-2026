import argparse
import os
import sys

from loguru import logger

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.backbones import DINOv3ConvNeXt
from utils.feature_viz import (
    find_visualization_images,
    get_dinov3_convnext_features,
    load_image,
    visualize_pca,
)


def parse_args():
    parser = argparse.ArgumentParser(description="测试 DINOv3 Backbone PCA 可视化脚本")
    parser.add_argument(
        "--weight-path",
        type=str,
        default="dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth",
        help="DINOv3 权重路径",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default="dataset/aug_data_extra/img_dir/train/DG100010000_256.jpg",
        help="测试图像路径",
    )
    parser.add_argument("--stage-idx", type=int, default=-1, help="可视化的 stage index")
    parser.add_argument(
        "--save-path",
        type=str,
        default="outputs/feature_pca_compare.png",
        help="输出图保存路径",
    )
    parser.add_argument(
        "--frozen-stages",
        type=int,
        default=0,
        help="冻结的 stage 数量 (-1=全部冻结, 0=不冻结)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="仅保存图片，不弹出窗口",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("开始加载模型与图像...")

    # 使用 DINOv3ConvNeXt 类而不是 load_dinov3_convnext_tiny 函数
    model = DINOv3ConvNeXt(weight_path=args.weight_path, frozen_stages=args.frozen_stages)
    model.eval()

    orig_img, img_tensor = load_image(args.image_path)
    vis_imgs = find_visualization_images(args.image_path)

    features = get_dinov3_convnext_features(model, img_tensor, stage_idx=args.stage_idx)
    logger.success(f"特征提取成功: {features.shape}")

    visualize_pca(
        features,
        original_img=orig_img,
        mask_overlay_img=vis_imgs["mask_overlay"],
        segmented_img=vis_imgs["segmented"],
        save_path=args.save_path,
        show=not args.no_show,
    )
    logger.success(f"可视化完成，输出: {args.save_path}")


if __name__ == "__main__":
    main()
