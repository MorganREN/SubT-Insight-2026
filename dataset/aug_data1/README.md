# aug_data1 数据集说明

隧道衬砌缺陷语义分割数据集，由 `coco2mask.py` 从 COCO 格式标注转换而来，按 70/15/15 比例随机划分（seed=42）。

---

## 基本信息

| 项目 | 值 |
|---|---|
| 图像尺寸 | 640 × 640 |
| 图像格式 | JPEG（RGB） |
| 掩码格式 | PNG（灰度，单通道） |
| 总样本数 | 324 |
| 总占用空间 | ~241 MB |

## 数据划分

| 划分 | 图像数 | 掩码数 |
|---|---|---|
| train | 226 | 226 |
| valid | 48 | 48 |
| test | 50 | 50 |

## 类别映射关系

掩码中每个像素的值对应一个类别：

| 像素值 | 类别 | 可视化颜色 (BGR) |
|---|---|---|
| 0 | background（背景） | (0, 0, 0) 黑色 |
| 1 | cracks-MKjm（裂缝） | (0, 0, 255) 红色 |
| 2 | cracks-MKjm（裂缝） | (0, 255, 0) 绿色 |
| 3 | leakage（渗漏） | (255, 0, 0) 蓝色 |
| 4 | spalling（剥落） | (0, 255, 255) 黄色 |


## 目录结构

```
aug_data1/                  # ~241 MB
├── README.md
├── img_dir/                # 原始图像 (~18 MB)
│   ├── train/              # 226 张 .jpg
│   ├── valid/              # 48 张 .jpg
│   └── test/               # 50 张 .jpg
├── ann_dir/                # 语义分割掩码 (~1.7 MB)
│   ├── train/              # 226 张 .png
│   ├── valid/              # 48 张 .png
│   └── test/               # 50 张 .png
└── visualization/          # 可视化结果 (~222 MB)
    ├── train/
    │   ├── mask_overlay/   # 226 张 掩码叠加图 (.png, RGB)
    │   └── segmented/      # 226 张 分割抠图 (.png, RGB)
    ├── valid/
    │   ├── mask_overlay/   # 48 张
    │   └── segmented/      # 48 张
    └── test/
        ├── mask_overlay/   # 50 张
        └── segmented/      # 50 张
```

## 文件命名规则

图像与掩码通过文件名一一对应（仅扩展名不同）：

- 图像: `{name}.jpg`（在 `img_dir/` 下）
- 掩码: `{name}.png`（在 `ann_dir/` 下）
- 可视化: `{name}.png`（在 `visualization/` 下）

示例: `1_1_033_30_jpg.rf.1d28c146e1ea8247eaa8acf9cfffaa2c`

## 生成方式

由项目根目录下的 `coco2mask.py` 脚本生成，该脚本读取 COCO JSON 标注文件并：
1. 将多边形标注转换为像素级语义分割掩码
2. 按 70/15/15 比例随机划分 train/valid/test
3. 生成 mask_overlay（掩码叠加到原图）和 segmented（按类别分割抠图）可视化结果