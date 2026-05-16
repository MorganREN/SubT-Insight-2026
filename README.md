# SubT-Insight-2026

Subterranean Insight for 2026 — 复杂环境下隧道衬砌智能多缺陷检测

## 目录

- [项目简介](#项目简介)
- [环境要求](#环境要求)
- [部署方式](#部署方式)
- [预训练权重](#预训练权重)
- [项目结构](#项目结构)
- [数据集](#数据集)
  - [类别定义](#类别定义与映射tongji)
  - [数据准备](#数据准备)
- [模型架构](#模型架构)
- [训练](#训练)
- [评估](#评估)
- [推理](#推理)
  - [推理结果转 labelme 标注](#推理结果转-labelme-标注可选)
- [进阶：模型量化](#进阶模型量化)
- [常用命令](#常用命令)

---

## 项目简介

本项目基于深度学习语义分割技术，实现对隧道衬砌在复杂环境下多种缺陷（裂缝、渗漏、衬砌脱落、管片损伤）的智能检测。

主推模型为 **TMDS（Topology-aware Morphological Decoupled Segmentation）**，通过形态感知的双流解码架构，将线型缺陷（裂缝）与面型缺陷（渗漏等）解耦处理。架构细节见 [TMDS_report.md](TMDS_report.md)。

---

## 环境要求

- Python 3.10+、PyTorch 2.2.0
- GPU 部署需要 CUDA 12.1
- 详见 `requirements.txt`

---

## 部署方式

### 方式一：Dev Container 部署（推荐）

项目预置了三套 Dev Container 配置，适配不同硬件环境，开箱即用。

#### 前置条件

1. 安装 [Docker Desktop](https://www.docker.com/products/docker-desktop/)（Windows/Mac）或 Docker Engine（Linux）
2. 安装 [Visual Studio Code](https://code.visualstudio.com/)
3. 在 VS Code 中安装 [Dev Containers 扩展](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

#### 选择适合你的配置

| 配置名称 | 路径 | 适用场景 | 基础镜像 |
|---|---|---|---|
| **NVIDIA GPU** | `.devcontainer/cuda-gpu/` | 有 NVIDIA 显卡的 Linux/Windows 主机 | `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime` |
| **Windows/Linux CPU** | `.devcontainer/windows-cpu/` | 无显卡的 Windows 或 Linux 主机 | `mcr.microsoft.com/devcontainers/python:3.10` |
| **Mac ARM** | `.devcontainer/mac-arm/` | Apple Silicon (M1/M2/M3) Mac | `mcr.microsoft.com/devcontainers/python:3.10` |

#### 部署步骤

```bash
git clone <仓库地址>
cd SubT-Insight-2026
code .
```

VS Code 打开后选择 `Reopen in Container`，根据硬件选择对应配置目录。

### 方式二：本地手动部署

```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 安装 PyTorch（根据硬件选择一条执行）
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121   # NVIDIA GPU
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cpu    # CPU
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0                                                      # Mac ARM

# 安装项目依赖
pip install -r requirements.txt
```

### 验证

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

---

## 预训练权重

模型骨干使用 DINOv3 自监督预训练权重，需提前下载并放置于项目根目录：

| 文件名 | 骨干 | 对应配置 |
|---|---|---|
| `dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth` | DINOv3 ConvNeXt-Tiny | `backbone_type="convnext_tiny"`（默认） |
| `dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth` | DINOv3 ViT-S+/16 | `backbone_type="vit_s16plus"` |

两种骨干均输出 4 级多尺度特征（通道数 96 / 192 / 384 / 768），可与所有解码头搭配使用。

---

## 项目结构

```
SubT-Insight-2026/
├── train.py                      # 训练入口
├── infer.py                      # 数据集评估入口
├── predict_image.py              # 单图推理入口
├── predict_dataset.py            # 批量推理入口
├── quantize.py                   # 模型量化入口（PTQ）
│
├── dataset_convert_awesome.py    # 数据集转换（分块 tiling，主推）
├── dataset_convert_raw.py        # 数据集转换（保留原始分辨率，用于推理评估）
├── filter_background_patches.py  # tiling 后背景 patch 筛选
├── analyze_dataset_resolution.py # 数据集分辨率统计
├── merge_splits.py               # split 合并工具
├── visualize_annotations.py      # 标注可视化
│
├── trainer/                      # 训练主流程（循环、验证、checkpoint、三阶段调度）
├── inference/                    # 数据集评估流程
├── predictor/                    # 单图预测、tiling 推理、可视化
├── models/                       # 骨干网络、解码头、分割器封装
├── criteria/                     # 损失函数与分割指标
├── dataload/                     # Dataset、数据增强、DataLoader 工厂
├── utils/                        # checkpoint 加载、优化器、调度器、量化、可视化工具
└── tools/                        # 辅助工具脚本
```

---

## 数据集

### 类别定义与映射（Tongji）

当前统一为 **1 个背景 + 6 类病害**（共 7 类，mask 像素值 `0..6`）：

| ID | 类别名 | 说明 |
|---|---|---|
| 0 | `background` | 背景 |
| 1 | `crack` | 裂缝 |
| 2 | `leakage_b` | 渗漏 B |
| 3 | `leakage_w` | 渗漏 W |
| 4 | `leakage_g` | 渗漏 G |
| 5 | `lining_falling_off` | 衬砌脱落 |
| 6 | `segment_damage` | 管片损伤 |

### 数据准备

项目使用两套数据集，服务于不同阶段：

| 数据集 | 脚本 | 用途 |
|---|---|---|
| `dataset/tongji_data_awesome/` | `dataset_convert_awesome.py` | **训练**（自适应 tiling，图像尺寸统一为 512×512） |
| `dataset/tongji_data_raw/` | `dataset_convert_raw.py` | **推理评估**（保留原始分辨率，与 tiling 推理配合使用） |

#### 训练数据（tongji_data_awesome）

```bash
# Step 1：将原始 Tongji 数据集转换为 tiling patch 格式
python dataset_convert_awesome.py

# Step 2：筛除冗余背景 patch（可选，降低数据集噪声）
python filter_background_patches.py
```

转换流程：图像按分辨率聚类（Tiny / Mobile / DSLR / HighRes）→ 来源级 80/20 分层划分 →
自适应滑动窗口 tiling（不同群组使用不同 patch_size 和 stride，镜像填充）→ 输出
`img_dir/{train,valid}/` + `ann_dir/{train,valid}/` 目录结构。

#### 评估数据（tongji_data_raw）

```bash
# 从原始数据集划分出与 awesome 一致的 train/valid，不做 tiling
python dataset_convert_raw.py
```

原始分辨率图像，划分规则与 `dataset_convert_awesome.py` 完全对齐（相同的来源分层策略）。
用于 tiling 推理时将预测结果与 GT 在原始尺寸下直接对比。

---

## 模型架构

### TunnelSegmentor（标准）

单路骨干 + 单解码头架构，适合快速实验和基线对比。

- **骨干**：DINOv3 ConvNeXt-Tiny 或 DINOv3 ViT-S+/16
- **解码头**：
  - `UPerHead`（`head_type="uper"`）：PPM + FPN 金字塔，精度高
  - `MLPHead`（`head_type="mlp"`）：轻量 MLP 融合，参数少
- **解码头通道数参考**（`head_channels`）：128 ≈ 0.6M / 160 ≈ 0.9M / 256 ≈ 2.2M

### TMDSSegmentor（主推）

双流解耦架构，专为隧道缺陷的形态异质性设计。详细原理见 [TMDS_report.md](TMDS_report.md)。

- **MRM**（Morphological Routing Module）：软路由骨干特征至线型流与面型流
- **线型流**（DSADecoder）：Deformable Strip Attention，捕获裂缝长程连续性
- **面型流**（ArealDecoder）：FPN + PPM，捕获渗漏等块状区域
- **CMIM**（Cross-Morphology Interaction Module）：双流双向上下文交互
- 训练时输出多分支 dict（主输出 + 两路辅助输出），推理时仅返回主输出张量

---

## 训练

修改 `train.py` 中对应的配置后直接运行：

```bash
python train.py
```

`train.py` 内置两套配置：

| 配置变量 | 模型 | 说明 |
|---|---|---|
| `RUN` | TunnelSegmentor | 标准单流模型，适合基线 |
| `TMDS_RUN` | TMDSSegmentor | 双流主推模型，切换方式：`main(TMDS_RUN)` |

### 三阶段渐进解冻训练

两套模型均支持三阶段训练，通过逐步解冻骨干避免破坏预训练表征：

| 阶段 | 骨干状态 | 说明 |
|---|---|---|
| Stage 1 | 全冻结（`frozen_stages=-1`） | 解码头热身，高 LR 安全收敛 |
| Stage 2 | 深层解冻（`frozen_stages=1`） | 骨干深层特征向任务适配 |
| Stage 3 | 全解冻（`frozen_stages=0`） | 全网络低 LR 精细联合优化 |

标准模型启用三阶段：配置 `use_stages=True`；TMDS 模型始终使用三阶段（由 `use_tmds=True` 自动触发）。

各阶段的 epoch 数、学习率、损失函数均可独立配置（`stage_epochs`、`stage_base_lrs`、`stage_loss_names`）。
每阶段均有独立的 cosine 退火调度。

### 损失函数

支持 7 种组合（`loss_name` / `stage_loss_names` 字段）：

| 配置值 | 组成 | 推荐场景 |
|---|---|---|
| `"ce"` | Cross-Entropy | 基线 |
| `"dice"` | Dice | 类别不均衡 |
| `"focal"` | Focal | 困难样本增强 |
| `"ce+dice"` | CE + Dice | 均衡精度 |
| `"ce+focal"` | CE + Focal | — |
| `"dice+focal"` | Dice + Focal（2:1） | **推荐**，类别极度不均衡时 |
| `"ce+dice+focal"` | 三者组合 | 全量监督 |

开启 `use_class_weights=True` 可自动计算像素频率类别权重传给 CE / Dice。

### Checkpoint

- `outputs/<run_name>/best.pth`：最优验证集 mIoU 对应权重
- `outputs/<run_name>/last.pth`：最近一次 epoch 权重（用于断点续训：`resume="outputs/.../last.pth"`）

---

## 评估

```bash
python infer.py
```

`infer.py` 内置三套配置：

| 配置变量 | 数据集 | 推理方式 | 说明 |
|---|---|---|---|
| `RUN` | tongji_data_awesome | 标准 batch | 标准模型评估 |
| `TMDS_RUN` | tongji_data_awesome | 标准 batch | TMDS 模型评估 |
| `RAW_RUN` | tongji_data_raw | **Tiling** | 原始分辨率下的评估（激活方式：`main(RAW_RUN)`） |

**Tiling 推理**（`use_tiling=True`）：对原始分辨率图像按图像群组参数做高斯加权滑动窗口推理，预测结果与 GT 均在原图尺寸下比较，无信息损失。

输出：mIoU / per-class IoU / pixel Accuracy，结果保存至 `output_dir/metrics.json`。

---

## 推理

### 单图推理

```bash
python predict_image.py
```

在 `predict_image.py` 的 `RUN` 中配置图片路径和 checkpoint，支持两种推理模式：

| 参数 | 说明 |
|---|---|
| `use_tiling=False` | 标准推理（图片 resize 到 input_size） |
| `use_tiling=True` | Tiling 推理（保留原始分辨率，高斯加权拼合） |

提供 GT mask 时（`mask=` 参数）自动计算单图 mIoU / Accuracy；
不提供时仅输出彩色预测叠加图。结尾日志包含图片分辨率、磁盘大小、推理耗时。

### 批量推理

```bash
python predict_dataset.py
```

对整个 split 批量推理，输出按 mIoU 排名的可视化面板：

- 每张图输出文件名包含 mIoU：`rank001_C100_iou82.3_panel.png`
- 各 split 结果分目录保存

配置 `BatchPredictConfig`（`predict_dataset.py` 内定义），主要字段：

| 字段 | 说明 |
|---|---|
| `img_root` | 图片根目录（如 `dataset/tongji_data_raw/img_dir`） |
| `splits` | 推理的 split 列表（默认 `["valid"]`） |
| `ckpt` | checkpoint 路径 |
| `use_tiling` | 是否使用 tiling 推理 |

### 推理结果转 labelme 标注（可选）

把推理生成的类别索引 mask（`{stem}_pred_mask.png`）反向转成 labelme 5.x JSON，
方便人工在 labelme 里复核 / 微调，并把它当作下一轮迭代的种子标注。
所有类别一律 `shape_type="polygon"`，标签名对齐 [dataset/tongji/](dataset/tongji/) 原始风格
（`crack` / `leakageB` / `leakageW` / `leakageG` / `lining falling off` / `segment damage`）。

**方式 A：推理时同步生成**

`predict_image.py` 与 `predict_dataset.py` 的配置（`PredictConfig` / `BatchPredictConfig`）
新增三个字段：

| 字段 | 默认 | 说明 |
|---|---|---|
| `save_labelme` | `False` | 设为 `True` 即在每张图推理后写出 `{image_stem}.json` |
| `labelme_epsilon` | `1.0` | `cv2.approxPolyDP` 简化阈值（像素）；`0` 表示不简化 |
| `labelme_embed_image` | `False` | `True` 时把原图 base64 嵌入 `imageData` 字段（JSON 自包含，文件较大） |

**方式 B：独立工具（已有 mask 也能补生成 JSON）**

```bash
# 单图模式
python data_tools/mask_to_labelme.py \
    --mask  outputs/.../{stem}_pred_mask.png \
    --image dataset/tongji_data_raw/img_dir/train/{stem}.jpg \
    --out   outputs/.../{stem}.json

# 批量模式：自动剥离 rank{N}_ / _iou{F} 等装饰，按 stem 与原图配对
python data_tools/mask_to_labelme.py \
    --mask_dir  outputs/ablation_tmds_full_small/predict_dataset/train \
    --image_dir dataset/tongji_data_raw/img_dir/train \
    --out_dir   outputs/ablation_tmds_full_small/predict_dataset_labelme/train
```

**输出 JSON schema**（labelme 5.x，老版 labelme 也能打开）：

```json
{
  "version": "5.4.1",
  "flags": {},
  "shapes": [
    {"label": "crack", "points": [[x,y], ...],
     "group_id": null, "shape_type": "polygon", "flags": {}}
  ],
  "imagePath": "C100.jpg",
  "imageData": null,
  "imageHeight": 2177,
  "imageWidth": 2184
}
```

`imagePath` 自动写为相对于 JSON 所在目录的相对路径，便于 labelme 打开 JSON 时定位原图；
若 JSON 与原图不在邻近目录，建议用 `--embed_image_data` / `labelme_embed_image=True` 让 JSON 自包含。

---

## 进阶：模型量化

对训练好的浮点模型执行 Post-Training Quantization（PTQ），压缩模型体积、降低 CPU 推理延迟：

```bash
python quantize.py
```

在 `quantize.py` 的 `RUN` 中配置源 checkpoint 路径，主要参数：

| 字段 | 默认值 | 说明 |
|---|---|---|
| `mode` | `"dynamic"` | 量化模式：`"dynamic"`（推荐，无需校准数据）/ `"static"` |
| `backend` | `"fbgemm"` | 量化后端：`"fbgemm"`（x86）/ `"qnnpack"`（ARM / 移动端） |

输出至 `output_dir/`：

- `model_int8.pth`：量化 checkpoint（含元信息，可直接用于 `infer.py` / `predict_image.py`）
- `quantize_summary.json`：浮点 / 量化模型体积对比报告

量化模型加载与原始模型接口完全一致，框架会自动检测并强制切换至 CPU 推理。

---

## 常用命令

```bash
# ── 数据准备 ──────────────────────────────────────────────────────────────────
# 训练数据（tiling）
python dataset_convert_awesome.py
python filter_background_patches.py   # 可选：筛除冗余背景 patch

# 评估数据（原始分辨率）
python dataset_convert_raw.py

# ── 训练 ─────────────────────────────────────────────────────────────────────
python train.py                        # 默认运行 main(TMDS_RUN)

# ── 评估 ─────────────────────────────────────────────────────────────────────
python infer.py                        # 标准评估
# 原图 tiling 评估：将 main(RAW_RUN) 写入 infer.py 后运行

# ── 推理 ─────────────────────────────────────────────────────────────────────
python predict_image.py                # 单图推理
python predict_dataset.py             # 批量推理（按 mIoU 排名输出）

# ── 推理结果转 labelme 标注（可选） ───────────────────────────────────────────
# 方式 A：在 predict_image.py / predict_dataset.py 的 RUN 中设 save_labelme=True
# 方式 B：对已有 *_pred_mask.png 批量补生成
python data_tools/mask_to_labelme.py \
    --mask_dir  outputs/<run>/predict_dataset/<split> \
    --image_dir dataset/tongji_data_raw/img_dir/<split> \
    --out_dir   outputs/<run>/predict_dataset_labelme/<split>

# ── 量化（进阶） ──────────────────────────────────────────────────────────────
python quantize.py

# ── TensorBoard ───────────────────────────────────────────────────────────────
tensorboard --logdir outputs/tmds_run_awesome/tensorboard
```
