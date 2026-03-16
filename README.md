# SubT-Insight-2026

Subterranean Insight for 2026 — 复杂环境下隧道衬砌智能多缺陷检测

## 目录

- [项目简介](#项目简介)
- [环境要求](#环境要求)
- [部署方式](#部署方式)
  - [方式一：Dev Container 部署（推荐）](#方式一dev-container-部署推荐)
  - [方式二：本地手动部署](#方式二本地手动部署)
- [代码说明（比赛版）](#代码说明比赛版)
- [数据集类别定义与映射（Tongji）](#数据集类别定义与映射tongji)
- [常用命令](#常用命令)

---

## 项目简介

本项目基于深度学习语义分割技术，实现对隧道衬砌在复杂环境下的多种缺陷进行智能检测。核心依赖包括 PyTorch、segmentation-models-pytorch 和 albumentations。

---

## 环境要求

- Python 3.10+、PyTorch 2.2.0
- GPU 部署需要 CUDA 12.1
- 详见 `requirements.txt`

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

1. **克隆仓库**

```bash
git clone <仓库地址>
cd SubT-Insight-2026
code .
```

VS Code 打开后选择 `Reopen in Container`，根据硬件选择配置：

| 配置 | 适用场景 |
|---|---|
| `.devcontainer/cuda-gpu/` | NVIDIA GPU（需安装 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)） |
| `.devcontainer/windows-cpu/` | Windows / Linux CPU |
| `.devcontainer/mac-arm/` | Apple Silicon Mac |

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

## 代码说明

这套仓库是比赛导向，结构上优先“能快速迭代和复现实验”：

- `train.py`：训练入口（直接改 `RUN` 配置后运行）。
- `infer.py`：整套验证/测试集评估入口。
- `predict_image.py`：单图推理与可视化入口。
- `dataset_convert.py`：Tongji 标注转换脚本。
- `trainer/`：训练主流程（epoch 循环、验证、保存最优模型）。
- `inference/`：评估流程。
- `predictor/`：单图预测流程。
- `models/`：backbone + segmentation head + 整体封装。
- `dataload/`：dataset、增强和 dataloader。
- `criteria/`：损失函数与分割指标。
- `utils/`：运行时、checkpoint、优化器、scheduler、可视化通用工具。

## 数据集类别定义与映射（Tongji）

当前统一为 **1 个背景 + 6 类病害**（共 7 类，mask 值 `0..6`）：

| ID | 类别名 | 说明 |
|---|---|---|
| 0 | `background` | 背景 |
| 1 | `crack` | 裂缝 |
| 2 | `leakage_b` | 渗漏 B |
| 3 | `leakage_w` | 渗漏 W |
| 4 | `leakage_g` | 渗漏 G |
| 5 | `lining_falling_off` | 衬砌脱落 |
| 6 | `segment_damage` | 管片损伤 |

`dataset_convert.py` : 将原始下载的 Tongji 数据集（包含 6 类病害的标注）`dataset.zip`解压后的文件转换为上述统一类别定义的格式，输出到 `tongji_data/` 目录。

## 常用命令

```bash
# 数据转换（tongji -> tongji_data）
python dataset_convert.py

# 训练
python train.py

# 评估
python infer.py

# 单图预测
python predict_image.py
```
