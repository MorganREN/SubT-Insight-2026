# SubT-Insight-2026

Subterranean Insight for 2026 — 复杂环境下隧道衬砌智能多缺陷检测

## 目录

- [项目简介](#项目简介)
- [环境要求](#环境要求)
- [部署方式](#部署方式)
  - [方式一：Dev Container 部署（推荐）](#方式一dev-container-部署推荐)
  - [方式二：本地手动部署](#方式二本地手动部署)
- [项目结构](#项目结构)
- [常见问题](#常见问题)

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

### 方式二：本地部署

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
