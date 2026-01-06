# TC-Light 批量推理 (Sim2Real) 使用指南

本文档旨在介绍 `examples/batch_inference.py` 脚本的使用方法，并对 TC-Light 代码库进行简要分析，帮助你更好地理解和修改代码。

## 1. 代码库简析

TC-Light 是一个基于视频生成的 Sim2Real（从模拟到真实）框架，其核心逻辑建立在 `VidToMe` 项目之上。

### 核心文件结构

*   **`generate.py`**:
    *   **作用**: 这是单视频推理的核心入口。
    *   **逻辑**: 它定义了 `Generator` 类，继承自 `VidToMeGenerator`。它负责初始化数据集、加载模型、进行 DDIM 采样（`ddim_sample`）、以及后处理优化（`exposure_align`, `unique_tensor_optimization`）。
    *   **关键点**: 如果你需要修改生成的具体逻辑（例如采样步数内部的操作），主要修改这里。

*   **`utils/VidToMe/`**:
    *   **作用**: 包含底层的模型管道和工具函数。
    *   **`generate_utils.py`**: 定义了 `VidToMeGenerator` 基类，处理模型加载、Prompt 编码、Latent 编解码等基础操作。
    *   **`controlnet_utils.py`**: 管理 ControlNet 的映射和预处理。我们刚刚在此文件中添加了对 `seg` (语义分割) 的支持。

*   **`utils/dataparsers/`**:
    *   **作用**: 负责数据的加载和解析。
    *   **`video_dataparser.py`**: 专门用于处理视频输入 (`.mp4` 等)，计算光流 (`raft`/`memflow`)，为优化过程提供数据支持。

*   **`configs/`**:
    *   **作用**: 存放 `.yaml` 配置文件。`tclight_default.yaml` 定义了默认的参数，如分辨率、ControlNet 类型、优化步数等。

### 批量推理脚本的设计 (`examples/batch_inference.py`)

为了满足你批量处理 Sim2Real 数据的需求，我们设计了 `batch_inference.py`。
*   **多卡并行 (DDP)**: 脚本会自动检测环境变量 `RANK` 和 `WORLD_SIZE`，利用 `torch.distributed` 将视频列表切分到不同的 GPU 上并行处理。
*   **动态配置**: 它会读取 `base_config`，然后针对每一个视频动态修改 `rgb_path` (视频路径) 和 `prompt` (提示词)，从而复用同一个模型管道，避免重复加载模型的开销。

---

## 2. 模型权重下载（手动）

由于服务器网络环境限制，请手动下载以下模型权重，然后放置到服务器的指定目录。

### 必需模型

| 模型名称 | HuggingFace 地址 | 推荐本地保存路径 |
| :--- | :--- | :--- |
| **Stable Diffusion v1.5** | [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) | `/data/models/stable-diffusion-v1-5` |

### ControlNet 模型 (按需下载)

根据你使用的 `--control_type` 下载对应的 ControlNet 模型：

| ControlNet 类型 | HuggingFace 地址 | 推荐本地保存路径 |
| :--- | :--- | :--- |
| `seg` (语义分割) | [lllyasviel/control_v11p_sd15_seg](https://huggingface.co/lllyasviel/control_v11p_sd15_seg) | `/data/models/control_v11p_sd15_seg` |
| `depth` (深度) | [lllyasviel/control_v11f1p_sd15_depth](https://huggingface.co/lllyasviel/control_v11f1p_sd15_depth) | `/data/models/control_v11f1p_sd15_depth` |
| `softedge` (软边缘) | [lllyasviel/control_v11p_sd15_softedge](https://huggingface.co/lllyasviel/control_v11p_sd15_softedge) | `/data/models/control_v11p_sd15_softedge` |
| `canny` (Canny 边缘) | [lllyasviel/control_v11p_sd15_canny](https://huggingface.co/lllyasviel/control_v11p_sd15_canny) | `/data/models/control_v11p_sd15_canny` |

### 下载方法

推荐使用 `huggingface-cli` 或 `git lfs`:

```bash
# 方法一：使用 huggingface-cli (推荐)
pip install huggingface_hub
huggingface-cli download runwayml/stable-diffusion-v1-5 --local-dir /data/models/stable-diffusion-v1-5

huggingface-cli download lllyasviel/control_v11p_sd15_seg --local-dir /data/models/control_v11p_sd15_seg

# 方法二：使用 git lfs (需要安装 git-lfs)
git lfs install
git clone https://huggingface.co/runwayml/stable-diffusion-v1-5 /data/models/stable-diffusion-v1-5
git clone https://huggingface.co/lllyasviel/control_v11p_sd15_seg /data/models/control_v11p_sd15_seg
```

---

## 3. 环境配置

本项目需要 PyTorch (CUDA 版本) 和相关依赖。以下指南适用于 **8卡 A6000 + CUDA 12.x** 环境，使用国内镜像源。

### 3.1 配置 pip 国内镜像源

首先配置 pip 使用清华镜像源：

```bash
# 创建或编辑 pip 配置文件
mkdir -p ~/.pip
cat > ~/.pip/pip.conf << EOF
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF
```

### 3.2 创建 Conda 环境 (推荐)

```bash
# 创建新的 conda 环境
conda create -n tclight python=3.10 -y
conda activate tclight
```

### 3.3 安装 PyTorch (CUDA 12.x)

使用清华/阿里云镜像安装 PyTorch CUDA 12.1 版本：

```bash
# 方法一：使用 pip + 清华镜像 (推荐)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 方法二：使用 conda + 清华镜像
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

验证安装：
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"
# 预期输出: PyTorch: 2.x.x+cu121, CUDA: True, Devices: 8
```

### 3.4 安装项目依赖

```bash
# 进入项目目录
cd /home/fch/research/video_gen/TC-Light

# 安装 requirements.txt 中的依赖 (使用清华源)
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 如果 torch_scatter 安装失败，单独安装
pip install torch_scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3.5 常见问题

| 问题 | 解决方案 |
| :--- | :--- |
| `torch_scatter` 安装失败 | 使用 `pip install torch_scatter -f https://data.pyg.org/whl/torch-{TORCH_VERSION}+{CUDA_VERSION}.html` |
| `xformers` 未安装 | `pip install xformers -i https://pypi.tuna.tsinghua.edu.cn/simple` (可选，用于加速) |
| CUDA 版本不匹配 | 确保 `nvcc --version` 输出的 CUDA 版本与 PyTorch 编译版本兼容 |

---

## 4. 使用方法

该脚本支持在单机多卡（单台服务器，8张卡）环境下运行。

### 参数说明

| 参数 | 说明 | 示例 |
| :--- | :--- | :--- |
| `--video_dir` | **[必填]** 输入视频文件夹路径 | `/path/to/sim_videos` |
| `--prompt_dir` | **[必填]** 提示词 txt 文件夹路径 | `/path/to/prompts` |
| `--output_dir` | **[必填]** 结果保存路径 | `/path/to/results` |
| `--sd_model_path` | **[关键]** SD 模型本地路径 | `/data/models/stable-diffusion-v1-5` |
| `--controlnet_path` | **[关键]** ControlNet 模型本地路径 | `/data/models/control_v11p_sd15_seg` |
| `--base_config` | 基础配置文件路径 (默认 `configs/tclight_default.yaml`) | `configs/tclight_default.yaml` |
| `--control_type` | ControlNet 类型 | `seg`, `depth`, `softedge`, `none` |
| `--seed` | 随机种子 (默认 12345) | `42` |

### 运行示例

假设你的服务器有 8 张 A6000 显卡，使用 `torchrun` 启动脚本：

```bash
# 进入项目根目录
cd /home/fch/research/video_gen/TC-Light

# 启动 8 卡并行推理 (使用本地模型)
torchrun --nproc_per_node=8 examples/batch_inference.py \
    --video_dir /data/sim_videos \
    --prompt_dir /data/sim_prompts \
    --output_dir /data/sim2real_results \
    --sd_model_path /data/models/stable-diffusion-v1-5 \
    --controlnet_path /data/models/control_v11p_sd15_seg \
    --control_type seg \
    --base_config configs/tclight_default.yaml
```

### 数据准备要求

1.  **视频文件夹 (`--video_dir`)**:
    *   包含 `.mp4`, `.avi`, `.mov` 或 `.mkv` 格式的视频。
    *   例如: `video_01.mp4`, `video_02.mp4`

2.  **Prompt 文件夹 (`--prompt_dir`)**:
    *   包含与视频同名的 `.txt` 文件。
    *   例如: `video_01.txt`, `video_02.txt`
    *   如果找不到对应的 txt 文件，脚本会尝试使用默认 Prompt 或 VLM 上采样（取决于配置），并在日志中打印警告。

### ControlNet 说明

对于 Sim2Real 任务，你提到有 `edge`, `seg`, `depth`, `blur` 数据。
*   **使用 Seg (语义分割)**: 设置 `--control_type seg`。请确保你的输入视频能够被 `lllyasviel/control_v11p_sd15_seg` 模型正确处理，或者你的 sim 数据本身就是 seg 色的（如果 sim 数据是 seg 图，直接作为输入即可；如果 sim 数据是渲染图，我们会对其进行 seg 预处理）。
    *   **注意**: 目前代码逻辑是读取 `video_dir` 的视频作为**内容条件** (Content Condition)，并对其进行 ControlNet 预处理（例如提取 Depth 或 Seg）。
    *   如果你的 `video_dir` 里已经是 Depth 或 Seg 这种控制信号视频了，你需要确保 ControlNet 预处理不会破坏它，或者修改 `controlnet_utils.py` 让其跳过预处理（`process_flag = False`）。

### 结果查看

推理完成后，结果将保存在 `--output_dir` 下，每个视频会有一个独立的子文件夹，结构如下：

```text
/data/sim2real_results/
├── video_01/
│   ├── config.yaml          # 该视频使用的配置
│   ├── output_opt_xxxx.mp4 # 最终生成的 Sim2Real 视频
│   ├── gt/                  # 原始输入视频 (用于对比)
│   └── latents/             # 倒置过程的 Latents
└── video_02/
    └── ...
```
