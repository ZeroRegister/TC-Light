[English Version](README.md)
<p align="center">
<h1 align="center"><strong> [NeurIPS`2025] TC-Light: Temporally Coherent Generative
Rendering for Realistic World Transfer</strong></h1>
  <p align="center">
    <em>Institute of Automation, Chinese Academy of Sciences; University of Chinese Academy of Sciences</em>
  </p>
</p>

<div id="top" align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2506.18904-b31b1b.svg)](https://arxiv.org/abs/2506.18904)
[![](https://img.shields.io/badge/%F0%9F%9A%80%20-Project%20Page-blue)](https://dekuliutesla.github.io/tclight/)
[![](https://img.shields.io/badge/📄-中文解读-red)](https://www.qbitai.com/2025/07/310873.html)
![GitHub Repo stars](https://img.shields.io/github/stars/Linketic/TC-Light)

</div>

[https://github.com/user-attachments/assets/9fc9c6ce-a83c-4ca5-9273-7cb672c99452](https://github.com/user-attachments/assets/9fc9c6ce-a83c-4ca5-9273-7cb672c99452)

本仓库包含 **TC-Light** 的官方实现。TC-Light 是一个用于操控视频光照分布的 one-shot 模型，可实现 **逼真的世界迁移（realistic world transfer）**。它尤其适用于**高动态视频**，例如运动剧烈、前景/背景频繁切换的场景。TC-Light 的优势包括：

* 🔥 在高动态场景中具有卓越的时间一致性。
* 🔥 计算效率高，可处理长视频（40G A100 上可处理分辨率 1280x720 的 300 帧视频）。

这些特性使其在 Embodied Agents 的 sim2real / real2real 增强，或用于生成视频对以训练更强大的视频重光照模型中具备很高价值。若您喜欢本项目，欢迎点个 ⭐！

## 📰 最新动态

**[2025.09.08]** 👏 TC-Light 已被 ICLR 2025 接收！

**[2025.06.23]** TC-Light 论文与代码正式开源！

## 💡 方法简介

<div align="center">
    <img src='assets/pipeline.png'/>
</div>

<b>TC-Light</b> 概述：给定源视频与文本提示 p，模型分别对输入隐变量在 xy 平面与 yt 平面进行编码。预测得到的噪声会被融合并用于去噪。其输出随后经过两个阶段的优化：第一阶段通过优化 appearance embedding 使曝光对齐；第二阶段通过优化基于时空关联性得到的<b>视频码本（即论文中的 Unique Video Tensor）</b>来对齐细节纹理与光照，该张量是视频的压缩表示。更多细节请参考论文。

## 💾 准备环境

首先按以下步骤安装运行环境：

```bash
git clone https://github.com/Linketic/TC-Light.git
cd TC-Light
conda create -n tclight python=3.10
conda activate tclight
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

然后从以下链接下载模型权重至 `./models`：

* **Hugging Face**: [https://huggingface.co/TeslaYang123/TC-Light](https://huggingface.co/TeslaYang123/TC-Light)
* **百度网盘**: [https://pan.baidu.com/s/1L-mk6Ilzd2o7KLAc7-gIHQ?pwd=rj99](https://pan.baidu.com/s/1L-mk6Ilzd2o7KLAc7-gIHQ?pwd=rj99)

## ⚡ 快速上手

你可以使用以下命令快速体验：

```bash
# 支持 .mp4, .gif, .avi，以及包含序列帧的文件夹
# --multi_axis 启用衰减式多轴去噪，可增强一致性但会降低速度
python run.py -i /path/to/your/video -p "your_prompt" \
              -n "your_negative_prompt" \  # 可选
              --multi_axis  # 可选
```

默认情况下，TC-Light 会以 960x720 分辨率重光照前 30 帧。默认 negative prompt 采用自 [Cosmos-Transfer1](https://github.com/nvidia-cosmos/cosmos-transfer1)，用于提升光照编辑的真实性。如果是第一次处理该视频，还会在视频所在目录生成并保存光流。

如需更精细的调控，可自定义 `.yaml` 配置文件并运行：

```bash
python run.py --config path/to/your_config.yaml
```

你可以参考 [configs/tclight_custom.yaml](configs/tclight_custom.yaml)，其中包含最常用的参数及详细说明。

<details>
<summary><span style="font-weight: bold;">示例</span></summary>

#### 重光照整个视场（FOV）

```bash
python run.py --config configs/examples/tclight_droid.yaml
```

```bash
python run.py --config configs/examples/tclight_navsim.yaml
```

```bash
python run.py --config configs/examples/tclight_scand.yaml
```

#### 并行重光照三个视频

```bash
bash scripts/relight.sh
```

#### 在静态背景条件下重光照前景

```bash
# 我们使用 IC-Light 的前景模式生成兼容的背景图像，然后移除前景并使用 sider.ai 等工具对图像进行修补（inpaint）
# 若想得到满意效果，需要一致且完整的前景分割；我们默认使用 BriaRMBG。
python run.py --config configs/examples/tclight_bkgd_robotwin.yaml
```

</details>

如需评估，可直接运行：

```bash
python evaluate.py --output_dir path/to/your_output_dir --eval_cost
```

## 🔎 使用注意事项

1. 更适用于分辨率高于 512x512 的视频，这也是 IC-Light 的训练分辨率下限；更高分辨率有助于保持图像内部属性的一致性。
2. 在真实场景上表现优于合成场景，无论是时间一致性还是物理合理性。
3. 难以对夜景或强投影阴影进行大幅度光照修改（IC-Light 同样存在该限制）。

## 📝 TODO 列表

* [x] 发布 arXiv 与项目页
* [x] 开源代码
* [ ] 发布数据集

## 🤗 引用方式

如果你觉得本仓库对你的研究有帮助，欢迎引用：

```
@inproceedings{
    liu2025tclight
    title={TC-Light: Temporally Coherent Generative Rendering for Realistic World Transfer},
    author={Yang Liu, Chuanchen Luo, Zimo Tang, Yingyan Li, Yuran Yang, Yuanyong Ning, Lue Fan, Junran Peng, Zhaoxiang Zhang},
    booktitle={The Thirty-Ninth Annual Conference on Neural Information Processing Systems},
    year={2025},
}
```

## 👏 致谢

本项目受益于以下优秀工作：[IC-Light](https://github.com/lllyasviel/IC-Light/)、[VidToMe](https://github.com/lixirui142/VidToMe/)、[Slicedit](https://github.com/fallenshock/Slicedit/)、[RAVE](https://github.com/RehgLab/RAVE)、[Cosmos](https://github.com/NVIDIA/Cosmos)。感谢他们的卓越贡献！本仓库仍在持续开发中，欢迎提出 PR 或讨论！

