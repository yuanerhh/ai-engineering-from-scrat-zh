---
name: skill-dcgan-scaffold
description: 根据 z_dim、image_size 和 num_channels，生成完整的 DCGAN 脚手架，包含训练循环和样本保存器
version: 1.0.0
phase: 4
lesson: 9
tags: [computer-vision, gan, dcgan, scaffolding]
---

# DCGAN 脚手架

给定三个参数，输出一个可运行的 DCGAN 项目骨架，架构大小根据目标图像分辨率自动调整。

## 使用时机

- 在小型数据集上开始新的生成式实验。
- 通过可运行的最小示例教授 DCGAN 基础知识。
- 原型条件 GAN（标签注入在同一脚手架中完成）。

## 输入

- `image_size`：32、64 或 128 之一（必须是 2 的幂次）。
- `num_channels`：1（灰度）或 3（RGB）。
- `z_dim`：通常为 64 或 128。
- `with_spectral_norm`：yes | no；默认为 yes。

## 架构尺寸

G 中转置卷积块数和 D 中步进卷积块数取决于 `image_size`：

| image_size | G 块数 | D 块数 |
|------------|--------|--------|
| 32         | 4      | 4      |
| 64         | 5      | 5      |
| 128        | 6      | 6      |

每增加一个块，G 的空间维度翻倍，D 的空间维度减半。特征数从 32 开始，按 `feat_base * 2^block_index` 缩放。

## 输出文件

- `model.py` — Generator + Discriminator 类
- `train.py` — 训练循环、损失、优化器设置
- `sample.py` — 样本网格保存器
- `config.json` — 超参数
- `README.md` — 10 行快速入门

## 报告

```
[scaffold]
  image_size:       <int>
  num_channels:     <int>
  z_dim:            <int>
  spectral_norm:    yes | no

[arch]
  G blocks:         <N>, channels: [列表]
  D blocks:         <N>, channels: [列表]
  G params (est):   <N>
  D params (est):   <N>

[training defaults]
  optimizer:   Adam(lr=2e-4, betas=(0.5, 0.999))
  batch_size:  64
  epochs:      50
  sample_every: 每 1 个 epoch

[files written]
  - model.py
  - train.py
  - sample.py
  - config.json
  - README.md
```

## 规则

- G 的输出始终使用 `nn.Tanh()`，训练时将数据缩放到 [-1, 1]。
- D 中始终使用 `LeakyReLU(0.2)`。
- 当 `with_spectral_norm == yes` 时，用 `spectral_norm()` 包裹 D 中的每个卷积，并去掉 D 中的 BatchNorm。G 中保留 BatchNorm。
- 不得为 `image_size > 128` 生成脚手架——DCGAN 在该分辨率以上会变得不稳定；请将用户引导至 StyleGAN 或扩散模型。
