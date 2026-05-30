---
name: diffusion-trainer
description: 配置扩散训练任务：噪声调度、预测目标、采样器和评估方案。
version: 1.0.0
phase: 8
lesson: 06
tags: [diffusion, ddpm, training]
---

给定数据集特征（模态、分辨率、数据集大小）、计算预算（GPU 小时数、最低显存）和质量要求（FID 目标或下游用途），输出以下内容：

1. 调度。线性、余弦（Nichol）或 sigmoid。步骤数 T（DDPM 基线使用 1000；更快变体使用 256）。
2. 预测目标。epsilon、v-prediction 或 x_0。理由与分辨率和调度中的信噪比挂钩。
3. 架构。像素级扩散使用 U-Net 深度 + 通道宽度，潜在扩散使用 DiT，视频使用 3D U-Net / DiT。包含时间嵌入方案（正弦 + MLP、FiLM 或 AdaLN）。
4. 采样器。DDIM（20-50 步）、DPM-Solver++（10-20 步）、Euler-A（创意）或蒸馏 1-4 步。包含引导缩放（CFG w）推荐。
5. 评估方案。FID / KID / CLIP 分数 / 人工偏好，样本数（FID 至少 1 万），CFG w 的扫描协议。

拒绝在潜在扩散能以 1/16 FLOP 达到相同质量时，推荐在 256×256 及以上分辨率训练像素级扩散。拒绝在没有 CFG 的情况下发布条件生成模型——无 CFG 的条件模型输出通常是退化的无条件样本。标记任何 beta_T > 0.1 的调度为可能产生饱和或不稳定的训练。
