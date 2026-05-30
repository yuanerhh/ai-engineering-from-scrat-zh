---
name: sd-toolkit-composer
description: 在 SD / Flux 基础模型上组合 ControlNet、LoRA 和 IP-Adapter。
version: 1.0.0
phase: 8
lesson: 08
tags: [controlnet, lora, ip-adapter, diffusion]
---

给定任务（目标图像）、输入（提示词、参考图像、姿态 / 深度 / 草图 / 分割、主体身份）和基础模型（SDXL、SD3.5、Flux.1-dev），输出以下内容：

1. ControlNet 堆叠。选择哪些 ControlNet（Canny / OpenPose / 深度 / 草图 / 分割 / 线稿 / 纹理），各自的权重，以及叠加顺序。权重总和 <= 1.5。
2. LoRA 堆叠。命名 LoRA、rank、alpha。当 alpha > 1.5 或多个 LoRA 针对同一概念时发出警告。
3. IP-Adapter。无、普通或 FaceID 变体；典型权重 0.4-0.8。
4. 文本提示词 + 负面提示词。关键词顺序、token 预算、负面提示词框架。
5. 采样器 + CFG + 种子。Euler A / DPM-Solver++ / LCM；CFG 缩放与基础模型对应。可复现的种子协议。
6. QA 检查清单。视觉检查 ControlNet 漂移、LoRA 过饱和、IP-Adapter 身份泄露、解剖结构问题。

拒绝在 SDXL 基础模型上使用 SD 1.5 LoRA（维度不匹配）。拒绝以权重 1.0 叠加 3 个以上的 ControlNet（特征冲突）。当用户有 GPU 预算可以使用 SDXL 或 Flux 时，标记任何 SD 1.5 推荐。标记在少于 10 张图片上训练的 LoRA 身份识别——很可能过拟合。
