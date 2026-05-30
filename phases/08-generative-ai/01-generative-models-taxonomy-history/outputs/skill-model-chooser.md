---
name: generative-model-chooser
description: 根据给定任务和预算选择生成模型家族、骨干网络和托管替代方案。
version: 1.0.0
phase: 8
lesson: 01
tags: [generative, taxonomy]
---

给定任务描述（模态、领域、延迟预算、计算预算、条件信号），输出以下内容：

1. 家族。显式可计算、显式近似（VAE / 扩散）、隐式（GAN）、分数 / 流匹配或 token 自回归。一句话说明理由，与模态 + 延迟挂钩。
2. 骨干网络 + 开源参考。一个用户今天可以微调的预训练开权重模型（例如 Stable Diffusion 3、Flux.1-dev、AudioCraft 2、StyleGAN3、3D 高斯溅射）。
3. 托管替代方案。三个生产 API，按质量 / 成本 / 延迟权衡排名（fal.ai、Replicate、Stability、Runway、Veo、Kling、ElevenLabs 等）。
4. 失败模式。所选家族的已知缺陷（模式崩溃、曝光偏差、采样漂移、分词器伪影、CLIP 分数博弈）。
5. 预算。单张 A100 上的大致训练小时数、每次生成的推理成本、最低显存要求。

拒绝在任务需要似然度评分时推荐 GAN。拒绝在高分辨率实时场景中推荐逐像素自回归。标记任何推荐"从头训练"的方案，如果列出的开源骨干网络已经覆盖该领域。
