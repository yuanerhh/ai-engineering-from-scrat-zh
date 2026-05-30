---
name: stylegan-inversion
description: 为预训练 StyleGAN 上的真实照片选择反演和编辑流水线。
version: 1.0.0
phase: 8
lesson: 05
tags: [stylegan, inversion, editing]
---

给定真实照片 + 预训练 StyleGAN 检查点（FFHQ-1024、StyleGAN-XL、自定义微调）和目标编辑（年龄、笑容、姿态、发型、身份保留），输出以下内容：

1. 反演方法。e4e（快速，低保真）、ReStyle（迭代编码器）、HyperStyle（超网络）、PTI（枢轴调优）或直接 W 优化。一句话说明理由，与保真度 vs 速度权衡挂钩。
2. 目标空间。W、W+ 或 StyleSpace。权衡：W = 最解耦但最低保真度，W+ = 每层独立的 w，StyleSpace = 通道级别。
3. 编辑方向。命名方向来源：InterFaceGAN（基于 SVM）、StyleSpace 通道、GANSpace PCA 或学习分类器。
4. 保真度预算。身份漂移前的 LPIPS 阈值；回退启发式。
5. 评估。ID 相似度（ArcFace 余弦）、与原图的 LPIPS、编辑强度（目标属性分类器得分）。

拒绝任何直接在 Z 空间编辑的流水线（高度纠缠）。拒绝在没有身份检查的情况下进行大幅度编辑（W 空间超过 1.5 sigma）。标记需要开放域编辑（例如"把他变成卡通"）的请求——这类需求需要扩散 + IP-Adapter，而非 StyleGAN。
