---
name: tokenizer-vs-adapter-picker
description: 为 VLM 项目在 Chameleon 风格的早期融合（共享词汇分词器）和 LLaVA 风格的后期融合（冻结 LLM 上的适配器）之间进行选择。
version: 1.0.0
phase: 12
lesson: 11
tags: [chameleon, early-fusion, vq-vae, late-fusion, adapter]
---

给定产品规格（仅理解或理解+生成）、目标图像质量（社交帖子 / 杂志 / 印刷 / 广播）和成本预算（训练 + 推理），推荐 Chameleon 系列或 LLaVA 系列，并附具体的架构概述。

输出：

1. 结论。早期融合（Chameleon / Emu3 / AnyGPT）还是后期融合（LLaVA / BLIP-2 / Qwen-VL）系列。
2. 分词器选择（针对早期融合结论）。VQ-VAE（Chameleon）、MAGVIT-v2、IBQ 或 SBER-MoVQGAN；引用预期的 PSNR 重建上限。
3. 训练稳定性计划。大规模早期融合的 QK-Norm、Dropout 放置和 LayerNorm 顺序。
4. 成本估算。训练 GPU 小时数和每张图像的推理延迟，与后期融合替代方案比较。
5. 生成质量上限。用户可以期望的 PSNR / FID 范围；产品的质量要求是否可以通过离散 token 实现，还是需要连续（Transfusion 风格）生成。
6. 迁移路径。如果用户扩展后期融合变得受限（他们需要图像输出），迁移是什么样的。

硬性拒绝：
- 对仅理解产品推荐 Chameleon 风格。后期融合对纯理解来说更简单、更便宜且上限更高。
- 提出 K<4096 的 VQ-VAE 用于生产图像生成。码本太小，伪影可见。
- 声称早期融合推理是免费的。VQ 解码器每张生成图像增加 50-200ms，通常超过 LLM 输出时间。

拒绝规则：
- 如果用户想要前沿质量的图像生成（FID < 15，印刷就绪），拒绝离散 token 并指向 Transfusion / Stable Diffusion 3 / MMDiT（第 12.13 课）。
- 如果产品永远不需要图像输出，拒绝早期融合——复杂性是不必要的。
- 如果用户想要插入现有的 Llama / Qwen LLM 权重，拒绝早期融合——它需要从头预训练一个新模型。

输出：一页计划，包含结论、分词器选择、稳定性检查清单、成本估算、质量上限和迁移路径。结尾附上 arXiv 2405.09818（Chameleon）和 2408.11039（Transfusion）供比较阅读。
