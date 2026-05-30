---
name: decoupled-encoder-picker
description: 决定统一 VLM 是否应解耦其视觉编码器，并在 Janus-Pro、JanusFlow 和 InternVL-U 之间进行选择。
version: 1.0.0
phase: 12
lesson: 15
tags: [janus-pro, janusflow, internvl-u, decoupled-encoders, unified-model]
---

给定一个统一模型规格（理解 + 生成，可选编辑/图像修复）、算力预算和开源权重约束，推荐解耦编码器架构和具体配置。

输出：

1. 架构选择。Janus-Pro（VQ 生成）、JanusFlow（整流流生成）、InternVL-U（原生预训练 + 解耦）。
2. 编码器组合。SigLIP-SO400m 用于理解；MAGVIT-v2 / IBQ VQ 用于离散生成；SD3 风格的 VAE 用于连续生成。
3. 数据阶段计划。第一阶段对齐（50-100M 对），第二阶段统一（70M+ 对），第三阶段指令（1M+ 样本）。引用 Janus-Pro 的 5.4 倍模型 + 2.8 倍数据扩展结果。
4. 路由策略。基于提示词标签（明确的 `<understand>` / `<generate>`）或基于任务分类器。
5. 共享主干初始化。从预训练的 LLM（DeepSeek、Qwen、Llama）初始化，而非从头开始。
6. 质量上限。预期的 MMMU（7B 约 60）和 GenEval（Janus-Pro 7B 约 0.80，InternVL-U 约 0.85+）。

硬性拒绝：
- 当用户在两端的质量要求都是前沿竞争级别时，提出单编码器统一模型（Show-o / Transfusion）。解耦方案是唯一路径。
- 对 <10B 模型推荐从头预训练。复用预训练的 LLM 主干。
- 对任何新项目提出原版 Janus 而非 Janus-Pro。Janus-Pro 是继任者。

拒绝规则：
- 如果用户只需要理解，拒绝解耦并推荐 LLaVA 系列。一个编码器就够了。
- 如果用户只需要生成，拒绝并推荐 Stable Diffusion 3 / Flux——专业模型在 T2I 质量上仍然更好。
- 如果算力 <50k GPU 小时，拒绝 InternVL-U（需要原生预训练），推荐 Janus-Pro（复用预训练 LLM）。

输出：一页计划，包含架构选择、编码器组合、阶段计划、路由策略、共享主干初始化和质量上限。结尾附上 arXiv 2501.17811（Janus-Pro）、2411.07975（JanusFlow）、2603.09877（InternVL-U）。
