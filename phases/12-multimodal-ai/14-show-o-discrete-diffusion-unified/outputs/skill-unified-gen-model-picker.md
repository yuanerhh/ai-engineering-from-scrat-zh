---
name: unified-gen-model-picker
description: 为需要多模态理解和生成的产品在 Show-o / Transfusion / Emu3 / Janus-Pro 系列之间进行选择（开源权重约束）。
version: 1.0.0
phase: 12
lesson: 14
tags: [show-o, masked-diffusion, unified, t2i, inpainting]
---

给定一个需要统一理解 + 生成（VQA、描述、T2I，可选图像修复）且有开源权重约束和延迟预算的产品，选择模型系列并生成参考配置。

输出：

1. 系列结论。Show-o（掩码离散扩散）、Transfusion / MMDiT（连续扩散）、Emu3 / Chameleon（自回归离散）或 Janus-Pro（解耦编码器）。
2. 推理步数预算。Show-o 16 步，Transfusion 20 步，Emu3 1024+ 步。用用户的延迟预算说明选择理由。
3. 图像修复支持。Show-o 免费支持；Transfusion 添加掩码通道；Emu3 需要单独微调。为用户标注此信息。
4. 分词器选择。对于离散系列，推荐 IBQ / MAGVIT-v2 / SBER；对于连续系列，推荐 SD3 的 VAE。
5. 训练稳定性。双损失（Transfusion）需要权重调优；Show-o 的单损失更简洁。
6. 用户规模增长时的迁移路径。从 Show-o 迁移到 Transfusion，当质量成为瓶颈时。

硬性拒绝：
- 在推理延迟 <10 秒/图像时提出 Emu3 / Chameleon。在约 1024 个 token 上自回归太慢。
- 声称 Show-o 在前沿图像质量上与 Transfusion 持平。它不是。分词器是上限。
- 为需要 VQA 的产品推荐 Stable Diffusion。SD 无法推理图像。

拒绝规则：
- 如果用户想要每张图像生成 <2 秒，拒绝 Show-o 并推荐 Stable Diffusion + 独立的 VLM 用于理解。接受多模型复杂性。
- 如果用户想要开源权重的「最佳质量」，拒绝 Show-o / Emu3 并推荐 Transfusion 系列（MMDiT）或 JanusFlow。
- 如果用户无法承诺分词器（担心许可证、质量上限），拒绝纯离散系列并推荐 Transfusion。

输出：一页选择，包含系列结论、步数预算、图像修复支持、分词器推荐、稳定性计划和迁移路径。结尾附上 arXiv 2408.12528（Show-o）、2408.11039（Transfusion）、2501.17811（Janus-Pro）。
