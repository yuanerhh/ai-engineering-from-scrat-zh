# Janus-Pro：解耦编码器的统一多模态模型

> 统一多模态模型面临一个不可回避的张力：理解任务需要语义特征——SigLIP 或 DINOv2 输出富含概念级信息的向量；生成任务需要重建友好的编码——能够合成清晰像素的 VQ 词元。两个目标在单一编码器中无法兼顾。Janus（DeepSeek，2024 年 10 月）和 Janus-Pro（DeepSeek，2025 年 1 月）给出的答案是：停止用一个编码器兼顾两端——解耦两个编码器。理解任务通过 SigLIP 路由，生成任务通过 VQ 分词器路由，Transformer 主体由两种任务共享。7B 规模下，Janus-Pro 在 GenEval 上超越 DALL-E 3，同时在 MMMU 上媲美 LLaVA。本课解读为何两个编码器能解决一个编码器会失败的问题。

**类型：** 构建
**编程语言：** Python（标准库，双编码器路由 + 共享主体信号）
**前置知识：** Phase 12 · 13（Transfusion）、Phase 12 · 14（Show-o）
**预计时间：** 约 120 分钟

## 学习目标

- 解释为何单一共享编码器会损害理解或生成质量中的至少一方。
- 描述 Janus-Pro 的路由方案：理解侧输入使用 SigLIP 特征，生成侧输入和输出均使用 VQ 词元。
- 追踪使 Janus-Pro 成功而 Janus 不足的数据规模扩展路径。
- 对比解耦（Janus-Pro）、连续耦合（Transfusion）和离散耦合（Show-o）三种架构。

## 问题背景

统一模型在理解和生成之间共享一个 Transformer 主体。以往的尝试（Chameleon、Show-o、Transfusion）都对两个方向使用同一个视觉分词器，而这个分词器必然是一种折中：

- 为重建优化（生成）：VQ-VAE 能捕捉细粒度像素细节，但产生的词元语义一致性较弱。
- 为语义优化（理解）：SigLIP 嵌入能将"猫"的图像聚集在"猫"词元附近，但无法支持良好的重建。

Show-o 和 Transfusion 都为此付出了某个方向上明显的质量代价。Janus-Pro 的反问：既然两个任务的需求不同，为何要强求一个分词器？

## 核心概念

### 解耦视觉编码

Janus-Pro 的架构将两个编码器分开：

- 理解路径：输入图像 → SigLIP-SO400m → 两层 MLP → Transformer 主体。
- 生成路径：输入图像（条件生成时）→ VQ 分词器 → 词元 ID → Transformer 主体。
- 输出生成：Transformer 预测的图像词元 → VQ 解码器 → 像素。

Transformer 主体由两种路径共享，主体上下游的一切均为任务专用。

输入通过提示格式加以区分：`<understand>` 标签通过 SigLIP 路由；`<generate>` 标签通过 VQ 路由。或者根据任务隐式路由。

### 为何奏效

理解损失获得 SigLIP 特征——CLIP 风格的预训练已将其调优为语义相似性的良好衡量。模型在感知基准上的表现优于 Show-o 和 Transfusion，因为输入特征更适合该任务。

生成损失获得 VQ 词元——分词器已将其调优为重建。图像质量优于 Show-o，因为 VQ 编码能干净地还原为像素。

共享的 Transformer 主体接收两种输入分布（SigLIP 和 VQ），学会同时处理两者。主张是：足够的数据 + 足够的参数，主体能够吸收模式切换带来的分布差异。

### 数据规模——Janus vs Janus-Pro

Janus（原始版本，arXiv 2410.13848）引入了解耦思路，但规模较小（1.3B 参数，数据有限）。Janus-Pro（arXiv 2501.17811）进行了规模扩展：

- 7B 参数（vs 1.3B）。
- 阶段 1（对齐）图文对：9000 万（vs 7200 万）。
- 阶段 2（统一）：7200 万（vs 2600 万）。
- 阶段 3 新增 20 万条图像生成指令样本。

结果：Janus-Pro-7B 在 MMMU 上媲美 LLaVA（60.3 vs ~58），在 GenEval 上超越 DALL-E 3（0.80 vs 0.67）。一个开放模型，在统一频谱的两端均具竞争力。

### JanusFlow——整流流变体

JanusFlow（arXiv 2411.07975）将 VQ 生成路径替换为整流流（连续）生成路径。分拆变为：理解使用 SigLIP + 生成使用整流流。质量上限进一步提升。架构仍然是解耦编码器 + 共享主体的范式。

### 共享主体的工作

Transformer 主体处理统一序列，但面对两种输入分布。它的职责是：

- 理解任务：消费 SigLIP 特征 + 文本词元 → 自回归生成文本。
- 生成任务：消费文本词元 + 可选图像 VQ 词元 → 自回归生成图像 VQ 词元。

主体在每个块内不含模态专用权重，就是 Qwen 或 Llama 内部标准的文本风格 Transformer，加上两个输入适配器。

值得注意的是，Janus-Pro 的主体可以从预训练 LLM 初始化——Janus-Pro 确实从 DeepSeek-MoE-7B 初始化。这一选择很重要：LLM 贡献了纯从头训练的统一模型难以达到的推理能力。

### 与 InternVL-U 的对比

InternVL-U（第 12.10 课）是 2026 年的后续工作，它集成了：

- 原生多模态预训练（InternVL3 骨干）。
- 解耦编码器路由（输入使用 SigLIP，输出使用 VQ + 扩散头）。
- 统一理解 + 生成 + 编辑能力。

InternVL-U 将 Janus-Pro 的架构选择纳入了更大的框架中。解耦编码器思路现已成为大规模统一模型的默认选择。

### 局限性

解耦编码器增加了架构复杂性——需要训练两个分词器，维护两条输入路径，应对两套失效模式。对于不需要生成能力的产品，Janus-Pro 过于复杂，选 LLaVA 系理解模型更合适。

对于不需要理解能力的产品，Janus-Pro 功能冗余，选 Stable Diffusion 3 / Flux 更为合适。

对于同时需要两种能力的产品，Janus-Pro 现已成为参考的开放架构。

## 动手实践

`code/main.py` 模拟 Janus-Pro 的路由过程：

- 两个模拟编码器：类 SigLIP（输出 256 维语义向量）和类 VQ（输出整数编码）。
- 根据任务标签选择编码器的提示路由器。
- 一个共享主体（占位符）处理任何编码器产生的词元序列。
- 从阶段 1（对齐）到阶段 3（指令微调）的加权采样调度切换。

打印三个示例的路由路径：图像 QA、文到图、图像编辑。

## 产出技能

本课产出 `outputs/skill-decoupled-encoder-picker.md`：给定一个需要前沿质量的统一生成 + 理解产品，在 Janus-Pro、JanusFlow 和 InternVL-U 之间做出选择，并提供具体的数据规模建议。

## 练习

1. Janus-Pro-7B 在 GenEval 上超越 DALL-E 3。解释为何 7B 的开放模型能在生成上媲美前沿专有模型，但理解能力上还有差距。

2. 实现一个路由函数：给定提示文本，将其分类为 `understand` 或 `generate`。如何处理"描述并草绘"这类模糊提示？

3. JanusFlow 用整流流替换了 VQ 路径。Transformer 主体现在输出什么，损失发生了哪些变化？

4. 提出 Janus-Pro 架构可以处理的第四种任务，需要额外的一个解耦编码器。示例：图像分割（DINO 风格）、深度估计（MiDaS 风格）。

5. 阅读 Janus-Pro 第 4.2 节关于数据规模的内容。哪个数据阶段对 T2I 质量提升（相比原始 Janus）贡献最大？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 解耦编码（Decoupled encoding） | "两个视觉编码器" | 每个方向使用独立的分词器或编码器：理解用语义编码，生成用重建编码 |
| 共享主体（Shared body） | "一个 Transformer" | 单一 Transformer 处理任意编码器的输出；没有模态专用权重 |
| SigLIP 用于理解 | "语义特征" | 提供丰富概念特征但重建效果差的 CLIP 系视觉塔 |
| VQ 用于生成 | "重建编码" | 能够干净还原为像素的向量量化词元 |
| JanusFlow | "整流流变体" | 用连续流匹配生成头替换 VQ 的 Janus-Pro 版本 |
| 路由标签（Routing tag） | "任务标签" | 提示中的标记（`<understand>` / `<generate>`），用于选择输入编码器 |

## 延伸阅读

- [Wu 等 — Janus（arXiv:2410.13848）](https://arxiv.org/abs/2410.13848)
- [Chen 等 — Janus-Pro（arXiv:2501.17811）](https://arxiv.org/abs/2501.17811)
- [Ma 等 — JanusFlow（arXiv:2411.07975）](https://arxiv.org/abs/2411.07975)
- [InternVL-U（arXiv:2603.09877）](https://arxiv.org/abs/2603.09877)
- [Dong 等 — DreamLLM（arXiv:2309.11499）](https://arxiv.org/abs/2309.11499)
