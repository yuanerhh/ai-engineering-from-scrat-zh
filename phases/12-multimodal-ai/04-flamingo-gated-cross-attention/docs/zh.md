# Flamingo 与少样本 VLM 的门控交叉注意力

> DeepMind 的 Flamingo（2022）率先完成了两件事：第一，它证明单个模型可以处理图像、视频与文本任意交错的序列；第二，它证明 VLM 可以进行上下文学习——给出三个（图像、描述）示例对作为少样本提示，模型无需任何梯度更新即可为新图像生成描述。实现机制是门控交叉注意力层，插入冻结 LLM 的现有层之间，以一个从零开始的可学习 tanh 门控，在初始化时保留 LLM 的文本能力。本课解读 Flamingo 的 Perceiver resampler 和门控交叉注意力架构——Gemini 交错输入和 Idefics2 视觉词元的共同前身。

**类型：** 学习
**编程语言：** Python（标准库，门控交叉注意力 + Perceiver resampler 演示）
**前置知识：** Phase 12 · 03（BLIP-2 Q-Former）
**预计时间：** 约 120 分钟

## 学习目标

- 解释门控交叉注意力如何通过 tanh(gate) = 0 在初始化时保留冻结 LLM 的文本能力。
- 逐步解读 Perceiver resampler：N 个图像块 → 通过交叉注意力得到 K 个固定"潜变量"查询。
- 描述 Flamingo 如何通过尊重图像位置的因果掩码处理图像-文本交错序列。
- 复现少样本多模态提示结构（3 个图文示例对，后接一张查询图像）。

## 问题背景

BLIP-2 将 32 个视觉词元送入冻结 LLM 的输入层，适用于每个提示包含一张图像的场景。但如果你想要将*多张*图像与文本交错输入，例如"这是图像 A，描述它；这是图像 B，描述它；现在这是图像 C，描述它"呢？LLM 的自注意力需要在单个序列中处理图像词元和文本词元，而哪些位置可以聚合哪些图像就变得复杂了。

Flamingo 的答案：完全不改变 LLM 的输入流。在现有 LLM 块之间插入额外的交叉注意力层，文本词元仍像往常一样流经 LLM 的因果自注意力。在每隔几个 LLM 块之后，文本词元还通过一个新的门控层对图像特征进行交叉注意力。门（初始化为零）意味着在第 0 步，新层是无操作的——模型表现得与预训练 LLM 完全相同。随着训练推进，门逐渐开放，视觉信息开始流入。

Flamingo 解答的第二个问题是：如何处理每个提示中可变数量的图像（0、1 或多张）？Perceiver resampler——一个小型交叉注意力模块，接受任意数量的块并生成固定数量的视觉潜变量词元。无论提示中有多少张图像，LLM 交叉注意力层看到的形状始终相同。

## 核心概念

### 冻结的 LLM

Flamingo 从一个冻结的 Chinchilla 70B LLM 开始，全部 700 亿权重不变，现有的文本自注意力和 FFN 正常运行。

### Perceiver resampler

对于提示中的每张图像，ViT 产生 N 个块词元。Perceiver resampler 有 K 个固定可学习潜变量（Flamingo 使用 K=64）。每个 resampler 块分两个子步骤：

1. 交叉注意力：K 个潜变量对 N 个块词元做注意力（Q 来自潜变量，K/V 来自块）。
2. 潜变量内部的自注意力 + FFN。

经过 6 个 resampler 块后，输出是 K=64 个维度为 1024 的视觉词元，与 ViT 产生的块数量无关。一张 224×224 图像（196 个块）和一张 480×480 图像（900 个块）都输出为 64 个 resampler 词元。

对于视频，resampler 在时间维度上应用：每帧的块产生 64 个潜变量，时间位置编码让模型区分 t=0 和 t=N。整个视频变成 T×64 个视觉词元。

### 门控交叉注意力

在冻结 LLM 的每 M 层之间（Flamingo 使用 M=4），插入一个新的门控交叉注意力块：

```
x_after_llm_block = llm_block(x_before)
cross = cross_attn(x_after, resampler_output)
gated = tanh(alpha) * cross + x_after
x_before_next_block = gated
```

- `alpha` 是一个初始化为零的可学习标量。
- `tanh(0) = 0`，因此在初始化时门控分支贡献为零。
- 随着 `alpha` 偏离零，交叉注意力的贡献平滑增加。
- 残差连接意味着即使门完全开放也不会覆盖 LLM 的文本表示，只是叠加了视觉信息。

这是 Flamingo 中最重要的设计决策：视觉条件化是加性的、有门控的，且在初始化时为零。第 0 步的 Flamingo 在纯文本输入上表现得与完美的 Chinchilla 70B 完全相同。

### 交错输入的掩码交叉注意力

在形如"<图像 A> 描述 A <图像 B> 描述 B <图像 C> ?"的提示中，每个文本词元应该只能看到序列中它之前出现的图像。交叉注意力掩码强制执行：位置 `t` 处的文本词元只聚合图像索引 `i < i_t` 的图像 resampler 词元，其中 `i_t` 是位置 `t` 之前最近一张图像的索引。"仅看最后一张前置图像"或"看所有前置图像"都是有效选择；Flamingo 选择了前者。

### 上下文少样本学习

Flamingo 提示的形式如下：

```
<image1> 这是一张猫的照片。<image2> 这是一张狗的照片。<image3> 这是一张
```

模型看到完形填充模式并输出"鸟"（或 image3 中实际显示的内容）。无需梯度更新。冻结 LLM 的上下文学习能力通过门控交叉注意力传递——这是这篇论文的精髓，也是它重要性的所在。

### 训练数据

Flamingo 在三个数据集上训练：

1. MultiModal MassiveWeb（M3W）：4300 万个带有交错图像和文本的网页，按阅读顺序重建。
2. 图文配对（ALIGN + LTIP）：44 亿对。
3. 视频-文本配对（VTP）：2700 万个短视频片段。

OBELICS（2023）是对交错网络语料库的开放复现，Idefics、Idefics2 和大多数开放的"类 Flamingo"模型都在其上训练。

### OpenFlamingo 与 Otter

OpenFlamingo（2023）是开放复现版本，架构完全相同（Perceiver resampler + 冻结 LLaMA 或 MPT 上的门控交叉注意力），在 3B、4B、9B 规模提供检查点，由于基础 LLM 较小且数据量少，质量不如 Flamingo。

Otter（2023）在 OpenFlamingo 基础上，用 MIMIC-IT（多模态指令数据集）进行指令微调，证明门控交叉注意力也适用于指令跟随。

### 后代模型

- Idefics / Idefics2 / Idefics3：Hugging Face 的门控交叉注意力系列，逐渐简化（Idefics2 放弃了 resampler，改用带自适应池化的直接块词元）。
- 从 Flamingo 到 Chameleon 的过渡：到 2024 年许多团队转向早期融合（12.11 课）；在需要冻结骨干的情况下，Flamingo 风格的门控交叉注意力仍在生产中使用。
- Gemini 的交错输入：概念上继承了 Flamingo 的交错格式灵活性，尽管具体机制是专有的。

### 与 BLIP-2 的对比

| | BLIP-2 | Flamingo |
|---|---|---|
| 视觉桥接 | 输入层一次性 Q-Former | 每 M 层的门控交叉注意力 |
| 每张图像的视觉词元 | 32 | 每个交叉注意力层 64 |
| 冻结 LLM | 是 | 是 |
| 上下文少样本学习 | 较弱 | 强——论文的核心 |
| 交错输入 | 无原生支持 | 是，设计目标 |
| 训练数据 | 1.3 亿对 | 13 亿对 + 4300 万交错页面 |
| 参数量 | 1.88 亿训练 | 约 100 亿训练（交叉注意力层） |
| 计算成本 | 8 块 A100 数天 | 数千块 TPUv4 数周 |

在预算有限的单图像 VQA 场景选 BLIP-2；在交错、少样本或多图像推理场景选 Flamingo/Idefics2。

## 动手实践

`code/main.py` 演示了：

1. 对 36 个虚假块词元使用 8 个可学习潜变量的 Perceiver resampler（纯 Python 交叉注意力）。
2. 门控交叉注意力步骤：`alpha = 0` 时输出等于输入（LLM 不变），`alpha = 2.0` 时视觉贡献混入。
3. 交错掩码构建器：为"(图像 1)(文本 1)(图像 2)(文本 2)"序列生成二维注意力掩码。

## 产出技能

本课产出 `outputs/skill-gated-bridge-diagnostic.md`：给定开放 VLM 的配置（有无 resampler、交叉注意力频率、门控方案），识别 Flamingo 系的元素并解释冻结策略。在调试微调导致文本性能下降时很有用（答案：门开得太快太大）。

## 练习

1. 计算 Flamingo-9B 的视觉参数量：9B LLM + 1.4B 门控交叉注意力层 + 64M resampler。训练参数占总参数的比例是多少？

2. 用 PyTorch 实现门控残差 `y = tanh(alpha) * cross + x`。通过实验证明，`alpha=0` 时 `y==x` 在初始化时精确成立。

3. 阅读 OpenFlamingo 第 3.2 节（arXiv:2308.01390），了解他们如何在每个提示图像数量不同时处理一批数据。描述填充策略。

4. 为什么 Flamingo 的交叉注意力掩码让文本词元只聚合*最近一张*前置图像，而不是所有前置图像？阅读 Flamingo 论文第 2.4 节并解释权衡。

5. 上下文少样本学习：为一个新 Flamingo 变体构建一个包含 4 个"图像 → 主要物体颜色"示例的提示。描述将示例数量从 0 变化到 8 时预期的准确率变化规律。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Perceiver resampler | "固定潜变量交叉注意力" | 从可变数量的输入块生成 K 个固定词元的模块 |
| 门控交叉注意力 | "tanh 门控桥接" | 残差层 `y = tanh(alpha)*cross + x`，可学习 alpha，初始化为 0 |
| 交错输入（Interleaved input） | "混合序列" | 图像与文本在阅读顺序中自由混合的提示格式 |
| 冻结 LLM（Frozen LLM） | "无 LLM 梯度" | 文本 LLM 的权重不更新；只有 resampler + 交叉注意力层训练 |
| 少样本（Few-shot） | "上下文示例" | 在提示中给出几个（图像，答案）对；模型无需微调即可泛化 |
| OBELICS | "交错网络语料库" | 包含图像和文本按阅读顺序排列的 1.41 亿网页的开放数据集 |
| Chinchilla | "70B 冻结基础模型" | Flamingo 的冻结文本 LLM，来自 DeepMind 的 Chinchilla 论文 |
| 门控调度（Gate schedule） | "alpha 的变化方式" | 训练过程中交叉注意力门开放的速率 |
| 交叉注意力频率 | "每 M 层" | 门控交叉注意力块插入的频率；Flamingo 使用 M=4 |
| OpenFlamingo | "开放复现" | MosaicML/LAION 的 3-9B 开放检查点；与 Flamingo 架构相同 |

## 延伸阅读

- [Alayrac 等 — Flamingo（arXiv:2204.14198）](https://arxiv.org/abs/2204.14198) — 原始论文。
- [Awadalla 等 — OpenFlamingo（arXiv:2308.01390）](https://arxiv.org/abs/2308.01390) — 开放复现。
- [Laurençon 等 — OBELICS（arXiv:2306.16527）](https://arxiv.org/abs/2306.16527) — 交错网络语料库。
- [Jaegle 等 — Perceiver IO（arXiv:2107.14795）](https://arxiv.org/abs/2107.14795) — 通用 Perceiver 架构。
- [Li 等 — Otter（arXiv:2305.03726）](https://arxiv.org/abs/2305.03726) — 指令微调的 Flamingo 后代。
- [Laurençon 等 — Idefics2（arXiv:2405.02246）](https://arxiv.org/abs/2405.02246) — Flamingo 方案的现代简化版。
