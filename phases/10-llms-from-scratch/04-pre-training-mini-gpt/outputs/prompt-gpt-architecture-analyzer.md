---
name: prompt-gpt-architecture-analyzer
description: 分析任何 GPT 风格 Transformer 模型的架构选择
version: 1.0.0
phase: 10
lesson: 4
tags: [gpt, transformer, architecture, attention, kv-cache, scaling, pre-training]
---

# GPT 架构分析器

在从技术报告、模型卡片或训练日志评估 GPT 风格模型时，使用此框架分解架构并识别设计权衡。

## 分析流程

### 1. 参数分配分解

计算每个组件的精确参数数量：

- **Token 嵌入**：vocab_size × embed_dim
- **位置嵌入**：max_seq_len × embed_dim
- **每块注意力**：4 × embed_dim × embed_dim（Q、K、V、输出投影）
- **每块 FFN**：2 × embed_dim × ff_dim + embed_dim + ff_dim（两个线性层 + 偏置）
- **每块 LayerNorm**：4 × embed_dim（两个归一化，每个有缩放 + 偏置）
- **最终 LayerNorm**：2 × embed_dim
- **输出头**：vocab_size × embed_dim（如果与 token 嵌入权重绑定则为 0）

如果任何单个组件超过总参数的 40% 则标记。嵌入矩阵在小模型中占主导地位。注意力和 FFN 在大模型中占主导地位。

### 2. 注意力设计分析

评估注意力配置：

- **头部维度**：embed_dim / num_heads。标准是 64（GPT-2）或 128（Llama 3）。低于 32 限制了每头的表达能力。高于 128 浪费计算而收益甚微。
- **每层头数**：更多头 = 更多样的注意力模式，但 KV 缓存内存更多。
- **分组查询注意力（GQA）**：模型是否在多个 Q 头之间共享 K/V 头？Llama 3 对 32 个 Q 头使用 8 个 KV 头的 GQA。这将 KV 缓存减少了 4 倍。
- **上下文长度**：最大位置嵌入数。RoPE 允许外推到训练长度之外。绝对位置嵌入不能。

### 3. 内存预算

在模型最大上下文长度下进行推理的内存：

- **权重（FP16）**：total_params × 2 字节
- **KV 缓存（FP16）**：2 × num_layers × num_kv_heads × head_dim × max_seq_len × 2 字节
- **激活值**：batch_size × seq_len × embed_dim × 2 字节 × num_layers（近似）

如果 KV 缓存超过权重内存则标记。这发生在长上下文模型（128K+）中，表明模型在解码时受内存限制。

### 4. 计算特征

- **每 token 预填充 FLOP**：约 2 × total_params（每个参数一次矩阵乘法，前向传播）
- **每 token 解码 FLOP**：与预填充相同，但针对单个 token
- **预填充瓶颈**：受计算限制（GPU TFLOP）
- **解码瓶颈**：受内存限制（GPU 内存带宽）
- **算术强度**：每访问字节的 FLOP 数。低于 100 = 受内存限制。

### 5. 缩放决策

对照已知缩放定律评估：

- **Chinchilla 最优**：对于给定的计算预算 C，最优模型大小 N 和 token 数 D 满足 N ~ D（大致相等缩放）。7B 模型需要约 1400 亿 token。
- **Llama 3 过度训练**：Meta 在 1.5 万亿 token 上训练了 Llama 3 8B（Chinchilla 最优的 100 倍）。在更多数据上过度训练小模型可以获得更好的每 token 推理成本。
- **宽度 vs 深度**：更深的模型（更多层）通常比更宽的模型（更大的 embed_dim）在相同参数数量下样本效率更高。

## 红色警报

- **FFN 比例不是 4×**：标准是 ff_dim = 4 × embed_dim。Llama 使用 SwiGLU 时为 8/3 × embed_dim。偏差应有理由。
- **无权重绑定**：输出头应与 token 嵌入共享权重，除非 vocab_size 相对于 embed_dim 非常大。
- **13B 以上无 GQA**：13B 以上没有分组查询注意力的模型 KV 缓存会过于庞大。
- **长上下文无 RoPE**：绝对位置嵌入无法外推到训练长度之外。目标上下文 32K+ 的模型应使用旋转位置嵌入。
- **模型大小对应的学习率过高**：更大的模型需要更低的峰值学习率。GPT-2 Small 使用 6e-4。Llama 3 405B 使用 8e-5。

## 输出格式

1. **参数表格**：逐组件参数数量及百分比
2. **内存预算**：最大上下文长度下的权重、KV 缓存和激活值内存
3. **计算特征**：A100/H100 上的预填充和解码吞吐量估算
4. **设计评估**：模型的正确之处和非标准之处
5. **缩放判断**：模型大小是否与其训练数据相匹配
