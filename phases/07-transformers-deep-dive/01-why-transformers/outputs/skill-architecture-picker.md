---
name: sequence-architecture-picker
description: 根据序列长度、吞吐量和训练预算选择序列架构（RNN、Transformer、SSM、混合架构）。
version: 1.0.0
phase: 7
lesson: 1
tags: [transformers, architecture, rnn, ssm]
---

给定一个序列问题（最大长度、批次形状、训练 token 预算、推理延迟目标、设备类型），输出以下内容：

1. 主要架构。以下之一：Transformer、状态空间模型（Mamba/RWKV）、混合 SSM+注意力、RNN。一句话说明理由，与主要约束挂钩。
2. 上下文长度策略。如果是 Transformer：全注意力截止长度、滑动窗口大小、RoPE 缩放因子。如果是 SSM：扫描块大小。如果是 RNN：隐藏层宽度。
3. 训练 FLOP 分析。架构 + 上下文对应的每 token 近似 FLOP 量；说明该规格是否符合计算预算。
4. 推理内存分析。Transformer 的 KV 缓存、SSM 的状态大小、RNN 的每 token 内存。标记目标设备是否能容纳单批次大小为 1 的推理。
5. 风险说明。该选择在给定规格下的一个已知失败模式（例如：在没有 Flash Attention 的 24GB GPU 上，64K 上下文的 Transformer 会 OOM）。

拒绝为超过 10 亿 token 的训练任务推荐纯 RNN，除非明确说明梯度流和并行性惩罚。拒绝为超过 64K 上下文的任务推荐全注意力 Transformer，除非说明 `O(N^2)` 内存成本。拒绝为生产环境推荐发布不足 12 个月的全新架构，除非有具体的备选方案。
