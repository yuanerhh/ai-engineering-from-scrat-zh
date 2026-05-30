---
name: positional-encoding-picker
description: 根据上下文长度和训练预算选择位置编码方式（RoPE、ALiBi、正弦）及扩展策略。
version: 1.0.0
phase: 7
lesson: 4
tags: [transformers, positional-encoding, rope, alibi]
---

给定 Transformer 规格（推理时目标上下文长度、训练时上下文长度、外推需求、微调 token 预算），输出以下内容：

1. 基础编码。以下之一：RoPE、ALiBi、正弦式、学习绝对位置编码。一句话说明理由。
2. 超参数。如果是 RoPE：`base` 值、`d_head` 的偶数分割要求。如果是 ALiBi：斜率公式。如果是正弦式：`max_len`。
3. 扩展策略。如果目标长度 > 训练长度：NTK-aware 缩放因子、YaRN 配置、LongRoPE 规格或位置插值比例。说明微调 token 预算。
4. 测试方案。目标上下文长度下的 NIAH（大海捞针）通过率目标，与训练长度基线相比的困惑度差异。
5. 备选方案。如果长上下文评估失败应怎么做：用更大的 `base` 重新训练、切换到 ALiBi 或限制部署上下文长度。

拒绝在 2026 年为新模型推荐正弦式或学习绝对位置编码——它们不能外推，且现代所有框架都假设使用 RoPE 或 ALiBi。拒绝在没有微调阶段的情况下将 RoPE 缩放超过训练长度的 8 倍。拒绝在未对完整部署长度运行 NIAH 测试的情况下发布长上下文配置。
