---
name: mha-configurator
description: 为新 Transformer 推荐注意力头数、KV 头数和投影策略（MHA / MQA / GQA / MLA）。
version: 1.0.0
phase: 7
lesson: 3
tags: [transformers, attention, mha, gqa]
---

给定 Transformer 规格（参数预算、隐藏维度 `d_model`、目标上下文长度、推理设备内存、训练 vs 推理优先级），输出以下内容：

1. 投影变体。以下之一：MHA、GQA、MQA、MLA。一句话说明理由，与 KV 缓存约束挂钩。
2. 注意力头配置。`n_heads`、`n_kv_heads`、`d_head`。数值必须满足 `d_model = n_heads * d_head` 且 `n_heads % n_kv_heads == 0`。
3. KV 缓存估算。所选变体在目标上下文长度下每 token 每层的字节数（fp16）。标记一个批次是否超过目标设备内存。
4. 初始化。Q、K、V、O 矩阵的 Xavier / Kaiming 缩放。注明是否包含偏置项（大多数 2026 年的模型已去掉偏置）。
5. 可测试性钩子。一个合成任务（例如归纳头模式 `A B A ? → B`），训练好的该配置两层版本应能以 ≥95% 的概率解决该任务。

拒绝推荐 `d_head < 32`——注意力动态会退化。拒绝在上下文长度超过 32K 时推荐 `n_heads > 16` 的 MHA，除非明确评估 KV 缓存成本并建议使用 GQA 或 MLA。除非用户明确在基准测试，否则拒绝为低于 10 亿参数的模型建议 MLA。
