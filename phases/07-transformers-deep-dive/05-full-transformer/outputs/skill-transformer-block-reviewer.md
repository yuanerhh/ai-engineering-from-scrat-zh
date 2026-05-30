---
name: transformer-block-reviewer
description: 对照 2026 年默认标准审查 Transformer 模块实现并标记偏差。
version: 1.0.0
phase: 7
lesson: 5
tags: [transformers, architecture, review]
---

给定 Transformer 模块源码（PyTorch / JAX / numpy / 伪代码）及其预期角色（编码器 / 解码器 / 编码器-解码器），输出以下内容：

1. 连接检查。前置归一化还是后置归一化。每个子层周围的残差连接。将后置归一化标记为 2026 年非默认设置，除非作者说明理由。
2. 归一化。LayerNorm vs RMSNorm。优先使用 RMSNorm。标记 Q/K/V/O 投影中存在偏置项的情况——大多数 2026 年的模型已去掉偏置。
3. 注意力形状。MHA / GQA / MQA / MLA。对于解码器模块：确认因果遮掩已应用。对于交叉注意力：确认 Q 来自解码器，K/V 来自编码器。
4. 前馈网络。激活函数（ReLU / GELU / SwiGLU / GeGLU）。扩展比例。SwiGLU 约 2.67× 是现代默认值；4× ReLU/GELU 是经典配置。
5. 位置信号。确认 RoPE / ALiBi / 绝对位置编码在预期位置应用（RoPE 通常应用于 Q、K 投影）。

拒绝批准超过 12 层后置归一化且没有预热学习率调度的模块——训练会发散。拒绝没有因果遮掩的解码器模块。标记 FFN 扩展比例低于 2× 的模块，可能容量不足。如果模块硬编码 `d_model` 而没有可配置字段，发出警告。
