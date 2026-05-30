---
name: transformer-review
description: 对照第 7 阶段 13 节课程审查从头实现的 Transformer。
version: 1.0.0
phase: 7
lesson: 14
tags: [transformers, review, capstone]
---

给定一个从头实现的 Transformer 代码库（PyTorch / JAX），对照 2026 年默认标准进行审查并标记缺失或错误的部分：

1. 注意力。因果遮掩存在。按 `sqrt(d_head)` 缩放。多头分割正确。如果可用，使用了 Flash Attention。如果 d_model ≥ 1024，提到 GQA。
2. 位置编码。RoPE（2026 年首选）或学习绝对位置编码（小模型可接受）。将正弦式标记为历史遗留。
3. 模块连接。前置归一化（非后置归一化）。RMSNorm（非 LayerNorm）。SwiGLU 前馈网络（非 ReLU/GELU）。每个子层周围有残差连接。线性层无偏置项（现代默认）。
4. 训练。AdamW（或 2026 年起的 Muon）、带线性预热的余弦学习率调度、梯度裁剪为 1.0、bf16 自动混合精度。token 嵌入与 lm_head 之间的权重绑定。
5. 损失。每个位置的移位一位交叉熵。如有填充则遮掩。在固定间隔记录训练和验证损失。

拒绝批准任何包含以下情况的代码库：没有明确理由的后置归一化、2026 年生产代码中没有理由使用 LayerNorm、解码器自注意力中缺少因果遮掩、小型语言模型中嵌入未绑定。标记：无验证集划分、无梯度裁剪、没有预热的学习率 > 1e-3，或块大小超过位置编码范围而无回退。建议运行 `python code/main.py` 端到端检查，确认 nano 配置在 tinyshakespeare 上的最终验证损失低于 2.5。
