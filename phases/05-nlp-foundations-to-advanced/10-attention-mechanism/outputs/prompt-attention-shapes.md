---
name: attention-shapes
description: 调试注意力实现中的形状错误。
phase: 5
lesson: 10
---

给定一个有问题的注意力实现，识别形状不匹配。输出：

1. 哪个矩阵形状错误。说明张量名称。
2. 其应有的形状，由 `(d_s, d_h, d_attn, T_enc, T_dec, batch_size)` 推导得出。
3. 一行修复方案。转置、reshape 或投影。
4. 用于捕获回归的测试。通常断言 `output.shape == (batch, T_dec, d_h)` 且 `weights.shape == (batch, T_dec, T_enc)` 且 `weights.sum(dim=-1)` 接近 1。

拒绝推荐会静默广播的修复方案。广播隐藏的 bug 会在后续表现为静默的精度下降。

对于 Bahdanau 的混淆，坚持解码器输入为 `s_{t-1}`（步骤前状态）。对于 Luong，为 `s_t`（步骤后状态）。点积注意力中最常见的初次错误是查询/键维度不匹配——需明确标记。
