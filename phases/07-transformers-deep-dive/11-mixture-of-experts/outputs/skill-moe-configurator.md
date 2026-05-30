---
name: moe-configurator
description: 为新的 MoE Transformer 选择专家数量、top-k、负载均衡策略和共享专家布局。
version: 1.0.0
phase: 7
lesson: 11
tags: [transformers, moe, mixture-of-experts, scaling]
---

给定 Transformer 规格（总参数预算、每 token 期望激活参数、可用训练 token、推理硬件），输出以下内容：

1. MoE 布局。`n_experts`、`top_k`、`n_shared`。前沿规模选择细粒度（256+ 专家，top-8）；较小规模选择经典（8 专家，top-2）。一句话说明理由。
2. 负载均衡策略。无辅助损失（DeepSeek-V3，默认）、Switch 风格辅助损失或专家容量 + token 丢弃。如果使用无辅助损失，说明 `γ` 值。
3. 专家并行方案。如何根据显存在 GPU 间分片专家。说明每个专家的显存开销和所需总 GPU 数量。
4. 路由精度。fp32 路由器分数 vs fp16。路由精度在大规模时至关重要。
5. 失败模式检查。具体风险：路由崩溃、专家饥饿、all-to-all 网络瓶颈、路由开销导致的推理延迟、检查点内存占用。

拒绝在激活参数低于 40 亿时推荐 MoE——相同计算量下密集模型更好。拒绝在 2026 年新项目中只使用辅助损失均衡（无辅助损失已成为默认方案）。拒绝在总参数超过 80 GB 时没有专家并行方案的 MoE 部署。将 MoE 用于延迟敏感的单用户路径时发出警告——可能比等效密集模型更慢。
