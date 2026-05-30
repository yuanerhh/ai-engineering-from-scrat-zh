---
name: checkpointing-planner
description: 给定训练配置和 HBM 预算，为每层选择激活重计算策略（none / selective / full / offload）。
version: 1.0.0
phase: 10
lesson: 34
tags: [gradient-checkpointing, activation-recomputation, selective-checkpoint, fsdp-offload, training-memory]
---

给定训练配置（层数 L、隐藏层大小 d、序列长度 S、微批量 B、每值 dtype 字节数、注意力内核、张量并行度 TP、流水线并行度 PP、MoE 的专家并行度 EP）以及每 rank 在权重和优化器状态之后的 HBM 预算，输出：

1. 每层策略。对堆栈中每个层族（嵌入层、注意力层、FFN、MoE 专家、归一化层、输出头）选择 none、selective、full 或 offload。默认：S 超过 4096 时注意力层使用 selective；残差流和归一化层默认 none；仅当 FFN 激活的 PCIe 传输时间小于其重计算时间时，FFN 才默认 offload。
2. 分段大小 k。若启用 full 检查点，对均匀层成本选取 k = round(sqrt(L))；当激活内存主导预算时取较小 k。以前向 FLOPs 的 (1/k) 报告额外 FLOP 百分比。
3. FlashAttention 交互。确认注意力内核是否已重计算 softmax。若是，selective 注意力检查点收益有限；降级为 none。按名称说明内核（FlashAttention-2/3、xFormers memory-efficient、原始实现）。
4. TP / PP 方案。对 TP，说明重计算时需要 gather 或 rescatter 的激活，以及每步增加的通信字节数。对 PP，确认哪些流水线阶段端到端进行检查点，以便反向微批量在回流前释放激活内存。
5. 预算计算。预测策略执行前后的激活内存（每 rank 的 MB 数）。预测 FLOP 开销占前向+反向传播的百分比。拒绝任何在 HBM 预算下没有 10% 余量的方案。

当仅在注意力上做 selective 就能满足预算时，拒绝对每层做 full 检查点；性能分析表明，相同内存节省下 full 的 FLOP 开销远高于 selective，具体倍数与工作负载相关。当目标 PCIe 链路上的激活传输时间超过重计算时间时，拒绝 offload；重计算更合算。当所选框架不对 amax 历史进行快照时，拒绝 FP8 训练下的"全处检查点"；重计算会漂移缩放因子并静默污染梯度。

示例输入："L=64, d=8192, S=8192, B=1, bf16, FlashAttention-3, TP=8, PP=4, 权重后每 rank HBM 预算 32 GB，MoE 含 8 个专家，EP=8。"

示例输出：
- 每层策略：注意力 selective，FFN none，MoE 专家 full，嵌入层 none，输出头 offload。
- 分段大小：full 仅应用于 MoE，k=8；专家路径 FLOP 开销 12%，其他路径为 0。
- FlashAttention 交互：FA-3 已重计算 softmax；selective 应用于层封装器层面，而非内核内部。
- TP / PP 方案：重计算时 TP gather 注意力输入，每步额外通信 0.3 GB；PP 各阶段分别对整个前向过程做检查点；PP 第 3 阶段保留其激活用于最终反向传播。
- 预算计算：策略前激活 38 GB，策略后 11 GB。总 FLOP 开销 7.5%（前向+反向）。
