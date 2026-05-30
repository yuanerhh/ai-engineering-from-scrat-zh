---
name: dualpipe-planner
description: 为训练集群规划流水线并行策略（1F1B、Zero Bubble、DualPipe、DualPipeV）。
version: 1.0.0
phase: 10
lesson: 19
tags: [pipeline-parallelism, dualpipe, dualpipev, zero-bubble, expert-parallelism, distributed-training]
---

给定训练集群规格（总 GPU 数量、互连拓扑、加速器型号、每 GPU 显存）、模型形态（总参数量、活跃参数量、MoE 或密集型、预期层数）以及目标训练数据量，推荐流水线并行策略并确认预期气泡比例。

输出：

1. 流水线深度 P。依据 GPU 显存预算（每个 rank 必须能容纳一个流水线阶段）、MoE 与密集型，以及互连带宽进行选择。范围：小集群为 4，前沿 MoE 训练为 16-32。
2. 微批量数 M。DualPipe 和 DualPipeV 要求 M 为 2 的倍数。典型比例 M/P 在 8 到 16 之间。依据梯度累积目标和目标序列长度下的激活内存进行论证。
3. 调度方案选择。从 1F1B、Zero Bubble、DualPipe、DualPipeV 中选择。决策表：500 GPU 以下的密集训练 → Zero Bubble；带专家并行的 MoE → DualPipe；500 GPU 以上无大量 all-to-all 的密集训练 → DualPipeV；100 GPU 以下的小规模训练 → 1F1B 即可。
4. 预期气泡比例。在目标 P 和 M 下计算所选调度的气泡比例。以百分比表示，并以总训练预算下相对于 1F1B 节省的绝对 GPU 小时数表示。
5. 参数复制方案（仅 DualPipe）。确认 2 倍参数复制在可用 VRAM 中可容纳。报告给定所选 P 时每 GPU 的有效参数密度。

强拒绝：
- 没有专家并行的 DualPipe。没有 EP 密集通信需要隐藏时，2 倍复制不合理。
- 任何训练运行中 P > 64。无论何种调度，气泡比例随 P 线性增长。
- DualPipe/DualPipeV 的微批量数不能被 2 整除。调度将无法闭合。
- 当模型可以放入单 GPU 显存时使用流水线并行。仅使用数据并行即可。

拒绝规则：
- 若每 GPU 互连速度为 200Gbps 或更低，拒绝 DualPipe 并推荐 DualPipeV。all-to-all 重叠窗口太窄，不足以证明复制的合理性。
- 若用户无法为其集群拓扑提供自定义 all-to-all 内核，推荐 Zero Bubble 而非 DualPipe。
- 若训练运行低于 10 亿 token，完全拒绝流水线并行规划，推荐数据并行加张量并行。

输出：一页方案，列出 P、M、调度、预期气泡比例、参数复制成本（如为 DualPipe）和 all-to-all 内核推荐。最后附一段"回滚触发条件"，说明若在前 1000 步内聚合 GPU 利用率未达到目标值，则应切换至更简单调度的具体利用率指标。
