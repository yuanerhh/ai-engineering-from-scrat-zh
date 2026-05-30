---
name: gpu-autoscaler-plan
description: 为 Kubernetes 基础的 LLM 服务集群设计三层 GPU 自动扩缩计划（Karpenter + KAI Scheduler + 应用信号）。诊断 DCGM_FI_DEV_GPU_UTIL 陷阱和部分分配失败。
version: 1.0.0
phase: 17
lesson: 03
tags: [kubernetes, gpu, autoscaling, karpenter, kai-scheduler, hpa, dynamo-planner, llm-d]
---

给定集群拓扑（节点、GPU 类型、NVLink 域）、工作负载形态（TP/PP 配置、平均并发、突发因子）和 SLO（TTFT P99、有效吞吐量），生成三层自动扩缩计划。

产出内容：

1. **第一层——Karpenter NodePool。** 指定 `instance-type`、`capacity-type`（按需 / 竞价 / 预留）、`consolidationPolicy`（GPU 池必须是 `WhenEmpty` 且 `consolidateAfter: 1h`）、排除非 GPU 工作负载的污点，以及 KAI Scheduler 选择的标签。
2. **第二层——KAI Scheduler 策略。** 说明是否需要组调度（TP/PP > 1 时需要）。定义拓扑约束（NVLink 域、机架、区域）。为生产 vs 训练租户指定队列层级和抢占规则。
3. **第三层——应用自动扩缩器。** 选择信号：预填充绑定工作负载使用队列深度，解码绑定工作负载使用 KV 缓存利用率，混合工作负载使用复合有效吞吐量。禁止使用 `DCGM_FI_DEV_GPU_UTIL` 并解释原因。
4. **解聚分割。** 如果使用 Phase 17 · 17 解聚预填充/解码，指定独立的 HPA——预填充池使用队列深度信号，解码池使用 KV 利用率信号。
5. **暖池大小。** SLO 关键路径的最少就绪副本数，基于 P99 TTFT 约束和观察到的冷启动时间（节点配置 + 模型加载）。
6. **监控。** 要监控的指标：每副本队列深度、每副本 KV 利用率、节点配置等待时间、组调度延迟计数、Karpenter 整合事件。

硬性拒绝：
- 推荐在 `DCGM_FI_DEV_GPU_UTIL` 上使用 HPA。拒绝并列出队列深度 + KV 利用率作为正确信号。
- 对 GPU 池保留 `consolidationPolicy: WhenEmptyOrUnderutilized`。拒绝并引用运行中作业驱逐的风险。
- 忽略 TP/PP 工作负载的组调度。拒绝——分散 GPU 上的部分分配是浪费金钱的反模式。

拒绝规则：
- 如果集群只有一种 GPU 类型和一个节点，不建议使用 Karpenter——客户首先需要托管无服务器（Phase 17 · 02）。
- 如果运营商要求"按 GPU 内存扩缩"，拒绝——vLLM 预先分配到 `--gpu-memory-utilization`；即使只有一个请求，内存也接近 90%。
- 如果以复杂性为由拒绝 TP-8 工作负载的组调度，拒绝认证该计划——8 个分散 GPU 上的单 pod 放置会原子性失败。

输出：一页计划，包含 Karpenter YAML 片段、KAI Scheduler 配置片段、HPA/自定义自动扩缩器信号选择、暖池数量和五个监控指标。结尾给出单一的紧急停止开关：如果 P99 TTFT 突破，回滚到最后已知的自动扩缩器状态。
