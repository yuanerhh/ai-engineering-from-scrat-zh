---
name: disaggregation-decider
description: 针对给定工作负载和集群，决策是否采用解耦预填充/解码（Dynamo 或 llm-d）。量化预填充:解码比率、KV 传输成本及预期节省。
version: 1.0.0
phase: 17
lesson: 17
tags: [disaggregated-serving, dynamo, llm-d, nixl, kv-transfer, prefill-decode]
---

给定工作负载概况（提示词/输出长度分布、模型、并发度）、集群拓扑（GPU、网络架构、RDMA 可用性）及当前服务成本，生成解耦决策报告。

输出内容：

1. 是否解耦？是/否，附编号说明。基线条件：提示词 >512 tokens 且输出 >200 tokens。网络架构：有 RDMA 有利；仅 TCP 会延长盈亏平衡点。
2. 技术栈选择。NVIDIA Dynamo（vLLM/SGLang/TRT-LLM 上层的托管编排器）或 llm-d（Kubernetes 原生 Services）。结合运维背景加以匹配。
3. 预填充:解码比率。使用 Dynamo Planner Profiler 读数，或根据工作负载形状计算（预填充 TFLOPS vs 解码 bytes/sec）。示例：RAG 密集型场景为 2 预填充 : 1 解码；输出密集型场景为 1:2。
4. KV 传输方案。指定传输方式（InfiniBand 上的 NIXL / RDMA / TCP 回退）。计算提示词 P99 下的单请求传输开销。
5. 路由器集成。缓存感知路由器（Phase 17 · 11）必须置于前端——不带前缀匹配的解耦会丧失缓存收益。
6. 预期节省。与并置基线相比进行计算；引用已发布案例（相同 SLA 下节省 30-40%）。

强制拒绝：
- 对短提示词工作负载（<512 tokens）进行解耦。拒绝——传输开销会占主导。
- 部署时不配置缓存感知路由器。拒绝——盲目路由会抵消 KV 局部性收益。
- 忽略拓扑结构（机架布局）。拒绝——多机架跨跳的 KV 传输成本高于同机架 RDMA。

拒绝规则：
- 若集群 GPU 数量 < 4，拒绝——池多样性不足，解耦无法带来回报。
- 若无 RDMA/InfiniBand 且无规划，注明 TCP 将盈亏平衡提示词长度提高至 >2K；重新评估。
- 若团队无法运维两个具备独立角色扩缩的 GPU 池，拒绝 llm-d，要求使用 Dynamo 作为托管替代方案。

输出：一页决策报告，列明解耦 Y/N、技术栈选择、比率、传输方式、路由器、预期节省。最后给出唯一验证指标：KV 传输 P99 延迟；超过方案规定阈值时设置门禁告警。
