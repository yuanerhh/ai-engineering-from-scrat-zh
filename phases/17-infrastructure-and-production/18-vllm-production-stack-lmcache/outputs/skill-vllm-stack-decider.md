---
name: vllm-stack-decider
description: 根据工作负载和机队规模，决策 vLLM 部署布局——生产环境 Helm chart、KV 卸载（原生 CPU 或 LMCache）、路由器/可观测性集成。
version: 1.0.0
phase: 17
lesson: 18
tags: [vllm, production-stack, lmcache, kv-offload, connector-api]
---

给定工作负载（提示词形状、并发度、前缀复用模式）、机队（引擎数量、GPU 类型）及运维背景（Kubernetes 原生、多租户、预算），生成 vLLM 技术栈方案。

输出内容：

1. 技术栈。使用 vLLM 生产环境 Helm chart（推荐用于新部署）或自行搭建。说明适用的 operators/CRDs。
2. KV 卸载。选择：
   - 不卸载（短提示词、低并发——开销超过收益）。
   - 原生 vLLM CPU 卸载（单引擎 HBM 压力大、配置简单）。
   - LMCache connector（多引擎前缀复用、抢占频繁或多租户共享提示词场景）。
3. HBM 利用率监控。设置 `--gpu-memory-utilization`，预留余量；持续高于 92% 时告警，作为抢占前的预警信号。
4. 路由器集成。缓存感知路由器（Phase 17 · 11）。确认 KV 事件通道已配置。
5. 可观测性。每引擎 Prometheus 抓取，OTel GenAI 属性（Phase 17 · 13），使用生产环境 Helm chart 内置的 Grafana 仪表板模板。
6. 预期影响。量化与当前相比的预期吞吐量提升——参照 16x H100 基准测试形态（当 KV 占用超过 HBM 时，LMCache 效果显著）。

强制拒绝：
- 在没有共享前缀或抢占场景下部署 LMCache。拒绝——只有开销，没有收益。
- 运行 vLLM 时不监控 HBM 压力。拒绝——首次抢占将是意外事件。
- 在 Helm chart 能覆盖该用例时自行手动搭建生产环境技术栈。拒绝——无谓的重复造轮子。

拒绝规则：
- 若机队 <2 个引擎，拒绝 LMCache——跨引擎复用才是核心价值；单引擎使用原生卸载。
- 若工作负载提示词 <1K tokens 且并发 <100，拒绝任何形式的卸载——HBM 余量足够。
- 若团队不具备 Kubernetes 能力，拒绝生产环境 Helm chart——从单引擎 vLLM + 简单代理开始。

输出：一页方案，列明技术栈、KV 卸载选择、HBM 监控、路由器集成、可观测性、预期影响。最后给出唯一门禁指标：过去 24 小时 HBM 利用率 P99。
