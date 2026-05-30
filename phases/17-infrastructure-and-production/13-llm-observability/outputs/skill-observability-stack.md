---
name: observability-stack
description: 根据技术栈、规模、预算和许可证立场，选择 LLM 可观测性技术栈（开发平台 + 网关 + 可选规模层），并定义 OpenTelemetry GenAI 属性集。
version: 1.0.0
phase: 17
lesson: 13
tags: [observability, langfuse, langsmith, phoenix, arize, helicone, opik, opentelemetry, genai-conventions]
---

给定技术栈（LangChain / DSPy / 原生 SDK）、规模（traces/天）、预算、许可证立场（仅 MIT 或商业可接受）以及自托管需求，生成可观测性方案。

输出内容：

1. 开发平台选择。Langfuse（OSS）、LangSmith（LangChain 优先商业版）、Opik（Comet OSS）或不选。结合技术栈和许可证加以说明。
2. 网关/遥测选择。Helicone（代理 + 网关）、SigNoz（完整 APM）、OpenLLMetry（纯 OTel）。若已使用 AI 网关（Phase 17 · 19），说明集成方式。
3. 规模/数据湖层。可选；Arize AX 或原生 Iceberg 用于长期分析，Phoenix 用于 RAG 漂移检测。
4. OTel GenAI 规范。指定最小属性集：`gen_ai.system`、`gen_ai.request.model`、`gen_ai.usage.input_tokens`、`gen_ai.usage.output_tokens`、`gen_ai.request.temperature`、`gen_ai.response.finish_reasons`，以及组织专属属性（tenant_id、user_id、task）。
5. 采样策略。100% 错误，100% 高成本（>$0.10/次调用），N% 成功采样率。原始数据保留窗口（14天/30天/90天），聚合数据保留时间更长。
6. 告警。必须设置告警的五个指标：错误率、P99 TTFT、每次请求成本、提示词缓存命中率、拒绝率。

强制拒绝：
- 在框架专属 SDK 内埋点却没有 OTel 回退方案。拒绝——会造成框架锁定。
- 对非受监管工作负载以 Datadog 级别定价（>$500/月）保留 100% traces。拒绝——建议采样。
- 忽略 OpenTelemetry GenAI 规范。拒绝——2026 年互操作性要求必须遵循。

拒绝规则：
- 若 traces/天 > 5M 且团队坚持全量 Datadog 保留，拒绝，除非提供成本预测。
- 若团队仅接受 MIT 许可却选择 LangSmith，拒绝——Langfuse 是 MIT 等效替代。
- 若团队没有 AI 网关，同时将 Helicone 用作网关和可观测性，接受——代理在约 500 RPS 以下可兼作网关使用（Phase 17 · 19 覆盖网关规模问题）。

输出：一页方案，列明开发平台、网关、规模层（如有）、OTel 属性集、采样规则、五个告警。最后给出唯一指标，用于检测技术栈漂移：过去 7 天内携带完整 OTel GenAI 属性的 LLM 调用占比。
