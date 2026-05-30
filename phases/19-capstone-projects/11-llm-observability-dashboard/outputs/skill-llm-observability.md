---
name: llm-observability
description: 构建一个自托管 LLM 可观测性仪表板，能够摄取 OpenTelemetry GenAI span、运行评估，并在五分钟内捕获注入的回退。
version: 1.0.0
phase: 19
lesson: 11
tags: [capstone, observability, otel, langfuse, phoenix, evals, drift, clickhouse]
---

给定跨至少六个 SDK 家族（OpenAI、Anthropic、Google GenAI、LangChain、LlamaIndex、vLLM）的生产 LLM 流量，部署一个自托管的可观测性平面，能够摄取 OTLP GenAI 语义约定 span、运行评估、检测漂移并发出告警。

构建计划：

1. OpenTelemetry Collector，配置 OTLP HTTP 接收器、尾部采样处理器（保留 100% 错误、10% 成功、100% 高毒性/PII），以及到 ClickHouse + S3 的导出器。
2. ClickHouse span schema 镜像 GenAI 语义约定：gen_ai.system、gen_ai.request.model、usage.input/output_tokens、latency_ms、user_id、app_id，以及用于提示/补全的 JSON 包。
3. Postgres 元数据存储，包含应用、用户、会话、标注队列。
4. 在每个 SDK 家族的客户端应用上使用 OpenLLMetry 自动插桩；验证规范 span 落地。
5. DeepEval + RAGAS + Phoenix 评估器包，按计划对采样追踪运行；针对 PII 和违反策略的内容使用自定义 LLM 裁判。
6. 每周对汇总提示嵌入进行 PSI / KL 漂移检测；告警阈值 0.2。
7. 用于评估分数聚合和延迟百分位数的 Prometheus 导出器；Alertmanager 接入 Slack（警告）+ PagerDuty（严重）。
8. Next.js 15 App Router 仪表板：概览、追踪搜索 + 瀑布图、评估趋势、漂移图表、告警。
9. 回退探测：注入 1% 的时间泄漏虚假 SSN 的响应模式；测量 MTTR（告警触发时间）。

评估标准：

| 权重 | 评估项 | 度量方式 |
|:-:|---|---|
| 25 | 追踪 schema 覆盖率 | 生成规范 GenAI span 的 SDK 家族数量（目标 6+） |
| 20 | 评估正确性 | DeepEval / RAGAS 分数与人工标注集的对比 |
| 20 | 仪表板用户体验 | 注入回退的 MTTR（目标低于 5 分钟） |
| 20 | 成本/规模 | 在无积压情况下持续摄取 1k span/秒 |
| 15 | 告警 + 漂移检测 | Prometheus/Alertmanager 链路端到端验证 |

硬性拒绝条件：

- 使用 OpenTelemetry GenAI 语义约定以外的属性名称的 span schema。
- 丢弃错误的尾部采样策略（这是已知的反模式）。
- 以摄取速率运行评估而不进行采样（成本不可接受）。
- 仅显示"延迟"而未区分 p50/p95/p99 的仪表板。

拒绝规则：

- 拒绝在没有 PII 脱敏策略的情况下持久化提示词或补全内容。
- 拒绝声称"多 SDK 支持"而没有每个 SDK 的规范 span 回归测试。
- 拒绝在没有基线窗口的情况下发布漂移检测；无基线的漂移检测毫无意义。

输出：一个包含 Collector 配置、ClickHouse schema、Next.js 15 仪表板、评估任务、漂移检测器、告警链路、带标注回退的 1 万追踪演示数据集的代码库，以及一份记录注入 PII 回退的 MTTR，以及迭代过程中降低 MTTR 的三大仪表板用户体验改进的报告。
