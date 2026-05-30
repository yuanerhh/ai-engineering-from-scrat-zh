---
name: otel-genai-instrumentation
description: 为智能体代码库生成端到端发送 OTel GenAI Span 的埋点方案。
version: 1.0.0
phase: 13
lesson: 19
tags: [otel, observability, gen-ai, tracing]
---

给定一个智能体代码库（LLM 调用、工具分发、MCP 客户端、子智能体），生成 OTel GenAI 埋点方案。

输出内容：

1. **Span 层级**。根节点 `agent.invoke_agent`（INTERNAL），子节点：`llm.chat`（CLIENT）、`tool.execute`（INTERNAL）、`mcp.call`（CLIENT）、`subagent.invoke`（INTERNAL）。
2. **每个 Span 的属性检查清单**。`gen_ai.operation.name`、`gen_ai.provider.name`、`gen_ai.request.model`、`gen_ai.response.model`、`gen_ai.usage.*`、`gen_ai.tool.name`、`gen_ai.agent.name`。
3. **传播规则**。在每次远程调用中注入 W3C traceparent；对于 MCP stdio，使用 `_meta.traceparent` 作为过渡字段。
4. **内容捕获策略**。默认关闭；记录启用所需的环境变量；列出 PII 风险。
5. **导出器选择**。Jaeger / Tempo / Langfuse / Phoenix / Datadog / Honeycomb；使用 OTLP 作为传输协议。

硬性拒绝：
- 任何缺少跨 MCP 或子智能体边界的追踪传播方案。
- 任何默认开启内容捕获的方案。会泄漏 Prompt 和 PII。
- 任何不带 `gen_ai.` 前缀或明确供应商前缀的自定义属性。

拒绝规则：
- 若代码库使用了内置 OTel 自动埋点的框架（Pydantic AI、LangGraph、AgentOps），建议优先使用框架钩子。
- 若导出器后端为本地部署且团队没有 SRE 支持，建议使用托管后端。
- 若用户要求在生产环境捕获内容用于调试，在没有明确的同意策略和 PII 脱敏管道的情况下拒绝。

输出：一页方案，包含 Span 层级、每 Span 属性检查清单、传播规则、内容捕获策略和导出器选择。结尾给出最值得告警的核心指标（通常为 p95 `gen_ai.client.operation.duration`）。
