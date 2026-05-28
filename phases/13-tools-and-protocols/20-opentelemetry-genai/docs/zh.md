# OpenTelemetry GenAI——端到端追踪工具调用

> 一个智能体调用了五个工具、三个 MCP 服务器和两个子智能体。你需要一条贯穿所有这些的追踪链。OpenTelemetry GenAI 语义约定（v1.37 及以上版本中的稳定属性）是 2026 年的标准，得到了 Datadog、Langfuse、Arize Phoenix、OpenLLMetry 和 AgentOps 的原生支持。本课列出必需属性，讲解 span 层级（智能体 → LLM → 工具），并提供一个可插入任何 OTel 导出器的标准库 span 发射器。

**类型：** 构建
**编程语言：** Python（标准库，OTel span 发射器）
**前置知识：** Phase 13 · 07（MCP 服务器）、Phase 13 · 08（MCP 客户端）
**预计时间：** 约 75 分钟

## 学习目标

- 列出 LLM span 和工具执行 span 的必需 OTel GenAI 属性。
- 构建涵盖智能体循环、LLM 调用、工具调用和 MCP 客户端分发的追踪层级。
- 决定哪些内容要捕获（选择性加入）vs 哪些要隐藏（默认值）。
- 在不重写工具代码的情况下，将 span 发送到本地收集器（Jaeger、Langfuse）。

## 问题背景

2026 年 2 月的一次调试：用户报告"我的智能体有时需要 30 秒响应，有时只需 3 秒"。没有追踪记录。日志显示了 LLM 调用，但没有工具分发、没有 MCP 服务器往返、没有子智能体。你只能猜测。最终你发现：一个 MCP 服务器偶尔会在冷启动时挂起。

没有端到端追踪，你无法发现这一点。OTel GenAI 解决了这个问题。

这些约定在 2025-2026 年由 OpenTelemetry 语义约定小组确定。它们定义了稳定的属性名称，使 Datadog、Langfuse、Phoenix、OpenLLMetry 和 AgentOps 都能解析相同的 span。一次插桩，发送到任何后端。

## 核心概念

### Span 层级

```
agent.invoke_agent  （顶层，INTERNAL span）
 ├── llm.chat       （CLIENT span）
 ├── tool.execute   （INTERNAL）
 │    └── mcp.call  （CLIENT span）
 ├── llm.chat       （CLIENT span）
 └── subagent.invoke （INTERNAL）
```

整个层级嵌套在一个 trace id 下。Span ID 链接父子关系。

### 必需属性

根据 2025-2026 语义约定：

- `gen_ai.operation.name`——`"chat"`、`"text_completion"`、`"embeddings"`、`"execute_tool"`、`"invoke_agent"`。
- `gen_ai.provider.name`——`"openai"`、`"anthropic"`、`"google"`、`"azure_openai"`。
- `gen_ai.request.model`——请求的模型字符串（如 `"gpt-4o-2024-08-06"`）。
- `gen_ai.response.model`——实际提供服务的模型。
- `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens`。
- `gen_ai.response.id`——提供商响应 ID，用于关联。

工具 span 的属性：

- `gen_ai.tool.name`——工具标识符。
- `gen_ai.tool.call.id`——具体调用 ID。
- `gen_ai.tool.description`——工具描述（可选）。

智能体 span 的属性：

- `gen_ai.agent.name` / `gen_ai.agent.id` / `gen_ai.agent.description`。

### Span 类型

- `SpanKind.CLIENT`：用于跨进程边界的调用（LLM 提供商、MCP 服务器）。
- `SpanKind.INTERNAL`：用于智能体自身的循环步骤和工具执行。

### 选择性加入的内容捕获

默认情况下，span 携带指标和时序数据——而非提示词或补全结果。大载荷和 PII 默认关闭。设置 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` 和特定的内容捕获环境变量以包含内容。在生产环境中启用前请仔细审查。

### Span 上的事件

令牌级别的事件可以作为 span 事件添加：

- `gen_ai.content.prompt`——输入消息。
- `gen_ai.content.completion`——输出消息。
- `gen_ai.content.tool_call`——记录的工具调用。

事件在 span 内按时间排序，用于详细重放。

### 导出器

OTel span 导出到：

- **Jaeger / Tempo。** 开源，本地部署。
- **Langfuse。** LLM 可观测性专用；可视化令牌使用情况。
- **Arize Phoenix。** 评估 + 追踪结合。
- **Datadog。** 商业产品；原生解析 `gen_ai.*` 属性。
- **Honeycomb。** 列式存储；查询友好。

所有这些都支持 OTLP（线路格式）。你的代码不需要关心具体后端。

### 跨 MCP 的传播

当 MCP 客户端调用服务器时，将 W3C traceparent 头注入请求。Streamable HTTP 支持标准头。Stdio 原生不携带 HTTP 头；规范的 2026 年路线图讨论在 JSON-RPC 调用上添加 `_meta.traceparent` 字段。

在此功能发布之前：手动在每个请求的 `_meta` 中包含 traceparent。服务器记录 trace id。

### 指标

除 span 外，GenAI 语义约定还定义了指标：

- `gen_ai.client.token.usage`——直方图。
- `gen_ai.client.operation.duration`——直方图。
- `gen_ai.tool.execution.duration`——直方图。

这些用于不需要每次调用细节的仪表盘。

### AgentOps 层

AgentOps（2024 年成立）专注于 GenAI 可观测性。它包装流行框架（LangGraph、Pydantic AI、CrewAI）以自动发送 OTel span。如果你的技术栈使用受支持的框架，这很有用；否则使用手动插桩。

## 动手实践

`code/main.py` 将 OTel 格式的 span 发送到 stdout（以类 OTLP-JSON 格式），用于一个调用 LLM、分发两个工具并进行一次 MCP 往返的智能体。没有真实的导出器——本课重点在于 span 形态和属性集。将输出粘贴到兼容 OTLP 的查看器中，或直接阅读。

重点关注：
- Trace id 在所有 span 中共享。
- 父子链接通过 `parentSpanId` 编码。
- 必需的 `gen_ai.*` 属性已填充。
- 内容捕获默认关闭；一个场景通过环境变量启用它。

## 产出技能

本课产出 `outputs/skill-otel-genai-instrumentation.md`：给定一个智能体代码库，该技能生成插桩计划：在哪里添加 span、填充哪些属性，以及目标导出器。

## 练习

1. 运行 `code/main.py`。统计 span 数量，识别哪些是 CLIENT span，哪些是 INTERNAL span。

2. 启用内容捕获（环境变量），确认 `gen_ai.content.prompt` 和 `gen_ai.content.completion` 事件出现。注意对 PII 的影响。

3. 添加工具执行指标 `gen_ai.tool.execution.duration`，并将其作为每次调用的直方图样本发送。

4. 将父智能体 span 的 traceparent 传播到 MCP 请求的 `_meta.traceparent` 字段。验证 MCP 服务器能看到相同的 trace id。

5. 阅读 OTel GenAI 语义约定规范。找出规范中列出的、但本课代码未发送的一个属性。添加它。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| OTel | "OpenTelemetry" | 追踪、指标、日志的开放标准 |
| GenAI 语义约定 | "GenAI 语义约定" | LLM / 工具 / 智能体 span 的稳定属性名称 |
| `gen_ai.*` | "属性命名空间" | 所有 GenAI 属性共享此前缀 |
| Span | "定时操作" | 具有开始、结束和属性的一个工作单元 |
| Trace | "跨 span 的祖先关系" | 共享同一 trace id 的 span 树 |
| SpanKind | "CLIENT / SERVER / INTERNAL" | 关于 span 方向的提示 |
| OTLP | "OpenTelemetry 线路协议" | 导出器的线路格式 |
| 选择性加入内容 | "提示词/补全捕获" | 默认关闭；通过环境变量启用 |
| traceparent | "W3C 头" | 跨服务传播追踪上下文 |
| 导出器（Exporter） | "后端特定发送器" | 将 span 发送到 Jaeger / Datadog 等的组件 |

## 延伸阅读

- [OpenTelemetry——GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — GenAI span、指标和事件的权威约定
- [OpenTelemetry——GenAI span](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) — LLM 和工具执行 span 属性列表
- [OpenTelemetry——GenAI 智能体 span](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/) — 智能体级别的 `invoke_agent` span
- [open-telemetry/semantic-conventions——GenAI span](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md) — GitHub 托管的权威来源
- [Datadog——LLM OTel 语义约定](https://www.datadoghq.com/blog/llm-otel-semantic-convention/) — 生产集成详解
