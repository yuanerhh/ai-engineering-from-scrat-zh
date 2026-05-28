# OpenTelemetry GenAI 语义约定

> OpenTelemetry 的 GenAI SIG（2024 年 4 月启动）定义了智能体遥测的标准 schema。Span 名称、属性和内容捕获规则在各供应商之间收敛，使得智能体追踪在 Datadog、Grafana、Jaeger 和 Honeycomb 中具有相同含义。

**类型：** 学习 + 构建
**编程语言：** Python（标准库）
**前置知识：** Phase 14 · 13（LangGraph）、Phase 14 · 24（可观测性平台）
**预计时间：** 约 60 分钟

## 学习目标

- 列出 GenAI span 类别：模型/客户端、智能体、工具。
- 区分 `invoke_agent` 的 CLIENT 和 INTERNAL span 及各自的适用场景。
- 列出顶层 GenAI 属性：提供商名称、请求模型、数据源 ID。
- 解释内容捕获契约：选择加入、`OTEL_SEMCONV_STABILITY_OPT_IN`、外部引用推荐。

## 问题背景

每家供应商都发明自己的 span 名称。运营团队最终为每个框架构建各自的仪表板。OpenTelemetry 的 GenAI SIG 通过定义整个生态系统都遵循的一个标准来解决这个问题。

## 核心概念

### Span 类别

1. **模型/客户端 span。** 涵盖原始 LLM 调用。由提供商 SDK（Anthropic、OpenAI、Bedrock）和框架模型适配器发出。
2. **智能体 span。** `create_agent`（智能体构建时）和 `invoke_agent`（智能体运行时）。
3. **工具 span。** 每次工具调用一个；通过父子关系与智能体 span 连接。

### 智能体 span 命名

- Span 名称：如果有名称则为 `invoke_agent {gen_ai.agent.name}`；否则回退到 `invoke_agent`。
- Span 类型：
  - **CLIENT**——用于远程智能体服务（OpenAI Assistants API、Bedrock Agents）。
  - **INTERNAL**——用于进程内智能体框架（LangChain、CrewAI、本地 ReAct）。

### 关键属性

- `gen_ai.provider.name`——`anthropic`、`openai`、`aws.bedrock`、`google.vertex`。
- `gen_ai.request.model`——模型 ID。
- `gen_ai.response.model`——解析后的模型（由于路由可能与请求不同）。
- `gen_ai.agent.name`——智能体标识符。
- `gen_ai.operation.name`——`chat`、`completion`、`invoke_agent`、`tool_call`。
- `gen_ai.data_source.id`——用于 RAG：查询了哪个语料库或存储。

Anthropic、Azure AI Inference、AWS Bedrock、OpenAI 都有特定于技术的约定。

### 内容捕获

默认规则：instrumentation 默认不应捕获输入/输出。通过以下方式选择加入捕获：

- `gen_ai.system_instructions`
- `gen_ai.input.messages`
- `gen_ai.output.messages`

推荐的生产模式：将内容存储在外部（S3、你的日志存储），在 span 上记录引用（指针 ID，而非完整内容）。这是将第 27 课的内容中毒防御与可观测性结合的方式。

### 稳定性

截至 2026 年 3 月，大多数约定仍处于实验阶段。通过以下方式选择加入稳定预览版：

```
OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
```

Datadog v1.37+ 将 GenAI 属性原生映射到其 LLM 可观测性 schema 中。其他后端（Grafana、Honeycomb、Jaeger）支持原始属性。

### 这个模式在哪里出错

- **在 span 中捕获完整提示词。** PII、密钥、客户数据出现在运营人员可以读取的追踪中。存储在外部。
- **没有 `gen_ai.provider.name`。** 当缺少归因时，多提供商仪表板会失效。
- **没有父链接的 span。** 孤立的工具 span。始终传播上下文。
- **没有设置稳定性选项。** 在后端升级时你的属性可能被重命名。

## 动手实践

`code/main.py` 实现了一个符合 GenAI 约定的标准库 span 发射器：

- 带 GenAI 属性 schema 的 `Span`。
- 带 `start_span` 和嵌套上下文的 `Tracer`。
- 一个脚本化智能体运行，发出：`create_agent`、`invoke_agent`（INTERNAL）、每工具 span、LLM 调用的 `chat` span。
- 一种内容捕获模式，将提示词存储在外部并在 span 上记录 ID。

运行：

```
python3 code/main.py
```

输出：包含所有必需 GenAI 属性的 span 树，以及展示选择加入内容引用的"外部存储"。

## 使用建议

- **Datadog LLM 可观测性**（v1.37+）原生映射属性。
- **Langfuse / Phoenix / Opik**（第 24 课）——自动检测生态系统。
- **Jaeger / Honeycomb / Grafana Tempo**——原始 OTel 追踪；从 GenAI 属性构建仪表板。
- **自托管**——运行带 GenAI 处理器的 OTel Collector。

## 产出技能

`outputs/skill-otel-genai.md` 将 OTel GenAI span 集成到现有智能体中，带内容捕获默认值和外部引用存储。

## 练习

1. 用 `invoke_agent`（INTERNAL）+ 每工具 span 检测第 01 课的 ReAct 循环。发送到 Jaeger 实例。
2. 以"仅引用"模式添加内容捕获：提示词到 SQLite，span 属性只携带行 ID。
3. 阅读 `gen_ai.data_source.id` 的规范。将其接入第 09 课的 Mem0 搜索。
4. 设置 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`，验证你的属性在 collector 升级时不会被重命名。
5. 构建仪表板："哪些工具错误与哪些模型相关"——仅从 GenAI 属性中提取。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| GenAI SIG | "OpenTelemetry GenAI 工作组" | 定义 schema 的 OTel 工作组 |
| invoke_agent | "智能体 span" | 表示智能体运行的 span 名称 |
| CLIENT span | "远程调用" | 调用远程智能体服务的 span |
| INTERNAL span | "进程内" | 进程内智能体运行的 span |
| gen_ai.provider.name | "提供商" | anthropic / openai / aws.bedrock / google.vertex |
| gen_ai.data_source.id | "RAG 源" | 检索命中了哪个语料库/存储 |
| 内容捕获 | "提示词日志" | 选择加入的消息捕获；在生产中存储于外部 |
| 稳定性选项 | "预览模式" | 固定实验性约定的环境变量 |

## 延伸阅读

- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 规范
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — 默认发出 GenAI span
- [AutoGen v0.4（Microsoft Research）](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — 内置 OTel span
- [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) — W3C 追踪上下文传播
