---
name: otel-genai
description: 使用 OpenTelemetry GenAI 语义约定对 Agent 进行插桩——生成包含正确属性和可选内容捕获的 invoke_agent、chat、tool_call span。
version: 1.0.0
phase: 14
lesson: 23
tags: [opentelemetry, genai, observability, tracing, semantic-conventions]
---

给定一个 Agent 运行时，接入 OTel GenAI 语义约定。

输出内容：

1. 每次 Agent 运行一个 `invoke_agent` span。远程 Agent 服务用 CLIENT，进程内用 INTERNAL。名称格式：`invoke_agent {gen_ai.agent.name}`。
2. 每次 LLM 调用一个 `chat` span，包含 `gen_ai.operation.name=chat`、`gen_ai.provider.name`、`gen_ai.request.model`、`gen_ai.response.model`。
3. 每次工具调用一个 `tool_call` span，包含 `gen_ai.tool.name`，以及在适用时包含 `gen_ai.data_source.id`（RAG 语料库/记忆存储）。
4. 可选内容捕获：默认关闭；开启时，将输入/输出存储在外部，并在 span 上记录 `*.reference_id`。
5. 上下文传播：使用 W3C trace context 请求头，使多进程运行（Claude Agent SDK CLI 子进程）合并为一条完整链路。

硬性拒绝：

- 默认将完整提示词/输出内联捕获。存在 PII 和密钥泄漏风险，且违反规范。
- 缺少 `gen_ai.provider.name`。多提供商 Dashboard 会出错。
- 孤立的 tool span。必须始终通过活动上下文设置父子关系。

拒绝规则：

- 如果运行时无法跨进程边界传播上下文，拒绝。Claude Agent SDK + CLI 用户需要多进程链路拼接。
- 如果产品有监管约束（HIPAA、GDPR），拒绝内联内容捕获。只允许使用带访问控制的外部存储。
- 如果后端未设置 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`，发出警告：collector 升级后属性名可能变更。

输出：`tracer.py`、`attributes.py`、`content_store.py`、`README.md`，说明 span 结构、稳定性选项和内容捕获策略。最后附"下一步阅读"，指向第 24 课（后端：Langfuse、Phoenix、Opik）或第 17 课了解 Claude Agent SDK 链路上下文传播。
