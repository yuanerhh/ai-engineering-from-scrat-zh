# 函数调用深度解析——OpenAI、Anthropic、Gemini

> 三家前沿提供商在 2024 年收敛到相同的工具调用循环，随后在其他所有方面产生了分歧。OpenAI 使用 `tools` 和 `tool_calls`，Anthropic 使用 `tool_use` 和 `tool_result` 块，Gemini 使用 `functionDeclarations` 和唯一 ID 关联。本课并排对比三家的差异，使得针对一家提供商编写的代码在移植时不会出错。

**类型：** 构建
**编程语言：** Python（标准库，schema 翻译器）
**前置知识：** Phase 13 · 01（工具接口）
**预计时间：** 约 75 分钟

## 学习目标

- 列出 OpenAI、Anthropic 和 Gemini 函数调用载荷的三处形态差异（声明、调用、结果）。
- 将一个工具声明翻译到三家提供商的格式，并预测严格模式约束的差异所在。
- 在各提供商中使用 `tool_choice` 强制、禁止或自动选择工具调用。
- 了解各提供商的硬性限制（工具数量、schema 深度、参数长度）以及违反限制时各家的错误信息。

## 问题背景

函数调用请求的形态因提供商而异。以下是 2026 年生产技术栈的三个具体示例：

**OpenAI Chat Completions / Responses API。** 传入 `tools: [{type: "function", function: {name, description, parameters, strict}}]`。模型响应中包含 `choices[0].message.tool_calls: [{id, type: "function", function: {name, arguments}}]`，其中 `arguments` 是需要解析的 JSON 字符串。严格模式（`strict: true`）通过约束解码强制执行 schema 合规。

**Anthropic Messages API。** 传入 `tools: [{name, description, input_schema}]`。响应以 `content: [{type: "text"}, {type: "tool_use", id, name, input}]` 的形式返回。`input` 已解析（是对象而非字符串）。你用包含 `{type: "tool_result", tool_use_id, content}` 块的新 `user` 消息回复。

**Google Gemini API。** 传入 `tools: [{functionDeclarations: [{name, description, parameters}]}]`（嵌套在 `functionDeclarations` 下）。响应以 `candidates[0].content.parts: [{functionCall: {name, args, id}}]` 的形式返回，其中 `id` 在 Gemini 3 及以上版本中对于并行调用关联是唯一的。你用 `{functionResponse: {name, id, response}}` 回复。

同样的循环，不同的字段名、不同的嵌套、不同的字符串-对象约定、不同的关联机制。一个在 OpenAI 上编写天气智能体的团队，仅为了管道适配，移植到 Anthropic 需要两天，再移植到 Gemini 又需要一天。

本课构建一个将三种格式统一为一种规范工具声明的翻译器，并在边界进行路由。Phase 13 · 17 将同样的模式推广为 LLM 网关。

## 核心概念

### 通用结构

每家提供商都需要五样东西：

1. **工具列表。** 每个工具的名称、描述和输入 schema。
2. **工具选择。** 强制使用特定工具、禁止工具或让模型决定。
3. **调用输出。** 命名工具和参数的结构化输出。
4. **调用 ID。** 将响应与正确的调用关联（并行调用时很重要）。
5. **结果注入。** 将结果与调用绑定的消息或块。

### 形态差异，逐字段对比

| 方面 | OpenAI | Anthropic | Gemini |
|------|--------|-----------|--------|
| 声明信封 | `{type: "function", function: {...}}` | `{name, description, input_schema}` | `{functionDeclarations: [{...}]}` |
| Schema 字段名 | `parameters` | `input_schema` | `parameters` |
| 响应容器 | assistant 消息上的 `tool_calls[]` | 类型为 `tool_use` 的 `content[]` | 类型为 `functionCall` 的 `parts[]` |
| 参数类型 | 字符串化的 JSON | 已解析的对象 | 已解析的对象 |
| ID 格式 | `call_...`（OpenAI 生成） | `toolu_...`（Anthropic） | UUID（Gemini 3+） |
| 结果块 | 角色 `tool`，带 `tool_call_id` | `user` 中含 `tool_result`，带 `tool_use_id` | 带匹配 `id` 的 `functionResponse` |
| 强制指定工具 | `tool_choice: {type: "function", function: {name}}` | `tool_choice: {type: "tool", name}` | `tool_config: {function_calling_config: {mode: "ANY"}}` |
| 禁止工具 | `tool_choice: "none"` | `tool_choice: {type: "none"}` | `mode: "NONE"` |
| Schema 强制执行 | `strict: true` | schema 即契约（始终强制执行） | 请求级的 `responseSchema` |

### 你实际会遇到的限制

- **OpenAI。** 每请求 128 个工具；schema 深度 5 层；参数字符串 ≤ 8192 字节；严格模式要求不含 `$ref`、不含带重叠的 `oneOf`/`anyOf`/`allOf`、每个属性都列在 `required` 中。
- **Anthropic。** 每请求 64 个工具；schema 深度实际上无限制，但实用上限约 10 层；没有严格模式标志，schema 是契约，模型倾向于遵守。
- **Gemini。** 每请求 64 个函数；schema 类型是 OpenAPI 3.0 子集（与 JSON Schema 2020-12 略有差异）；Gemini 3 起并行调用具有唯一 ID。

### `tool_choice` 行为

三家都支持三种模式，只是命名不同：

- **Auto（自动）。** 模型选择工具或文本。默认模式。
- **Required / Any（必须/任意）。** 模型必须至少调用一个工具。
- **None（禁止）。** 模型不得调用工具。

另外每家各有一种独特模式：

- **OpenAI。** 按名称强制指定某个工具。
- **Anthropic。** 按名称强制指定某个工具；`disable_parallel_tool_use` 标志用于分隔单次与多次调用。
- **Gemini。** `mode: "VALIDATED"` 无论模型意图如何，都将每个响应通过 schema 验证器路由。

### 并行调用

OpenAI 的 `parallel_tool_calls: true`（默认）在一条 assistant 消息中输出多个调用。你全部运行，然后用包含每个 `tool_call_id` 一个条目的批量 tool 角色消息回复。Anthropic 历史上只支持单次调用；`disable_parallel_tool_use: false`（Claude 3.5 起为默认）启用多次调用。Gemini 2 允许并行调用但没有稳定 ID；Gemini 3 添加了 UUID，使得乱序响应能够干净地关联。

### 流式传输

三家都支持流式工具调用，但线路格式不同：

- **OpenAI。** `tool_calls[i].function.arguments` 的增量块逐渐到达，累积到 `finish_reason: "tool_calls"` 为止。
- **Anthropic。** block-start / block-delta / block-stop 事件，`input_json_delta` 块携带部分参数。
- **Gemini。** `streamFunctionCallArguments`（Gemini 3 新增）通过 `functionCallId` 输出块，使多个并行调用可以交错。

Phase 13 · 03 深入讲解并行与流式重组，本课重点介绍声明和单次调用的形态。

### 错误与修复

无效参数错误的表现形式也各不相同：

- **OpenAI（非严格模式）。** 模型返回 `arguments: "{bad json}"`，JSON 解析失败，注入错误消息后重新调用。
- **OpenAI（严格模式）。** 验证在解码过程中完成，无效 JSON 不可能出现，但可能出现 `refusal`。
- **Anthropic。** `input` 可能包含意外字段；schema 是建议性的。在服务端验证。
- **Gemini。** OpenAPI 3.0 的怪癖：对象字段上的 `enum` 被静默忽略；自己验证。

### 翻译器模式

你代码中的规范工具声明如下所示（你自选形态）：

```python
Tool(
    name="get_weather",
    description="Use when ...",
    input_schema={"type": "object", "properties": {...}, "required": [...]},
    strict=True,
)
```

三个小函数将其翻译为三家提供商的形态。`code/main.py` 中的框架正是这样做的，然后通过每家提供商的响应形态循环处理一个假工具调用。无需网络——本课讲的是形态，不是 HTTP。

生产团队将这个翻译器包装在 `AbstractToolset`（Pydantic AI）、`UniversalToolNode`（LangGraph）或 `BaseTool`（LlamaIndex）中。Phase 13 · 17 构建一个在三家之上暴露 OpenAI 形态 API 的网关。

## 动手实践

`code/main.py` 定义了一个规范的 `Tool` 数据类和三个分别输出 OpenAI、Anthropic、Gemini 声明 JSON 的翻译器。然后将每家提供商形态的手工构造响应解析为相同的规范调用对象，演示三者在表皮之下语义相同。运行并并排对比三种声明。

重点关注：
- 三个声明块只在信封和字段名上有所不同。
- 三个响应块在调用所在位置上有所不同（顶层 `tool_calls`、`content[]` 块、`parts[]` 条目）。
- 一个 `canonical_call()` 函数从三种响应形态中提取 `{id, name, args}`。

## 产出技能

本课产出 `outputs/skill-provider-portability-audit.md`：给定一个针对某家提供商的函数调用集成，该技能产出可移植性审计：它依赖哪家提供商的限制，哪些字段需要重命名，移植到每家其他提供商时什么会出问题。

## 练习

1. 运行 `code/main.py`，验证三家提供商的声明 JSON 都序列化了同一个底层 `Tool` 对象。修改规范工具以添加枚举参数，确认只有 Gemini 翻译器需要处理 OpenAPI 的怪癖。

2. 为每家提供商添加 `ListToolsResponse` 解析器，提取模型在 `list_tools` 或发现调用后返回的工具列表。注意 OpenAI 原生没有这个功能，并记录这种不对称性。

3. 实现 `tool_choice` 转换：将规范的 `ToolChoice(mode="force", tool_name="x")` 映射到三家提供商的形态，再映射 `mode="any"` 和 `mode="none"`。对照本课的差异对比表检查。

4. 选一家提供商，从头到尾阅读其函数调用指南。找出另外两家不支持的一个 schema 规范字段。候选：OpenAI 的 `strict`，Anthropic 的 `disable_parallel_tool_use`，Gemini 的 `function_calling_config.allowed_function_names`。

5. 编写一个测试向量：一个参数违反声明 schema 的工具调用。通过每家提供商的验证器运行它（第 01 课的标准库版本可用作代理），记录触发了哪些错误。写明你在生产中会用哪家提供商来保证严格性。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 函数调用（Function calling） | "工具使用" | 提供商级 API，用于输出结构化工具调用 |
| 工具声明（Tool declaration） | "工具规范" | 名称 + 描述 + JSON Schema 输入载荷 |
| `tool_choice` | "强制/禁止" | auto / required / none / 指定名称 四种模式 |
| 严格模式（Strict mode） | "Schema 强制执行" | OpenAI 的约束解码标志，强制输出符合 schema |
| `tool_use` 块 | "Anthropic 的调用形态" | 包含 id、name、input 的内联内容块 |
| `functionCall` 部件 | "Gemini 的调用形态" | 包含 name、args 和 id 的 `parts[]` 条目 |
| 参数为字符串 | "字符串化 JSON" | OpenAI 以 JSON 字符串而非对象形式返回参数 |
| 并行工具调用 | "一轮扇出" | 一条 assistant 消息中的多个工具调用 |
| 拒绝（Refusal） | "模型拒绝" | 仅在严格模式下出现的拒绝块，代替调用 |
| OpenAPI 3.0 子集 | "Gemini schema 怪癖" | Gemini 使用的类 JSON Schema 方言，有细微差异 |

## 延伸阅读

- [OpenAI — 函数调用指南](https://platform.openai.com/docs/guides/function-calling) — 包含严格模式和并行调用的权威参考
- [Anthropic — 工具使用概述](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) — `tool_use` 和 `tool_result` 块语义
- [Google — Gemini 函数调用](https://ai.google.dev/gemini-api/docs/function-calling) — 并行调用、唯一 ID 和 OpenAPI 子集
- [Vertex AI — 函数调用参考](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling) — Gemini 的企业级接口
- [OpenAI — 结构化输出](https://platform.openai.com/docs/guides/structured-outputs) — 严格模式 schema 强制执行详情
