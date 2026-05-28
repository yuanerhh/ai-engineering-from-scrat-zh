# MCP 基础——原语、生命周期、JSON-RPC 基础

> MCP 出现之前，每次集成都是一次性的。模型上下文协议（Model Context Protocol）由 Anthropic 于 2024 年 11 月首次发布，现已由 Linux Foundation 的 Agentic AI Foundation 负责维护，它对发现和调用进行了标准化，使任何客户端都能与任何服务器通信。2025-11-25 规范定义了六种原语（三种服务器端、三种客户端端）、一个三阶段生命周期，以及 JSON-RPC 2.0 的线路格式。掌握这些，本阶段其余的 MCP 章节便不在话下。

**类型：** 学习
**编程语言：** Python（标准库，JSON-RPC 解析器）
**前置知识：** Phase 13 · 01 至 05（工具接口与函数调用）
**预计时间：** 约 45 分钟

## 学习目标

- 说出全部六种 MCP 原语（服务器端：tools、resources、prompts；客户端：roots、sampling、elicitation），并各举一个使用场景。
- 逐步介绍三阶段生命周期（initialize、operation、shutdown），并说明每个阶段各自发送哪条消息。
- 解析并生成 JSON-RPC 2.0 的请求、响应和通知信封。
- 解释 `initialize` 中的能力协商是什么，以及缺少它会出什么问题。

## 问题背景

MCP 出现之前，每个使用工具的智能体都有自己的协议。Cursor 有一个形似 MCP 但不兼容的工具体系，Claude Desktop 自带另一种，VS Code 的 Copilot 扩展又是第三种。一个团队构建"Postgres 查询"工具时，需要按照三个不同宿主的 API 分别写三遍。复用它意味着复制代码。

结果便是一次集成爆炸，并形成了生态速度的天花板。

MCP 通过标准化线路格式解决了这个问题。一个 MCP 服务器可在每个 MCP 客户端中运行：Claude Desktop、ChatGPT、Cursor、VS Code、Gemini、Goose、Zed、Windsurf——截至 2026 年 4 月已有 300+ 客户端。每月 SDK 下载量 1.1 亿次，公开服务器超过 10,000 个。Linux Foundation 于 2025 年 12 月在新成立的 Agentic AI Foundation 旗下接管了该项目的维护。

本阶段使用的规范版本为 **2025-11-25**，新增了异步任务（SEP-1686）、URL 模式引导（SEP-1036）、带工具的采样（SEP-1577）、增量范围授权（SEP-835）以及 OAuth 2.1 资源指示语义。Phase 13 · 09 至 16 涵盖这些扩展，本课停留在基础层面。

## 核心概念

### 三种服务器原语

1. **工具（Tools）。** 可调用的动作，与 Phase 13 · 01 中的四步循环相同。
2. **资源（Resources）。** 暴露的数据。按 URI 寻址的只读内容：`file:///path`、`db://query/...`、自定义 Scheme。
3. **提示（Prompts）。** 可复用的模板。在宿主 UI 中呈现为斜杠命令；服务器提供模板，客户端填充参数。

### 三种客户端原语

4. **根（Roots）。** 服务器被允许访问的 URI 集合。客户端声明，服务器遵守。
5. **采样（Sampling）。** 服务器请求客户端的模型执行一次补全。使服务器无需持有 API 密钥即可托管智能体循环。
6. **引导（Elicitation）。** 服务器在运行过程中向客户端用户请求结构化输入，以表单或 URL 形式呈现（SEP-1036）。

MCP 中的每一种能力都恰好属于这六种原语之一。Phase 13 · 10 至 14 分别深入讲解每种原语。

### 线路格式：JSON-RPC 2.0

每条消息都是一个 JSON 对象，包含以下字段：

- 请求：`{jsonrpc: "2.0", id, method, params}`
- 响应：`{jsonrpc: "2.0", id, result | error}`
- 通知：`{jsonrpc: "2.0", method, params}`——没有 `id`，不期待响应

基础规范中约有 15 个方法，按原语分组，重要的包括：

- `initialize` / `initialized`（握手）
- `tools/list`、`tools/call`
- `resources/list`、`resources/read`、`resources/subscribe`
- `prompts/list`、`prompts/get`
- `sampling/createMessage`（服务器发往客户端）
- `notifications/tools/list_changed`、`notifications/resources/updated`、`notifications/progress`

### 三阶段生命周期

**阶段一：initialize（初始化）**

客户端发送 `initialize`，携带自己的 `capabilities` 和 `clientInfo`。服务器回复自己的 `capabilities`、`serverInfo` 以及所用的规范版本。客户端消化响应后，发送 `notifications/initialized`。此后，任意一方均可按协商好的能力发送请求。

**阶段二：operation（运行）**

双向通信阶段。客户端调用 `tools/list` 进行发现，再调用 `tools/call` 执行。如果服务器声明了该能力，可以向客户端发送 `sampling/createMessage`。当工具集发生变化时，服务器可发送 `notifications/tools/list_changed`；当用户更改根范围时，客户端可发送 `notifications/roots/list_changed`。

**阶段三：shutdown（关闭）**

任意一方关闭传输连接。MCP 没有结构化的关闭方法；传输层（stdio 或 Streamable HTTP，见 Phase 13 · 09）负责传递连接结束信号。

### 能力协商

`initialize` 握手中的 `capabilities` 字段就是双方的契约。服务器示例：

```json
{
  "tools": {"listChanged": true},
  "resources": {"subscribe": true, "listChanged": true},
  "prompts": {"listChanged": true}
}
```

服务器声明它可以发出 `tools/list_changed` 通知，并支持 `resources/subscribe`。客户端通过声明自己的能力来表示同意：

```json
{
  "roots": {"listChanged": true},
  "sampling": {},
  "elicitation": {}
}
```

如果客户端未声明 `sampling`，服务器就不得调用 `sampling/createMessage`。反之亦然：如果服务器未声明 `resources.subscribe`，客户端就不得尝试订阅。

这正是防止生态漂移的机制。不支持 sampling 的客户端仍是合法的 MCP 客户端；不调用 `sampling` 的服务器仍是合法的 MCP 服务器，只是不共同使用该功能。

### 结构化内容与错误格式

`tools/call` 返回包含类型化块的 `content` 数组：`text`、`image`、`resource`。Phase 13 · 14 将 MCP Apps（`ui://` 交互式 UI）添加到该列表。

错误使用 JSON-RPC 错误码，规范新增的错误码包括：`-32002` "Resource not found"（资源未找到）、`-32603` "Internal error"（内部错误），以及作为 `error.data` 的 MCP 专用错误数据。

### 客户端能力 vs 工具调用细节

常见混淆点：`capabilities.tools` 表示客户端是否支持工具列表变更通知，而不是客户端会不会调用某个具体工具——这是运行时由模型决定的，不是能力标志。能力标志是规范层面的契约，模型的选择与之正交。

### 为何选用 JSON-RPC 而非 REST？

JSON-RPC 2.0（2010 年）是一个轻量的双向协议。REST 是客户端发起的单向协议。MCP 需要服务器发起的消息（采样、通知），因此具有对称请求/响应形态的 JSON-RPC 自然成为合适的选择。JSON-RPC 还能干净地在 stdio 和 WebSocket/Streamable HTTP 之上组合，无需重新设计 HTTP 的请求形态。

## 动手实践

`code/main.py` 提供一个最小的 JSON-RPC 2.0 解析器和生成器，然后手动演示 `initialize` → `tools/list` → `tools/call` → `shutdown` 的完整序列，并打印每条消息。无需真实传输，只有消息格式。对照延伸阅读中的规范链接，逐一验证每个信封。

重点关注：
- `initialize` 双向声明能力；响应中包含 `serverInfo` 和 `protocolVersion: "2025-11-25"`。
- `tools/list` 返回 `tools` 数组，每项包含 `name`、`description`、`inputSchema`。
- `tools/call` 使用 `params.name` 和 `params.arguments`。
- 响应的 `content` 是 `{type, text}` 块的数组。

## 产出技能

本课产出 `outputs/skill-mcp-handshake-tracer.md`：给定一份 pcap 风格的 MCP 客户端-服务器交互记录，该技能为每条消息标注它所属的原语、生命周期阶段以及它所依赖的能力。

## 练习

1. 运行 `code/main.py`，找到能力协商发生的那一行，并描述如果服务器未声明 `tools.listChanged` 会有什么变化。

2. 扩展解析器以处理 `notifications/progress`。消息格式为：`{method: "notifications/progress", params: {progressToken, progress, total}}`。在长时间运行的 `tools/call` 过程中发出它，并确认客户端处理器会显示进度条。

3. 从头到尾阅读 MCP 2025-11-25 规范——整个文档约 80 页。找出大多数服务器**不需要**的那一个能力标志。（提示：与资源订阅有关。）

4. 在纸上画出一个假设的"定时任务"功能应该属于哪种原语。（提示：服务器希望客户端在定时时间调用它。现有六种原语都不完全适合。）MCP 的 2026 年路线图中有一个对应的草案 SEP。

5. 在 GitHub 上找一个开源 MCP 服务器的会话日志，解析其内容，统计请求、响应和通知消息的数量，计算生命周期流量与运行流量各占多少比例。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| MCP | "模型上下文协议" | 用于模型与工具发现和调用的开放协议 |
| 服务器原语 | "服务器暴露的内容" | tools（动作）、resources（数据）、prompts（模板） |
| 客户端原语 | "服务器可使用的客户端能力" | roots（范围）、sampling（LLM 回调）、elicitation（用户输入） |
| JSON-RPC 2.0 | "线路格式" | 对称的请求/响应/通知信封 |
| `initialize` 握手 | "能力协商" | 第一个消息对；服务器和客户端声明各自支持的功能 |
| `tools/list` | "发现" | 客户端向服务器请求当前工具集 |
| `tools/call` | "调用" | 客户端请求服务器以参数执行某个工具 |
| `notifications/*_changed` | "变更事件" | 服务器通知客户端其原语列表已发生变化 |
| 内容块（Content block） | "类型化结果" | 工具结果中的 `{type: "text" | "image" | "resource" | "ui_resource"}` |
| SEP | "规范演进提案" | 命名草案提案（例如 SEP-1686 代表异步任务） |

## 延伸阅读

- [模型上下文协议——规范 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — 权威规范文档
- [模型上下文协议——架构概念](https://modelcontextprotocol.io/docs/concepts/architecture) — 六原语思维模型
- [Anthropic — 介绍模型上下文协议](https://www.anthropic.com/news/model-context-protocol) — 2024 年 11 月发布帖
- [MCP 博客 — MCP 一周年](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/) — 一年回顾与 2025-11-25 规范变更
- [WorkOS — MCP 2025-11-25 规范更新](https://workos.com/blog/mcp-2025-11-25-spec-update) — SEP-1686、1036、1577、835 和 1724 摘要
