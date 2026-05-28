# MCP 资源与提示——超越工具的上下文暴露

> 工具占据了 MCP 90% 的注意力。另外两种服务器原语解决的是不同问题。资源用于暴露供读取的数据；提示将可复用的模板以斜杠命令的形式暴露出来。许多服务器应该用资源代替将读取操作包装进工具，用提示代替将工作流硬编码进客户端提示。本课给出决策规则，并逐步讲解 `resources/*` 和 `prompts/*` 消息。

**类型：** 构建
**编程语言：** Python（标准库，资源 + 提示处理器）
**前置知识：** Phase 13 · 07（MCP 服务器）
**预计时间：** 约 45 分钟

## 学习目标

- 针对给定领域，决定将某项能力以工具、资源还是提示的形式暴露。
- 实现 `resources/list`、`resources/read`、`resources/subscribe`，并处理 `notifications/resources/updated`。
- 实现带参数模板的 `prompts/list` 和 `prompts/get`。
- 识别宿主何时将提示作为斜杠命令呈现，何时作为自动注入的上下文呈现。

## 问题背景

一个简单的笔记应用 MCP 服务器将所有内容都以工具形式暴露：`notes_read`、`notes_list`、`notes_search`。这将每次数据访问都包装成了一次模型驱动的工具调用。后果：

- 模型必须决定是否对每个可能受益于上下文的查询调用 `notes_read`。
- 只读内容无法被订阅或流式传输到宿主的侧边栏。
- 客户端 UI（Claude Desktop 的资源附件面板、Cursor 的"包含文件"选择器）无法呈现这些数据。

正确的划分：将数据以资源形式暴露，将变更或计算操作以工具形式暴露，将可复用的多步骤工作流以提示形式暴露。每种原语都有其 UX 适用场景和访问模式。

## 核心概念

### 工具 vs 资源 vs 提示——决策规则

| 能力 | 原语 |
|------|------|
| 用户想搜索、过滤或转换数据 | 工具 |
| 用户希望宿主将此数据作为上下文包含进来 | 资源 |
| 用户希望有一个可反复执行的模板化工作流 | 提示 |

指导原则：如果模型能从每次相关查询中调用它而获益，它是工具；如果用户能从将其附加到对话中获益，它是资源；如果整个多步骤工作流是用户想要复用的单元，它是提示。

### 资源

`resources/list` 返回 `{resources: [{uri, name, mimeType, description?}]}`。`resources/read` 接受 `{uri}` 并返回 `{contents: [{uri, mimeType, text | blob}]}`。

URI 可以是任何可寻址的内容：

- `file:///Users/alice/notes/mcp.md`
- `postgres://my-db/query/SELECT ...`
- `notes://note-14`（自定义 scheme）
- `memory://session-2026-04-22/recent`（服务器特定）

`contents[]` 同时支持文本和二进制。二进制使用 base64 编码的 `blob` 字符串加上 `mimeType`。

### 资源订阅

在能力中声明 `{resources: {subscribe: true}}`。客户端调用 `resources/subscribe {uri}`。资源发生变化时，服务器发送 `notifications/resources/updated {uri}`，客户端重新读取。

使用场景：资源是磁盘上文件的笔记服务器；文件监视器触发更新通知；当文件在宿主外部被编辑时，Claude Desktop 重新将其拉入上下文。

### 资源模板（2025-11-25 新增）

`resourceTemplates` 允许暴露参数化的 URI 模式：`notes://{id}`，其中 `id` 是一个补全目标。客户端可以在资源选择器中自动补全 ID。

### 提示

`prompts/list` 返回 `{prompts: [{name, description, arguments?}]}`。`prompts/get` 接受 `{name, arguments}` 并返回 `{description, messages: [{role, content}]}`。

提示是一个模板，填充后生成宿主传给模型的消息列表。例如，`code_review` 提示接受一个 `file_path` 参数，并返回三条消息的序列：一条系统消息、一条包含文件内容的用户消息，以及一条带有推理模板的助手开场消息。

### 宿主与提示

Claude Desktop、VS Code 和 Cursor 将提示作为斜杠命令暴露在聊天 UI 中。用户输入 `/code_review` 并从表单中选择参数。服务器的提示就是"用户快捷方式"和"发送给模型的完整提示"之间的契约。

并非所有客户端都支持提示——需检查能力协商。声明了提示能力的服务器与不支持提示的客户端配对时，斜杠命令将不可见。

### "列表变更"通知

资源和提示都会在集合发生变化时发出 `notifications/list_changed`。刚导入了 20 条新笔记的笔记服务器会发出 `notifications/resources/list_changed`；客户端重新调用 `resources/list` 以获取新增项。

### 内容类型约定

- 文本：`mimeType: "text/plain"`、`text/markdown`、`application/json`
- 二进制：`image/png`、`application/pdf`，加上 `blob` 字段
- MCP Apps（第 14 课）：`ui://` URI 中的 `text/html;profile=mcp-app`

### 动态资源

资源 URI 不必对应静态文件。`notes://recent` 每次读取时可以返回最新的五条笔记。`db://query/users/active` 可以执行参数化查询。服务器可以自由动态计算内容。

规则：如果客户端可以按 URI 缓存，URI 必须稳定。如果计算是一次性的，URI 应包含时间戳或随机数，以防客户端缓存过期。

### 订阅 vs 轮询

支持订阅的客户端通过 `notifications/resources/updated` 获得服务器推送。不支持订阅的客户端或宿主通过重新读取进行轮询。两者都符合规范。服务器的能力声明告诉客户端它支持哪种方式。

订阅的成本：服务器需要维护每会话的状态（谁订阅了什么）。要控制订阅集合的规模；断开连接的客户端应超时处理。

### 提示 vs 系统提示

MCP 中的提示不是系统提示。宿主的系统提示（其自身的操作指令）和 MCP 提示（服务器提供的、由用户调用的模板）并列存在。行为良好的客户端不允许服务器提示覆盖其自身的系统提示；而是将它们叠加。

## 动手实践

`code/main.py` 在第 07 课的笔记服务器基础上扩展了：

- 每条笔记的资源（`notes://note-1` 等），支持 `resources/subscribe`。
- 渲染为三条消息模板的 `review_note` 提示。
- 模拟文件监视器，当笔记被修改时发出 `notifications/resources/updated`。
- 始终返回最新五条笔记的动态资源 `notes://recent`。

运行演示即可看到完整流程。

## 产出技能

本课产出 `outputs/skill-primitive-splitter.md`：给定一个 MCP 服务器提案，该技能将每种能力分类为工具/资源/提示，并附上理由说明。

## 练习

1. 运行 `code/main.py`。观察初始资源列表，然后触发一次笔记编辑，验证 `notifications/resources/updated` 事件是否触发。

2. 添加 `resources/list_changed` 发射器：当新笔记被创建时，发送通知，使客户端重新发现资源列表。

3. 为 GitHub MCP 服务器设计三个提示：`summarize_pr`、`triage_issue`、`release_notes`，每个都带参数 schema。提示体应无需进一步编辑即可运行。

4. 取第 07 课服务器中的一个现有工具，判断它应该保留为工具，还是应该拆分为资源加工具对。用一句话说明理由。

5. 阅读规范的 `server/resources` 和 `server/prompts` 章节。找出 `resources/read` 中一个很少使用但规范支持的字段。（提示：看资源内容上的 `_meta`。）

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 资源（Resource） | "暴露的数据" | 宿主可读取的 URI 可寻址内容 |
| 资源 URI | "数据指针" | 带 scheme 前缀的标识符（`file://`、`notes://` 等） |
| `resources/subscribe` | "监视变更" | 客户端选择订阅特定 URI 的服务器推送更新 |
| `notifications/resources/updated` | "资源已变更" | 通知客户端已订阅的资源有新内容 |
| 资源模板 | "参数化 URI" | 带有供宿主选择器自动补全的 URI 模式 |
| 提示（Prompt） | "斜杠命令模板" | 带参数槽的命名多消息模板 |
| 提示参数 | "模板输入" | 宿主在渲染前收集的类型化参数 |
| `prompts/get` | "渲染模板" | 服务器返回填充好的消息列表 |
| 内容块（Content block） | "类型化数据块" | `{type: text | image | resource | ui_resource}` |
| 斜杠命令 UX | "用户快捷方式" | 宿主将提示以 `/` 开头的命令形式呈现 |

## 延伸阅读

- [MCP——概念：资源](https://modelcontextprotocol.io/docs/concepts/resources) — 资源 URI、订阅和模板
- [MCP——概念：提示](https://modelcontextprotocol.io/docs/concepts/prompts) — 提示模板与斜杠命令集成
- [MCP——服务器资源规范 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/resources) — `resources/*` 消息完整参考
- [MCP——服务器提示规范 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/prompts) — `prompts/*` 消息完整参考
- [MCP——协议信息站：资源](https://modelcontextprotocol.info/docs/concepts/resources/) — 对官方文档的社区补充指南
