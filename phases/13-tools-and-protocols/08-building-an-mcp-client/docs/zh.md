# 构建 MCP 客户端——发现、调用、会话管理

> 大多数 MCP 内容提供服务器教程，对客户端则一笔带过。而真正复杂的编排逻辑恰恰在客户端：进程启动、能力协商、跨多个服务器的工具列表合并、采样回调、重连以及命名空间冲突解决。本课构建一个多服务器客户端，将三个不同的 MCP 服务器合并为一个扁平的工具命名空间提供给模型使用。

**类型：** 构建
**编程语言：** Python（标准库，多服务器 MCP 客户端）
**前置知识：** Phase 13 · 07（构建 MCP 服务器）
**预计时间：** 约 75 分钟

## 学习目标

- 将 MCP 服务器作为子进程启动，完成 `initialize` 握手，发送 `notifications/initialized`。
- 维护每个服务器的会话状态（能力、工具列表、最近收到的通知 ID）。
- 将多个服务器的工具列表合并到一个命名空间，并处理冲突。
- 将工具调用路由到拥有该工具的服务器，并重新组装响应。

## 问题背景

真实的智能体宿主（Claude Desktop、Cursor、Goose、Gemini CLI）会同时加载多个 MCP 服务器。用户可能同时运行文件系统服务器、Postgres 服务器和 GitHub 服务器。客户端的工作是：

1. 启动每个服务器。
2. 分别与每个服务器握手。
3. 对每个服务器调用 `tools/list`，将结果展平。
4. 当模型输出 `notes_search` 时，在合并后的命名空间中查找并路由到正确的服务器。
5. 在不阻塞的情况下处理来自任意服务器的通知（`tools/list_changed`）。
6. 在传输失败时重连。

手动实现上述这一切，是区分"玩具"和"可用产品"的关键。官方 SDK 封装了这些逻辑，但思维模型必须由你自己建立。

## 核心概念

### 子进程启动

使用 `subprocess.Popen`，设置 `stdin=PIPE, stdout=PIPE, stderr=PIPE`。设置 `bufsize=1` 并使用文本模式逐行读取。每个服务器对应一个进程，客户端持有每个服务器的一个 `Popen` 句柄。

### 每服务器的会话状态

每个服务器对应一个 `Session` 对象，包含：

- `process`——Popen 句柄。
- `capabilities`——服务器在 `initialize` 中声明的能力。
- `tools`——上次 `tools/list` 的结果。
- `pending`——请求 ID 到等待响应的 Promise/Future 的映射。

请求本质上是异步的；向服务器 A 发送 `tools/call` 时，服务器 B 正在处理另一个调用，不应阻塞。可以使用带队列的线程，也可以使用 asyncio。

### 合并命名空间

当客户端汇聚所有工具列表时，名称可能冲突。两个服务器可能都暴露了 `search`。客户端有三种选择：

1. **按服务器名称添加前缀。** `notes/search`、`files/search`。清晰但冗长。
2. **静默优先。** 后加载的服务器的 `search` 覆盖前一个。有风险，会隐藏冲突。
3. **冲突拒绝。** 拒绝加载第二个服务器，通知用户。对安全敏感的宿主最安全。

Claude Desktop 使用按服务器名称前缀的方式；Cursor 使用冲突拒绝并显示清晰的错误；VS Code MCP 也采用按服务器名称前缀。

### 路由

合并后，分发表将 `tool_name -> session` 进行映射。模型按名称输出调用；客户端找到对应的会话，向该服务器的 stdin 写入 `tools/call` 消息，然后等待响应。

### 采样回调

如果服务器在 `initialize` 时声明了 `sampling` 能力，它可能会发送 `sampling/createMessage`，请求客户端运行其 LLM。客户端必须：

1. 阻塞对该服务器的进一步请求，直到采样完成；或者若实现支持并发，则进行流水线处理。
2. 调用其 LLM 提供商。
3. 将响应发回服务器。

第 11 课端到端讲解采样，本课为其提供存根以保证完整性。

### 通知处理

`notifications/tools/list_changed` 表示需要重新调用 `tools/list`。`notifications/resources/updated` 表示如果该资源正在使用中需要重新读取。通知不得产生响应——不要尝试回执它们。

客户端的一个常见 bug：在处理 `tools/call` 时阻塞读取循环，而此时一条通知正等在流中。应使用后台读取线程，将每条消息推送到队列中；主线程从队列中取出消息并分发处理。

### 重连

传输可能失败：服务器崩溃、OS 杀死进程、stdio 管道断开。客户端检测到 stdout 的 EOF，将该会话标记为已死亡。选项：

- 静默重启服务器并重新握手。适用于纯只读服务器。
- 向用户显示错误。适用于有用户可见会话的有状态服务器。

Phase 13 · 09 讲解 Streamable HTTP 的重连语义；stdio 更为简单。

### 保活与会话 ID

Streamable HTTP 使用 `Mcp-Session-Id` 请求头。Stdio 没有会话 ID——进程标识就是会话。保活 ping 是可选的；stdio 管道在空闲时不会断开。

## 动手实践

`code/main.py` 将三个模拟 MCP 服务器作为子进程启动，与每个服务器握手，合并工具列表，并将工具调用路由到正确的服务器。这些"服务器"实际上是运行简单响应器的其他 Python 进程（无真实 LLM）。运行后可以看到：

- 三次各自独立能力集的初始化。
- 三个 `tools/list` 结果合并成一个包含 7 个工具的命名空间。
- 基于工具名称的路由决策。
- 通过命名空间前缀阻止的冲突。

重点关注：
- `Session` 数据类干净地保存每服务器的状态。
- 后台读取线程不阻塞主线程地从 stdout 逐行读取消息。
- 分发表是一个简单的 `dict[str, Session]`。
- 冲突处理是明确的：当两个服务器声明相同名称时，后者被添加前缀重命名。

## 产出技能

本课产出 `outputs/skill-mcp-client-harness.md`：给定一个声明式的 MCP 服务器列表（名称、命令、参数），该技能生成一个能够启动服务器、合并工具列表并提供带冲突解决的路由函数的测试套件。

## 练习

1. 运行 `code/main.py`，观察服务器启动日志。向一个模拟服务器进程发送 SIGTERM，观察客户端如何检测到 EOF 并将该会话标记为已死亡。

2. 实现命名空间前缀。当两个服务器都暴露 `search` 时，将第二个重命名为 `<server>/search`。更新分发表并验证工具调用能正确路由。

3. 为服务器重启添加连接池风格的退避策略：连续失败时指数退避，上限 30 秒，三次失败后向用户发出通知。

4. 设想一个支持 100 个并发 MCP 服务器的客户端，用什么数据结构替代简单的分发 dict？（提示：用于前缀命名的 trie，加上每服务器工具数量的指标。）

5. 将客户端移植到官方 MCP Python SDK。SDK 封装了 `stdio_client` 和 `ClientSession`。代码应从约 200 行压缩到约 40 行，同时保留多服务器路由能力。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| MCP 客户端 | "智能体宿主" | 启动服务器并编排工具调用的进程 |
| 会话（Session） | "每服务器状态" | 能力、工具列表和待处理请求的簿记 |
| 合并命名空间 | "一个工具列表" | 所有活动服务器的工具名称的扁平集合 |
| 命名空间冲突 | "两个服务器同名工具" | 客户端必须添加前缀、拒绝或采用先到先得策略 |
| 路由（Routing） | "这个调用发给谁？" | 从工具名称分发到拥有该工具的服务器 |
| 后台读取器 | "非阻塞 stdout" | 将服务器 stdout 的内容转入队列的线程或任务 |
| 采样回调 | "LLM 即服务" | 客户端对服务器发送的 `sampling/createMessage` 的处理器 |
| `notifications/*_changed` | "原语发生变更" | 通知客户端需要重新发现或重新读取的信号 |
| 重连策略 | "服务器挂掉时" | 传输失败时的重启语义 |
| Stdio 会话 | "进程即会话" | 无会话 ID；子进程的生命周期就是会话的生命周期 |

## 延伸阅读

- [模型上下文协议——客户端规范](https://modelcontextprotocol.io/specification/2025-11-25/client) — 规范客户端行为
- [MCP——客户端快速入门指南](https://modelcontextprotocol.io/quickstart/client) — 使用 Python SDK 的 hello-world 客户端教程
- [MCP Python SDK——client 模块](https://github.com/modelcontextprotocol/python-sdk) — 参考 `ClientSession` 和 `stdio_client`
- [MCP TypeScript SDK——Client](https://github.com/modelcontextprotocol/typescript-sdk) — TypeScript 对应实现
- [VS Code——扩展中的 MCP](https://code.visualstudio.com/api/extension-guides/ai/mcp) — VS Code 如何在单个编辑器宿主中多路复用多个 MCP 服务器
