# 构建 MCP 服务器——Python 与 TypeScript SDK

> 大多数 MCP 教程只展示 stdio 版的 hello world。一个真正的服务器要暴露工具、资源和提示，处理能力协商，输出结构化错误，并在各种 SDK 间保持一致行为。本课端到端构建一个笔记服务器：标准库 stdio 传输、JSON-RPC 分发、三种服务器原语，以及一种纯函数风格——只需稍作改动，便可迁移到 Python SDK 的 FastMCP 或 TypeScript SDK。

**类型：** 构建
**编程语言：** Python（标准库，stdio MCP 服务器）
**前置知识：** Phase 13 · 06（MCP 基础）
**预计时间：** 约 75 分钟

## 学习目标

- 实现 `initialize`、`tools/list`、`tools/call`、`resources/list`、`resources/read`、`prompts/list` 和 `prompts/get` 方法。
- 编写一个从 stdin 读取 JSON-RPC 消息并将响应写入 stdout 的分发循环。
- 按照 JSON-RPC 2.0 规范和 MCP 附加错误码输出结构化错误响应。
- 在不重写工具逻辑的情况下，将标准库实现迁移到 FastMCP（Python SDK）或 TypeScript SDK。

## 问题背景

在使用远程传输（Phase 13 · 09）或认证层（Phase 13 · 16）之前，需要一个干净的本地服务器。本地意味着 stdio：服务器由客户端作为子进程启动，消息以换行符分隔的方式在 stdin/stdout 上传输。

2025-11-25 规范规定 stdio 消息被编码为以显式 `\n` 为分隔符的 JSON 对象。这里没有 SSE；SSE 是旧的远程模式，将于 2026 年中期移除（Atlassian 的 Rovo MCP 服务器于 2026 年 6 月 30 日废弃，Keboola 于 2026 年 4 月 1 日废弃）。对于 stdio，每行一个 JSON 对象就是完整的线路格式。

笔记服务器是一个很好的示例形态，因为它覆盖了全部三种服务器原语。工具执行变更操作（`notes_create`），资源暴露数据（`notes://{id}`），提示提供模板（`review_note`）。本课的形态可推广到任何领域。

## 核心概念

### 分发循环

```
loop:
  line = stdin.readline()
  msg = json.loads(line)
  if has id:
    handle request -> write response
  else:
    handle notification -> no response
```

三条规则：

- 向 stdout 输出的内容必须全部是 JSON-RPC 信封。调试日志输出到 stderr。
- 每个请求**必须**配有携带相同 `id` 的响应。
- **不得**对通知进行响应。

### 实现 `initialize`

```python
def initialize(params):
    return {
        "protocolVersion": "2025-11-25",
        "capabilities": {
            "tools": {"listChanged": True},
            "resources": {"listChanged": True, "subscribe": False},
            "prompts": {"listChanged": False},
        },
        "serverInfo": {"name": "notes", "version": "1.0.0"},
    }
```

只声明你实际支持的能力。客户端依赖这个能力集来决定是否启用对应功能。

### 实现 `tools/list` 和 `tools/call`

`tools/list` 返回 `{tools: [...]}`，每项包含 `name`、`description`、`inputSchema`。`tools/call` 接受 `{name, arguments}`，返回 `{content: [blocks], isError: bool}`。

内容块是类型化的，最常见的几种：

```json
{"type": "text", "text": "Found 2 notes"}
{"type": "resource", "resource": {"uri": "notes://14", "text": "..."}}
{"type": "image", "data": "<base64>", "mimeType": "image/png"}
```

工具错误有两种形式。协议级错误（未知方法、参数错误）使用 JSON-RPC 错误格式。工具级错误（调用合法但工具本身失败）以 `{content: [...], isError: true}` 的形式返回，这样模型就能在上下文中看到失败信息。

### 实现资源

资源按设计是只读的。`resources/list` 返回清单；`resources/read` 返回内容。URI 可以是 `file://...`、`http://...` 或自定义 scheme，如 `notes://`。

将数据作为资源而非工具暴露时的优势：

- 模型不会主动"调用"它，客户端可在用户请求时将其注入到上下文中。
- 订阅允许服务器在资源发生变化时推送更新（Phase 13 · 10）。
- Phase 13 · 14 通过 `ui://` 扩展了交互式资源。

### 实现提示

提示是带命名参数的模板，宿主以斜杠命令的形式呈现给用户。例如，`review_note` 提示可以接受一个 `note_id` 参数，生成一个多消息的提示模板，由客户端传递给其模型。

### Stdio 传输的细节

- 换行符分隔的 JSON，无长度前缀帧。
- 不要缓冲，每次写入后调用 `sys.stdout.flush()`。
- 客户端控制进程生命周期。当 stdin 关闭（EOF）时，干净退出。
- 不要静默处理 SIGPIPE，应记录日志并退出。

### 注解（Annotations）

每个工具可以携带描述安全属性的 `annotations`：

- `readOnlyHint: true`——纯读取，可安全重试。
- `destructiveHint: true`——不可逆的副作用；客户端应请求确认。
- `idempotentHint: true`——相同输入产生相同输出。
- `openWorldHint: true`——与外部系统交互。

客户端使用这些信息决定 UX（确认对话框、状态指示器）和路由策略（Phase 13 · 17）。

### 迁移路径

`code/main.py` 中的标准库服务器大约 180 行。FastMCP（Python）将同样的逻辑压缩到装饰器风格：

```python
from fastmcp import FastMCP
app = FastMCP("notes")

@app.tool()
def notes_search(query: str, limit: int = 10) -> list[dict]:
    ...
```

TypeScript SDK 有等价的形态。当你准备好时，迁移是直接替换；概念（能力、分发、内容块）完全一致。

## 动手实践

`code/main.py` 是一个完整的基于 stdio 的笔记 MCP 服务器，仅使用标准库。它处理 `initialize`、`tools/list`、三个工具的 `tools/call`（`notes_list`、`notes_search`、`notes_create`）、`resources/list` 和每条笔记的 `resources/read`，以及 `review_note` 提示。可以通过管道传递 JSON-RPC 消息来驱动它：

```
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | python main.py
```

重点关注：
- 分发器是一个以方法名为键的 `dict[str, Callable]`。
- 每个工具执行器返回内容块列表，而不是裸字符串。
- 执行器抛出异常时设置 `isError: true`。

## 产出技能

本课产出 `outputs/skill-mcp-server-scaffolder.md`：给定一个领域（笔记、工单、文件、数据库），该技能为其搭建包含工具/资源/提示合理划分的 MCP 服务器骨架，并附带 SDK 迁移路径。

## 练习

1. 运行 `code/main.py`，用手写的 JSON-RPC 消息驱动它。先执行 `notes_create`，然后用 `resources/read` 获取新创建的笔记。

2. 添加带 `annotations: {destructiveHint: true}` 的 `notes_delete` 工具。验证客户端会弹出确认对话框（需要真实宿主；Claude Desktop 可以使用）。

3. 实现 `resources/subscribe`，使服务器在笔记被修改时推送 `notifications/resources/updated`。添加一个保活任务。

4. 将服务器移植到 FastMCP。Python 文件应压缩到 80 行以内。线路行为必须完全一致；用相同的 JSON-RPC 测试套件验证。

5. 阅读规范的 `server/tools` 部分，找出本课服务器未实现的工具定义字段之一。（提示：有好几个；选一个并添加它。）

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| MCP 服务器 | "暴露工具的那个进程" | 通过 stdio 或 HTTP 使用 MCP JSON-RPC 通信的进程 |
| stdio 传输 | "子进程模型" | 服务器由客户端启动；通过 stdin/stdout 通信 |
| 分发器（Dispatcher） | "方法路由器" | JSON-RPC 方法名到处理函数的映射 |
| 内容块（Content block） | "工具结果块" | 工具响应 `content` 数组中的类型化元素 |
| `isError` | "工具级失败" | 表示工具失败；区别于 JSON-RPC 错误 |
| 注解（Annotations） | "安全提示" | readOnly / destructive / idempotent / openWorld 标志 |
| FastMCP | "Python SDK" | 基于装饰器的 MCP 协议上层框架 |
| 资源 URI | "可寻址数据" | `file://`、`db://` 或标识资源的自定义 scheme |
| 提示模板（Prompt template） | "斜杠命令简报" | 服务器提供的带参数槽的模板，供宿主 UI 使用 |
| 能力声明（Capability declaration） | "功能开关" | 在 `initialize` 中声明的每个原语的标志 |

## 延伸阅读

- [模型上下文协议——Python SDK](https://github.com/modelcontextprotocol/python-sdk) — 参考 Python 实现
- [模型上下文协议——TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk) — 对应的 TS 实现
- [FastMCP——服务器框架](https://gofastmcp.com/) — MCP 服务器的装饰器风格 Python API
- [MCP——服务器快速入门指南](https://modelcontextprotocol.io/quickstart/server) — 使用任意 SDK 的端到端教程
- [MCP——服务器工具规范](https://modelcontextprotocol.io/specification/2025-11-25/server/tools) — tools/* 消息的完整参考
