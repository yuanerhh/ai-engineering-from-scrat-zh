# 根与引导——范围控制与飞行中的用户输入

> 硬编码路径在用户打开不同项目时立刻失效。预填充的工具参数在用户描述不够具体时也会失效。根（Roots）将服务器限定在用户可控的 URI 集合内；引导（Elicitation）则在工具调用过程中暂停，通过表单或 URL 向用户请求结构化输入。两个客户端原语，解决两种常见的 MCP 失败模式。SEP-1036（URL 模式引导，2025-11-25）在 2026 年上半年仍处于实验阶段——使用前请检查 SDK 版本。

**类型：** 构建
**编程语言：** Python（标准库，roots + elicitation 演示）
**前置知识：** Phase 13 · 07（MCP 服务器）
**预计时间：** 约 45 分钟

## 学习目标

- 声明 `roots` 并响应 `notifications/roots/list_changed`。
- 将服务器文件操作限制在声明的根集合内的 URI 范围内。
- 使用 `elicitation/create` 在工具调用中途向用户请求确认或结构化输入。
- 在表单模式和 URL 模式引导之间做出选择（后者是实验性的，注意漂移风险）。

## 问题背景

一个笔记 MCP 服务器在生产中遇到的两个具体失败案例。

**路径假设失效。** 服务器是针对 `~/notes` 编写的。笔记放在 `~/Documents/Notes` 的用户会遇到静默失败的工具调用（找不到文件），甚至更糟——写入了错误的位置。

**用户才知道的缺失参数。** 用户说"删除那个旧的 TPS 报告笔记"。模型调用 `notes_delete(title: "TPS report")`，但存在来自 2023、2024 和 2025 年的三条匹配笔记。工具无法猜测。以"不明确"失败让人恼火；对三条全部执行则是灾难性的。

根解决了第一个问题：客户端在 `initialize` 时声明服务器可以访问的 URI 集合。引导解决了第二个问题：服务器暂停工具调用，发送 `elicitation/create` 让用户选择哪一条。

## 核心概念

### 根

客户端在 `initialize` 时声明根列表：

```json
{
  "capabilities": {"roots": {"listChanged": true}}
}
```

然后服务器可以调用 `roots/list`：

```json
{"roots": [{"uri": "file:///Users/alice/Documents/Notes", "name": "Notes"}]}
```

服务器**必须**将根视为边界：拒绝根集合之外的任何文件读写。这不由客户端强制执行（服务器仍然是用户信任的代码），但符合规范的服务器会遵守。

当用户添加或删除根时，客户端发送 `notifications/roots/list_changed`。服务器重新调用 `roots/list` 并更新其边界。

### 根为何是客户端原语

根由客户端声明，因为它代表了用户的授权模型。用户告诉 Claude Desktop"给这个笔记服务器访问这两个目录的权限"。服务器不能扩大该范围。

### 引导：默认的表单模式

`elicitation/create` 接受一个表单 schema 加上自然语言提示：

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "删除'TPS 报告'？找到多条匹配的笔记，请选择一条。",
    "requestedSchema": {
      "type": "object",
      "properties": {
        "note_id": {
          "type": "string",
          "enum": ["note-3", "note-7", "note-14"]
        },
        "confirm": {"type": "boolean"}
      },
      "required": ["note_id", "confirm"]
    }
  }
}
```

客户端渲染表单，收集用户答案，返回：

```json
{
  "action": "accept",
  "content": {"note_id": "note-14", "confirm": true}
}
```

三种可能的行动：`accept`（用户填写了表单）、`decline`（用户关闭了表单）、`cancel`（用户中止了整个工具调用）。

表单 schema 是扁平的——v1 中不支持嵌套对象。SDK 通常会拒绝超过单层结构的内容。

### 引导：URL 模式（SEP-1036，实验性）

2025-11-25 新增。服务器发送 URL 而非 schema：

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "登录 GitHub",
    "url": "https://github.com/login/oauth/authorize?client_id=..."
  }
}
```

客户端在浏览器中打开该 URL，等待操作完成，用户返回后恢复。适用于表单不足以处理的 OAuth 流程、支付授权和文件签署场景。

漂移风险提示：SEP-1036 的响应格式仍在稳定中；某些 SDK 返回回调 URL，其他 SDK 返回完成令牌。在生产中使用 URL 模式前，请阅读 SDK 的发布说明。

### 引导的正确使用场景

- 破坏性操作前的用户确认（配合 destructive 提示）。
- 消歧义（从 N 个匹配中选择一个）。
- 首次运行设置（API 密钥、目录、偏好设置）。
- OAuth 风格的流程（URL 模式）。

### 引导的错误使用场景

- 填充模型本可以用文本重新询问的工具必填参数。用正常的重新提示，而不是引导对话框。
- 高频调用。引导会打断对话；不要在循环中触发。
- 服务器事后可以验证的内容。验证、返回错误、让模型以文本形式询问用户。

### 人在环桥梁

引导加上采样共同实现了 MCP 的"人在环"模型。服务器的智能体循环既可以为用户输入暂停（引导），也可以为模型推理暂停（采样）。Phase 13 · 11 讲解了采样，本课讲解引导。将两者结合，即可实现完整的循环中控制。

## 动手实践

`code/main.py` 在笔记服务器基础上扩展了：

- 服务器在根列表变更通知后重新查询的 `roots/list` 响应。
- 当多条笔记匹配时使用 `elicitation/create` 进行消歧义的 `notes_delete` 工具。
- 使用 URL 模式引导打开首次运行配置页面（模拟）的 `notes_setup` 工具。
- 拒绝对声明根之外 URI 进行操作的边界检查。

演示运行三种场景：正常路径（一条匹配）、消歧义（三条匹配，触发引导）、超出根范围的写入（被拒绝）。

## 产出技能

本课产出 `outputs/skill-elicitation-form-designer.md`：给定一个可能需要用户确认或消歧义的工具，该技能设计引导表单 schema 和消息模板。

## 练习

1. 运行 `code/main.py`。触发消歧义路径；确认模拟的用户答案被正确路由回工具。

2. 添加一个每次都需要引导确认的新工具 `notes_archive`（带 destructive 提示）。检查 UX：与模型以文本形式重新询问相比，体验如何？

3. 为首次运行的 OAuth 流程实现 URL 模式引导。注意漂移风险，添加 SDK 版本检查。

4. 扩展 `roots/list` 处理：当通知到达时，服务器应原子性地重新读取并扫描可能已超出范围的打开文件句柄。

5. 阅读 GitHub 上的 SEP-1036 讨论线程。找出一个影响服务器应如何处理 URL 模式回调的未解决问题。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 根（Root） | "授权边界" | 客户端允许服务器访问的 URI |
| `roots/list` | "服务器请求范围" | 客户端返回当前根集合 |
| `notifications/roots/list_changed` | "用户更改了范围" | 客户端发出根集合已变化的信号 |
| 引导（Elicitation） | "调用中途询问用户" | 服务器发起的结构化用户输入请求 |
| `elicitation/create` | "该方法" | 引导请求的 JSON-RPC 方法 |
| 表单模式 | "Schema 驱动的表单" | 以表单形式在客户端 UI 中渲染的扁平 JSON Schema |
| URL 模式 | "浏览器跳转" | SEP-1036 实验性功能；打开 URL 并等待 |
| `accept` / `decline` / `cancel` | "用户响应结果" | 服务器处理的三个分支 |
| 消歧义（Disambiguation） | "选择一个" | 当工具有 N 个候选时常见的引导使用场景 |
| 扁平表单 | "仅顶层属性" | 引导 schema 不能嵌套 |

## 延伸阅读

- [MCP——客户端根规范](https://modelcontextprotocol.io/specification/draft/client/roots) — 根的权威参考
- [MCP——客户端引导规范](https://modelcontextprotocol.io/specification/draft/client/elicitation) — 引导的权威参考
- [Cisco——MCP 引导、结构化内容、OAuth 增强功能中的新特性](https://blogs.cisco.com/developer/whats-new-in-mcp-elicitation-structured-content-and-oauth-enhancements) — 2025-11-25 新增功能逐步解析
- [MCP——GitHub SEP-1036](https://github.com/modelcontextprotocol/modelcontextprotocol) — URL 模式引导提案（实验性，注意漂移风险）
- [The New Stack——引导如何为 AI 工具引入人在环](https://thenewstack.io/how-elicitation-in-mcp-brings-human-in-the-loop-to-ai-tools/) — UX 详细说明
