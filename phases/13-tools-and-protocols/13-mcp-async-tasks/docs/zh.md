# 异步任务（SEP-1686）——立即调用、稍后获取的长时间运行工作

> 真实的智能体工作需要数分钟乃至数小时：CI 运行、深度研究综合、批量导出。同步工具调用会断开连接、超时或阻塞 UI。2025-11-25 合并的 SEP-1686 添加了任务（Tasks）原语：任何请求都可以被增强为任务，结果可以稍后获取，或通过状态通知进行流式传输。漂移风险提示：任务在 2026 年上半年仍处于实验阶段；SDK 接口仍在围绕规范进行设计。

**类型：** 构建
**编程语言：** Python（标准库，异步任务状态机）
**前置知识：** Phase 13 · 07（MCP 服务器）、Phase 13 · 09（传输层）
**预计时间：** 约 75 分钟

## 学习目标

- 识别何时应将工具从同步升级为支持任务的方式（服务器端工作超过 30 秒）。
- 逐步讲解任务生命周期：`working` → `input_required` → `completed` / `failed` / `cancelled`。
- 持久化任务状态，使崩溃不会丢失进行中的工作。
- 正确轮询 `tasks/status` 并获取 `tasks/result`。

## 问题背景

一个 `generate_report` 工具运行一个需要几分钟的提取流水线。在同步模型下的选项：

1. 保持连接开放三分钟。远程传输会断开连接；客户端超时；UI 冻结。
2. 立即返回占位符；要求客户端轮询自定义端点。破坏了 MCP 的统一性。
3. 发后不管；没有结果。

这些都不是好方案。SEP-1686 添加了第四种：任务增强。任何请求（通常是 `tools/call`）都可以被标记为任务。服务器立即返回一个任务 ID。客户端轮询 `tasks/status`，完成后获取 `tasks/result`。服务器端状态在重启后仍然存在。

## 核心概念

### 任务增强

通过设置 `params._meta.task.required: true`（或 `optional: true`，由服务器决定），请求变为任务。服务器立即响应：

```json
{
  "jsonrpc": "2.0", "id": 1,
  "result": {
    "_meta": {
      "task": {
        "id": "tsk_9f7b...",
        "state": "working",
        "ttl": 900000
      }
    }
  }
}
```

`ttl` 是服务器承诺保留状态的时限；超过 ttl 后，任务结果将被丢弃。

### 每工具的选择性支持

工具注解可以声明任务支持：

- `taskSupport: "forbidden"`——此工具始终同步运行。适用于快速工具。
- `taskSupport: "optional"`——客户端可以请求任务增强。
- `taskSupport: "required"`——客户端**必须**使用任务增强。

`generate_report` 工具应设为 `required`，`notes_search` 工具应设为 `forbidden`。

### 状态

```
working  -> input_required -> working  （通过引导循环）
working  -> completed
working  -> failed
working  -> cancelled
```

状态机是追加性的：一旦进入 `completed`、`failed` 或 `cancelled`，任务即为终态。

### 方法

- `tasks/status {taskId}`——返回当前状态和进度提示。
- `tasks/result {taskId}`——阻塞等待，或在未完成时返回 404。
- `tasks/cancel {taskId}`——幂等；终态忽略。
- `tasks/list`——可选；枚举活动的和最近完成的任务。

### 流式状态变更

当服务器支持时，客户端可以订阅状态通知：

```
server -> notifications/tasks/updated {taskId, state, progress?}
```

通过流式方式而非轮询的客户端可以获得更好的 UX。轮询始终作为最小接口被支持。

### 持久化状态

规范要求声明任务支持的服务器持久化状态。崩溃不应在 ttl 内丢失已完成的结果。存储方案从 SQLite 到 Redis 到文件系统都可以。第 13 课的测试套件使用文件系统。

### 取消语义

`tasks/cancel` 是幂等的。如果任务正在执行中，服务器会尝试停止（检查执行器是否支持协作取消）。如果已经是终态，该请求是无操作。

### 崩溃恢复

服务器进程重启时：

1. 加载所有持久化的任务状态。
2. 将进程死亡前仍处于 `working` 状态的任务标记为 `failed`，错误为 `CRASH_RECOVERY`。
3. 在 ttl 内保留 `completed` / `failed` / `cancelled` 状态。

### 异步任务加采样

任务本身可以调用 `sampling/createMessage`。这正是长时间运行的研究任务的工作方式：服务器的任务线程按需对客户端的模型进行采样，同时客户端 UI 显示任务为 `working` 并定期更新进度。

### 为何仍处于实验阶段

SEP-1686 在 2025-11-25 发布，但更广泛的路线图指出了三个未解决的问题：持久订阅原语、子任务（父子任务关系）和结果 TTL 标准化。预计规范将在 2026 年持续演进。生产代码应将任务视为仅对常见情况稳定，并对未来 SDK 中子任务的变化进行防御性处理。

## 动手实践

`code/main.py` 实现了一个持久化任务存储（基于文件系统）和一个在后台线程中运行的 `generate_report` 工具。客户端调用工具后立即得到任务 ID，在工作线程更新进度的同时轮询 `tasks/status`，完成后获取 `tasks/result`。取消功能可用；通过终止工作线程并重新加载状态来模拟崩溃恢复。

重点关注：
- 任务状态 JSON 持久化到 `/tmp/lesson-13-tasks/<id>.json`。
- 工作线程更新 `progress` 字段；轮询显示进度在推进。
- 客户端侧的取消设置一个事件；工作线程检查并提前退出。
- "崩溃"时的状态重新加载将进行中的任务标记为带有 `CRASH_RECOVERY` 的 `failed` 状态。

## 产出技能

本课产出 `outputs/skill-task-store-designer.md`：给定一个长时间运行的工具（研究、构建、导出），该技能设计任务存储（状态形态、ttl、持久性），选择正确的 `taskSupport` 标志，并勾勒进度通知方案。

## 练习

1. 运行 `code/main.py`。启动一个 `generate_report` 任务，轮询状态，然后获取结果。

2. 在运行中途添加 `tasks/cancel` 调用。验证工作线程响应取消，状态变为 `cancelled`。

3. 模拟崩溃恢复：终止工作线程，重启加载器，观察 `CRASH_RECOVERY` 失败模式。

4. 将存储扩展为 SQLite。持久性优势相同；查询选项增加（列出会话 X 的所有任务）。

5. 阅读 2026 年 MCP 路线图博文。找出最有可能在下一年影响 SDK API 设计的那个与任务相关的未解决问题。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 任务（Task） | "长时间运行的工具调用" | 使用 `_meta.task` 增强的用于异步执行的请求 |
| SEP-1686 | "任务规范" | 在 2025-11-25 中添加任务的规范演进提案 |
| `_meta.task` | "任务信封" | 包含 id、状态、ttl 的每请求元数据 |
| taskSupport | "工具标志" | 每个工具的 `forbidden` / `optional` / `required` |
| `tasks/status` | "轮询方法" | 获取当前状态和可选的进度提示 |
| `tasks/result` | "获取结果" | 返回已完成的载荷，或在未完成时返回 404 |
| `tasks/cancel` | "停止它" | 幂等的取消请求 |
| ttl | "保留预算" | 服务器承诺保留任务状态的毫秒数 |
| `notifications/tasks/updated` | "状态推送" | 服务器发起的状态变更事件 |
| 持久化存储 | "崩溃安全状态" | 文件系统 / SQLite / Redis 持久化层 |

## 延伸阅读

- [MCP——GitHub SEP-1686 议题](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1686) — 原始提案与完整讨论
- [WorkOS——用于 AI 智能体工作流的 MCP 异步任务](https://workos.com/blog/mcp-async-tasks-ai-agent-workflows) — 设计详解与设计理由
- [DeepWiki——MCP 任务系统与异步操作](https://deepwiki.com/modelcontextprotocol/modelcontextprotocol/2.7-task-system-and-async-operations) — 机制与状态机
- [FastMCP——任务](https://gofastmcp.com/servers/tasks) — SDK 级别的任务实现模式
- [MCP 博客——2026 年路线图](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — 未解决问题和 2026 年优先项（含子任务）
