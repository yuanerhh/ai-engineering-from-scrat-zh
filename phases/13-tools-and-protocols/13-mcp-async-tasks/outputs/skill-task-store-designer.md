---
name: task-store-designer
description: 为长时运行的 MCP 工具设计任务存储：状态结构、TTL、持久化、取消及崩溃恢复。
version: 1.0.0
phase: 13
lesson: 13
tags: [mcp, tasks, durable-store, long-running, sep-1686]
---

给定一个长时运行工具（研究、构建、导出、报告生成），设计支持 SEP-1686 任务增强的任务存储。

输出内容：

1. **状态结构**。最小字段：`id`、`state`、`progress`、`result`、`error`、`ttl`、`created_at`。可选项：`request_meta`、`parent_task_id`（用于未来子任务）。
2. **持久化选择**。玩具项目用文件系统；单进程用 SQLite；多副本用 Redis。需说明理由。
3. **taskSupport 标志**。每个工具取 `forbidden`、`optional` 或 `required` 之一，附单行理由。
4. **取消方案**。工作进程如何检测取消信号；部分进度下的处理方式。
5. **崩溃恢复**。启动时重载规则；`CRASH_RECOVERY` 失败对客户端的表现。

硬性拒绝：
- 任何在 TTL 内丢失已完成结果的存储方案。
- 任何未明确定义终止状态（`completed`、`failed`、`cancelled`）的任务状态设计。
- 任何不具有幂等性的取消操作。

拒绝规则：
- 若工具运行时间不超过 5 秒，拒绝将其提升为任务。同步方式更简单。
- 若任务将产生超过 10 MB 的结果，拒绝并建议改用流式内容块。
- 若服务器没有能够持久化状态的进程（无状态边缘函数），拒绝并建议迁移到持久化运行时。

输出：一页存储设计文档，包含状态结构、持久化选择、taskSupport 标志、取消方案和崩溃恢复规则。结尾给出一句话建议，说明 SEP-1686 子任务发布后是否会影响该设计。
