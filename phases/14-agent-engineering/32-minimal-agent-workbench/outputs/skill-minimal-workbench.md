---
name: minimal-workbench
description: 为任意仓库搭建三文件最小可行 Agent 工作台——简短的 AGENTS.md 路由器、持久化的 agent_state.json，以及与项目当前待办事项关联的 JSON task_board.json。
version: 1.0.0
phase: 14
lesson: 32
tags: [workbench, agents-md, state, task-board, scaffold]
---

给定一个仓库路径和简短的待办列表，搭建最小可行 Agent 工作台。

输出内容：

1. `AGENTS.md` 不超过 80 行。必须路由到：状态文件、任务看板、更深层的规则文档（即使为空），以及验证命令。该文件中不放教程性散文。
2. `agent_state.json` 包含以下键：`active_task_id`、`touched_files`、`assumptions`、`blockers`、`next_action`。所有可选字段默认为空数组或空字符串，数组字段绝不使用 `null`。
3. `task_board.json` 为 JSON 数组，每个任务包含 `id`、`goal`、`owner`（`builder` | `reviewer` | `human`）、`acceptance`（字符串列表）和 `status`（`todo` | `in_progress` | `done` | `blocked`）。
4. `docs/agent-rules.md` 占位文件，每个工作台面各一个 H2 标题，供后续课程填充内容。

硬性拒绝：

- `AGENTS.md` 超过 80 行或少于 10 行。太长 Agent 会跳过；太短则无法承载路由信息。
- 状态文件引用聊天历史而非仓库。仓库是权威数据源。
- 没有 `acceptance` 的任务看板。没有验收标准的任务会变成"看起来还行"的橡皮图章。
- `owner` 为 `agent` 或 `model` 的任务。Owner 是角色，不是实体。

拒绝规则：

- 如果仓库没有验证命令，拒绝编写 `AGENTS.md`，直到提供或存根化一个验证命令。一个指向缺失门控的路由器比没有路由器更糟糕。
- 如果待办列表有超过 12 个开放任务，拒绝并要求用户拆分。超过一屏的看板会退化为规划作秀。
- 如果项目在已追踪文件中存储密钥，拒绝编写状态文件，并首先将密钥泄漏列为阻塞性发现。

输出结构：

```
<repo>/
├── AGENTS.md
├── agent_state.json
├── task_board.json
└── docs/
    └── agent-rules.md
```

最后附"下一步阅读"，指向：

- 第 33 课，将规则占位文件转化为可执行约束。
- 第 34 课，了解持久化状态 Schema。
- 第 36 课，了解每个任务的范围契约。
