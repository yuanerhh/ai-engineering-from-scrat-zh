---
name: state-schema
description: 为 Agent 状态和任务看板生成项目专属 JSON Schema，提供支持原子写入的 Python StateManager，以及防止 Schema 升级损坏工作台的迁移脚手架。
version: 1.0.0
phase: 14
lesson: 34
tags: [state, schema, json-schema, atomic-writes, migrations]
---

给定一个仓库和在其中运行的 Agent 产品，为工作台生成 Schema 优先的状态文件。

输出内容：

1. `schemas/agent_state.schema.json`，涵盖必填键、允许的状态值、数组与 null 的约束，以及整数类型的 `schema_version`。
2. `schemas/task_board.schema.json`，涵盖任务 ID 模式、允许的 owner、允许的状态，以及验收数组。
3. `tools/state_manager.py`，暴露 `load`、`commit` 和 `update` 方法，使用先写临时文件再重命名的原子写入方式。
4. `tools/migrate_state.py`，下次 Schema 升级时的迁移脚手架，遇到未知版本时大声报错。
5. `agent_state.json` 和 `task_board.json`，以 `schema_version: 1` 和全新待办列表为初始值。

硬性拒绝：

- 没有 `schema_version` 字段的 Schema。迁移不是可选项。
- 在预期为数组的地方允许 `null`。`null` 是写入时的 bug，伪装成数据。
- 使用普通 `open(path, "w")` 的写入器。只允许原子写入；部分文件会损坏权威数据源。
- 在状态中存储 token、原始聊天记录或 PII。状态只用于与仓库相关的事实。

拒绝规则：

- 如果仓库没有版本控制，拒绝交付状态文件。原子写入加 git diff 是持久性保障。
- 如果项目没有至少一条验证 `done` 状态转换的验收命令，拒绝 `status: done` 枚举值。没有验收检查就添加 `done` 是走过场。
- 如果项目打算跨进程共享状态而没有锁策略，在交付前呈现这一发现；原子重命名是必要条件，但不是充分条件。

输出结构：

```
<repo>/
├── agent_state.json
├── task_board.json
├── schemas/
│   ├── agent_state.schema.json
│   └── task_board.schema.json
└── tools/
    ├── state_manager.py
    └── migrate_state.py
```

最后附"下一步阅读"，指向：

- 第 35 课，了解在启动时调用管理器的初始化脚本。
- 第 38 课，了解读取状态以评分完成度的验证门控。
- 第 40 课，了解使用相同 Schema 的交接生成器。
