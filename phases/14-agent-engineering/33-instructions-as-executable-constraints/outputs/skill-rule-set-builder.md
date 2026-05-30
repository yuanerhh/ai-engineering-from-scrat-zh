---
name: rule-set-builder
description: 访谈项目负责人，将其现有散文指令分类为五个操作类别，并生成带版本的 agent-rules.md 及 Python 检查器存根。
version: 1.0.0
phase: 14
lesson: 33
tags: [rules, instructions, constraints, checker, workbench]
---

给定一个仓库和任何现有散文指令（`AGENTS.md`、`CONTRIBUTING.md`、入职文档），生成工作台可执行的五类规则集。

五个类别：

1. `startup` —— 工作开始前必须满足的条件。
2. `forbidden` —— 绝对不能发生的事情。
3. `definition_of_done` —— 证明任务完成的依据。
4. `uncertainty` —— Agent 不确定时的处理方式。
5. `approval` —— 需要人工审批的事项。

输出内容：

1. `docs/agent-rules.md`，每条规则一个 `##` 标题。每条规则包含 `category`、`check` 和一行描述。
2. `tools/rule_checker.py`，包含 `RuleChecker` 类，每个 `check` 对应一个方法。每个方法接收 `TurnTrace` 数据类并返回 `bool`。
3. `tools/rule_report.py` 运行器，加载规则，对链路运行检查器，输出 `rule_report.json`。
4. 迁移说明文件：哪些散文行变成了哪条规则，哪些被舍弃为仅有抱负意义的内容，以及原因。

硬性拒绝：

- 没有 `check` 字段的规则。只有抱负意义的规则属于入职文档，不属于工作台规则集。
- 单一的"要小心"规则。需要指定类别和检查，否则删除它。
- 需要 LLM 调用的检查。规则检查必须是确定性的且廉价的，以便每轮都能运行。
- 规则文件超过 200 行。按类别拆分为 `agent-rules.{startup,forbidden,done,uncertainty,approval}.md`，并从父索引路由。

拒绝规则：

- 如果 Agent 产品无法提供 `TurnTrace`（没有插桩），拒绝接入检查器，直到至少记录了 `read_state_file`、`edited_files` 和 `tests_exit_code`。
- 如果现有指令大多是抱负性的（>50%），在生成规则前先呈现这一发现。规则集看起来会很单薄；这是正确的。
- 如果某条规则是因为单次历史事故而添加的，附上事故 ID，以便未来审查时决定是否仍然需要它。

输出结构：

```
<repo>/
├── docs/
│   └── agent-rules.md
├── tools/
│   ├── rule_checker.py
│   └── rule_report.py
└── docs/migration-notes.md
```

最后附"下一步阅读"，指向：

- 第 36 课，了解扩展 forbidden 类别的每任务范围契约。
- 第 38 课，了解使用规则报告的验证门控。
- 第 39 课，了解对规则合规性进行评分的审查者 Agent。
