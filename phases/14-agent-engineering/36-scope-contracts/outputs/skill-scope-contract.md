---
name: scope-contract
description: 生成包含允许/禁止 glob、验收标准和回滚计划的单任务范围合约，以及在每次 Agent diff 上运行的支持 CI 的 glob 感知检查器。
version: 1.0.0
phase: 14
lesson: 36
tags: [scope, contract, globs, diff-check, ci]
---

给定任务描述和代码仓库布局，生成范围合约和差异感知检查器。

需产出：

1. 任务的 `scope_contract.json`，包含以下字段：`task_id`、`goal`、`allowed_files`（glob）、`forbidden_files`（glob）、`acceptance_criteria`、`rollback_plan`、`approvals_required`。
2. `tools/scope_check.py`，接受合约路径和已触及文件列表，返回 `ScopeReport`，并在任何违规时以非零状态退出。
3. CI 步骤（`.github/workflows/scope-check.yml` 或等效文件），针对合并差异运行检查器。
4. `outputs/scope/closed/<task_id>.json` 存档约定，使合约随变更历史一起交付。

强制拒绝：

- 没有 `forbidden_files` 的合约。负空间也是合约的一部分。
- 对代码目录使用原始路径而非 glob 的合约。重构会在一夜之间使原始路径失效。
- `rollback_plan` 字段为空或写着"见运维手册"。必须明确说明。
- 审批要求写成"视情况而定"。审批边界必须是可枚举的。

拒绝规则：

- 如果任务描述没有限定仓库中的某个区域，拒绝仅凭描述来编写 `allowed_files`。需要询问任务所在的目录。
- 如果仓库没有测试命令，在提供或补充测试命令之前拒绝添加 `acceptance_criteria`。无法验证的合约只是一个愿望。
- 如果 Agent 运行时无法遵守审批边界（无人工介入），在交付前暴露该缺口；范围蔓延到需要审批的操作将是最主要的失败原因。

输出结构：

```
<repo>/
├── scope_contract.json
├── outputs/scope/closed/
│   └── T-XXX.json
├── tools/
│   └── scope_check.py
└── .github/
    └── workflows/
        └── scope-check.yml
```

最后以"下一步阅读"结尾，指向：

- 第 37 课，了解将已运行命令与合约关联起来的运行时反馈。
- 第 38 课，了解消费范围报告的验证门控。
- 第 39 课，了解对已关闭合约存档进行审计的审查者 Agent。
