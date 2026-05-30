---
name: workbench-pack
description: 生成一个针对项目调优的即插即用 Agent 工作台包——规则针对团队历史记录进行了锐化，范围 glob 与仓库匹配，评分标准维度增加了一个领域专属条目。
version: 1.0.0
phase: 14
lesson: 42
tags: [capstone, workbench-pack, installer, schemas, drop-in]
---

给定一个代码仓库、团队的事故历史以及在其中运行的 Agent 产品，生成一个调优的 agent-workbench-pack 及其安装程序。

需产出：

1. `agent-workbench-pack/` 目录，匹配规范布局：AGENTS.md、docs/、schemas/、scripts/、bin/、README.md、VERSION。
2. `bin/install.sh`，在没有 `--force` 标志的情况下拒绝覆盖已有包，并将 `.workbench-version` 写入目标仓库。
3. 项目调优版本的 `agent-rules.md`（每个类别至少包含一条从团队最近六次事故中提炼的规则）、`reviewer-rubric.md`（包含第六个领域维度）以及 `scope_contract.schema.json`（包含项目专属 glob）。
4. `lint_pack.py` 脚本，当脚本与模式之间或 VERSION 与模式的 `schema_version` 之间存在漂移时失败。
5. 可选的 CI 集成，在演示分支上安装该包，并针对已知良好的任务运行验证门控。

强制拒绝：

- 包含项目专属任务的包。任务存放在目标仓库的任务板上。
- 绑定到单一厂商 SDK 的包。仅限框架无关；SDK 接线是目标仓库的职责。
- 会修改状态文件的安装程序。安装程序是幂等的表面操作；状态属于 Agent 和人类。
- 没有对应检查函数的规则。立志向的规则属于入职培训，不属于包的内容。

拒绝规则：

- 如果事故历史为空，拒绝交付调优的 `agent-rules.md`。使用规范默认值并暴露该缺口。
- 如果目标仓库的 CI 与安装不兼容（没有 `.github/workflows/` 或等效目录），拒绝可选的 CI 步骤并记录手动操作路径。
- 如果团队使用该包的私有 fork，拒绝编写公开安装程序。私有安装程序携带私有不变量。

输出结构：

```
agent-workbench-pack/
├── AGENTS.md
├── docs/
├── schemas/
├── scripts/
├── bin/install.sh
├── lint_pack.py
├── VERSION
└── README.md
```

最后以"下一步阅读"结尾，指向：

- 第 41 课，了解本包所改进的前后对比基准测试。
- 第 30 课（评估驱动的 Agent 开发），了解消费本包裁决结果的评估循环。
- [SkillKit](https://github.com/rohitg00/skillkit)，了解如何将该包分发给 32 个 AI Agent。
