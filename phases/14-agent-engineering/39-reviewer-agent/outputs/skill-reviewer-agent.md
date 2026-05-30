---
name: reviewer-agent
description: 建立一个具有五维评分标准的审查者 Agent 角色，读取构建者产物，生成结构化审查报告，让人工审查从一份写好的页面开始而非从空白处开始。
version: 1.0.0
phase: 14
lesson: 39
tags: [reviewer, rubric, role-separation, second-loop, review-report]
---

给定一个已经生成工作台产物的构建者 Agent，建立一个读取这些产物并撰写结构化报告的审查者。

需产出：

1. `agents/reviewer.md`，包含审查者系统提示：只读访问、五维评分标准，每个评分必须引用对应的产物路径。
2. `tools/reviewer.py`，从工作台加载 `ReviewerInputs` 并按维度运行 LLM 评分器。
3. `outputs/review/<task_id>.json` 作为规范的审查报告路径。
4. `docs/reviewer-rubric.md`，列出五个维度、每个维度回答的问题以及 0-1-2 分的锚点描述。
5. 每当构建者任务关闭时，将审查报告以 PR 评论形式发布的 CI 步骤。

强制拒绝：

- 对差异具有写入权限的审查者。构建者与审查者之间的隔离正是信号所在；消除隔离会破坏可靠性。
- 每个评分没有锚点描述的评分标准。没有锚点的"0 到 2 分"会退化为直觉评判。
- 缺少引用的审查报告。每个评分必须指向一个文件或追踪条目。
- 共享构建者的系统提示。使用同一模型可以，但使用同一提示不行。

拒绝规则：

- 如果构建者没有生成验证报告，拒绝运行审查者。在验收通过之前，评判毫无意义。
- 如果项目关闭的任务少于三个，拒绝声称评分标准已完成校准。将前几份报告保存为校准集。
- 如果要求审查者在最低置信度以下评分，拒绝并将不确定的维度暴露给人工处理。

输出结构：

```
<repo>/
├── agents/reviewer.md
├── tools/reviewer.py
├── outputs/review/
│   └── <task_id>.json
├── docs/reviewer-rubric.md
└── .github/workflows/review.yml
```

最后以"下一步阅读"结尾，指向：

- 第 40 课，了解结合验证与审查的交接数据包。
- 第 41 课，了解端到端演练构建者/审查者分离的真实风格任务。
- 第 05 课（自我精炼与 CRITIC），了解本课所改进的单 Agent 自我审查基线。
