---
name: workbench-benchmark
description: 在项目自身的示例应用上，分别通过纯提示词和工作台引导两条流水线运行同一任务，并生成包含五个结果的前后对比报告。
version: 1.0.0
phase: 14
lesson: 41
tags: [benchmark, before-after, evaluation, workbench, sample-app]
---

给定一个代码仓库、一个 Agent 产品和一个小型示例应用，生成一个可移植的评估框架，对比纯提示词流水线与工作台引导流水线。

需产出：

1. `eval/sample_app/` — 一个从项目领域提炼的最小可行示例应用。
2. `eval/run_prompt_only.py` 和 `eval/run_workbench.py`，各自接受任务描述并返回 `TaskOutcome`。
3. `eval/report.py`，运行两条流水线并写入 `before-after-report.md` 和 `comparison.json`。
4. 当工作台结果在固定任务集上出现回归时失败的 CI 工作流。
5. `docs/benchmark.md`，解释五种结果以及什么构成回归。

强制拒绝：

- 只有一条流水线的基准测试。对比才是重点。
- 没有分母的百分比结果。始终报告 `n / m`。
- Agent 产品已训练过的示例应用。使用领域专属的测试夹具。
- 隐藏假阴性的报告。纯提示词更快的任务必须被列举出来。

拒绝规则：

- 如果项目没有验收命令，拒绝交付基准测试。没有任何东西可以衡量。
- 如果工作台流水线在中位任务上耗时超过纯提示词流水线的 3 倍，暴露该发现；工作台需要简化，而不是模型。
- 如果框架无法离线运行，拒绝将其接入 CI。网络不稳定会破坏对比的有效性。

输出结构：

```
<repo>/
├── eval/
│   ├── sample_app/
│   ├── run_prompt_only.py
│   ├── run_workbench.py
│   └── report.py
├── outputs/eval/
│   ├── before-after-report.md
│   └── comparison.json
├── docs/benchmark.md
└── .github/workflows/benchmark.yml
```

最后以"下一步阅读"结尾，指向：

- 第 42 课，了解捆绑了工作台流水线所有组件的 Capstone 包。
- 第 19 课（SWE-bench、GAIA、AgentBench），了解与之互补的宏观基准测试。
- 第 30 课（评估驱动的 Agent 开发），了解基准测试接入后的持续评估循环。
