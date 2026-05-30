---
name: multi-agent-team
description: 构建一个包含架构师、并行编码者、评审者和测试者的多智能体软件团队，在 SWE-bench Pro 上进行测量，并生成交接失败事后分析报告。
version: 1.0.0
phase: 19
lesson: 10
tags: [capstone, multi-agent, swe-bench, langgraph, a2a, worktree, roles]
---

给定一个 GitHub issue URL 和并行度级别，部署一个能生成可合并 PR 的多智能体软件团队。在 50 个 SWE-bench Pro 问题上进行评估，并发布交接失败直方图。

构建计划：

1. 任务看板：文件支持（或 Redis）的 JSONL 存储，包含类型化消息。消息类型：plan_request、subtask、diff_ready、review_needed、review_feedback、approved、test_needed、test_passed、test_failed、replan_needed。
2. 架构师（Opus 4.7）：读取 issue，撰写计划，生成包含明确接口（涉及文件、公共函数、测试影响）的子任务 DAG。
3. N 个编码者（Sonnet 4.7）：各自认领子任务，生成全新的 `git worktree add` + Daytona 沙箱，独立实现。
4. 合并协调者：三方合并；仅对文件级别重叠进行 LLM 调解冲突解决。
5. 评审者（GPT-5.4）：读取合并后的差异；不能审批其自己编写的差异；生成 approved 或 review_feedback 并路由到相关编码者。
6. 测试者（Gemini 2.5 Pro）：在干净沙箱中运行测试套件；生成 test_passed 或带产物的 test_failed。
7. 交接计费：每条跨角色消息成为一个带有载荷大小和模型信息的 Langfuse span。计算 Token 放大倍数 = total_tokens / single_agent_baseline_tokens。
8. 注入明显 Bug 探测（10% 的运行）以测量评审者误批率。
9. 在 50 个 SWE-bench Pro 问题上运行；发布 pass@1、与单智能体基线的墙钟时间对比、各角色 Token 分解、交接失败直方图。

评估标准：

| 权重 | 评估项 | 度量方式 |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 | 50 个问题子集的 pass@1 |
| 20 | 并行加速 | 与单智能体基线相比的墙钟时间 |
| 20 | 评审质量 | 注入 Bug 探测上的误批率 |
| 20 | Token 效率 | 每个解决问题的总 Token 数与单智能体相比 |
| 15 | 协调工程 | 合并冲突解决、交接失败直方图 |

硬性拒绝条件：

- 能够审批其自己编写或提议差异的评审者。硬性约束。
- 没有对照单智能体基线运行的报告。多智能体必须在*每美元*上胜出，而不仅仅是 pass@1。
- 消息为自由格式字符串而非类型化 A2A 消息的任务看板。
- 静默丢弃冲突差异而非路由回去重新规划的合并协调者。

拒绝规则：

- 拒绝在没有每角色预算上限（Token + 美元）的情况下运行。
- 拒绝开启测试者未在干净沙箱中验证通过的 PR。
- 拒绝在单次运行中将编码者扩展到超过 8 个。超过此数量后协调开销占主导。

输出：一个包含任务看板 + 角色 Worker、50 个问题 SWE-bench Pro 运行日志、对照单智能体基线运行、带角色标记 span 和各角色 Token 分解的 Langfuse 仪表板、注入 Bug 探测报告的代码库，以及一份事后分析，说明哪三个交接最常出错以及减少每种问题的消息模式或提示变更。
