---
name: migration-agent
description: 构建一个代码库级别的代码迁移智能体，将确定性配方与智能体回退循环相结合，通过 MigrationBench 评测并发布失败分类体系。
version: 1.0.0
phase: 19
lesson: 09
tags: [capstone, code-migration, openrewrite, libcst, migrationbench, agent, sandbox]
---

给定一个 Java 8 或 Python 2 代码库，生成一个已迁移分支（至 Java 17 或 Python 3.12），具备绿色测试套件和最小覆盖率回退。在 50 个代码库的 MigrationBench 子集上进行评估。

构建计划：

1. 确定性阶段：先运行 OpenRewrite（Java）或 libcst（Python）执行机械性重写。以"配方提交"的形式提交，包含干净的差异。
2. Daytona 沙箱：预安装目标运行时；按分支构建；只读源码挂载。
3. 智能体循环：基于 Claude Opus 4.7 + GPT-5.4-Codex 的 LangGraph 或 OpenAI Agents SDK。工具：`run_build`、`read_file`、`edit_file`、`run_test`、`git_diff`。将失败分类（依赖、语法、测试、构建工具），应用针对性修复，重新运行。
4. 预算上限：30 分钟、8 美元、20 轮。超出任意上限时停止，并以 `budget_exhausted` 标签归档当前差异。
5. 测试 + 覆盖率关口：先构建通过再测试通过；覆盖率下降不得超过 2%。
6. 开启 PR，包含配方提交 + 智能体提交 + 摘要注释。
7. 失败分类：按代码库标记，分类包括 `{dep_upgrade_required, build_tool_drift, custom_annotation, test_flake, syntax_edge_case, budget_exhausted, coverage_regression}`。
8. 在 MigrationBench 上运行 50 个代码库；发布每类通过率、每代码库成本和覆盖率保留情况；与仅确定性基线进行对比。

评估标准：

| 权重 | 评估项 | 度量方式 |
|:-:|---|---|
| 25 | MigrationBench 通过率 | 50 个代码库子集的 pass@1 |
| 20 | 测试覆盖率保留 | 与基础分支相比的平均覆盖率变化 |
| 20 | 每个迁移代码库的成本 | 通过运行的平均每代码库美元成本 |
| 20 | 智能体/确定性工具集成 | OpenRewrite 与智能体处理修复的比例 |
| 15 | 失败分析报告 | 带示例的分类体系完整性 |

硬性拒绝条件：

- 跳过确定性阶段的管道。OpenRewrite 处理机械性的 70-80% 比任何智能体都更便宜、更可靠。
- 将覆盖率回退超过 2% 视为通过。
- 将机械性和智能体编写的变更合并到一个提交中的 PR。必须分开。
- 未在同 50 个代码库上运行对照仅确定性基线就报告通过率。

拒绝规则：

- 拒绝将已迁移分支强制推送覆盖基础分支。始终使用新分支 + PR。
- 拒绝开启 CI 在沙箱中未变为绿色的 PR。
- 拒绝在没有明确修改授权的情况下在企业代码库上运行。

输出：一个包含双层迁移管道、50 个代码库 MigrationBench 运行日志、失败分类仪表板、对照仅确定性基线运行的代码库，以及一份说明三种最常见失败类别及能消除每种问题的配方变更的报告。
