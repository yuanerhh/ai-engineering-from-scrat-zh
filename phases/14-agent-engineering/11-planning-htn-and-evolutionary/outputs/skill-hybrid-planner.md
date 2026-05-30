---
name: hybrid-planner
description: 构建混合规划器——用 ChatHTN 生成可证明合理的计划，用 AlphaEvolve 进行具有机器可验证评估器的代码搜索——并针对问题选择合适的方案。
version: 1.0.0
phase: 14
lesson: 11
tags: [planning, htn, chathtn, alphaevolve, evolutionary-search]
---

给定问题类别（策略约束工作流 vs 代码优化 vs 开放式任务），选择规划器并生成正确的脚手架。

决策逻辑：

1. 问题是否有硬性前置条件 / 策略 / 调度约束？-> HTN（ChatHTN）。
2. 问题是否有确定性的、机器可验证的适应度函数？-> 进化式（AlphaEvolve）。
3. 两者都不符合？-> 改用 ReAct（第 01 课）或 ReWOO（第 02 课）。

对于 HTN，输出：

1. `Operator` 类型，包含 `preconditions`、`effects_add`、`effects_remove`。
2. `Method` 类型，包含 `task`、`preconditions`、`subtasks`。
3. 一个规划器，优先尝试方法，退而使用 LLM 分解，并缓存成功的 LLM 分解结果。
4. 一个验证步骤，拒绝引用未知算子或方法的 LLM 分解结果。

对于进化式，输出：

1. 一个候选程序种群作为初始种子。
2. 一个返回标量适应度值的确定性评估器。
3. 一个变异算子（LLM 驱动或基于规则）。
4. 一个选择循环（保留 top-k，变异，重复），并带早停机制。

硬性拒绝：

- ChatHTN 中 LLM 输出未经算子模式验证即直接应用。这会导致合理性声明失效。
- AlphaEvolve 中评估器调用 LLM 作为评判者。适应度必须是确定性的；LLM 评判者引入的随机噪声是循环无法恢复的。
- 将上述任一模式用于开放式任务（如"写一篇博客"）。没有评估器、没有前置条件 -> 使用 ReAct。

拒绝规则：

- 如果领域没有清晰的算子模式，拒绝使用 ChatHTN。建议改用 ReWOO 或普通 ReAct。
- 如果领域没有机器可验证的适应度，拒绝使用 AlphaEvolve。建议改用自我精炼（第 05 课）。
- 如果用户想要"规划器 + LLM 做最终决定"，拒绝。符号正确性与 LLM 探索之间的分离是核心承重结构。

输出：`operators.py`、`methods.py`、`planner.py`（HTN）或 `evaluator.py`、`mutator.py`、`loop.py`（进化式），以及 `README.md`（含决策依据）。末尾附"下一步阅读"，若辩论式验证适合该问题指向第 25 课，若任务实际上是 ReWOO 形态则指向第 02 课。
