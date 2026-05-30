---
name: orchestration-picker
description: 针对给定问题选择最小编排拓扑（supervisor、swarm、hierarchical、debate 或无），并最简化地实现它。
version: 1.0.0
phase: 14
lesson: 28
tags: [orchestration, supervisor, swarm, hierarchical, debate]
---

给定一个产品领域和任务类别，选择最小拓扑。

决策：

1. 1 个 Agent + 工作流模式（第 12 课）就够用？-> 完全不需要拓扑。
2. 2-4 个职责明确的专家？-> **supervisor-worker（主管-工作者）**。
3. 延迟敏感且专家间可以干净地交接？-> **swarm（蜂群）**。
4. 10+ 个专家，supervisor 上下文预算不够？-> **hierarchical（分层）**。
5. 准确率比成本更重要，多提案 + 评论有帮助？-> **debate（辩论）**（第 25 课）。

输出内容：

1. 所选拓扑的脚手架。
2. swarm 的跳数计数器；hierarchical 的嵌套深度限制；debate 的轮次上限。
3. 每次交接或每步的可观测性钩子（OTel GenAI span，第 23 课）。
4. README 中的"为何选此，而非彼"章节。

硬性拒绝：

- 将 3 次顺序 LLM 调用称为"多 Agent"。那只是提示词链。
- 没有跳数计数器的 swarm。反弹是必然的。
- 每个分支只有 1 个专家的 hierarchical。把它展平。

拒绝规则：

- 如果用户想用多 Agent 处理单个 ReAct 循环就能搞定的任务，拒绝并建议第 01 课。
- 如果用户想用 supervisor 处理 2 步任务，拒绝并建议提示词链（第 12 课）。
- 如果领域有合规/审计要求，拒绝 swarm，建议使用 supervisor 或 hierarchical。

输出：拓扑脚手架 + README（包含决策理由）。最后附"下一步阅读"，指向第 13 课（LangGraph）了解 supervisor 实现，第 16 课（OpenAI Agents SDK）了解工具式交接，或第 25 课了解辩论细节。
