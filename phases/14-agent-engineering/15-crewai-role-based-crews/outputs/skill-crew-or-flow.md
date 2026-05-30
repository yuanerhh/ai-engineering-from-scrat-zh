---
name: crew-or-flow
description: 针对给定任务选择 CrewAI Crew 或 Flow，并构建最小化实现脚手架。
version: 1.0.0
phase: 14
lesson: 15
tags: [crewai, crews, flows, multi-agent, role-based]
---

给定任务描述，选择 Crew（自主式）或 Flow（确定性），然后构建脚手架。

决策逻辑：

1. 任务是否有 SLA、合规或确定性回放要求？-> Flow。
2. 任务是否具有探索性（研究、初稿、头脑风暴）？-> Crew。
3. 任务是否有 4 个以上专家且由 LLM 决定顺序？-> 层级式 Crew。
4. 任务是否有不超过 3 个专家且顺序固定？-> 顺序式 Crew 或 Flow——优先选 Flow。

对于 Crew，输出：

1. 智能体定义：role、goal、backstory（简洁，不超过 200 字）、tools。
2. 任务定义：description、expected_output、agent。
3. 使用正确 Process（Sequential | Hierarchical）的 Crew。
4. 一个测试用例，在样本输入上运行 Crew 并检查是否产出预期输出。

对于 Flow，输出：

1. `@start` 入口函数。
2. 构成 DAG 的 `@listen(topic)` 步骤。
3. 显式事件主题；不使用魔法式广播。
4. 一个回放测试用例：给定 kickoff 载荷，确定性地重新运行。

硬性拒绝：

- 没有 backstory 的 Crew。Backstory 是承重构件。
- 没有显式主题名称的 Flow。"隐式链式调用"会破坏审计目的。
- 只有 2 个专家的层级式 Crew。管理者的开销无法收回成本。

拒绝规则：

- 如果用户在纯合规生产任务上要求使用 Crew，拒绝并迁移到 Flow。
- 如果用户在开放式研究任务上要求使用 Flow，拒绝并迁移到 Crew。
- 如果 backstory 超过 200 字，拒绝并要求精简。上下文预算是有限的。

输出：`agents.py`、`tasks.py`、`crew.py` 或 `flow.py`，以及 `README.md`（含决策依据）。末尾附"下一步阅读"，若需要可观测性指向第 24 课（Langfuse/AgentOps），若 Flow 需要持久化恢复语义指向第 13 课。
