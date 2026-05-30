---
name: framework-picker
description: 通过将抽象与问题形态匹配，为智能体任务选择 LangGraph、CrewAI、AutoGen、Agno 或纯 Python。
version: 1.0.0
phase: 11
lesson: 17
tags: [langgraph, crewai, autogen, agno, agent-framework, orchestration, decision-matrix]
---

给定任务描述（问题形态、每次运行的 LLM 调用总数、分支模式、持久性和恢复需求、人工干预检查点、并行扇出、会话记忆、预期每日运行量），输出：

1. 形态匹配。一句话命名合适的抽象：图（类型化状态、命名转换）、组织图（专家角色、经理路由的交接）、聊天（智能体对话直到完成）、带工具的单智能体。如果无法选择，任务还没有形成智能体形态；停下来先分解。
2. 分支权威。谁选择下一步：开发者（明确边）、经理 LLM（CrewAI 层级）、对话涌现（AutoGen GroupChat）、工具调用自路由（Agno）。如果适用，引用 LLM 选择路由的每轮 token 成本。
3. 状态预算。确认是否需要重启后恢复、时间旅行或人工中断。如果需要，LangGraph 在状态优先抽象上获胜；Agno 只覆盖会话范围的记忆。
4. 框架选择。输出 langgraph、crewai、autogen、agno、plain_python 之一。包含一句将形态和状态答案映射到框架核心原语的理由。
5. 逃生舱。如果每日运行量超过 10,000 或任务是没有状态的两次以下 LLM 调用，推荐使用提供商 SDK 的纯 Python。当任务很小时，没有框架才是最快的框架。

拒绝为有已知 DAG 的确定性工作流推荐 AutoGen；GroupChatManager 花费 token 选择发言者，而开发者本可以静态连接好。CrewAI 确实通过 `output_pydantic` / `output_json` 支持结构化任务输出（参见 [docs.crewai.com/en/concepts/tasks](https://docs.crewai.com/en/concepts/tasks)），但其 `context` 通道仍然通过下一个任务的提示词字符串流动。当工作流依赖原始 `context` 在任务间携带结构化状态而没有这些输出 Schema 连接时，推回 CrewAI。对于两次调用的摘要器推回 LangGraph；StateGraph 开销是纯税。当任务在超过 4 个并行子工作者上扇出并带有归约语义时推回 Agno；Agno 提供了一个 `Parallel` 块，其输出按步骤名称键控连接到字典中（参见 [docs-v1.agno.com/workflows_2/overview](https://docs-v1.agno.com/workflows_2/overview) 和 [docs.agno.com/workflows/access-previous-steps](https://docs.agno.com/workflows/access-previous-steps)），但它没有暴露可与 LangGraph 的 Send 风格扇出归约原语相比的功能。

示例输入：「长期运行的研究工作流：计划，扇出到三个检索器，合成，人工批准简报，写报告，引用来源。必须在崩溃后恢复。生产绑定到每天 50 次运行。」

示例输出：
- 形态：图。类型化计划、三个并行检索器、合成和写作之间的命名转换。
- 分支：开发者通过条件边决定。没有每轮经理 LLM。
- 状态：需要恢复和人工中断。LangGraph 是必须的。
- 框架：langgraph。状态、Send 扇出、interrupt_before 和 PostgresSaver 都是一级原语。
- 逃生舱：不适用。每天 50 次运行远低于纯 Python 阈值，工作流太有状态而无法不用框架。
