---
name: runtime-picker
description: 根据技术栈、延迟预算和运营形态，为给定场景选择生产级智能体运行时（Agno、Mastra、LangGraph、提供商 SDK）。
version: 1.0.0
phase: 14
lesson: 18
tags: [agno, mastra, langgraph, runtime, selection]
---

给定技术栈、延迟预算、所需原语和运营形态，选择运行时。

决策逻辑：

1. Python + FastAPI + 每秒数千个短暂智能体 -> **Agno**。
2. TypeScript + Next.js/Vercel + 统一多提供商 -> **Mastra**。
3. 持久化状态、显式图、故障恢复 -> **LangGraph**（第 13 课）。
4. Claude 优先产品，想要 Claude Code 工具链形态 -> **Claude Agent SDK**（第 17 课）。
5. OpenAI 优先产品，需要交接 + 护栏 + 追踪 -> **OpenAI Agents SDK**（第 16 课）。
6. 多智能体团队、actor 模型并发、故障隔离 -> **AutoGen v0.4** / **Microsoft Agent Framework**（第 14 课）。
7. 基于角色的协作或事件驱动的确定性工作流 -> **CrewAI** Crew 或 Flow（第 15 课）。
8. 以上都不符合 -> 直接 API 调用 + 第 01 课的 stdlib 循环。

输出内容：

- 一份简短决策文档：技术栈、延迟目标、所需原语、观察到的权衡。
- 所选运行时的最小化脚手架。
- 如果当前使用的是其他运行时，提供迁移计划。

硬性拒绝：

- 仅凭"性能"选择 Agno 或 Mastra，而实际工作负载是每次请求一次慢调用。性能很少是瓶颈所在。
- 在 Python 单体仓库中选择 TypeScript 运行时而没有理由。混合语言的智能体代码是一种运营税。
- 对无状态短任务选择 LangGraph。检查点器增加的开销，用简单工作流（第 12 课）完全可以避免。

拒绝规则：

- 如果用户想要"全部五种运行时，以供比较"，拒绝。请在你的工作负载上做基准测试；框架厂商的基准数据只是方向性参考。
- 如果用户想自托管 Mastra 的 `ee/` 功能，拒绝，并指向许可证条款。
- 如果产品需要长时间运行的异步工作（数小时到数天），拒绝自托管，路由到 Claude Managed Agents 或基于队列的架构（第 29 课）。

输出：决策文档 + 脚手架 + README。末尾附"下一步阅读"，指向第 24 课（可观测性）和第 29 课（生产运行时），了解框架之上的运营层。
