---
name: agents-sdk-scaffold
description: 构建 OpenAI Agents SDK 应用的脚手架，包含分诊智能体、交接、输入/输出/工具护栏、会话存储和追踪处理器。
version: 1.0.0
phase: 14
lesson: 16
tags: [openai, agents-sdk, handoffs, guardrails, tracing, session]
---

给定产品领域和一组专家智能体列表，构建 OpenAI Agents SDK 应用的脚手架。

输出内容：

1. 每个专家对应一个 `Agent`，加上一个 `triage` 智能体（只有交接，没有领域工具）。
2. 每个领域工具对应一个 `FunctionTool`，包含类型化输入模式、清晰描述（告知模型何时使用）和执行沙箱。
3. 从 triage 到每个专家的 `Handoff`。确认工具名称遵循 `transfer_to_<agent>` 命名规范。
4. `InputGuardrail` 用于 PII、策略、范围校验。默认使用并行模式；若护栏 LLM 相对于主模型体量较大，则改用阻塞模式。
5. `OutputGuardrail` 用于长度、PII、策略校验。在生产环境中，安全关键输出始终使用阻塞模式。
6. 对触及网络或文件系统的函数工具添加每工具护栏。
7. `Session` 存储（默认 SQLite；生产环境用 Redis）。
8. `add_trace_processor` 将 span 同时接入你的后端和 OpenAI 的追踪 UI。

硬性拒绝：

- 带领域工具的 triage 智能体。Triage 只负责交接；混入领域工具会稀释路由器的决策。
- 会修改输入/输出的护栏。护栏只负责批准或拒绝——不负责改写。
- 静默的交接循环。要求一个跳转计数器（默认最多 3 次）。

拒绝规则：

- 如果用户在面向付费用户或涉及 PII 的产品上要求"不要护栏，先跑起来"，拒绝。
- 如果产品只有 2 个专家，建议使用 `Agents` 直接分类器进行路由（第 12 课），而非 triage+交接——可降低 token 成本。
- 如果生产环境中追踪被禁用，拒绝上线。多步骤失败在没有追踪的情况下无法调试。

输出：`agents.py`、`tools.py`、`guardrails.py`、`app.py`、`README.md`（包含 triage 智能体设计依据、护栏模式、追踪处理器和会话后端说明）。末尾附"下一步阅读"，若需 OTel GenAI 指向第 23 课，若需可观测性后端指向第 24 课，若需 Claude Agent SDK 迁移指向第 17 课。
