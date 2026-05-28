# 智能体框架权衡 — LangGraph vs CrewAI vs AutoGen vs Agno

> 每个框架都展示相同的演示（研究智能体生成报告），并隐藏相同的 bug（状态 Schema 与编排层冲突）。选择其核心抽象与你的问题形状匹配的框架；其他所有东西都是你要写两次的粘合代码。

**类型：** 学习
**语言：** Python
**前置条件：** Phase 11 · 09（函数调用）、Phase 11 · 16（LangGraph）
**预计时间：** ~45 分钟

## 问题所在

你有一个需要不止一次 LLM 调用的任务。也许是研究工作流（规划、搜索、摘要、引用）。也许是代码审查流水线（解析差异、批评、修补、验证）。也许是预订航班、写邮件和提交费用报告的多轮助手。你选择了一个框架。

三天后，你发现框架的抽象开始泄漏。CrewAI 给你角色，但当"研究员"需要将结构化计划交给"写手"时就和你作对。AutoGen 给你智能体间的聊天，但没有一等状态，所以你的检查点是对话日志的 pickle。LangGraph 给你状态图，但强迫你在知道智能体会做什么之前就命名每个转换。Agno 给你单智能体原语，当你尝试扇出到三个并发工作者时它就崩溃。

解决方案不是"选择最好的框架"，而是将框架的核心抽象与你的问题形状匹配。本节课绘制这张地图。

## 核心概念

四个框架主导着 2026 年的格局。它们的核心抽象并不相同。

| 框架 | 核心抽象 | 最适合 | 最不适合 |
|------|---------|-------|---------|
| **LangGraph** | `StateGraph` — 类型化状态、节点、条件边、检查点器 | 具有显式状态和人工参与中断的工作流；需要时间旅行调试的生产智能体 | 拓扑未知的松散角色驱动头脑风暴 |
| **CrewAI** | `Crew` — 角色（目标、背景故事）、任务、流程（顺序或层级） | 具有简短线性/层级计划的角色扮演或人设驱动工作流 | 超出团队轮次历史的任何有状态的东西；复杂分支 |
| **AutoGen** | `ConversableAgent` 对 — 两个或多个智能体轮流发言直到满足退出条件 | 多智能体*对话*（教师-学生、提议者-批评者、演员-评审者），思考从聊天中涌现 | 具有已知 DAG 的确定性工作流；任何需要跨重启持久状态的东西 |
| **Agno** | `Agent` — 单个 LLM + 工具 + 记忆，可组合成团队 | 快速构建的单智能体和轻量级团队；强大的多模态和内置存储驱动 | 具有自定义归约器的深度显式分支图 |

### "抽象"实际上意味着什么

框架的核心抽象是你在提出架构时在白板上画的东西。

- **LangGraph** → 你画一个图。节点是步骤，边是转换，每个点的状态对象都是类型化的。心智模型是状态机。
- **CrewAI** → 你画一个组织图。每个角色都有职位描述，管理者路由任务。心智模型是一支小型专家团队。
- **AutoGen** → 你画一个 Slack DM。两个智能体互发消息；如果需要主持人则加入第三个。心智模型是聊天。
- **Agno** → 你画一个带工具悬挂的单个框。把框放在一起组成团队。心智模型是"带电池的智能体"。

### 状态问题

状态是大多数框架选择在生产中崩溃的地方。

- **LangGraph。** 类型化状态（`TypedDict` 或 Pydantic 模型），每字段归约器，一等检查点器（SQLite/Postgres/Redis）。恢复、中断和时间旅行是免费的。（参见 Phase 11 · 16。）
- **CrewAI。** 状态通过 `context` 字段在任务间以字符串流动，或通过 `output_pydantic` 结构化传递。没有开箱即用的持久化每团队存储；如果团队必须在重启后存活，你自己添加。
- **AutoGen。** 状态是聊天历史和任何用户定义的 `context`。对话记录持久化；任意工作流状态不会，除非你编写适配器。
- **Agno。** 内置存储驱动（SQLite、Postgres、Mongo、Redis、DynamoDB），通过 `storage=` 附加到 `Agent`——会话和用户记忆自动持久化。不是完整的图检查点器；是会话存储。

### 分支问题

每个非平凡的智能体都会分支。谁决定分支很重要。

- **LangGraph** — 你决定，通过条件边。路由是带命名分支的 Python 函数。分支在编译图中是一等的；检查点器记录走了哪个分支。
- **CrewAI** — 在层级模式中管理者决定；在顺序模式中你在构建时决定。路由在任务列表中是隐式的；管理者提示词之外没有一等的"if"。
- **AutoGen** — 智能体通过聊天决定。分支从接下来谁发言中涌现。`GroupChatManager` 选择下一个发言者；你可以手写 `speaker_selection_method`，但默认是 LLM 驱动的。
- **Agno** — 智能体通过接下来调用哪个工具决定。团队有协调者/路由器/协作者模式；超出此范围的分支是开发者的责任。

### 可观测性问题

- **LangGraph** — 通过 LangSmith 或任何 OTel 导出器的 OpenTelemetry。每次节点转换都是一个追踪 span；检查点兼作可重放追踪。LangSmith 是一方选项；Langfuse/Phoenix 也有适配器。
- **CrewAI** — 自 2025 年底以来支持一等 OpenTelemetry；与 Langfuse、Phoenix、Opik、AgentOps 集成。
- **AutoGen** — 通过 `autogen-core` 集成 OpenTelemetry；AgentOps 和 Opik 有连接器。追踪粒度是每智能体消息，而不是每节点。
- **Agno** — 内置 `monitoring=True` 标志加 OpenTelemetry 导出器；与 Langfuse 紧密集成用于会话追踪。

### 成本和延迟

所有四个框架都增加每次调用的开销（框架逻辑、验证、序列化）。开销增加的粗略顺序：Agno ≈ LangGraph < CrewAI ≈ AutoGen。差异主要由框架进行多少额外 LLM 路由决定。CrewAI 的层级管理者花费 token 决定接下来谁执行；AutoGen 的 `GroupChatManager` 也是如此。LangGraph 只在你写 `llm.invoke` 的地方花费 token。Agno 的单智能体路径很精简。

当每次运行的成本重要时，优先使用显式路由（LangGraph 边，AutoGen `speaker_selection_method`）而不是 LLM 选择的路由。

### 互操作性

- **LangGraph** ↔ LangChain 工具、检索器、LLM。一等 MCP 适配器（工具作为 MCP 服务器导入）。
- **CrewAI** ↔ 工具继承自 `BaseTool`；LangChain 工具、LlamaIndex 工具和 MCP 工具都可以适配进来。通过 `allow_delegation=True` 进行 Crew 间委托。
- **AutoGen** → `FunctionTool` 包装任何 Python 可调用对象；MCP 适配器可用。与 AG2 生态系统紧密耦合用于智能体间模式。
- **Agno** → `@tool` 装饰器或 BaseTool 子类；MCP 适配器；工具可以在智能体和团队间共享。

## 技能

> 你能用一句话解释为什么某个框架适合某个智能体问题。

构建前检查清单：

1. **画出形状。** 这是图（类型化状态，命名转换）？角色扮演（专家交接工作）？聊天（智能体对话直到完成）？带工具的单个智能体？
2. **决定谁分支。** 开发者决定分支 → LangGraph。管理者智能体决定 → CrewAI 层级。聊天涌现 → AutoGen。工具调用决定 → Agno。
3. **检查状态预算。** 你需要从检查点恢复吗？时间旅行？运行中的人工中断？如果是，LangGraph 是默认选择；Agno 会话涵盖对话范围的状态。
4. **检查成本预算。** LLM 选择的路由每轮花费额外 token。如果智能体每天运行数千次，优先使用显式路由。
5. **估算框架开销。** 每个框架都是另一个依赖。如果任务是两次 LLM 调用和一个工具，写 30 行纯 Python；没有框架比没有框架更便宜。

在你能画出图、组织图、聊天或智能体框之前，拒绝使用框架。拒绝选择会强迫你与其状态模型搏斗以获得你实际需要的东西的框架。

## 决策矩阵

| 问题形状 | 首选框架 | 原因 |
|---------|---------|-----|
| 带类型化状态、人工批准、长时间运行的工作流 DAG | LangGraph | 一等状态、检查点器、中断、时间旅行 |
| 具有独特角色的研究/写作流水线 | CrewAI（顺序）或 LangGraph 子图 | 在 CrewAI 中每任务一角色很容易表达；分支复杂时用 LangGraph 扩展 |
| 提议者-批评者或教师-学生对话 | AutoGen | 两智能体聊天是其原生形状 |
| 带工具、会话、记忆的单智能体 | Agno | 最精简的设置，内置存储和记忆 |
| 带归约器的数千个并行扇出 | LangGraph + `Send` | 唯一具有一等并行调度原语的 |
| 快速原型，无框架承诺 | 纯 Python + 提供商 SDK | 没有框架是最快的框架 |

## 练习

1. **简单。** 用相同的任务——"研究 Anthropic 的总部，写一份 200 字简报，引用来源"——在 LangGraph（四个节点：规划、搜索、写作、引用）和 CrewAI（三个角色：研究员、写手、编辑）中实现。报告每次运行的 token 成本和代码行数。
2. **中等。** 在 AutoGen（研究员 ↔ 写手聊天，编辑通过 `GroupChat` 加入）和 Agno（带 `search_tools` 和 `write_tools` 的单智能体，加上会话存储）中构建相同的任务。就以下方面对四种实现排名：(a) 每次运行成本，(b) 崩溃后恢复的能力，(c) 在写作步骤前注入人工批准的能力。
3. **困难。** 构建一个决策树脚本 `pick_framework.py`，接受简短的问题描述（JSON：`{has_typed_state, has_roles, has_dialogue, has_parallel_fanout, needs_resume}`）并返回带一句话理由的推荐。用你自己设计的六个案例验证它。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 编排（Orchestration） | "智能体如何协调" | 决定接下来运行哪个节点/角色/智能体的层 |
| 持久状态（Durable state） | "重启后恢复" | 在进程死亡后存活的状态，附加到检查点或会话存储 |
| LLM 选择的路由（LLM-selected routing） | "让模型决定" | 规划器 LLM 每轮选择下一步；灵活但每次决策都花 token |
| 显式路由（Explicit routing） | "开发者决定" | Python 函数或静态边选择下一步；便宜且可审计 |
| Crew | "CrewAI 团队" | 角色 + 任务 + 流程（顺序或层级）绑定成单个可运行对象 |
| GroupChat | "AutoGen 的多智能体聊天" | 带发言者选择器的 N 个智能体之间的受管对话 |
| Team（Agno） | "多智能体 Agno" | 一组智能体上的路由/协调/协作模式 |
| StateGraph | "LangGraph 的图" | 类型化状态、节点、条件边、检查点器原语 |

## 延伸阅读

- [LangGraph 文档](https://langchain-ai.github.io/langgraph/) — StateGraph、检查点器、中断、时间旅行
- [CrewAI 文档](https://docs.crewai.com/) — Crew、Flow、Agent、Task、Process
- [AutoGen 文档](https://microsoft.github.io/autogen/) — ConversableAgent、GroupChat、团队、工具
- [Agno 文档](https://docs.agno.com/) — Agent、Team、Workflow、存储、记忆
- [Anthropic — 构建有效的智能体（2024 年 12 月）](https://www.anthropic.com/research/building-effective-agents) — 框架无关的模式库（提示词链、路由、并行化、编排者-工作者、评估者-优化器）
- [Yao 等，"ReAct：协同推理和行动"（ICLR 2023）](https://arxiv.org/abs/2210.03629) — 每个框架都装扮的原语
- [Wu 等，"AutoGen：通过多智能体对话实现下一代 LLM 应用"（2023）](https://arxiv.org/abs/2308.08155) — AutoGen 的设计论文
- [Park 等，"生成式智能体：人类行为的交互模拟"（UIST 2023）](https://arxiv.org/abs/2304.03442) — CrewAI 式人设堆叠构建的角色扮演基础
- Phase 11 · 16（LangGraph）— 本节课与之对比的框架
- Phase 11 · 19（Reflexion）— 一种完美映射到 LangGraph 但笨拙映射到 CrewAI 的模式
- Phase 11 · 22（生产可观测性）— 如何对你选择的任何框架进行检测
