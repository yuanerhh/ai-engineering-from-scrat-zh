---
name: state-graph
description: 构建 LangGraph 形态的状态机，包含类型化状态、条件边、按节点检查点和持久化恢复。
version: 1.0.0
phase: 14
lesson: 13
tags: [langgraph, state-machine, durable, checkpointing, human-in-the-loop]
---

给定目标运行时、状态结构、一组节点函数和检查点后端，构建一个有状态智能体图。

输出内容：

1. 类型化 `State`（dict 或 Pydantic）。为每个字段编写文档。节点读取状态；返回时返回更新内容。
2. `StateGraph`，含 `add_node`、`add_edge`、`add_conditional_edges`、`set_entry`，以及 `START`/`END` 哨兵节点。
3. `Checkpointer` 接口，含 `save(session_id, node, state)` 和 `load_latest(session_id)`。默认使用 SQLite；允许 Postgres/Redis/自定义实现。
4. `Runner`，逐步遍历图，每个节点后序列化状态，捕获 `PausedAtNode` 以支持人在回路，并支持带可选 `state_override` 的 `resume_from`。
5. 三种拓扑辅助器：supervisor（中央路由）、swarm（共享工具交接）、hierarchical（子图）。

硬性拒绝：

- 非确定性节点没有显式随机种子或挂钟时间捕获。恢复假设给定输入状态时节点输出是可复现的。
- 只保存"摘要"状态的检查点器。必须序列化完整状态，否则恢复会失败。
- 每条边都是条件边的图。应优先使用偶有分支的线性链。

拒绝规则：

- 如果用户要求不带持久化的状态图，拒绝。持久化恢复是其核心价值；如不需要恢复，使用第 12 课的工作流模式。
- 如果用户想"只在成功时检查点"，拒绝。失败时也需要状态——调试从那里开始。
- 如果图节点超过约 30 个，拒绝扁平布局，要求使用嵌套子图。30 节点的扁平图无法审阅。

输出：`state.py`、`graph.py`、`checkpointer.py`、`runner.py`、`README.md`（说明状态模式、检查点选择和恢复语义）。末尾附"下一步阅读"，若需要 actor 模型替代方案指向第 14 课，若需要交接/护栏层指向第 16 课，若需要图步骤的 OTel span 指向第 23 课。
