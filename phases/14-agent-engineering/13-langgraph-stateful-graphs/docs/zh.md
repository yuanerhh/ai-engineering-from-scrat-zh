# LangGraph：有状态图与持久化执行

> LangGraph 是 2026 年低级有状态编排的参考实现。智能体是一个状态机；节点是函数；边是转换；状态是不可变的，并在每步后进行检查点。从任何失败处恰好从中断处恢复。

**类型：** 学习 + 构建
**编程语言：** Python（标准库）
**前置知识：** Phase 14 · 01（智能体循环）、Phase 14 · 12（工作流模式）
**预计时间：** 约 75 分钟

## 学习目标

- 描述 LangGraph 的核心模型：带不可变状态、函数节点、条件边和步后检查点的状态机。
- 列出文档强调的四种能力：持久化执行、流式传输、人机协同、全面的记忆。
- 解释 LangGraph 支持的三种编排拓扑：监督者、点对点（群）、层次化（嵌套子图）。
- 用标准库实现一个带不可变状态、条件边和检查点/恢复周期的状态图。

## 问题背景

智能体和工作流共享一个问题：当 40 步的运行在第 38 步失败时，你想从第 38 步恢复，而不是重新开始。二等状态模型让操作人员在假设全新运行的库周围黑客般地处理重试。

LangGraph 的设计答案：状态是一等类型化对象，变更是显式的，检查点在每个节点后持久化。恢复就是一次 `load_state(session_id)` 调用。

## 核心概念

### 图

图由以下部分定义：

- **状态类型。** 每个节点读取和变更的类型化字典（或 Pydantic 模型）。
- **节点（Nodes）。** 纯函数 `(state) -> state_update`。返回后更新被合并进状态。
- **边（Edges）。** 节点之间的条件或直接转换。
- **入口和出口。** `START` 和 `END` 哨兵节点标记边界。

示例：一个带 `classify`、`refund`、`bug`、`sales`、`done` 节点的智能体——作为图的路由工作流。

### 持久化执行

每个节点返回后，运行时将状态序列化并写入检查点存储（SQLite、Postgres、Redis、自定义）。在第 N 步失败时，运行时可以 `resume(session_id)` 并从第 N+1 步以精确状态继续。

LangGraph 文档明确列举了这一点重要的生产用户：Klarna、Uber、摩根大通。关键不是图的形态；而是图形态加检查点使恢复变得廉价。

### 流式传输

每个节点可以产出部分输出。图向调用者流式传输每节点增量事件，使 UI 在图运行时实时更新。

### 人机协同

在节点之间检查和修改状态。实现方式：在关键节点前暂停，将状态呈现给人工，接受修改，然后恢复。检查点存储使这变得容易，因为状态已经序列化了。

### 记忆

短期（在一次运行内——状态中的对话历史）和长期（跨运行——通过检查点存储加独立的长期存储持久化）。LangGraph 通过工具与外部记忆系统（Mem0、自定义）集成。

### 三种拓扑

1. **监督者（Supervisor）。** 中央路由器 LLM 分发给专业子智能体。`langgraph-supervisor` 中的 `create_supervisor()`（不过 LangChain 团队在 2026 年建议直接通过工具调用来实现，以获得更多上下文控制）。
2. **群（Swarm）/ 点对点。** 智能体通过共享工具表面直接交接。没有中央路由器。
3. **层次化（Hierarchical）。** 监督者管理子监督者，实现为嵌套子图。

### 这个模式在哪里出错

- **检查点太小。** 只对对话轮次进行检查点，会让工具状态和记忆写入无法恢复。完整状态必须序列化。
- **非确定性节点。** 恢复假设节点输入产生相同的状态更新。随机种子、挂钟时间、外部 API 调用必须被捕获。
- **过度使用条件边。** 每条边都是条件的图是无法推理的状态机。优先使用带偶尔分支的线性链。

## 动手实践

`code/main.py` 实现了一个标准库有状态图：

- `State`——带 `messages`、`step`、`route`、`output`、`human_approval` 的类型化字典。
- `Node`——接受状态并返回更新字典的可调用对象。
- `StateGraph`——节点 + 边 + 条件边 + 运行 + 恢复。
- `SQLiteCheckpointer`（内存假实现）——每个节点后序列化状态；`load(session_id)` 恢复。
- 演示图：classify（分类）-> branch（分支：refund/bug/sales）-> human gate（人工门控）-> send（发送）。

运行：

```
python3 code/main.py
```

轨迹显示第一次运行在人工门控处失败，持久化，然后恢复产生最终输出。

## 使用建议

- **LangGraph**——参考实现，生产就绪。使用 `create_react_agent`、`create_supervisor`，或构建自己的图。
- **AutoGen v0.4**（第 14 课）——高并发场景的 Actor 模型替代方案。
- **Claude Agent SDK**（第 17 课）——带内置会话存储的托管框架。
- **自定义**——当你需要对状态形态或检查点存储后端进行精确控制时。

## 产出技能

`outputs/skill-state-graph.md` 在任何目标运行时生成 LangGraph 形态的状态图，内置检查点和恢复功能。

## 练习

1. 当分类置信度低于阈值时，从 `classify` 添加一条到 `end` 的条件边。在人工手动设置 `route` 后恢复运行。
2. 将 SQLite 类的假实现替换为真实的 SQLite 检查点存储。测量每步的序列化开销。
3. 实现并行边：两个节点并发运行，通过自定义 reducer 合并。不可变状态在这里带来了什么好处？
4. 阅读 `langgraph-supervisor` 参考文档。将玩具移植到 `create_supervisor`。比较轨迹形态。
5. 添加流式传输：每个节点在运行时产出部分状态。在增量到达时打印它们。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 状态图（State graph） | "智能体作为状态机" | 类型化状态 + 节点 + 边 + 归约器 |
| 检查点存储（Checkpointer） | "持久化后端" | 每个节点后序列化状态；启用恢复 |
| 归约器（Reducer） | "状态合并器" | 将当前状态与节点更新组合的函数 |
| 条件边（Conditional edge） | "分支" | 由状态函数选择的边 |
| 子图（Subgraph） | "嵌套图" | 用作另一个图内节点的图 |
| 持久化执行 | "从失败处恢复" | 以精确状态从最后成功的节点重新开始 |
| 监督者（Supervisor） | "路由器 LLM" | 专业子智能体的中央调度器 |
| 群（Swarm） | "P2P 智能体" | 智能体通过共享工具交接；没有中央路由器 |

## 延伸阅读

- [LangGraph 概述](https://docs.langchain.com/oss/python/langgraph/overview) — 参考文档
- [langgraph-supervisor 参考](https://reference.langchain.com/python/langgraph/supervisor/) — 监督者模式 API
- [AutoGen v0.4，微软研究院](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — Actor 模型替代方案
- [Claude Agent SDK 概述](https://platform.claude.com/docs/en/agent-sdk/overview) — 会话存储和子智能体
