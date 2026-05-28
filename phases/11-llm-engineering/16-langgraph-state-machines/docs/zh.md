# LangGraph — 智能体的状态机

> 手写的 ReAct 循环是一个 `while True`。用 LangGraph 写的 ReAct 循环是一个你可以检查点化、中断、分支和时间旅行的图。智能体没有改变，围绕它的框架改变了。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 11 · 09（函数调用）、Phase 11 · 14（模型上下文协议）
**预计时间：** ~75 分钟

## 问题所在

你发布了一个函数调用智能体。它运行了三个轮次，然后出了问题：模型尝试了一个返回 500 的工具，用户在任务中途改变了主意，或者智能体决定在没有人工签字的情况下退款。`while True:` 循环没有钩子。你无法暂停它，无法回退它，也无法分支到"如果模型选择了另一个工具会怎样"。一旦你将其投入生产，智能体就变成了一个要么成功要么失败的黑盒。

一旦你看到了，下一步就显而易见。智能体已经是一个状态机——系统提示词加上消息历史加上待处理的工具调用加上下一个动作。让状态机显式化："模型思考"、"工具运行"、"人工批准"的节点，以及它们之间条件转换的边。一旦图变得显式，框架就免费获得了四样东西：检查点化（在步骤间保存状态）、中断（为人工暂停）、流式传输（流式传输 token 和中间事件）和时间旅行（回退到先前状态并尝试不同分支）。

LangGraph 是提供这种抽象的库。它不是 LangChain 意义上的智能体框架（"这是一个 AgentExecutor，祝你好运"）。它是一个具有一等状态、一等持久化和一等中断的图运行时。智能体循环是你绘制的东西，而不是你手写的东西。

## 核心概念

`StateGraph` 有三样东西。

1. **状态（State）。** 流经图的类型化字典（TypedDict 或 Pydantic 模型）。每个节点接收完整状态并返回部分更新，LangGraph 使用每个字段的*归约器（reducer）*进行合并——列表用 `operator.add` 累积，默认用覆盖。
2. **节点（Nodes）。** Python 函数 `state -> partial_state`。每个都是一个离散步骤："调用模型"、"运行工具"、"摘要"。
3. **边（Edges）。** 节点间的转换。静态边只去一个地方。条件边接受路由函数 `state -> next_node_name`，使图可以基于模型输出分支。

你编译图。编译绑定拓扑，附加检查点器（可选但对生产必不可少），并返回可运行对象。你用初始状态和 `thread_id` 调用它。每个执行步骤都持久化以 `(thread_id, checkpoint_id)` 为键的检查点。

### 四大超能力

**检查点化（Checkpointing）。** 每次节点转换都将新状态写入存储（测试用内存，生产用 Postgres/Redis/SQLite）。通过用相同的 `thread_id` 再次调用图来恢复。图从暂停的地方继续。

**中断（Interrupts）。** 用 `interrupt_before=["human_review"]` 标记节点，执行在该节点运行前停止。状态持久化。你的 API 向用户响应"等待批准"。后来向同一 `thread_id` 发送带 `Command(resume=...)` 的请求会恢复执行。

**流式传输（Streaming）。** `graph.stream(state, mode="updates")` 在事件发生时产生状态增量。`mode="messages"` 在模型节点内流式传输 LLM token。`mode="values"` 产生完整快照。你选择在 UI 中呈现什么。

**时间旅行（Time-travel）。** `graph.get_state_history(thread_id)` 返回完整的检查点日志。将任何先前的 `checkpoint_id` 传递给 `graph.invoke` 并从该点分叉。非常适合调试（"如果模型选择了工具 B 会怎样？"）和重放生产追踪的回归测试。

### 归约器是关键

每个状态字段都有一个归约器。大多数默认值都是好的——新值覆盖旧值。但消息列表需要 `operator.add`，这样新消息会追加而不是替换。并行边通过归约器合并它们的更新。如果两个节点都更新了 `messages` 而你忘记了 `Annotated[list, add_messages]`，第二个会悄悄赢并且你丢失了半个轮次。归约器是库中唯一微妙的地方；搞定它，其余都能组合。

### 四个节点的 ReAct 图

生产级 ReAct 智能体是四个节点和两条边：

1. `agent` — 用当前消息历史调用 LLM。返回助手消息（可能包含 tool_calls）。
2. `tools` — 执行最后一条助手消息中的任何 tool_calls，将工具结果作为工具消息追加。
3. 从 `agent` 出发的条件边，如果最后一条消息有 tool_calls 则路由到 `tools`，否则路由到 `END`。
4. 从 `tools` 回到 `agent` 的静态边。

就这些。你用大约 40 行代码获得了完整的 ReAct 循环（思考 → 行动 → 观察 → 思考 → …），带有检查点化、中断和流式传输。

### StateGraph vs Send（扇出）

`Send(node_name, state)` 让节点调度并行子图。示例：智能体决定同时查询三个检索器。每个 `Send` 产生目标节点的并行执行；它们的输出通过状态归约器合并。这是 LangGraph 在不使用线程原语的情况下表达编排者-工作者模式的方式。

### 子图

编译后的图可以是另一个图中的节点。外部图看到一个单一节点；内部图有自己的状态和自己的检查点。这是团队构建主管-工作者智能体的方式：主管图将用户意图路由到每个域的工作者子图。

## 构建它

### 步骤 1：状态和节点

```python
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def agent_node(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

tool_node = ToolNode(tools=[search_web, read_file])

graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile(checkpointer=MemorySaver())
```

`add_messages` 是使消息列表累积而不是覆盖的归约器。忘记它是最常见的 LangGraph 错误。

### 步骤 2：用线程运行

```python
config = {"configurable": {"thread_id": "user-42"}}
for event in app.stream(
    {"messages": [HumanMessage("find the Anthropic headquarters address")]},
    config,
    stream_mode="updates",
):
    print(event)
```

每次更新都是一个字典 `{node_name: state_delta}`。你的前端可以将这些流式传输到 UI，这样用户就能看到"智能体正在思考…调用 search_web…获得结果…回答中"。

### 步骤 3：添加人工参与中断

标记一个节点使执行在运行前暂停。

```python
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],  # 在每次工具调用前暂停
)

state = app.invoke({"messages": [HumanMessage("delete the production database")]}, config)
# state["__interrupt__"] 已设置。检查提议的工具调用。
# 如果批准：
from langgraph.types import Command
app.invoke(Command(resume=True), config)
# 如果拒绝：写一条拒绝消息并恢复
app.update_state(config, {"messages": [AIMessage("Blocked by human reviewer.")]})
```

状态、检查点和线程在中断期间全部持久化。除执行期间外，没有任何东西在内存中。

### 步骤 4：用于调试的时间旅行

```python
history = list(app.get_state_history(config))
for snapshot in history:
    print(snapshot.values["messages"][-1].content[:80], snapshot.config)

# 从先前的检查点分叉
target = history[3].config  # 往回三步
for event in app.stream(None, target, stream_mode="values"):
    pass  # 从该点向前重放
```

将 `None` 作为输入传递会从给定检查点重放；传递一个值会在恢复前将其作为更新追加到该检查点的状态。这是在不重新运行整个对话的情况下重现糟糕智能体运行的方法。

### 步骤 5：为生产替换检查点器

```python
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string("postgresql://...") as checkpointer:
    checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)
```

SQLite、Redis 和 Postgres 都已提供。`MemorySaver` 用于测试。任何需要跨重启持久化的东西都需要真实的存储。

## 技能

> 你将智能体构建为图，而不是 `while True` 循环。

在使用 LangGraph 之前，进行 60 秒的设计：

1. **命名节点。** 每个离散决策或有副作用的动作都是一个节点。"智能体思考"、"工具运行"、"审核员批准"、"响应流式传输"。如果你无法列出它们，任务还不是智能体形状的。
2. **声明状态。** 最小 TypedDict，每个列表字段都有归约器。不要把所有东西都塞进 `messages`；将特定任务字段（工作 `plan`、`budget` 计数器、`retrieved_docs` 列表）提升到顶层。
3. **绘制边。** 除非下一步取决于模型输出，否则使用静态边。每条条件边都需要带有命名分支的路由函数。
4. **预先选择检查点器。** 测试用 `MemorySaver`，其他用 Postgres/Redis/SQLite。不要在没有检查点器的情况下发布——没有检查点器意味着没有恢复、没有中断、没有时间旅行。
5. **在工具运行前而不是运行后决定中断。** 批准放在进入有副作用节点的边上，这样你可以在造成伤害之前取消；验证放在从模型出来的边上，这样你可以廉价地拒绝糟糕的调用。
6. **默认流式传输。** `mode="updates"` 用于 UI，`mode="messages"` 用于模型节点内的 token 级流式传输，`mode="values"` 用于评估期间的完整快照。

拒绝发布没有检查点器的 LangGraph 智能体。拒绝发布在副作用*之后*中断的。拒绝发布没有 `add_messages` 作为归约器的 `messages` 字段。

## 练习

1. **简单。** 用计算器工具和网络搜索工具实现上面的四节点 ReAct 图。验证 `list(app.get_state_history(config))` 对于两轮对话返回至少四个检查点。
2. **中等。** 添加一个在 `agent` 之前运行的 `planner` 节点，并将结构化 `plan: list[str]` 写入状态。让 `agent` 将计划步骤标记为完成。如果 `plan` 在检查点恢复中丢失（归约器错误），则测试失败。
3. **困难。** 构建一个使用 `Send` 在三个子图（`researcher`、`writer`、`reviewer`）之间路由的主管图。每个子图都有自己的状态和检查点器。在外部图上添加 `interrupt_before=["writer"]`，以便人工可以批准研究简报。确认从先前检查点的时间旅行只重新运行分叉的分支。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| StateGraph | "LangGraph 的图" | 在编译前添加节点和边的构建器对象 |
| 归约器（Reducer） | "字段如何合并" | 当节点返回该字段的更新时应用的函数 `(old, new) -> merged`；默认是覆盖，`add_messages` 追加 |
| 线程（Thread） | "对话 ID" | 为一个会话范围化所有检查点的 `thread_id` 字符串 |
| 检查点（Checkpoint） | "暂停的状态" | 节点转换后完整图状态的持久化快照，以 `(thread_id, checkpoint_id)` 为键 |
| 中断（Interrupt） | "为人工暂停" | `interrupt_before` / `interrupt_after` 在节点边界停止执行；用 `Command(resume=...)` 恢复 |
| 时间旅行（Time-travel） | "从先前步骤分叉" | `graph.invoke(None, config_with_old_checkpoint_id)` 从该检查点向前重放 |
| Send | "并行子图调度" | 节点可以返回的构造函数，用于产生目标节点的 N 个并行执行 |
| 子图（Subgraph） | "编译图作为节点" | 用作另一个图中节点的编译 StateGraph；保留自己的状态范围 |

## 延伸阅读

- [LangGraph 文档](https://langchain-ai.github.io/langgraph/) — StateGraph、归约器、检查点器和中断的权威参考
- [LangGraph 概念：状态、归约器、检查点器](https://langchain-ai.github.io/langgraph/concepts/low_level/) — 本节课使用的心智模型，直接来自源头
- [LangGraph 持久化和检查点](https://langchain-ai.github.io/langgraph/concepts/persistence/) — Postgres/SQLite/Redis 存储、检查点命名空间和线程 ID 的详情
- [LangGraph 人工参与](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/) — `interrupt_before`、`interrupt_after`、`Command(resume=...)` 和编辑状态模式
- [Yao 等，"ReAct：在语言模型中协同推理和行动"（ICLR 2023）](https://arxiv.org/abs/2210.03629) — 每个 LangGraph 智能体实现的模式；阅读它了解推理追踪的原理
- [Anthropic — 构建有效的智能体（2024 年 12 月）](https://www.anthropic.com/research/building-effective-agents) — 哪些图形状（链、路由器、编排者-工作者、评估者-优化器）更好以及何时选择
- Phase 11 · 09（函数调用）— 每个 LangGraph 智能体节点复用的工具调用原语
- Phase 11 · 14（模型上下文协议）— 通过 MCP 适配器插入 LangGraph `ToolNode` 的外部工具发现
- Phase 11 · 17（智能体框架权衡）— 何时选择 LangGraph 而不是 CrewAI、AutoGen 或 Agno
