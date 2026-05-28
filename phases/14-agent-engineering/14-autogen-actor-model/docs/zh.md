# AutoGen v0.4：Actor 模型与智能体框架

> AutoGen v0.4（微软研究院，2025 年 1 月）围绕 Actor 模型重新设计了智能体编排。异步消息交换、事件驱动的智能体、故障隔离、天然并发。该框架现处于维护模式，微软智能体框架（2025 年 10 月公开预览）成为其继任者。

**类型：** 学习 + 构建
**编程语言：** Python（标准库）
**前置知识：** Phase 14 · 01（智能体循环）、Phase 14 · 12（工作流模式）
**预计时间：** 约 75 分钟

## 学习目标

- 描述 Actor 模型：智能体作为 Actor，消息是唯一的进程间通信方式，每个 Actor 的故障独立隔离。
- 列出 AutoGen v0.4 的三个 API 层——Core（核心）、AgentChat、Extensions（扩展）——及各自的用途。
- 解释为什么将消息投递与处理解耦能实现故障隔离和天然并发。
- 用标准库实现一个 Python Actor 运行时，并将一个双智能体代码审查流移植到上面。

## 问题背景

大多数智能体框架是同步的：一个智能体产生，一个智能体消费，在调用堆栈中。失败会使堆栈崩溃。并发是附加上去的。分布式需要重写。

AutoGen v0.4 的答案：Actor 模型。每个智能体是一个带私有收件箱的 Actor。消息是唯一的交互方式。运行时将投递与处理解耦。故障隔离到单个 Actor。并发是天然的。分布式只是不同的传输。

## 核心概念

### Actor

一个 Actor 有：

- 私有状态（永远不会从外部直接触碰）。
- 收件箱（消息队列）。
- 处理器：`receive(message) -> effects`，其中效果可以是"回复"、"发送给其他 Actor"、"生成新 Actor"、"更新状态"、"停止自身"。

两个 Actor 不能共享内存。它们只能发送消息。

### AutoGen v0.4 的三个 API 层

1. **Core（核心）。** 低级 Actor 框架。`AgentRuntime`、`Agent`、`Message`、`Topic`。异步消息交换，事件驱动。
2. **AgentChat。** 任务驱动的高级 API（替代 v0.2 的 ConversableAgent）。`AssistantAgent`、`UserProxyAgent`、`RoundRobinGroupChat`、`SelectorGroupChat`。
3. **Extensions（扩展）。** 集成——OpenAI、Anthropic、Azure、工具、记忆。

### 为什么解耦很重要

在 v0.2 模型中，调用 `agent_a.chat(agent_b)` 会同步阻塞 agent_a 直到 agent_b 返回。在 v0.4 中，`send(agent_b, msg)` 将消息放入 agent_b 的收件箱并返回。运行时稍后投递。三个后果：

- **故障隔离。** 智能体 B 崩溃不会让智能体 A 崩溃——运行时在 B 的处理器中捕获失败并决定如何处理（记录、重试、死信）。
- **天然并发。** 同时有多条消息在传输中；Actor 并发处理其收件箱。
- **分布式就绪。** 收件箱 + 传输是相同的抽象，无论 Actor 是在进程内还是在另一台主机上。

### 拓扑

- **RoundRobinGroupChat。** 智能体按固定轮换顺序轮流发言。
- **SelectorGroupChat。** 选择器智能体根据对话上下文选择谁下一个发言。
- **Magentic-One。** 用于网页浏览、代码执行、文件处理的参考多智能体团队。构建在 AgentChat 之上。

### 可观测性

内置 OpenTelemetry 支持。每条消息发出一个 span；工具调用按 2026 年 OTel GenAI 语义约定（第 23 课）携带 `gen_ai.*` 属性。

### 状态：维护模式

2026 年初：AutoGen v0.7.x 对研究和原型设计是稳定的。微软已将活跃开发转移到微软智能体框架（2025 年 10 月 1 日公开预览；1.0 GA 目标 2026 年第一季度末）。AutoGen 模式可以顺利向前移植——Actor 模型是持久的思想。

## 动手实践

`code/main.py` 实现了一个标准库 Actor 运行时：

- `Message`——带 `sender`、`recipient`、`topic`、`body` 的类型化载荷。
- `Actor`——带 `receive(message, runtime)` 的抽象。
- `Runtime`——带共享队列、投递、故障隔离的事件循环。
- 双 Actor 演示：`ReviewerAgent` 审查代码，`ChecklistAgent` 运行检查清单；它们交换消息直到达成共识。

运行：

```
python3 code/main.py
```

轨迹显示消息投递、一个 Actor 中的模拟失败（不会使另一个崩溃），以及对共同裁决的收敛。

## 使用建议

- **AutoGen v0.4/v0.7**（维护模式）——对研究、原型设计、多智能体模式是稳定的。
- **微软智能体框架**（公开预览）——前进路径；在更新 API 中体现相同的 Actor 模型思想。
- **LangGraph 群拓扑**（第 13 课）——通过共享工具交接的类似模式。
- **自定义 Actor 运行时**——当你需要特定传输（NATS、RabbitMQ、gRPC）时。

## 产出技能

`outputs/skill-actor-runtime.md` 为给定的多智能体任务生成最小 Actor 运行时加上团队模板（RoundRobin 或 Selector）。

## 练习

1. 添加死信队列：当处理器抛出异常时，将失败消息停放等待人工检查。在你的玩具中 DLQ 多频繁被触发？
2. 实现 `SelectorGroupChat`：一个选择器 Actor 根据对话状态选择谁处理下一条消息。
3. 添加分布式传输：将进程内队列替换为 JSON-over-HTTP 服务器，使 Actor 可以在独立进程中运行。
4. 为每条消息连接一个 OTel span（或无操作的替代）。按第 23 课发出 `gen_ai.agent.name`、`gen_ai.operation.name`。
5. 阅读 AutoGen v0.4 的架构文章。将你的玩具移植到真实的 `autogen_core` API。你跳过了什么在生产中重要的内容？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Actor | "智能体" | 私有状态 + 收件箱 + 处理器；没有共享内存 |
| 消息（Message） | "事件" | 类型化载荷；Actor 交互的唯一方式 |
| 收件箱（Inbox） | "邮箱" | 每个 Actor 的待处理消息队列 |
| 运行时（Runtime） | "智能体宿主" | 路由消息并隔离失败的事件循环 |
| 主题（Topic） | "通道" | Actor 之间的命名发布-订阅路由 |
| 故障隔离 | "让它崩溃" | 一个 Actor 失败不会使其他 Actor 崩溃 |
| RoundRobinGroupChat | "固定轮换团队" | 智能体按顺序轮流 |
| SelectorGroupChat | "上下文路由团队" | 选择器选择下一个发言者 |
| Magentic-One | "参考团队" | 用于网页 + 代码 + 文件的多智能体小组 |

## 延伸阅读

- [AutoGen v0.4，微软研究院](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — 重新设计文章
- [LangGraph 概述](https://docs.langchain.com/oss/python/langgraph/overview) — 图形态的替代方案
- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — AutoGen 默认发出的 span
