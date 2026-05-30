---
name: actor-runtime
description: 构建 AutoGen v0.4 形态的 actor 运行时，包含私有状态、每 actor 收件箱、纯消息 IPC、故障隔离和死信队列。
version: 1.0.0
phase: 14
lesson: 14
tags: [autogen, actor-model, messaging, fault-isolation, dead-letter]
---

给定一个多智能体任务，构建 actor 运行时和所需的智能体 actor。

输出内容：

1. `Message` 类型，包含 `sender`、`recipient`、`topic`、`body`、`mid`。
2. `Actor` 基类，含 `receive(message, runtime)`。Actor 状态私有。
3. `Runtime`，包含共享队列、`send()`、`run_until_idle()` 和死信队列。处理器中的异常进入 DLQ，不向外传播。
4. 一个拓扑辅助器：RoundRobin（固定轮转）、Selector（LLM 选择下一个）或自定义广播。
5. 每条消息的可观测性钩子：根据第 23 课，发出携带 `gen_ai.agent.name` 和 `gen_ai.operation.name` 的 OTel span。

硬性拒绝：

- 同步消息传递，发送方阻塞直到接收方返回。这是 v0.2 模型，会破坏故障隔离。
- actor 之间共享可变状态。Actor 只能通过消息读取状态，或根本不读取。
- 传播处理器异常的运行时。失败应进入 DLQ；让其他 actor 继续运行。

拒绝规则：

- 如果任务只有两个 actor 且固定来回交互，拒绝使用 actor 模型，建议改用提示链（第 12 课）。只有当有 >=3 个 actor 或需要异步并发时，actor 模型才值得其成本。
- 如果用户以"更易于调试"为由要求"同步模式"，拒绝。建议改用日志 + 追踪（第 23 课）。
- 如果领域严格是单一专家的请求/响应模式，建议使用路由（第 12 课）而非 actor 团队。

输出：`message.py`、`actor.py`、`runtime.py`、`teams.py`、`README.md`（说明 DLQ 策略、拓扑选择及 OTel span 的接入方式）。末尾附"下一步阅读"，若 actor 之间存在协商指向第 25 课（多智能体辩论），若需要追踪指向第 23 课（OTel），若希望使用前瞻性运行时指向 Microsoft Agent Framework。
