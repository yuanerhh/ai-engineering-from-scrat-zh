# OpenAI Agents SDK：Handoffs、Guardrails、Tracing

> OpenAI Agents SDK 是构建在 Responses API 之上的轻量级多智能体框架。五个原语：Agent（智能体）、Handoff（切换）、Guardrail（防护栏）、Session（会话）、Tracing（追踪）。Handoffs 是名为 `transfer_to_<agent>` 的工具。Guardrails 在输入或输出时触发。Tracing 默认开启。

**类型：** 学习 + 构建
**编程语言：** Python（标准库）
**前置知识：** Phase 14 · 01（智能体循环）、Phase 14 · 06（工具使用）
**预计时间：** 约 75 分钟

## 学习目标

- 列出 OpenAI Agents SDK 的五个原语。
- 解释 handoffs：为什么它们被建模为工具，模型看到的名称形态，以及上下文如何转移。
- 区分输入防护栏、输出防护栏和工具防护栏；解释 `run_in_parallel` 与阻塞模式的区别。
- 用标准库实现一个带 handoffs + guardrails + span 风格 tracing 的运行时。

## 问题背景

无法干净委托的智能体最终会将所有内容塞入一个提示词。没有防护栏的智能体会发出 PII、违反策略的输出或无限循环。OpenAI 的 SDK 将使多智能体工作可管理的三个原语规范化了。

## 核心概念

### 五个原语

1. **Agent（智能体）。** LLM + 指令 + 工具 + handoffs。
2. **Handoff（切换）。** 委托给另一个智能体。以名为 `transfer_to_<agent_name>` 的工具的形式呈现给模型。
3. **Guardrail（防护栏）。** 对输入（仅第一个智能体）、输出（仅最后一个智能体）或工具调用（每个函数工具）的验证。
4. **Session（会话）。** 跨轮次的自动对话历史。
5. **Tracing（追踪）。** 针对 LLM 生成、工具调用、handoffs、guardrails 的内置 span。

### Handoffs 作为工具

模型在其工具列表中看到 `transfer_to_billing_agent`。调用它会通知运行时：

1. 复制对话上下文（或通过 `nest_handoff_history` 测试版折叠它）。
2. 用目标智能体的指令初始化它。
3. 继续用目标智能体运行。

这是第 13 课/第 28 课的监督者模式的产品化。

### Guardrails

三种类型：

- **输入防护栏（Input guardrails）。** 在第一个智能体的输入上运行。在任何 LLM 调用之前拒绝不安全或超出范围的请求。
- **输出防护栏（Output guardrails）。** 在最后一个智能体的输出上运行。捕获 PII 泄露、策略违规、格式错误的响应。
- **工具防护栏（Tool guardrails）。** 每个函数工具运行。验证参数、检查权限、审计执行。

模式：

- **并行**（默认）。防护栏 LLM 与主 LLM 并行运行。更低的尾延迟。如果触发，主 LLM 的工作会被丢弃（token 浪费）。
- **阻塞**（`run_in_parallel=False`）。防护栏 LLM 先运行。如果触发，主调用不会浪费 token。

触发线引发 `InputGuardrailTripwireTriggered` / `OutputGuardrailTripwireTriggered`。

### Tracing

默认开启。每次 LLM 生成、工具调用、handoff 和 guardrail 都会发出一个 span。`OPENAI_AGENTS_DISABLE_TRACING=1` 可选择退出。`add_trace_processor(processor)` 将 span 扇出到你自己的后端以及 OpenAI 的后端。

### Sessions

`Session` 将对话历史存储在后端（SQLite、Redis、自定义）中。`Runner.run(agent, input, session=session)` 自动加载并追加。

### 这个模式在哪里出错

- **Handoff 漂移。** 智能体 A 切换到智能体 B，B 又切换回 A。添加跳数计数器。
- **防护栏绕过。** 工具防护栏只在函数工具上触发；内置工具（文件读取、网络获取）需要单独的策略。
- **过度追踪。** span 中的敏感内容。配合 OTel GenAI 内容捕获规则（第 23 课）——在外部存储，通过 ID 引用。

## 动手实践

`code/main.py` 在标准库中实现了 SDK 形态：

- `Agent`、`FunctionTool`、`Handoff`（作为带转移语义的函数工具）。
- `Runner` 带输入/输出/工具防护栏、handoff 分发和跳数计数器。
- 一个简单的 span 发射器，展示追踪形态。
- 一个根据用户查询切换到账单或支持的分流智能体；防护栏在某个输入上触发。

运行：

```
python3 code/main.py
```

轨迹显示两次成功的 handoff、一次输入防护栏触发，以及镜像真实 SDK 发出内容的 span 树。

## 使用建议

- **OpenAI Agents SDK** 用于 OpenAI 优先的产品。
- **Claude Agent SDK**（第 17 课）用于 Claude 优先的产品。
- **LangGraph**（第 13 课）当你想要显式状态和持久化恢复时。
- **自定义** 当你需要精确控制时（语音、多提供商、联邦部署）。

## 产出技能

`outputs/skill-agents-sdk-scaffold.md` 搭建一个 Agents SDK 应用，带有分流智能体、handoffs、输入/输出/工具防护栏、会话存储和追踪处理器。

## 练习

1. 添加 handoff 跳数计数器：N 次转移后拒绝。跟踪行为。
2. 将 `nest_handoff_history` 实现为一个选项——在转移前将先前消息折叠为一个摘要。
3. 编写一个阻塞输出防护栏。比较会触发它的提示词和通过的提示词的延迟。
4. 将 `add_trace_processor` 连接到 JSON 记录器。每个 span 发出什么形态？
5. 阅读 SDK 文档。将你的标准库玩具移植到 `openai-agents-python`。你对什么建模错了？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Agent（智能体） | "LLM + 指令" | SDK 中的智能体类型；拥有工具和 handoffs |
| Handoff（切换） | "转移" | 模型调用以委托给另一个智能体的工具 |
| Guardrail（防护栏） | "策略检查" | 对输入/输出/工具调用的验证 |
| Tripwire（触发线） | "防护栏触发" | 防护栏拒绝时抛出的异常 |
| Session（会话） | "历史存储" | 在运行之间持久化的对话记忆 |
| Tracing（追踪） | "Span" | LLM + 工具 + handoff + guardrail 的内置可观测性 |
| 阻塞防护栏 | "顺序检查" | 防护栏先运行；触发时不浪费 token |
| 并行防护栏 | "并发检查" | 防护栏并行运行；延迟更低，触发时浪费 token |

## 延伸阅读

- [OpenAI Agents SDK 文档](https://openai.github.io/openai-agents-python/) — 原语、handoffs、guardrails、tracing
- [Claude Agent SDK 概述](https://platform.claude.com/docs/en/agent-sdk/overview) — Claude 风格的对应物
- [Anthropic，构建有效智能体](https://www.anthropic.com/research/building-effective-agents) — 何时需要 handoffs
- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — Agents SDK span 映射到的标准
