# 并行工具调用与带工具的流式传输

> 三次独立的天气查询顺序执行就是三次往返。并行运行，总时间缩减为最慢单次调用的时间。每家前沿提供商现在都能在单次轮次中输出多个工具调用。收益是真实的，管道实现却有些微妙。本课介绍两个方面：并行扇出和流式参数重组，重点在于 ID 关联的陷阱。

**类型：** 构建
**编程语言：** Python（标准库，线程池 + 流式传输框架）
**前置知识：** Phase 13 · 02（函数调用深度解析）
**预计时间：** 约 75 分钟

## 学习目标

- 解释 `parallel_tool_calls: true` 存在的原因以及何时应禁用它。
- 在并行扇出过程中，将流式参数块与正确的工具调用 ID 关联。
- 在不提前解析的情况下，将不完整的 `arguments` 字符串重组为完整 JSON。
- 运行一个三城市天气基准测试，演示顺序与并行延迟的差异。

## 问题背景

如果没有并行调用，一个回答"班加罗尔、东京和苏黎世的天气如何"的智能体会这样运行：

```
用户 -> LLM
LLM -> 调用 get_weather(班加罗尔)
宿主 -> 运行执行器，回复结果
LLM -> 调用 get_weather(东京)
宿主 -> 运行执行器，回复结果
LLM -> 调用 get_weather(苏黎世)
宿主 -> 运行执行器，回复结果
LLM -> 最终文本答案
```

三次 LLM 往返，每次还要叠加执行器延迟，约为理想挂钟时间的 4 倍。

有了并行调用：

```
用户 -> LLM
LLM -> 调用 get_weather(班加罗尔); 调用 get_weather(东京); 调用 get_weather(苏黎世)
宿主 -> 并发运行三个执行器，回复三个结果
LLM -> 最终文本答案
```

一次 LLM 往返，执行器时间是三者的最大值而非总和。OpenAI、Anthropic 和 Gemini 的生产基准测试显示，扇出工作负载的挂钟时间减少 60%-70%。

代价是关联复杂性：三个调用乱序完成时，你的结果必须携带匹配的 `tool_call_id`，以便模型对应起来。流式传输时，必须在执行之前将不完整的参数片段组装成完整 JSON。Gemini 3 添加唯一 ID 部分是为了解决一个现实问题：对同一工具的两个并行调用无法区分。

## 核心概念

### 启用并行调用

- **OpenAI。** `parallel_tool_calls: true` 默认开启，设为 `false` 强制串行。
- **Anthropic。** 通过 `disable_parallel_tool_use: false`（Claude 3.5 及更高版本的默认值）启用并行，设为 `true` 强制串行。
- **Gemini。** 始终支持并行；`tool_config.function_calling_config.mode = "AUTO"` 让模型自行决定。

在以下情况下禁用并行：工具存在顺序依赖（`create_file` 然后 `write_file`）、一个调用的输出需要作为另一个的输入、或速率限制器无法处理扇出。

### ID 关联

模型输出的每个调用都有一个 `id`，宿主返回的每个结果必须包含相同的 `id`。没有这个，结果就是模糊的。

- **OpenAI。** 每条工具角色消息上的 `tool_call_id`。
- **Anthropic。** 每个 `tool_result` 块上的 `tool_use_id`。
- **Gemini。** 每个 `functionResponse` 上的 `id`（Gemini 3 及以上；Gemini 2 通过名称匹配，这在同名并行调用时会出问题）。

### 并发执行调用

宿主在各自的线程、协程或远程工作器上运行每个调用的执行器。最简单的框架使用线程池，生产环境使用 `asyncio.gather` 或结构化并发。完成顺序不可预测——ID 是标识符。

一个常见 bug：按调用列表顺序而非完成顺序回复结果。这通常可以工作，因为模型只关心 `tool_call_id`，但如果结果被丢弃或重复，乱序提交会使调试更加困难。建议以完成顺序回复并携带明确的 ID。

### 流式工具调用

当模型流式传输时，`arguments` 分块到达。三个并行调用各自的三股数据块在网络上交错。你需要为每个 ID 维护一个累加器。

各提供商的形态：

- **OpenAI。** 每个块是 `choices[0].delta.tool_calls[i].function.arguments`（部分字符串），块携带 `index`（在调用列表中的位置）。按索引累加，在首次出现时读取 `id`，当 `finish_reason = "tool_calls"` 时解析 JSON。
- **Anthropic。** 流式事件为 `message_start`，然后每个块各有一个 `content_block_start`（类型为 `tool_use`，包含 id、name、空 input）。`content_block_delta` 事件携带 `input_json_delta` 块，`content_block_stop` 关闭每个块。
- **Gemini。** `streamFunctionCallArguments`（Gemini 3 及以上）通过 `functionCallId` 输出块，使多个调用可以干净地交错。Gemini 3 之前，流式传输每次返回一个完整调用。

### 不完整 JSON 与提前解析陷阱

在 `arguments` 完整之前不能解析它。类似 `{"city": "Beng` 这样的不完整 JSON 是无效的，会抛出异常。正确的门控条件是提供商的调用结束信号：OpenAI 的 `finish_reason = "tool_calls"`、Anthropic 的 `content_block_stop` 或 Gemini 的流式结束事件。只有在那时才尝试 `json.loads`。更健壮的方法是使用在结构完成时产生事件的增量 JSON 解析器——OpenAI 的流式传输指南对希望显示实时"思考中"指示器的 UI 推荐这种方法。计算大括号作为完整性测试是不可靠的（引用字符串或转义内容中的大括号会导致误报），只能作为非正式调试启发方法。

### 乱序完成

```
call_A：快速 API，最先返回
call_B：慢速 API，第二返回
call_C：中速 API，第三返回
```

宿主的回复仍然必须引用 ID：

```
[{role: "tool", tool_call_id: "call_A", content: ...},
 {role: "tool", tool_call_id: "call_B", content: ...},
 {role: "tool", tool_call_id: "call_C", content: ...}]
```

在 OpenAI 或 Anthropic 上，回复中的顺序不影响正确性。Gemini 接受任意顺序，只要 ID 匹配即可。

### 基准测试：顺序 vs 并行

`code/main.py` 中的框架模拟三个延迟分别为 400、600 和 800ms 的执行器。顺序执行总计 1800ms，并行执行为 max(400, 600, 800) = 800ms。差距是固定的而非比例性的，因此节省量随工具数量增加而增大。

现实注意事项：并行调用会给下游 API 带来压力。对速率受限服务进行 10 路扇出会导致失败。Phase 13 · 17 介绍网关级背压控制，重试语义计划在后续章节中介绍。

### 流式扇出挂钟时间

如果模型本身在流式传输，可以在一个调用的参数完整后立即开始执行，而无需等待所有调用完成。这是 OpenAI 记录的优化，但并非所有 SDK 都暴露了这个特性。本课的框架实现了这一点：模拟流一旦产出完整的参数对象，宿主就立即启动该调用。

## 动手实践

`code/main.py` 分为两部分。第一部分使用 `concurrent.futures.ThreadPoolExecutor` 顺序和并行运行三个模拟天气调用，并打印挂钟时间。第二部分重放一个假流式响应——三个并行调用的 `arguments` 块交错在一条流上——并用 `StreamAccumulator` 按 ID 重组它们。无需 LLM，无需网络，只有重组逻辑。

重点关注：
- 顺序计时器达到 1.8 秒，并行计时器在相同的假延迟下达到 0.8 秒。
- 累加器通过按 ID 缓冲并仅在每个调用的 JSON 完整时解析来处理乱序到达的块。
- 执行器在一个 ID 的参数确定后立即启动，而不是等所有流结束。

## 产出技能

本课产出 `outputs/skill-parallel-call-safety-check.md`：给定一个工具注册表，该技能审核哪些工具可以安全并行化、哪些有顺序依赖、哪些会压垮下游速率限制——返回一个带有每个工具 `parallel_safe` 标志的修订注册表。

## 练习

1. 运行 `code/main.py` 并改变模拟延迟。确认并行与顺序的比率约为 `max/sum`（真实运行因线程调度、序列化和框架开销略有偏差）。在什么延迟分布下，并行化不再重要？

2. 扩展累加器，通过丢弃其缓冲区并发出 `cancelled` 事件来处理"调用在流中间被取消"的情况。哪家提供商明确记录了这种情况？检查 Anthropic 的 `content_block_stop` 语义和 OpenAI 的 `finish_reason: "length"` 行为。

3. 用 `asyncio.gather` 替换线程池。对两者进行基准测试。如果执行器做真实 I/O，你应该看到 async 有小幅优势，这源于更低的上下文切换开销。

4. 选择两个不应并行化的工具（例如 `create_file` 然后 `write_file`）。在注册表中添加 `ordering_dependency` 图，并根据该图门控并行扇出。这是依赖感知调度的最小机制，后续智能体工程阶段会将其正式化。

5. 阅读 OpenAI 的并行函数调用章节和 Anthropic 的 `disable_parallel_tool_use` 文档。找出 Anthropic 建议禁用并行的一种真实工具类型。（提示：对同一资源的重要性变更操作。）

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 并行工具调用 | "一轮扇出" | 模型在单条 assistant 消息中输出多个工具调用 |
| `parallel_tool_calls` | "OpenAI 的标志" | 启用或禁用多调用输出 |
| `disable_parallel_tool_use` | "Anthropic 的反向标志" | 退出标志；默认并行已启用 |
| 工具调用 ID | "关联句柄" | 每次调用的标识符，结果消息必须回传 |
| 累加器（Accumulator） | "流缓冲区" | 用于不完整 `arguments` 块的按 ID 字符串缓冲区 |
| 乱序完成 | "最快者先返回" | 并行调用以不可预测的顺序完成；ID 是纽带 |
| 依赖图 | "顺序约束" | 输出需要作为其他工具输入的工具；不能并行化 |
| 提前解析陷阱 | "JSON.parse 异常" | 尝试解析不完整的 `arguments` 字符串 |
| `streamFunctionCallArguments` | "Gemini 3 特性" | 每次调用带唯一 ID 的流式参数块 |
| 按完成顺序回复 | "不等全部完成" | 结果到达即按 ID 键回复，无需等待所有调用 |

## 延伸阅读

- [OpenAI — 并行函数调用](https://platform.openai.com/docs/guides/function-calling#parallel-function-calling) — 默认行为和退出标志
- [Anthropic — 工具使用：实现工具使用](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implementing-tool-use) — `disable_parallel_tool_use` 和结果批处理
- [Google — Gemini 函数调用并行章节](https://ai.google.dev/gemini-api/docs/function-calling) — Gemini 3 的 ID 关联并行调用
- [OpenAI — 带工具的流式响应](https://platform.openai.com/docs/api-reference/responses-streaming) — OpenAI 流的分块参数重组
- [Anthropic — 流式消息](https://docs.anthropic.com/en/api/messages-streaming) — `content_block_delta` 与 `input_json_delta`
