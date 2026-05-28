# MCP 采样——服务器请求的 LLM 补全与智能体循环

> 大多数 MCP 服务器只是哑执行器：接收参数、运行代码、返回内容。采样（Sampling）让服务器反转方向：它请求客户端的 LLM 做出决策。这使服务器无需拥有任何模型凭据，即可托管智能体循环。2025-11-25 合并的 SEP-1577 在采样请求中添加了工具支持，使循环可以包含更深层次的推理。漂移风险提示：SEP-1577 的采样中工具形态在 2026 年 Q1 仍处于实验阶段，SDK API 尚在稳定中。

**类型：** 构建
**编程语言：** Python（标准库，采样测试套件）
**前置知识：** Phase 13 · 07（MCP 服务器）、Phase 13 · 10（资源与提示）
**预计时间：** 约 75 分钟

## 学习目标

- 解释 `sampling/createMessage` 解决了什么问题（服务器托管循环，无需服务器端 API 密钥）。
- 实现一个请求客户端对多轮提示进行采样并返回补全结果的服务器。
- 使用 `modelPreferences`（成本/速度/智能优先级）引导客户端的模型选择。
- 构建一个在内部通过采样迭代、而不是硬编码行为的 `summarize_repo` 工具。

## 问题背景

一个有用的代码摘要工作流 MCP 服务器需要：遍历文件树、选择要读取的文件、合成摘要、返回结果。LLM 推理在哪里发生？

**选项 A：** 服务器调用自己的 LLM。需要 API 密钥，服务器端计费，每用户成本高昂。

**选项 B：** 服务器返回原始内容，客户端的智能体进行推理。可以工作，但将服务器逻辑移入了客户端提示，这很脆弱。

**选项 C：** 服务器通过 `sampling/createMessage` 请求客户端的 LLM。服务器保留算法（读取哪些文件、进行几轮处理），客户端保留计费和模型选择。服务器完全不需要凭据。

采样就是选项 C。这是受信任服务器在不成为完整 LLM 宿主的情况下托管智能体循环的机制。

## 核心概念

### `sampling/createMessage` 请求

服务器发送：

```json
{
  "jsonrpc": "2.0",
  "id": 42,
  "method": "sampling/createMessage",
  "params": {
    "messages": [{"role": "user", "content": {"type": "text", "text": "..."}}],
    "systemPrompt": "...",
    "includeContext": "none",
    "modelPreferences": {
      "costPriority": 0.3,
      "speedPriority": 0.2,
      "intelligencePriority": 0.5,
      "hints": [{"name": "claude-3-5-sonnet"}]
    },
    "maxTokens": 1024
  }
}
```

客户端运行其 LLM，返回：

```json
{"jsonrpc": "2.0", "id": 42, "result": {
  "role": "assistant",
  "content": {"type": "text", "text": "..."},
  "model": "claude-3-5-sonnet-20251022",
  "stopReason": "endTurn"
}}
```

### `modelPreferences`

三个加起来为 1.0 的浮点数：

- `costPriority`：倾向更便宜的模型。
- `speedPriority`：倾向更快的模型。
- `intelligencePriority`：倾向更有能力的模型。

加上 `hints`：服务器偏好的命名模型。客户端可以采纳也可以忽略提示；客户端的用户配置始终优先。

### `includeContext`

三个值：

- `"none"`——仅使用服务器提供的消息。默认值。
- `"thisServer"`——包含来自该服务器会话的先前消息。
- `"allServers"`——包含所有会话上下文。

由于存在跨服务器上下文泄漏的安全隐患，`includeContext` 从 2025-11-25 起被软废弃。建议使用 `"none"`，并在消息中明确传递所需上下文。

### 采样中的工具（SEP-1577）

2025-11-25 新增：采样请求可以包含 `tools` 数组。客户端使用这些工具运行完整的工具调用循环。这让服务器可以通过客户端的模型托管 ReAct 风格的智能体循环。

```json
{
  "messages": [...],
  "tools": [
    {"name": "fetch_url", "description": "...", "inputSchema": {...}}
  ]
}
```

客户端循环：采样、如果调用了工具则执行、再次采样，直到返回最终助手消息。此功能在 2026 年 Q1 仍处于实验阶段；SDK 签名可能仍在变化。实现时请对照 2025-11-25 规范的 client/sampling 章节确认。

### 人在环（Human-in-the-loop）

客户端**必须**在运行采样之前向用户展示服务器请求模型做什么。恶意服务器可以利用采样操纵用户会话（"向用户说 X，使其点击 Y"）。Claude Desktop、VS Code 和 Cursor 将采样请求呈现为用户可以拒绝的确认对话框。

2026 年的共识：没有人工确认的采样是一个警示信号。网关（Phase 13 · 17）可以自动批准低风险采样，自动拒绝任何可疑内容。

### 无 API 密钥的服务器托管循环

典型用例：一个没有 LLM 访问权限的代码摘要 MCP 服务器。它执行：

1. 遍历代码库结构。
2. 调用 `sampling/createMessage`，内容为"选出五个最可能描述该代码库目的的文件"。
3. 读取这些文件。
4. 调用 `sampling/createMessage`，携带文件内容，内容为"用三段话总结这个代码库"。
5. 将摘要作为 `tools/call` 的结果返回。

服务器从不访问 LLM API。客户端的用户使用自己的凭据为补全买单。

### 安全风险（Unit 42 披露，2026 年 Q1）

- **隐蔽采样。** 某个工具总是调用采样并要求"从会话上下文中响应用户的电子邮件"。Phase 13 · 15 讲解攻击向量。
- **通过采样盗用资源。** 服务器请求客户端摘要攻击者的载荷，由用户承担费用。
- **循环炸弹。** 服务器在紧循环中调用采样。客户端**必须**强制执行每会话的速率限制。

## 动手实践

`code/main.py` 提供一个伪服务器-到-客户端的采样测试套件。一个模拟的 `summarize_repo` 工具触发两轮采样（选择文件，然后摘要），伪客户端返回预设的响应。该套件展示了：

- 服务器发送携带 `modelPreferences` 的 `sampling/createMessage`。
- 客户端返回补全结果。
- 服务器继续其循环。
- 速率限制器对每次工具调用的采样总次数进行上限控制。

重点关注：
- 服务器只暴露一个工具（`summarize_repo`）；所有推理都在采样调用中进行。
- 模型偏好权重影响客户端的模型选择；hints 列出了偏好的模型。
- 循环在 `stopReason: "endTurn"` 时终止。
- `max_samples_per_tool = 5` 的限制防止失控循环。

## 产出技能

本课产出 `outputs/skill-sampling-loop-designer.md`：给定一个需要 LLM 调用的服务器端算法（研究、摘要、规划），该技能设计基于采样的实现方案，包含正确的 modelPreferences、速率限制和安全确认。

## 练习

1. 运行 `code/main.py`。将 `max_samples_per_tool` 改为 2，观察速率限制截断效果。

2. 实现 SEP-1577 的采样中工具变体：采样请求携带 `tools` 数组。验证客户端循环在返回最终补全之前执行了这些工具。注意漂移风险：SDK 签名可能在 2026 年上半年仍有变化。

3. 添加人在环确认：在服务器的第一次 `sampling/createMessage` 之前，暂停并等待用户批准。被拒绝的调用返回一个类型化的拒绝结果。

4. 添加一个以客户端会话为键的每用户速率限制器。同一用户对同一服务器的循环应共享一个预算。

5. 设计一个使用采样选取要包含的块的 `summarize_pdf` 工具。勾勒发送的消息序列。`modelPreferences.intelligencePriority` 设为 0.1 和 0.9 时行为会有何不同？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 采样（Sampling） | "服务器到客户端的 LLM 调用" | 服务器请求客户端的模型提供补全结果 |
| `sampling/createMessage` | "该方法" | 采样请求的 JSON-RPC 方法 |
| `modelPreferences` | "模型优先级" | 成本/速度/智能权重加上名称提示 |
| `includeContext` | "跨会话泄漏" | 软废弃的上下文包含模式 |
| SEP-1577 | "采样中的工具" | 允许在采样内部使用工具以支持服务器托管的 ReAct |
| 人在环 | "用户确认" | 客户端在运行前向用户展示采样请求 |
| 循环炸弹 | "失控采样" | 服务器端的无限采样循环；客户端必须设置速率限制 |
| 隐蔽采样 | "隐藏推理" | 恶意服务器在采样提示中隐藏意图 |
| 资源盗用 | "占用用户的 LLM 预算" | 服务器强迫客户端为不需要的采样买单 |
| `stopReason` | "生成停止的原因" | `endTurn`、`stopSequence` 或 `maxTokens` |

## 延伸阅读

- [MCP——概念：采样](https://modelcontextprotocol.io/docs/concepts/sampling) — 采样的高层次概述
- [MCP——客户端采样规范 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/client/sampling) — `sampling/createMessage` 的规范形态
- [MCP——GitHub SEP-1577](https://github.com/modelcontextprotocol/modelcontextprotocol) — 采样中工具的规范演进提案（实验性）
- [Unit 42——MCP 攻击向量](https://unit42.paloaltonetworks.com/model-context-protocol-attack-vectors/) — 隐蔽采样和资源盗用模式
- [Speakeasy——MCP 采样核心概念](https://www.speakeasy.com/mcp/core-concepts/sampling) — 含客户端代码示例的详细讲解
