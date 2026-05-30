---
name: a2a-agent-spec
description: 为需要通过 A2A 调用的智能体生成 Agent Card 和技能模式。
version: 1.0.0
phase: 13
lesson: 18
tags: [a2a, agent-card, task-lifecycle, delegation]
---

给定智能体的能力和预期协作者，生成其 A2A Agent Card 和技能定义。

输出内容：

1. **Agent Card**。`name`、`description`、`url`、`version`、`schemaVersion`、`capabilities`（streaming、pushNotifications）、`skills[]`。
2. **技能列表**。每项包含 `id`、`name`、`description`、`inputModes`、`outputModes`。在描述中使用"在 X 情况下使用，不适用于 Y。"模式。
3. **任务状态方案**。每个技能的预期状态转换路径和 input_required 路径。
4. **签名方案**。是否通过 AP2 对 Card 签名（推荐用于对外可调用的智能体）。
5. **传输协议**。HTTP 上的 JSON-RPC（默认）或 gRPC。注意与 v1.0 的向后兼容性。

硬性拒绝：
- 任何没有稳定 URL 的 Agent Card。破坏服务发现。
- 任何未声明输入和输出模式的技能。调用方无法判断兼容性。
- 任何对外可调用的智能体未制定 AP2 签名方案。存在身份冒充风险。

拒绝规则：
- 若智能体用途仅为单次工具调用，拒绝搭建 A2A；建议使用 MCP。
- 若智能体暴露了不应公开的内部信息（工具调用追踪、思维链），拒绝并要求不透明化。
- 若智能体需要 A2A 用于支付（AP2 用例），确认 AP2 扩展版本，并说明 AP2 与核心 A2A 是相互独立的。

输出：一页 Agent Card JSON、每个操作的技能模式、状态转换方案、签名和传输选择。结尾给出该智能体承诺的最低 v1.0 向后兼容性保证。
