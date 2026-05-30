---
name: mcp-handshake-tracer
description: 给定 MCP 客户端-服务端对话的 pcap 风格记录，对每条消息注释其原语、生命周期阶段和能力依赖。
version: 1.0.0
phase: 13
lesson: 06
tags: [mcp, json-rpc, lifecycle, capabilities]
---

给定从 MCP 会话中捕获的 JSON-RPC 2.0 消息序列，生成一份逐步解析，命名每条消息的原语、生命周期阶段和底层能力标志。

输出内容：

1. 逐消息注释。对每个 `{request, response, notification}`，说明：方向（客户端到服务端或服务端到客户端）、原语（tools / resources / prompts / roots / sampling / elicitation / lifecycle）、生命周期阶段，以及此消息有效所需已协商的能力标志。
2. 能力检查。从记录中重建 `initialize` 交换并列出所有已协商的能力。标记任何违反缺失能力的消息。
3. 错误诊断。对每个 JSON-RPC 错误，根据上下文命名错误码及最可能的原因。
4. 完整性审计。标记缺少以下任一项的记录：`initialize`、`initialized` 通知、至少一个 `tools/list` 或等价调用、优雅关闭。
5. 规范合规性。按 2025-11-25 规范的最小字段集检查每个请求的参数。标记遗漏项。

硬性拒绝：
- 任何使用规范允许集合之外的方法且没有 `x-` 前缀的消息。
- 任何客户端未声明 `sampling` 能力时出现的 `sampling/createMessage` 消息。
- 任何在 `notifications/initialized` 到达之前发生的调用。

拒绝规则：
- 如果被要求审计非 MCP 协议的记录，拒绝并指向 A2A 规范（Phase 13 · 19）作为替代。
- 如果被要求"修复"记录，拒绝。本技能仅注释，不重写。将修正路由到实现 SDK。

输出：按到达顺序对每条消息单独一行进行注释：`[phase/primitive/capability] <method or result shape>`。以三行摘要作为结尾，说明任何能力违规和缺失的生命周期步骤。
