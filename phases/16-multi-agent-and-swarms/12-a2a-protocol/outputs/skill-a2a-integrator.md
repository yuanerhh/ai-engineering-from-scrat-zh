---
name: a2a-integrator
description: 设计两个智能体之间的 A2A 集成——Agent Card、任务模式、认证、流式传输或轮询。
version: 1.0.0
phase: 16
lesson: 12
tags: [multi-agent, a2a, protocol, interoperability, google]
---

给定两个需要互操作的智能体系统，生成 A2A 集成计划：Agent Card 内容、任务模式、认证、传输模式。

产出内容：

1. **Agent Card。** 名称、版本、技能、端点、支持的模态（文本、结构化、图像、音频、视频）、protocol_version、认证声明。
2. **每项技能的任务模式。** 输入 JSON 模式 + 工件 JSON 模式。要明确——客户端会进行验证。
3. **认证选择。** Bearer 令牌（OAuth2 或不透明）、mTLS 或签名请求。根据威胁模型（公共互联网、VPC、混合）说明理由。
4. **传输模式。** 轮询 vs SSE 流式传输 vs webhook 回调。流式传输用于长期运行或进度密集型任务；轮询用于短任务。
5. **速率限制。** 每客户端和每任务限制。防止滥用。
6. **幂等性。** 重复 `POST /tasks` 请求的处理策略（客户端侧任务键、服务器端去重）。
7. **故障处理。** `failed` 之外的任务状态（可重试 vs 致命）、死信策略、错误工件模式。
8. **MCP vs A2A 分割。** 如果远程智能体内部使用 MCP，说明哪些工具是暴露的，哪些保留在内部。

硬性拒绝：

- 没有声明协议版本的 Agent Card。
- 在用例需要结构化时使用自由格式文本的任务模式。
- 公共互联网部署上 Auth=none。

拒绝规则：

- 如果两个智能体在同一进程中运行，拒绝 A2A 并推荐直接 Python/JS 调用。A2A 用于跨系统边界。
- 如果延迟要求是亚 100ms 往返，拒绝 A2A 并推荐使用共享模式的直接 RPC。
- 如果远程智能体不声明 Agent Card，拒绝集成并推荐先发布 Agent Card。

输出：一页集成简报。结尾内联粘贴 Agent Card JSON，以便工程师可以直接放入 `/.well-known/agent.json`。
