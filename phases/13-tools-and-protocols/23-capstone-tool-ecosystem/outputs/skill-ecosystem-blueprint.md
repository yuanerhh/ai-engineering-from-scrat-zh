---
name: ecosystem-blueprint
description: 根据产品需求生成完整的 Phase 13 生态系统架构，涵盖原语、安全态势、遥测和打包方式。
version: 1.0.0
phase: 13
lesson: 22
tags: [mcp, capstone, ecosystem, architecture, a2a, otel]
---

给定一个产品需求（研究、摘要、自动化，或任何由智能体驱动的工作流），生成完整架构。

输出内容：

1. **MCP 原语**。需要哪些工具、资源、Prompt 和任务。是否需要 `ui://` 应用？是否有异步任务？
2. **安全态势**。OAuth 2.1 scope 集合、网关 RBAC 矩阵、固定哈希清单、Rule of Two 审计。
3. **A2A 协作**。识别子智能体调用。定义其 Agent Card。
4. **遥测**。OTel GenAI Span 层级。导出器和后端选择。
5. **打包**。AGENTS.md、SKILL.md 和部署面（Docker Compose、K8s）。
6. **Phase 13 课程映射**。每个设计选择可追溯回哪一课。

硬性拒绝：
- 任何在单次轮次中将不可信输入、敏感数据和高危操作相结合的架构（Rule of Two）。
- 任何缺少跨 MCP 和 A2A 跃点追踪传播的架构。
- 任何 LLM 层没有至少一个回退提供商的架构。

拒绝规则：
- 若产品需求更适合直接调用 LLM，拒绝搭建完整生态系统。
- 若团队缺少网关的 SRE 支持，建议使用托管网关（Cloudflare MCP Portals、Portkey）。
- 若架构涉及支付，将 AP2 标注为存在偏差风险的 A2A 扩展，并建议单独进行审批。

输出：一页蓝图，包含原语、安全态势、A2A 跃点、遥测方案、打包方式和课程映射。结尾用一句话指出该部署最难应对的单一运营风险。
