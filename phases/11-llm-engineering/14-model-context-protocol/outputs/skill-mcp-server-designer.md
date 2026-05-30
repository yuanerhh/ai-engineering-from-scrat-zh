---
name: mcp-server-designer
description: 设计并搭建包含工具、资源和安全默认配置的 MCP 服务器。
version: 1.0.0
phase: 11
lesson: 14
tags: [llm-engineering, mcp, tool-use]
---

给定一个领域（内部 API、数据库、文件来源）和将挂载该服务器的宿主，输出：

1. 原语映射。哪些能力变成 `tools`（动作），哪些变成 `resources`（只读数据），哪些变成 `prompts`（用户调用的模板）。每个原语一行。
2. 认证方案。Stdio（受信任的本地）、带 API 密钥的流式 HTTP，或带 PKCE 的 OAuth 2.1。选择并说明理由。
3. Schema 草稿。每个工具参数的 JSON Schema，`description` 字段针对模型工具选择进行调优（而非 API 文档）。
4. 破坏性操作列表。每个会改变状态的工具；要求 `destructiveHint: true` 和人工审批。
5. 测试计划。每个工具：一个仅 Schema 的合约测试、一个通过 MCP 客户端的端到端测试、一个红队提示词注入测试用例。

拒绝发布在没有审批路径的情况下写入磁盘或调用外部 API 的服务器。拒绝在单个服务器上暴露超过 20 个工具；改为拆分成按领域划分的服务器。
