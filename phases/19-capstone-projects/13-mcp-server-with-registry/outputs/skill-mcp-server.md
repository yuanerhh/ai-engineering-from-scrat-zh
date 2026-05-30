---
name: mcp-server-platform
description: 部署一个生产级 MCP 服务器，具备 StreamableHTTP、OAuth 2.1 作用域、OPA 策略、破坏性工具的人工审批门控和用于发现的注册表。
version: 1.0.0
phase: 19
lesson: 13
tags: [capstone, mcp, fastmcp, streamablehttp, oauth, opa, registry, governance]
---

给定一个企业环境，交付一个包含 10 个内部工具的 MCP 服务器、一个用于发现的注册表服务，以及一个通过 Slack 审批对破坏性工具进行门控的治理层。

构建计划：

1. FastMCP 服务器暴露 10 个只读工具（Postgres、S3、Jira、Linear、Datadog、PagerDuty、GitHub、Notion、Slack、Salesforce），每个工具都有类型化 schema 和所需作用域。
2. StreamableHTTP 传输，无状态，部署在负载均衡器后方。
3. OAuth 2.1 Token 自省中间件；通过 SPIFFE / SPIRE 实现工作负载身份。
4. 对每次工具调用进行 OPA / Rego 策略决策：作用域执行、PII 脱敏、载荷大小限制。
5. 破坏性工具（Jira 创建、Linear 创建、Postgres 写入）在单独的 MCP 服务器上，需要在 15 分钟内通过 Slack 卡片提升的 `approved:by:human` 作用域。
6. 注册表服务，轮询每个服务器的 `.well-known/mcp-capabilities`，使用 JSON Schema 验证，并提供列出/搜索/验证/启用的 UI。
7. 每租户 JSONL 审计日志，写入前使用 Presidio 进行 PII 脱敏。
8. 100 客户端压力测试，演示水平扩展；通过 MCP 一致性套件。

评估标准：

| 权重 | 评估项 | 度量方式 |
|:-:|---|---|
| 25 | 规范一致性 | StreamableHTTP + 能力清单通过 MCP 一致性测试 |
| 20 | 安全性 | 每个工具的作用域执行、OPA 覆盖率、密钥卫生 |
| 20 | 可观测性 | 带 PII 脱敏写入的每次工具调用审计日志 |
| 20 | 规模 | 100 客户端压力测试与水平扩展演示 |
| 15 | 注册表用户体验 | 发现/验证/启用-禁用工作流验证 |

硬性拒绝条件：

- 需要有状态会话的服务器（违反 2026 年 StreamableHTTP 无状态合约）。
- 破坏性工具与只读工具共享同一认证面的单服务器拓扑。
- 持久化原始 PII 的审计日志。
- 忽略能力清单；注册表集成是硬性要求。

拒绝规则：

- 拒绝在没有 OAuth 的情况下部署；匿名访问是取消资格的条件。
- 拒绝发布没有 Slack 审批流程的破坏性工具。
- 拒绝暴露作用域或描述未在能力清单中的工具。

输出：一个包含两个 MCP 服务器（只读 + 破坏性）、注册表服务、Slack 审批集成、OPA 策略、100 客户端压力测试框架、一致性测试结果的代码库，以及一份说明你考虑过但未暴露的工具（及原因），以及在试运行中捕获到近未遂情况的三大 OPA 规则的报告。
