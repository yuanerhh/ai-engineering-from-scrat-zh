# 顶点项目 13 — MCP 服务器（含注册中心与治理）

> Model Context Protocol 在 2026 年停止了"未来技术"的身份，成为默认的工具调用规范。Anthropic、OpenAI、Google 以及每个主流 IDE 都内置了 MCP 客户端。Pinterest 发布了其内部 MCP 服务器生态系统。AAIF 注册中心在 `.well-known` 下形式化了能力元数据。AWS ECS 发布了参考无状态部署方案。Block 的 goose-agent 将同一协议内置进了一个托管助手。2026 年的生产形态是：StreamableHTTP 传输，OAuth 2.1 作用域，OPA 策略门控，以及一个让平台团队能够发现、验证和启用服务器的注册中心。端到端构建这一切。

**类型：** 顶点项目  
**语言：** Python（服务器，通过 FastMCP）或 TypeScript（@modelcontextprotocol/sdk），Go（注册中心服务）  
**前置条件：** 第 11 阶段（LLM 工程），第 13 阶段（工具与 MCP），第 14 阶段（智能体），第 17 阶段（基础设施），第 18 阶段（安全）  
**涵盖阶段：** P11 · P13 · P14 · P17 · P18  
**预计时间：** 25 小时

## 问题背景

MCP 成为了工具调用的通用语。Claude Code、Cursor 3、Amp、OpenCode、Gemini CLI 以及每个托管智能体现在都消费 MCP 服务器。生产挑战不在于编写服务器（FastMCP 使这变得容易），而在于以企业需求规模化部署：每租户 OAuth 作用域、破坏性工具的 OPA 策略、StreamableHTTP 无状态扩缩容、用于发现的注册中心、每次工具调用的审计日志。Pinterest 的内部 MCP 生态系统和 AAIF 注册中心规范设定了 2026 年的标准。

你将构建一个暴露 10 个内部工具（Postgres 只读、S3 列表、Jira、Linear、Datadog 等）的 MCP 服务器，一个供平台发现使用的注册中心 UI，以及一个针对破坏性工具的人工审批门控。负载测试展示 StreamableHTTP 水平扩缩容。审计追踪满足企业安全审查。

## 核心概念

MCP 2026 修订版将 StreamableHTTP 指定为默认传输。与早期的 stdio 和 SSE 形态不同，StreamableHTTP 默认无状态：单个 HTTP 端点接受 JSON-RPC 请求，流式传输响应，并支持通知的长连接。无状态意味着在负载均衡器后面可以水平扩缩容。

授权使用带每工具作用域的 OAuth 2.1。一个 token 携带如 `jira:read`、`s3:list`、`postgres:query:readonly` 这样的作用域。MCP 服务器在工具调用时而不仅仅是会话开始时检查作用域。对于高风险工具，服务器拒绝任何在过去 N 分钟内未将作用域提升到 `approved:by:human` 的调用——该提升来自 Slack 审查卡片。

注册中心是一个独立的服务。每个 MCP 服务器在 `.well-known/mcp-capabilities` 上暴露一个带工具清单、传输 URL、认证要求的文档。注册中心轮询、验证和索引。平台团队使用注册中心 UI 查看哪些工具可用、需要哪些作用域以及哪些团队拥有它们。

## 架构图

```
MCP client (Claude Code, Cursor 3, ...)
          |
          v
StreamableHTTP over HTTPS (JSON-RPC + streaming)
          |
          v
MCP server (FastMCP) behind load balancer
          |
   +------+------+---------+----------+------------+
   v             v         v          v            v
Postgres    S3 listing  Jira       Linear     Datadog
(read-only) (paged)     (read)     (read)     (query)
          |
   +------+-------------+
   v                    v
 OPA policy gate   destructive tool MCP (separate server)
                        |
                        v
                   human approval via Slack
                        |
                        v
                   audit log (append-only, per-tenant)

  registry service
     |
     v  GET /.well-known/mcp-capabilities from each server
     v
     UI: search / validate / enable-disable / ownership
```

## 技术栈

- 服务器框架：FastMCP（Python）或 `@modelcontextprotocol/sdk`（TypeScript）
- 传输：StreamableHTTP over HTTPS（无状态）
- 认证：OAuth 2.1 带通过 SPIFFE/SPIRE 的工作负载身份
- 策略：每工具 OPA/Rego 规则；每请求策略决策服务
- 注册中心：自托管，消费 `.well-known/mcp-capabilities` 清单
- 人工审批：破坏性工具的 Slack 交互消息
- 部署：AWS ECS Fargate 或 Fly.io，每租户一个服务器或带租户作用域的共享服务器
- 审计：每租户存储桶的结构化 JSONL，带每次调用的溯源记录

## 构建步骤

1. **工具表面。** 暴露 10 个内部工具：Postgres 只读查询、S3 列表对象、Jira 搜索/获取、Linear 搜索/获取、Datadog 指标查询、PagerDuty 值班查询、GitHub 只读、Notion 搜索、Slack 搜索、Salesforce 读取。每个工具都有类型化模式和作用域标签。

2. **FastMCP 服务器。** 挂载工具。配置 StreamableHTTP 传输。添加 OAuth token 内省和作用域执行的中间件。

3. **OPA 策略。** 每工具 Rego 策略：哪些作用域允许调用，适用哪些 PII 脱敏，适用哪些 payload 大小上限。每次工具调用时调用决策服务。

4. **注册中心服务。** 独立的 Go 或 TS 服务，从注册的服务器轮询 `.well-known/mcp-capabilities`，用 JSON Schema 验证，并暴露列表/搜索/验证/启用-禁用 UI。

5. **能力清单。** 每个服务器暴露 `.well-known/mcp-capabilities`，包含：工具列表、认证要求、传输 URL、所有者团队、SLO。

6. **破坏性工具分离。** 改变状态的工具（Jira 创建、Linear 创建、Postgres 写入）位于第二个带更严格认证流的 MCP 服务器：token 必须在 15 分钟内通过 Slack 卡片提升到 `approved:by:human` 作用域。

7. **审计日志。** 每租户追加式 JSONL：`{timestamp, user, tool, args_redacted, response_redacted, outcome}`。写入前通过 Presidio 进行 PII 脱敏。

8. **负载测试。** StreamableHTTP 上 100 个并发客户端。通过添加第二个副本演示水平扩缩容；展示负载均衡器无需会话粘性即可重新分配。

9. **一致性测试。** 对两个服务器运行官方 MCP 一致性套件。通过所有强制性部分。

## 使用示例

```
$ curl -H "Authorization: Bearer eyJhbGc..." \
       -X POST https://mcp.internal.example.com/ \
       -d '{"jsonrpc":"2.0","method":"tools/call",
            "params":{"name":"postgres.readonly","arguments":{"sql":"SELECT 1"}}}'
[registry]   capability validated: postgres.readonly v1.2
[policy]    scope postgres:query:readonly present; allowed
[audit]     logged: user=u42 tool=postgres.readonly outcome=ok
response:    { "result": { "rows": [[1]] } }
```

## 交付物

`outputs/skill-mcp-server.md` 描述了交付物。一个面向内部工具的生产级 MCP 服务器 + 注册中心 + 审计层，带 OAuth 2.1 作用域和 OPA 门控。

| 权重 | 评分标准 | 衡量方式 |
|:-:|---|---|
| 25 | 规范一致性 | StreamableHTTP + 能力清单通过 MCP 一致性测试 |
| 20 | 安全性 | 作用域执行、所有工具的 OPA 覆盖、密钥卫生 |
| 20 | 可观测性 | 带 PII 脱敏的每次工具调用审计日志 |
| 20 | 规模 | 100 个客户端负载测试水平扩缩容演示 |
| 15 | 注册中心体验 | 发现/验证/启用-禁用工作流 |
| **100** | | |

## 练习题

1. 添加一个新工具（Confluence 搜索）。在不触碰核心服务器的情况下，通过注册中心验证流程发布它。

2. 编写一个 OPA 策略，对包含名为 `email`、`ssn` 或 `phone` 列的 Postgres 查询结果进行脱敏。用探测查询进行验证。

3. 基准测试 StreamableHTTP vs stdio 的本地延迟。报告每次调用的 p50/p95。

4. 实现每租户配额：每个工具每租户每分钟最多 N 次调用。通过第二条 OPA 规则执行。

5. 运行 [mcp-conformance-tests](https://github.com/modelcontextprotocol/conformance) 的 MCP 一致性套件，修复每一个失败。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| StreamableHTTP | "2026 年 MCP 传输" | 无状态 HTTP + 流式传输；为网络服务器替代 SSE + stdio |
| 能力清单 | "well-known 文档" | `.well-known/mcp-capabilities`，包含工具列表、认证、传输 URL |
| OPA/Rego | "策略引擎" | Open Policy Agent，用于针对外部规则授权工具调用 |
| 作用域提升 | "经人工审批" | 通过 Slack 审批授予的短期作用域，破坏性工具必需 |
| 注册中心 | "工具发现" | 从能力清单索引 MCP 服务器的服务 |
| 工作负载身份 | "SPIFFE/SPIRE" | 用于 OAuth token 颁发的加密服务身份 |
| 一致性套件 | "规范测试" | StreamableHTTP + 工具清单正确性的官方 MCP 测试套件 |

## 延伸阅读

- [Model Context Protocol 2026 路线图](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — StreamableHTTP、能力元数据、注册中心
- [AAIF MCP 注册中心规范](https://github.com/modelcontextprotocol/registry) — 2026 年注册中心规范
- [AWS ECS 参考部署](https://aws.amazon.com/blogs/containers/deploying-model-context-protocol-mcp-servers-on-amazon-ecs/) — 参考生产部署
- [Pinterest 内部 MCP 生态系统](https://www.infoq.com/news/2026/04/pinterest-mcp-ecosystem/) — 参考内部部署
- [Block goose MCP 用法](https://block.github.io/goose/) — 参考智能体消费模式
- [FastMCP](https://github.com/jlowin/fastmcp) — Python 服务器框架
- [Open Policy Agent](https://www.openpolicyagent.org/) — 策略引擎参考
- [SPIFFE/SPIRE](https://spiffe.io) — 工作负载身份参考
