# MCP 网关与注册表——企业控制平面

> 企业不能让每个开发者随意安装 MCP 服务器。网关将认证、RBAC、审计、速率限制、缓存和工具投毒检测集中管理，然后将合并后的工具集以单一 MCP 端点的形式暴露出来。官方 MCP 注册表（Anthropic + GitHub + PulseMCP + Microsoft，命名空间已验证）是权威的上游来源。本课介绍网关的位置、最小实现方案，并调研 2026 年的供应商生态。

**类型：** 学习
**编程语言：** Python（标准库，最小网关）
**前置知识：** Phase 13 · 15（工具投毒）、Phase 13 · 16（OAuth 2.1）
**预计时间：** 约 45 分钟

## 学习目标

- 解释 MCP 网关所处的位置（介于 MCP 客户端和多个后端 MCP 服务器之间）。
- 实现网关的五项职责：认证、RBAC、审计、速率限制、策略。
- 在网关层强制执行工具哈希锁定清单。
- 区分官方 MCP 注册表与元注册表（Glama、MCPMarket、MCP.so、Smithery、LobeHub）。

## 问题背景

一家财富 500 强企业有 30 个已批准的 MCP 服务器、5000 名开发者、合规和审计要求，以及需要集中策略管理的安全团队。让每个开发者在 IDE 中随意安装服务器根本不可行。

网关模式：

1. 网关作为单一 Streamable HTTP 端点运行，供开发者连接。
2. 网关持有每个后端 MCP 服务器的凭据。
3. 每个开发者请求通过网关自身的 OAuth 进行认证和范围限定。
4. 网关将调用路由到后端服务器，同时应用策略。
5. 所有调用都记录日志用于审计。

Cloudflare MCP Portals、Kong AI Gateway、IBM ContextForge、MintMCP、TrueFoundry、Envoy AI Gateway——这些公司都在 2025-2026 年发布了网关或网关功能。

与此同时，官方 MCP 注册表作为权威上游来源启动：经过策划、命名空间验证、以反向 DNS 命名的服务器，网关可以从中拉取。元注册表（Glama、MCPMarket、MCP.so、Smithery、LobeHub）聚合来自多个来源的服务器。

## 核心概念

### 网关的五项职责

1. **认证（Auth）。** OAuth 2.1 用于识别开发者；映射到用户角色。
2. **RBAC。** 每用户策略：哪些服务器、哪些工具、哪些范围。
3. **审计（Audit）。** 记录每次调用的主体、内容、时间和结果。
4. **速率限制（Rate limit）。** 每用户/每工具/每服务器的上限，防止滥用。
5. **策略（Policy）。** 拒绝投毒描述、强制执行二规则、编辑 PII。

### 网关作为单一端点

对开发者而言，网关看起来像一个 MCP 服务器。在内部，它路由到 N 个后端。会话 ID（Phase 13 · 09）在边界处被重写。

### 凭据保险库

开发者永远不会看到后端令牌。网关持有这些令牌（或代理到持有令牌的身份提供商）。在网关上拥有 `notes:read` 的开发者可以通过网关自己的后端凭据间接访问笔记 MCP 服务器——但只能在约束传递访问的策略下进行。

### 网关层的工具哈希锁定

网关持有已批准工具描述的清单（SHA256 哈希）。在发现阶段，它获取每个后端的 `tools/list`，将哈希与清单进行比较，并移除任何描述已发生变化的工具。这是 Phase 13 · 15 中地毯抽走防御的集中应用。

### 策略即代码

高级网关使用 OPA/Rego、Kyverno 或 Styra 表达策略。"用户 `alice` 只能在 `acme` 组织的仓库上调用 `github.open_pr`"这样的规则以声明方式编码。简单网关使用手写的 Python。两种形式都有效。

### 会话感知路由

当用户会话包含多个服务器时，网关进行多路复用：开发者的单个 MCP 会话持有 N 个后端会话，每个服务器一个。来自任何后端的通知都通过网关路由到开发者的会话。

### 命名空间合并

网关合并来自所有后端的工具命名空间，通常在发生冲突时添加前缀：`github.open_pr`、`notes.search`。这使路由无歧义。

### 注册表

- **官方 MCP 注册表（`registry.modelcontextprotocol.io`）。** 在 Anthropic、GitHub、PulseMCP、Microsoft 的监管下启动。命名空间已验证（反向 DNS：`io.github.user/server`）。预先过滤了基本质量。
- **Glama。** 以搜索为核心的元注册表，聚合多个来源。
- **MCPMarket。** 偏商业导向的目录，包含供应商列表。
- **MCP.so。** 社区目录；开放提交。
- **Smithery。** 包管理器风格的安装流程。
- **LobeHub。** 集成在其 LobeChat 应用中的 UI 注册表。

企业网关默认从官方注册表拉取，允许管理员从元注册表中策划添加，并拒绝所有未锁定的内容。

### 反向 DNS 命名

官方注册表为公开服务器强制要求反向 DNS 名称：`io.github.alice/notes`。命名空间防止抢注，使信任委托更加清晰。

### 供应商调研，2026 年 4 月

| 供应商 | 优势 |
|--------|------|
| Cloudflare MCP Portals | 边缘托管；OAuth 集成；有免费层 |
| Kong AI Gateway | Kubernetes 原生；细粒度策略；日志输出到 OpenTelemetry |
| IBM ContextForge | 企业 IAM；合规；审计导出 |
| TrueFoundry | 偏 DevOps；指标优先 |
| MintMCP | 面向开发者平台 |
| Envoy AI Gateway | 开源；可定制过滤器 |

Phase 17（生产基础设施）对网关运维进行了更深入的讲解。

## 动手实践

`code/main.py` 以约 150 行代码实现了一个最小网关：通过伪 Bearer 令牌认证用户，持有每用户的 RBAC 策略，将请求路由到两个后端 MCP 服务器，将每次调用写入审计日志，强制执行速率限制，并拒绝任何描述哈希与锁定清单不匹配的后端工具。

重点关注：
- `RBAC` 字典以 `user_id` 为键，包含允许的 `server_tool` 条目。
- `AUDIT_LOG` 是一个仅追加的事件列表。
- 速率限制对每个用户使用令牌桶算法。
- 锁定清单是 `server::tool -> hash` 的字典。

## 产出技能

本课产出 `outputs/skill-gateway-bootstrap.md`：给定一个企业 MCP 计划（用户、后端、合规要求），该技能生成网关配置规范。

## 练习

1. 运行 `code/main.py`。以允许的用户发起调用；以不允许的用户发起调用；进行超出速率限制的突发请求。验证三种流程均符合预期。

2. 添加一个在返回客户端之前从结果中编辑 PII 的策略。对 SSN 格式的字符串使用简单的正则表达式处理；注意缺口（电子邮件、电话号码）。

3. 扩展审计日志以发出 OpenTelemetry GenAI Span。Phase 13 · 20 讲解具体属性。

4. 为一个有 50 名开发者和五个后端（笔记、github、postgres、jira、slack）的团队设计 RBAC 策略。谁在每个后端上获得只读权限？谁获得写入权限？

5. 从头到尾阅读 Cloudflare 企业 MCP 博文。找出 Cloudflare 提供但此标准库网关没有的一个功能。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 网关（Gateway） | "MCP 代理" | 介于客户端和后端之间的集中管理服务器 |
| 凭据保险库 | "后端令牌留在服务器端" | 开发者永远不会看到上游令牌 |
| 会话感知路由 | "多后端会话" | 网关为每个开发者会话多路复用 N 个后端会话 |
| 工具哈希锁定 | "已批准清单" | 每个已批准工具描述的 SHA256；集中阻止地毯抽走 |
| RBAC | "每用户策略" | 工具和服务器的基于角色的访问控制 |
| 策略即代码 | "声明式规则" | OPA/Rego、Kyverno、Styra 策略在网关层强制执行 |
| 审计日志 | "谁、做了什么、何时" | 用于合规的仅追加事件日志 |
| 速率限制 | "每用户令牌桶" | 每分钟上限，防止滥用 |
| 官方 MCP 注册表 | "权威上游" | `registry.modelcontextprotocol.io`，命名空间已验证 |
| 反向 DNS 命名 | "注册表命名空间" | `io.github.user/server` 约定 |

## 延伸阅读

- [官方 MCP 注册表](https://registry.modelcontextprotocol.io/) — 权威上游，命名空间已验证
- [Cloudflare——企业 MCP](https://blog.cloudflare.com/enterprise-mcp/) — 带 OAuth 和策略的网关模式
- [agentic-community——MCP 网关注册表](https://github.com/agentic-community/mcp-gateway-registry) — 开源参考网关
- [TrueFoundry——什么是 MCP 网关？](https://www.truefoundry.com/blog/what-is-mcp-gateway) — 功能对比文章
- [IBM——MCP context forge](https://github.com/IBM/mcp-context-forge) — IBM 的企业网关
