---
name: mcp-auth-iii-wiring
description: 将生产级 MCP 授权（RFC 8414、7591、8707、7636 PKCE、9728）对接到 iii 原语——用 registerTrigger 处理 HTTP/cron，用 registerFunction 处理验证，用 state::* 处理 JWKS 缓存。
version: 1.0.0
phase: 13
lesson: 18
tags: [mcp, oauth, dcr, jwks, iii, rfc8414, rfc7591, rfc8707, rfc7636, rfc9728]
---

给定 MCP 服务器配置和 IdP 能力集，生成构成生产级鉴权面的 iii 原语和拒绝规则。

输入：

- `mcp_resource_url` — 规范资源 URL（无路径），用作 `aud` 和受保护资源元数据的 `resource` 值。
- `idp_metadata_url` — IdP 的 `/.well-known/oauth-authorization-server` URL。
- `idp_capabilities` — `code_challenge_methods_supported`、`grant_types_supported`、`registration_endpoint`、`response_types_supported` 的观察值。
- `tools` — MCP 工具列表，包含每个工具所需的 scope。

输出内容：

1. **拒绝门控**。若以下四个条件任意一项未满足，拒绝接线并停止：
   - `code_challenge_methods_supported` 中缺少 `S256`。
   - `grant_types_supported` 中缺少 `authorization_code`。
   - 无 `registration_endpoint`（不支持 RFC 7591 DCR）。
   - `response_types_supported` 不恰好为 `["code"]`。

2. **受保护资源元数据文档**（RFC 9728），供 MCP 服务器在 `/.well-known/oauth-protected-resource` 发布。包含 `resource`、`authorization_servers`（颁发者允许列表）、`scopes_supported`、`bearer_methods_supported: ["header"]`。

3. **iii 触发器注册**。逐字生成每条调用：
   - `iii.registerTrigger("http", {"path": "/.well-known/oauth-protected-resource", "method": "GET"}, "auth::serve-protected-resource")`
   - `iii.registerTrigger("http", {"path": "/mcp", "method": "POST"}, "mcp::dispatch")` — 调度器在任何工具运行前调用 `iii.trigger("auth::validate-jwt", ...)`。
   - `iii.registerTrigger("cron", {"schedule": "<rotation_schedule>"}, "auth::rotate-jwks")` — 默认计划为 `0 */6 * * *`；高轮换 IdP 收紧至 `*/15 * * * *`。

4. **iii 函数注册**。逐字生成每条调用：
   - `iii.registerFunction("auth::validate-jwt", handler)` — 检查 `iss` 允许列表、针对缓存 JWKS 的签名、`aud == mcp_resource_url`、`exp`、所需 scope。
   - `iii.registerFunction("auth::rotate-jwks", handler)` — 获取 `jwks_uri`，写入 `state::set("auth/jwks/<iss>", {keys, fetched_at})`。
   - `iii.registerFunction("auth::serve-protected-resource", handler)` — 返回第 (2) 条中的文档。
   - `iii.registerFunction("auth::issue-step-up", handler)` — 仅当工具列表中存在用户初始未授予 scope 所控的操作时启用。

5. **状态键方案**。每个接受的颁发者对应一个键：`auth/jwks/<issuer>`，存储 `{keys, fetched_at}`。记录读取模式：验证器从 `state::get` 读取；`kid` 未命中时回退到同步 `iii.trigger("auth::rotate-jwks", ...)`。

6. **Scope 映射**。将每个工具映射到其所需 scope。输出表格：
   `| tool | required_scope | rationale |`。将破坏性工具归入独立 scope；读取 scope 不得复用于写入工具。

7. **运行时拒绝规则**（验证器必须编码这些规则——在处理器体内生成）：
   - `aud != mcp_resource_url` 时拒绝。
   - `iss not in authorization_servers` 时拒绝。
   - 经过一次轮换回退后 `kid` 仍不在缓存 JWKS 中时拒绝。
   - 所需 scope 缺失时 → 403 `Bearer error="insufficient_scope", scope="<required>", resource="<mcp_resource_url>"`。
   - 拒绝任何不携带 `code_verifier` 或 `resource` 参数的令牌请求。

硬性拒绝（永不接线，拒绝请求并记录原因）：

- 以明文在 iii 状态存储中存储 `client_secret`。公开客户端使用 `token_endpoint_auth_method: none`；机密客户端使用 `private_key_jwt`。`state::*` 或注册响应日志中不得存储明文共享密钥。
- 跳过验证器的 `aud` 检查。混淆代理正是 RFC 8707 + RFC 9728 存在的原因。
- 允许无 PKCE 的授权码请求。OAuth 2.1 明确禁止；验证器必须拒绝任何存储的授权码记录中缺少 `code_challenge` 的 `/token` 兑换。
- 在没有刷新任务的情况下缓存 JWKS。要么随附 cron 触发器，要么不部署鉴权面。
- 不经允许列表信任 `iss` 声明。任何接受任意 `iss` 令牌的验证器，都允许攻击者搭建自己的 IdP 并伪造令牌。
- 以明文存储 `registration_access_token`。静态哈希；每次更新时需要明文。

输出：一页接线方案，包含受保护资源文档、三条 `registerTrigger` 调用、四条 `registerFunction` 调用、状态键方案、scope 映射表和编码后的运行时拒绝规则。结尾指出针对所选 IdP 最可能出现的单一部署阻塞缺口——通常是企业 SSO 的 DCR 可用性问题。
