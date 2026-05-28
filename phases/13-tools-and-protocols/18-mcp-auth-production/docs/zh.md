# 生产环境 MCP 认证——DCR、JWKS 轮换、iii 原语上的受众锁定令牌

> 第 16 课在内存中搭建了 OAuth 2.1 状态机。到 2026 年，你发布给真实组织的每个 MCP 服务器都需要生产级认证：动态客户端注册（RFC 7591）、授权服务器元数据发现（RFC 8414）、不会破坏凌晨三点令牌验证的 JWKS 轮换，以及拒绝混淆代理复用的受众锁定令牌。本课将这一切通过 iii 原语串联起来——`iii.registerTrigger` 用于 HTTP 和 cron，`iii.registerFunction` 用于认证逻辑，`state::set/get` 用于缓存密钥——使认证面板像引擎中的所有其他工作负载一样可观测、可重启、可重放。

**类型：** 构建
**编程语言：** Python（标准库，iii 原语在课程环境中模拟）
**前置知识：** Phase 13 · 16（OAuth 2.1 状态机）、Phase 13 · 17（网关）
**预计时间：** 约 90 分钟

## 学习目标

- 通过 RFC 8414 元数据发现授权服务器并验证契约。
- 实现 RFC 7591 动态客户端注册，使 MCP 客户端无需管理员干预即可注册。
- 使用 cron 触发器缓存和轮换 JWKS 密钥，使签名验证在密钥轮换后仍能正常工作。
- 使用 RFC 8707 资源指示器将令牌锁定到单一 MCP 资源，拒绝混淆代理复用。
- 将每个端点和后台任务都设计为 iii 原语（HTTP 触发器、cron 触发器、命名函数和 `state::*` 读取），使单次重启即可重建认证面板。
- 读取 IdP 能力矩阵，当 IdP 无法满足 MCP 认证配置文件时拒绝部署。

## 问题背景

第 16 课的模拟器在内存中运行 OAuth 2.1。生产环境存在三个仅内存模拟器看不到的运营缺口。

**第一个缺口是注册。** 真实组织运行数百个 MCP 服务器和数千个 MCP 客户端。运营者不会手动将每个 Cursor 用户注册为 OAuth 客户端。RFC 7591 动态客户端注册让客户端 `POST /register` 到授权服务器，立即获得 `client_id`（可选 `client_secret`）。服务器在其 RFC 8414 元数据中发布 `registration_endpoint`；客户端无需带外配置即可发现它。

**第二个缺口是密钥轮换。** JWT 验证依赖于授权服务器的签名密钥，以 JSON Web Key Set（JWKS）形式发布。授权服务器按计划轮换这些密钥（通常每小时，事故响应时可能更快）。在启动时只获取一次 JWKS 的 MCP 服务器在轮换窗口前验证正常——然后在重启前所有请求都会失败。生产环境将 JWKS 作为缓存值处理，并配置刷新任务在前一批密钥过期前覆盖缓存，同时在缓存未命中时提供同步回退（处理由比缓存更新的密钥签名的令牌到达的情况）。

**第三个缺口是受众绑定。** 第 16 课介绍了 RFC 8707 资源指示器。在生产环境中，该指示器变为每个请求上的硬性声明检查。MCP 服务器将 `token.aud` 与自身的规范资源 URL 进行比较，不匹配时返回 HTTP 401。这是防止上游 MCP 服务器（或持有本应用于某服务器令牌的恶意客户端）将令牌重放给同一信任网格中另一服务器的唯一防御措施。

本课将上述每个缺口都作为 iii 原语处理。

## 核心概念

### RFC 8414——OAuth 授权服务器元数据

`/.well-known/oauth-authorization-server` 处的文档描述了客户端所需的一切：

```json
{
  "issuer": "https://auth.example.com",
  "authorization_endpoint": "https://auth.example.com/authorize",
  "token_endpoint": "https://auth.example.com/token",
  "jwks_uri": "https://auth.example.com/.well-known/jwks.json",
  "registration_endpoint": "https://auth.example.com/register",
  "response_types_supported": ["code"],
  "grant_types_supported": ["authorization_code", "refresh_token"],
  "code_challenge_methods_supported": ["S256"],
  "scopes_supported": ["mcp:tools.read", "mcp:tools.invoke"],
  "token_endpoint_auth_methods_supported": ["none", "private_key_jwt"]
}
```

客户端获得 MCP 资源 URL 后，链式发现：RFC 9728 的 `oauth-protected-resource`（资源服务器文档）命名 issuer，然后 `oauth-authorization-server`（此 RFC）命名每个端点。客户端不需要硬编码授权 URL。

在信任 MCP IdP 之前验证的契约：

- `code_challenge_methods_supported` 包含 `S256`（RFC 7636 PKCE）。
- `grant_types_supported` 包含 `authorization_code` 且排除 `password` 和 `implicit`。
- `registration_endpoint` 存在（RFC 7591 支持）。
- `response_types_supported` 对于 OAuth 2.1 精确为 `["code"]`。

如果任何一项缺失，MCP 服务器拒绝部署到此 IdP。是部署清单有问题，不是代码。

### RFC 9728（回顾）——受保护资源元数据

第 16 课已讲解 RFC 9728。生产环境的增量点：此文档是客户端查找被*这个* MCP 服务器信任的授权服务器的唯一地方。单个 MCP 服务器可以接受来自多个 IdP 的令牌（一个用于员工，一个用于合作伙伴）。RFC 9728 声明该集合；RFC 8414 记录每个 IdP 的支持内容。

### RFC 7591——动态客户端注册

没有 DCR，每个 MCP 客户端（Cursor、Claude Desktop、自定义智能体）都需要与 IdP 管理员进行带外交换。有了 DCR，客户端 POST：

```json
POST /register
Content-Type: application/json

{
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none",
  "scope": "mcp:tools.invoke",
  "client_name": "Cursor",
  "software_id": "com.cursor.cursor",
  "software_version": "0.42.0"
}
```

服务器响应 `client_id` 和用于后续更新的 `registration_access_token`。

`token_endpoint_auth_method: none` 是在用户设备上运行的 MCP 客户端的正确默认值。它们只获得 `client_id`——没有 `client_secret` 可被窃取。PKCE 提供了公共客户端所需的持有证明。

三个生产陷阱：

- 注册端点必须按源 IP 进行速率限制。否则，恶意行为者可以脚本化数百万个假注册，耗尽 `client_id` 命名空间。
- 某些企业 IdP 要求 `software_statement`（为客户端背书的已签名 JWT）。课程模拟跳过了这一步；生产环境添加验证步骤，拒绝来自 localhost 重定向 URI 以外的未签名注册。
- `registration_access_token` 必须存储为哈希，而非明文。此令牌被盗意味着攻击者可以重写客户端的重定向 URI。

### RFC 8707（回顾）——资源指示器

第 16 课建立了形态。生产规则：每个令牌请求都包含 `resource=<规范-mcp-url>`，MCP 服务器在每次调用时验证 `token.aud` 与自身资源 URL 匹配。如果 MCP 服务器可通过 `https://notes.example.com/mcp` 访问，规范 URL 为 `https://notes.example.com`——路径组件被排除，以便单个服务器在一个受众下托管多个路径。

### iii 原语接线（本课的核心内容）

五个原语组成认证面板：

```python
# 1. RFC 8414 元数据文档
iii.registerTrigger(
    "http",
    {"path": "/.well-known/oauth-authorization-server", "method": "GET"},
    "auth::serve-asm",
)

# 2. RFC 7591 动态客户端注册
iii.registerTrigger(
    "http",
    {"path": "/register", "method": "POST"},
    "auth::register-client",
)

# 3. JWT 验证作为可调用函数（资源服务器触发它）
iii.registerFunction("auth::validate-jwt", validate_jwt_handler)

# 4. 增量范围的步进授权颁发（第 16 课的 SEP-835）
iii.registerFunction("auth::issue-step-up", issue_step_up_handler)

# 5. Cron 驱动的 JWKS 轮换
iii.registerTrigger(
    "cron",
    {"schedule": "0 */6 * * *"},
    "auth::rotate-jwks",
)
iii.registerFunction("auth::rotate-jwks", rotate_jwks_handler)
```

MCP 服务器本身从不直接调用验证，而是：

```python
result = iii.trigger("auth::validate-jwt", {"token": bearer_token, "resource": self.resource})
if not result["valid"]:
    return {"status": 401, "WWW-Authenticate": result["www_authenticate"]}
```

这种间接性是 iii 的价值所在。明天你可以将验证器替换为并行查询两个 IdP 的扇出，或添加跨度发射器，或缓存正验证结果。MCP 服务器不需要改变。

### JWKS 轮换模式

两个密钥同时存在是稳定状态。授权服务器通过在撤销旧密钥（`k_2026_03`）之前引入下一个密钥（`k_2026_04`）来轮换，使在旧密钥下颁发的令牌在过期前仍然有效。缓存保存两者的并集；验证器按 `kid` 选择。

每六小时，cron 触发器调用 `auth::rotate-jwks`，获取 `<issuer>/.well-known/jwks.json` 并写入 `state::set("auth/jwks/<issuer>", {keys, fetched_at})`。验证器从 `state::get` 读取。如果令牌的 `kid` 在缓存中缺失，触发同步的 `auth::rotate-jwks` 调用作为回退。

### IdP 能力矩阵

| IdP 类别 | RFC 8414 元数据 | RFC 7591 DCR | RFC 8707 资源 | RFC 7636 S256 PKCE | 备注 |
|---------|---------------|-------------|-------------|-------------------|------|
| 自托管（Keycloak） | 是 | 是 | 是（24.x 起） | 是 | 本课 MCP 配置文件的参考 IdP |
| 企业 SSO（Microsoft Entra ID） | 是 | 是（高级层） | 是 | 是 | DCR 可用性因租户层不同而异 |
| 企业 SSO（Okta） | 是 | 是（Okta CIC/Auth0） | 是 | 是 | DCR 在 Auth0 可用；经典 Okta 组织需要管理员预注册 |
| 社交登录 IdP（通用） | 各异 | 很少 | 很少 | 是 | 大多数社交 IdP 将客户端视为静态合作伙伴；不依赖 DCR |
| 自定义/内部开发 | 取决于实现 | 取决于实现 | 取决于实现 | 取决于实现 | 如果自己开发，实现完整配置文件 |

拒绝规则：如果选择的 IdP 没有返回 `registration_endpoint` 且没有在 `code_challenge_methods_supported` 中列出 `S256`，MCP 服务器拒绝启动。没有降级模式。

### 混淆代理场景与受众绑定

服务器 A（`notes.example.com`）和服务器 B（`tasks.example.com`）都针对同一授权服务器注册。服务器 A 被攻陷。攻击者获取用户的笔记令牌，并将其重放给服务器 B。

服务器 B 的验证器：

1. 解码 JWT，通过 `kid` 获取 JWKS，验证签名。
2. 对照其受保护资源元数据的 `authorization_servers` 检查 `iss`。（通过——同一 IdP。）
3. 检查 `aud == "https://tasks.example.com"`。（**失败**——令牌的 `aud` 是 `https://notes.example.com`。）
4. 返回 401，带 `WWW-Authenticate: Bearer error="invalid_token", error_description="audience mismatch"`。

受众声明是协议层面对此攻击的唯一防御。为了性能而跳过它是最常见的生产错误；验证器必须在每个请求上运行，而不仅仅在会话开始时。

### 失败模式

- **JWKS 过期。** 密钥轮换后验证器拒绝有效令牌。修复方案是上述的 cron+回退模式。永远不要在没有刷新任务的情况下缓存 JWKS。
- **缺少 `aud` 声明。** 某些 IdP 默认省略 `aud`，除非令牌请求中存在 `resource`。验证器必须拒绝缺少 `aud` 的令牌，而不是将缺失视为通配符。
- **范围升级竞争。** 对同一用户同时进行的两个步进授权流程都可能成功，产生两个具有不同范围的访问令牌。验证器必须使用请求中提供的令牌，而不是查找"用户当前的范围"——那会产生 TOCTOU 窗口。
- **注册令牌被盗。** 泄露的 `registration_access_token` 让攻击者可以重写重定向 URI。静态存储时哈希化；要求客户端在每次更新时提供明文；在发现可疑时轮换。
- **未锁定 `iss`。** 接受任何 `iss` 的验证器让攻击者可以建立自己的授权服务器，为目标受众注册客户端，并颁发令牌。受保护资源元数据的 `authorization_servers` 列表是白名单；强制执行它。

## 动手实践

`code/main.py` 使用标准库 Python 和模拟 `iii.registerFunction`、`iii.registerTrigger`、`iii.trigger` 和 `state::set/get` 的小型 `iii_mock` 注册表演示完整的生产流程，共九步。最终，模拟 JWKS 轮换使密钥在不重启的情况下更新，混淆代理尝试因受众不匹配而失败并返回 401。

## 产出技能

本课产出 `outputs/skill-mcp-auth-iii.md`：给定 MCP 服务器配置和 IdP 能力集，该技能输出要注册的 iii 原语、JWKS 轮换计划、范围映射，以及当 IdP 不支持完整 RFC 配置文件时应用的拒绝规则。

## 练习

1. 运行 `code/main.py`。追踪九步流程。注意 `state::get` 在 `auth::rotate-jwks` 覆盖之前返回过期数据的位置，以及下一个请求如何针对新密钥进行验证。

2. 向受保护资源元数据的 `authorization_servers` 列表添加新 IdP。颁发一个由新 IdP 签名的令牌并确认验证器接受它。颁发一个由未列出的 IdP 签名的令牌并确认验证器拒绝，带有 `error_description="iss not allowed"`。

3. 将 `auth::rate-limit` 实现为 iii 函数，并在注册 HTTP 触发器内、注册器运行之前调用它。使用存储在 `state::set("auth/ratelimit/<ip>", ...)` 中的每源 IP 令牌桶。

4. 阅读 RFC 7591，找出课程 `/register` 处理器未验证的两个字段。添加验证。（提示：`software_statement` 和 `redirect_uris` URI scheme。）

5. 阅读 MCP 规范 2025-11-25 授权章节。找出课程验证器目前未发出的 `WWW-Authenticate` 头的一个规范性要求。添加它。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| ASM | "OAuth 元数据文档" | RFC 8414 `/.well-known/oauth-authorization-server` JSON |
| DCR | "自助客户端注册" | RFC 7591 `POST /register` 流程 |
| JWKS | "JWT 验证的公钥" | JSON Web Key Set，从 `jwks_uri` 获取，按 `kid` 索引 |
| 资源指示器 | "受众参数" | RFC 8707 `resource` 参数，将令牌锁定到单一服务器 |
| `aud` 声明 | "受众" | 验证器与规范资源 URL 比较的 JWT 声明 |
| 混淆代理 | "令牌重放" | 为服务器 A 颁发的令牌被提交给服务器 B 的攻击 |
| `iss` 白名单 | "受信任的授权服务器" | 受保护资源元数据 `authorization_servers` 中命名的集合 |
| 密钥轮换 | "滚动 JWKS" | 带重叠窗口的签名密钥定期替换 |
| 公共客户端 | "原生或浏览器客户端" | 没有 `client_secret` 的 OAuth 客户端；PKCE 补偿 |
| `WWW-Authenticate` | "401/403 响应头" | 携带 `Bearer error=...` 指令驱动客户端恢复 |

## 延伸阅读

- [MCP——授权规范（2025-11-25）](https://modelcontextprotocol.io/specification/draft/basic/authorization) — 本课实现的 MCP 认证配置文件
- [RFC 8414——OAuth 2.0 授权服务器元数据](https://datatracker.ietf.org/doc/html/rfc8414) — 发现契约
- [RFC 7591——OAuth 2.0 动态客户端注册协议](https://datatracker.ietf.org/doc/html/rfc7591) — DCR
- [RFC 7636——代码交换证明密钥（PKCE）](https://datatracker.ietf.org/doc/html/rfc7636) — 公共客户端持有证明
- [RFC 8707——OAuth 2.0 的资源指示器](https://datatracker.ietf.org/doc/html/rfc8707) — 受众锁定
- [RFC 9728——OAuth 2.0 受保护资源元数据](https://datatracker.ietf.org/doc/html/rfc9728) — 资源服务器发现
- [OAuth 2.1 草案](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-v2-1) — 合并的 OAuth 基础
