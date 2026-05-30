---
name: oauth-scope-planner
description: 为远程 MCP 服务器设计 OAuth 2.1 权限范围集、固定规则和步进升级策略。
version: 1.0.0
phase: 13
lesson: 16
tags: [oauth, pkce, resource-indicators, step-up, sep-835]
---

给定一个带有工具列表的远程 MCP 服务器，设计其授权模型。

输出内容：

1. **Scope 层级**。分级的 scope 集合（例如 `read` -> `write` -> `delete` -> `admin`）。每个操作类别对应一个 scope；不要过度拆分 scope 集合。
2. **Scope 到工具的映射**。每个工具标注其所需 scope。标记需要超过一个 scope 的工具。
3. **步进升级策略**。哪些操作需要步进升级而非初始同意。通常：破坏性操作需要步进升级。
4. **资源指示符值**。在 `resource` 参数中使用的规范 URL。确保该 URL 与 `.well-known/oauth-protected-resource` 的 resource 字段匹配。
5. **受保护资源元数据**。起草包含 `authorization_servers`、`scopes_supported` 和 `resource` 的 `.well-known/oauth-protected-resource` JSON。

硬性拒绝：
- 任何需要 admin scope 但在未显式确认对话框的情况下被调用的工具。需要步进升级。
- 任何覆盖超过一个操作类别的 scope。权限蔓延。
- 任何跳过受众验证的服务器。混淆代理漏洞。

拒绝规则：
- 若服务器为本地（stdio），拒绝使用 OAuth，并说明 stdio 继承父进程信任。
- 若服务器依赖旧版 OAuth 2.0 隐式流，拒绝并要求迁移至 2.1 + PKCE。
- 若用户要求仅使用"纯 API Key"鉴权，对远程服务器拒绝；对于用户授权访问，需使用带资源指示符的 OAuth 2.1 授权码 + PKCE。客户端凭据仅适用于无用户委托的机器间场景。

输出：一页授权方案，包含 scope 层级、scope 到工具的映射、步进升级策略、资源指示符和受保护资源元数据 JSON。结尾指出用户首次遇到时最可能感到意外的步进升级操作。
