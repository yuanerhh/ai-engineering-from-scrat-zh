---
name: mcp-transport-migrator
description: 生成从旧版 HTTP+SSE 迁移到 Streamable HTTP 的迁移方案，保持会话 id 连续性并进行 Origin 验证。
version: 1.0.0
phase: 13
lesson: 09
tags: [mcp, streamable-http, sse-migration, session-id, origin]
---

给定一个现有的 HTTP+SSE（旧版）MCP 服务器，生成迁移到单端点 Streamable HTTP 的方案。

输出内容：

1. 端点改写。将 `/messages` 和 `/sse` 合并为一个 `/mcp`。将 POST 映射到请求处理，GET 映射到 SSE 流，DELETE 映射到会话终止。
2. 会话连续性。在首次 POST 时生成新的 `Mcp-Session-Id`。拒绝客户端提供的 id。如果客户端首先发送旧版会话 cookie，保留桥接逻辑。
3. Origin 验证。将明确的生产来源（`https://app.company.com`、`https://claude.ai`、localhost 变体）加入允许列表。其他来源一律返回 403。
4. Last-event-id 重放。为每个会话保留最近事件的环形缓冲区，以便重连时可以恢复。
5. 废弃窗口。记录切换日期和 60 天宽限期，在此期间旧版端点以 301 重定向到新端点，并附带警告头。

硬性拒绝：
- 任何无限期保留两个端点的方案。旧版 SSE 将在 2026 年被移除。
- 任何会话 id 由客户端生成的方案。违反加密随机性要求。
- 任何没有 Origin 验证的方案。DNS 重绑定漏洞。

拒绝规则：
- 如果服务器仅限本地使用（stdio），拒绝迁移到 HTTP；stdio 对本地场景是正确的选择。
- 如果服务器尚未部署 OAuth，在公开暴露之前先完成 Phase 13 · 16。
- 如果托管目标不支持长连接 HTTP（例如 Vercel 免费套餐），拒绝并推荐 Cloudflare Workers。

输出：一份迁移操作手册，包含端点变更、Origin 允许列表、会话 id 方案、废弃时间表，以及覆盖 initialize、tools/list、流式通知、携带 last-event-id 的重连和显式 DELETE 的测试清单。
