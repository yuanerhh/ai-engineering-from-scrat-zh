# MCP 传输层——stdio vs Streamable HTTP vs SSE 迁移

> stdio 仅适用于本地，无法用于远程。Streamable HTTP（2025-03-26）是远程标准。旧的 HTTP+SSE 传输已被废弃，将于 2026 年中期移除。选错传输层意味着一次迁移代价；选对则可获得一个具备会话连续性和 DNS 重绑定防护的可远程托管的 MCP 服务器。

**类型：** 学习
**编程语言：** Python（标准库，Streamable HTTP 端点骨架）
**前置知识：** Phase 13 · 07、08（MCP 服务器与客户端）
**预计时间：** 约 45 分钟

## 学习目标

- 根据部署形态（本地 vs 远程、单进程 vs 集群）在 stdio 和 Streamable HTTP 之间做出选择。
- 实现 Streamable HTTP 的单端点模式：POST 用于请求，GET 用于会话流。
- 强制执行 `Origin` 验证和会话 ID 语义，防御 DNS 重绑定攻击。
- 在 2026 年中期截止期限前将遗留的 HTTP+SSE 服务器迁移到 Streamable HTTP。

## 问题背景

第一个 MCP 远程传输协议（2024-11）是 HTTP+SSE：两个端点，一个用于客户端的 POST，一个服务器发送事件（SSE）通道用于服务器到客户端的流。它能工作，但也很笨拙：每个会话两个端点，某些 CDN 前的缓存会出问题，以及对长连接 SSE 的强依赖（某些 WAF 会主动断开）。

2025-03-26 规范将其替换为 Streamable HTTP：一个端点，POST 用于客户端请求，GET 用于建立会话流，两者共享一个 `Mcp-Session-Id` 请求头。此后构建或迁移的每个服务器都使用 Streamable HTTP。旧的 SSE 模式正在被废弃——Atlassian Rovo 于 2026 年 6 月 30 日移除，Keboola 于 2026 年 4 月 1 日移除，大多数剩余的企业服务器将于 2026 年底移除。

stdio 仍然对本地服务器很重要。Claude Desktop、VS Code 以及每个 IDE 形态的客户端都通过 stdio 启动服务器。正确的思维模型：stdio 用于"本机"，Streamable HTTP 用于"网络传输"。两者不交叉使用。

## 核心概念

### stdio

- 子进程传输方式。客户端启动服务器，通过 stdin/stdout 通信。
- 每行一个 JSON 对象，换行符分隔。
- 无会话 ID；进程标识就是会话。
- 无需认证（子进程继承父进程的信任边界）。
- 永远不要用于远程服务器——如果必须远程，需要 SSH 或 socat 隧道，届时应直接使用 Streamable HTTP。

### Streamable HTTP

单端点 `/mcp`（或任意路径），支持三种 HTTP 方法：

- **POST /mcp。** 客户端发送 JSON-RPC 消息。服务器回复单个 JSON 响应，或一个包含一条或多条响应的 SSE 流（适用于批量响应和与该请求相关的通知）。
- **GET /mcp。** 客户端建立长连接 SSE 通道。服务器使用它发送服务器到客户端的请求（采样、通知、引导）。
- **DELETE /mcp。** 客户端显式终止会话。

会话通过服务器在第一个响应中设置的 `Mcp-Session-Id` 请求头来标识，客户端在后续每个请求中回传该头。会话 ID **必须**是加密随机的（128 位以上）；客户端选择的 ID 出于安全原因会被拒绝。

### 单端点 vs 双端点

旧规范的双端点模式在 2026 年仍可调用——规范将其声明为"遗留兼容"。但所有新服务器应使用单端点。官方 SDK 生成单端点；只有在与未迁移的远端通信时才使用遗留模式。

### `Origin` 验证与 DNS 重绑定防护

浏览器目前不是 MCP 客户端，但攻击者可以构造一个网页，让浏览器向 `localhost:1234/mcp`（用户本地 MCP 服务器的监听地址）发送 POST 请求。如果服务器不检查 `Origin`，浏览器的同源策略将无法保护它，因为 `Origin: http://evil.com` 是有效的跨源请求。

2025-11-25 规范要求服务器拒绝 `Origin` 不在白名单中的请求。白名单通常包含 MCP 客户端宿主（`https://claude.ai`、`vscode-webview://*`）以及本地 UI 的 localhost 变体。

### 会话 ID 生命周期

1. 客户端发送第一个请求，不携带 `Mcp-Session-Id`。
2. 服务器分配随机 ID，在响应头中设置 `Mcp-Session-Id`。
3. 客户端在后续所有请求和 `GET /mcp` 流请求中回传该头。
4. 服务器可以撤销会话；客户端在后续请求中会收到 404，必须重新初始化。
5. 客户端可以通过 DELETE 显式地干净关闭会话。

### 保活与重连

SSE 连接会断开。客户端通过携带相同 `Mcp-Session-Id` 重新发起 GET 请求来恢复。服务器**必须**在一个合理的窗口内排队并存储断线期间的事件，并通过客户端回传的 `last-event-id` 头进行重放。

Phase 13 · 13 讲解任务（Tasks），它允许长时间运行的工作甚至在整个会话重连后仍能存活。

### 向后兼容探测

希望同时支持新旧服务器的客户端：

1. POST 到 `/mcp`。
2. 如果响应是 `200 OK` 并带有 JSON 或 SSE 内容，这是 Streamable HTTP。
3. 如果响应是 `200 OK` 带 `Content-Type: text/event-stream` **且**有指向第二个端点的 `Location` 头，则是遗留 HTTP+SSE；跟随 `Location`。

### Cloudflare、ngrok 与托管

2026 年的生产远程 MCP 服务器运行在 Cloudflare Workers（使用其 MCP Agents SDK）、Vercel Functions 或容器化的 Node/Python 上。关键：托管平台必须支持 SSE GET 所需的长连接 HTTP。Vercel 免费层上限 10 秒，不适用。Cloudflare Workers 支持无限流。

### 网关组合

当你用网关（Phase 13 · 17）聚合多个 MCP 服务器时，网关就是单个 Streamable HTTP 端点，负责重写会话 ID 并多路复用上游服务器。工具在网关层合并；客户端看到的是单一逻辑服务器。

### 传输失败模式

- **stdio SIGPIPE。** 子进程在写入中途死亡会引发 SIGPIPE；服务器应干净退出。客户端应检测 EOF 并将会话标记为死亡。
- **HTTP 502 / 504。** Cloudflare、nginx 等代理在上游失败时发出这些响应。Streamable HTTP 客户端应在短暂退避后重试一次。
- **SSE 连接断开。** TCP RST、代理超时或客户端网络变化关闭了流。客户端携带 `Mcp-Session-Id` 和可选的 `last-event-id` 重连以恢复。
- **会话撤销。** 服务器使会话 ID 失效；客户端在下次请求时看到 404，必须重新握手。
- **时钟偏差。** 客户端计算的资源 TTL 与服务器不一致。客户端应将服务器时间戳视为权威。

### 何时绕过 Streamable HTTP

某些企业在其内部网络中通过 gRPC 或消息队列传输部署 MCP 服务器。这是非标准的——MCP 规范没有正式定义这些传输方式。网关可以向 MCP 客户端暴露 Streamable HTTP 接口，同时在内部使用 gRPC。保持外部接口符合规范，由网关负责转换。

## 动手实践

`code/main.py` 使用 `http.server`（标准库）实现了一个最小的 Streamable HTTP 端点。它在 `/mcp` 上处理 POST、GET 和 DELETE，在第一个响应中设置 `Mcp-Session-Id`，验证 `Origin`，并拒绝来自非白名单来源的请求。处理器复用了第 07 课笔记服务器的分发逻辑。

重点关注：
- POST 处理器读取 JSON-RPC 请求体，分发处理，并写入 JSON 响应（单响应变体；SSE 变体在结构上类似）。
- `Origin` 检查拒绝默认的 `http://evil.example` 探测，但接受 `http://localhost`。
- 会话 ID 是随机的 128 位十六进制字符串；服务器在内存中保存每会话的状态。

## 产出技能

本课产出 `outputs/skill-mcp-transport-migrator.md`：给定一个 HTTP+SSE（遗留）MCP 服务器，该技能生成迁移到 Streamable HTTP 的计划，包含会话 ID 连续性、Origin 检查和向后兼容探测支持。

## 练习

1. 运行 `code/main.py`。用 `curl` POST 一个 `initialize` 请求，观察 `Mcp-Session-Id` 响应头。再 POST 一个携带该头的请求，验证会话连续性。

2. 添加 GET 处理器，建立一个 SSE 流。每五秒发送一个 `notifications/progress` 事件。通过携带相同会话 ID 重新发起 GET 来重连，确认服务器接受该会话。

3. 实现 `last-event-id` 重放逻辑。在重连时，重放从该 ID 之后生成的所有事件。

4. 扩展 `Origin` 验证以支持通配符模式（`https://*.example.com`），确认它接受 `https://app.example.com` 但拒绝 `https://evil.example.com.attacker.net`。

5. 从官方注册表中找一个遗留 HTTP+SSE 服务器，勾勒其迁移方案：端点处理、会话 ID 生成和请求头语义各需要做哪些改动。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| stdio 传输 | "本地子进程" | 基于 stdin/stdout 的 JSON-RPC，换行分隔 |
| Streamable HTTP | "远程传输" | 单端点 POST + GET + 可选 SSE，2025-03-26 规范 |
| HTTP+SSE | "遗留" | 将于 2026 年中期移除的双端点模型 |
| `Mcp-Session-Id` | "会话头" | 服务器分配的随机 ID，后续每个请求都需回传 |
| `Origin` 白名单 | "DNS 重绑定防御" | 拒绝来源未获批准的请求 |
| 单端点 | "一个 URL" | `/mcp` 处理所有会话操作的 POST / GET / DELETE |
| `last-event-id` | "SSE 重放" | 用于在不丢失事件的情况下恢复中断流的请求头 |
| 向后兼容探测 | "新旧检测" | 客户端通过响应格式自动选择传输方式 |
| 长连接 HTTP | "SSE 流" | 服务器在一个 TCP 连接上推送事件，持续数分钟乃至数小时 |
| 会话撤销 | "强制重新初始化" | 服务器使会话 ID 失效；客户端必须重新握手 |

## 延伸阅读

- [MCP——基础传输规范 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports) — stdio 和 Streamable HTTP 的权威参考
- [MCP——基础传输规范 2025-03-26](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports) — 引入 Streamable HTTP 的版本
- [Cloudflare——MCP 传输](https://developers.cloudflare.com/agents/model-context-protocol/transport/) — Workers 托管的 Streamable HTTP 模式
- [AWS——MCP 传输机制](https://builder.aws.com/content/35A0IphCeLvYzly9Sw40G1dVNzc/mcp-transport-mechanisms-stdio-vs-streamable-http) — 跨部署形态的比较
- [Atlassian——HTTP+SSE 废弃通知](https://community.atlassian.com/forums/Atlassian-Remote-MCP-Server/HTTP-SSE-Deprecation-Notice/ba-p/3205484) — 具体迁移截止日期示例
