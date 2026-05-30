---
name: mcp-client-harness
description: 给定 MCP 服务器的声明式列表（名称、命令、参数），搭建包含握手、命名空间合并和路由的多服务器客户端。
version: 1.0.0
phase: 13
lesson: 08
tags: [mcp, client, multi-server, routing, namespace]
---

给定待运行的 MCP 服务器配置，生成一个客户端测试框架，负责启动每台服务器、与其握手、将工具列表合并到统一命名空间，并将每次调用路由到所属服务器。

输出内容：

1. 服务器配置解析器。映射 `name -> {command, args, env}`。验证命令在路径上存在。
2. 启动方案。使用 subprocess.Popen，配置 stdin/stdout/stderr 管道、`bufsize=1`、文本模式。每台服务器一个后台读取线程。
3. 握手流水线。对每个会话：发送 `initialize`，等待响应，持久化能力，发送 `notifications/initialized`。
4. 命名空间合并。选择冲突策略：`prefix-on-collision`（默认）、`reject-on-collision` 或 `silent-overwrite`（禁用）。启动时打印合并后的工具列表。
5. 路由函数。`client.call(canonical_name, arguments)` 查找所属会话，向其写入 `tools/call` 消息。通过待处理请求表中的 future 等待匹配 id 的响应。

硬性拒绝：
- 任何未在独立进程中启动每台服务器的测试框架。进程内多路复用会破坏隔离模型。
- 任何将 `silent-overwrite` 作为默认冲突策略的测试框架。存在安全风险。
- 任何在主线程上阻塞等待 stdout 读取的测试框架。通知将会卡死。

拒绝规则：
- 如果服务器命令不可信（不在固定的允许列表中），拒绝启动并路由到 Phase 13 · 15 进行安全检查。
- 如果用户配置了超过 10 台服务器且没有说明原因，警告并建议使用网关（Phase 13 · 17）。
- 如果被要求在此处处理 OAuth，拒绝并路由到 Phase 13 · 16。

输出：一个完整的客户端测试框架 Python 文件（约 150 行），包含 Session、合并逻辑、路由和主循环，对每台配置的服务器进行演练。以一行摘要说明冲突策略和合并工具总数作为结尾。
