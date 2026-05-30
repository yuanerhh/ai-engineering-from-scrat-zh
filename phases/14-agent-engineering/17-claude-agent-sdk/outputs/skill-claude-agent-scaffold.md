---
name: claude-agent-scaffold
description: 构建 Claude Agent SDK 应用的脚手架，包含子智能体、生命周期钩子、会话存储、MCP 服务器挂载和 W3C 追踪传播。
version: 1.0.0
phase: 14
lesson: 17
tags: [claude-agent-sdk, subagents, hooks, session-store, mcp]
---

给定产品领域和一组 MCP 服务器，构建 Claude Agent SDK 应用的脚手架。

输出内容：

1. 主智能体定义，包含指令、内置工具访问权限（read_file、write_file、shell、grep、glob、web fetch）和自定义函数工具。
2. 子智能体生成器，用于并行化和上下文隔离。当编排器自身的上下文预算不足时使用。
3. 已注册的生命周期钩子：PreToolUse + PostToolUse 用于审计，SessionStart 用于初始化，SessionEnd 用于清理，UserPromptSubmit 用于规则执行（参见 pro-workflow 模式）。
4. 会话存储（默认 SQLite），`list_subkeys` 接入后可渲染子智能体树。
5. 挂载 MCP 服务器，用于外部工具/资源接入。
6. W3C 追踪上下文传播，使调用方的 OTel span 可以贯通 CLI。

硬性拒绝：

- 为单工具任务生成子智能体。子智能体用于并行化或上下文隔离，不适用于"一次 read_file 调用"。
- 钩子中执行同步耗时操作。钩子应在微秒到毫秒级完成。耗时工作应放入子智能体。
- 没有级联删除策略的会话存储。孤立的子智能体会话会使存储臃肿。

拒绝规则：

- 如果产品需要长时间运行的异步工作（数小时到数天），拒绝使用自托管 SDK，路由到 Claude Managed Agents。
- 如果用户要求将 `--session-mirror` 指向共享位置，拒绝。会话记录包含 PII；镜像应写入每用户加密存储。
- 如果智能体依赖原始 LLM 流式输出来提供 UX 体验而不使用工具，拒绝使用 Agent SDK，推荐直接使用 Client SDK。

输出：`agent.py`、`tools.py`、`hooks.py`、`session.py`、`README.md`（说明子智能体策略、钩子注册表、会话后端、MCP 挂载和 OTel 接入）。末尾附"下一步阅读"，若需要语音交接指向第 22 课，若需要 OTel span 归因指向第 23 课，若产品需要生产运行时形态指向第 18 课。
