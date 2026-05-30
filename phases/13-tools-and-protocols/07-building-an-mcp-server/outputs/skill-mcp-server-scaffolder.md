---
name: mcp-server-scaffolder
description: 为特定领域搭建 MCP 服务器框架，合理拆分 tools/resources/prompts，并规划 SDK 升级路径。
version: 1.0.0
phase: 13
lesson: 07
tags: [mcp, server, fastmcp, scaffold]
---

给定一个领域（笔记、工单、文件、数据库等），生成一份 MCP 服务器规划：哪些能力作为工具暴露、哪些作为资源、哪些作为提示词，以及升级到 Python 或 TypeScript SDK 的路径。

输出内容：

1. 工具列表。用户明确要求执行的原子操作。包含名称、描述（使用"Use-when"模式）、输入 Schema 和注释提示。
2. 资源列表。用户希望读取的数据。URI 方案、mime 类型，以及是否启用 `resources/subscribe`。
3. 提示词列表。宿主应作为斜杠命令暴露的可复用模板。参数列表。
4. 能力声明。服务器在 `initialize` 中返回的精确 `capabilities` 对象。
5. 升级说明。每个部分对应的 FastMCP（Python）或 TypeScript SDK 等价物。命名一个 SDK 特性（例如 `lifespan`、`context`），用于替换框架中手写的标准库模式。

硬性拒绝：
- 任何仅作为工具而非资源暴露"数据库查询"的情况。正确拆分是：`/list` 和 `/read` 用资源，带参数的 `/query` 用工具。
- 任何在同一命名空间中混用用户输入工具与特权工具且没有注释的服务器。
- 任何声明 `resources/subscribe` 能力但没有持久通知机制的服务器框架。

拒绝规则：
- 如果领域没有只读的数据面，拒绝搭建资源；建议使用纯工具服务器。
- 如果领域没有自然的斜杠命令模板，拒绝搭建提示词。
- 如果用户要求认证方案，拒绝并路由到 Phase 13 · 16（OAuth 2.1）。

输出：一页服务器规划，包含三个原语列表、能力对象，以及一个 10 行的 `@app.tool()` 装饰器风格升级代码片段。以服务器应设置的最重要注释标志作为结尾。
