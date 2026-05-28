# 综合项目——构建完整的工具生态系统

> Phase 13 讲解了每一个组件。本综合项目将它们整合为一个生产形态的系统：带有工具 + 资源 + 提示词 + 任务 + UI 的 MCP 服务器、边缘处的 OAuth 2.1、RBAC 网关、多服务器客户端、A2A 子智能体调用、OTel 追踪到收集器、CI 中的工具投毒检测，以及 AGENTS.md + SKILL.md 包。最终你能够为每一个架构决策进行辩护。

**类型：** 构建
**编程语言：** Python（标准库，端到端生态系统测试框架）
**前置知识：** Phase 13 · 01 至 21
**预计时间：** 约 120 分钟

## 学习目标

- 构建一个暴露工具、资源、提示词和带 `ui://` 应用任务的 MCP 服务器。
- 用强制执行 RBAC 和哈希固定的 OAuth 2.1 网关在服务器前端部署。
- 编写一个端到端使用 OTel GenAI 属性进行追踪的多服务器客户端。
- 通过 A2A 将部分工作负载委托给子智能体；验证不透明性得到保持。
- 使用 AGENTS.md + SKILL.md 打包整个技术栈，使其他智能体可以驱动它。

## 问题背景

交付"研究与报告"系统：

- 用户问："总结 2026 年 arXiv 上关于智能体协议被引用最多的三篇论文。"
- 系统：通过 MCP 搜索 arXiv；通过 A2A 将论文摘要任务委托给专门的写作智能体；汇聚结果；将交互式报告渲染为 MCP Apps 的 `ui://` 资源；将每一步记录到 OTel。

Phase 13 的所有原语都会出现。这不是玩具——2026 年 Anthropic（Claude Research 产品）、OpenAI（带 Apps SDK 的 GPTs）以及第三方生产的研究助手系统就是这个形态。

## 核心概念

### 架构

```
[用户] -> [客户端] -> [网关（OAuth 2.1 + RBAC）] -> [研究型 MCP 服务器]
                                                     |
                                                     +- MCP 工具：arxiv_search（纯函数）
                                                     +- MCP 资源：notes://recent
                                                     +- MCP 提示词：/research_topic
                                                     +- MCP 任务：generate_report（长时任务）
                                                     +- MCP Apps UI：ui://report/current
                                                     +- A2A 调用：writer-agent（tasks/send）
                                                     |
                                                     +- OTel GenAI span
```

### 追踪层级

```
agent.invoke_agent
 ├── llm.chat（启动）
 ├── mcp.call -> tools/call arxiv_search
 ├── mcp.call -> resources/read notes://recent
 ├── mcp.call -> prompts/get research_topic
 ├── a2a.tasks/send -> writer-agent
 │    └── 任务状态转换（不透明内部状态）
 ├── mcp.call -> tools/call generate_report（带任务增强）
 │    └── tasks/status 轮询
 │    └── tasks/result（已完成，返回 ui:// 资源）
 └── llm.chat（最终综合）
```

单一 trace id。每个 span 带有正确的 `gen_ai.*` 属性。

### 安全态势

- OAuth 2.1 + PKCE，资源指示符将受众固定到网关。
- 网关持有上游凭证；用户永远看不到它们。
- RBAC：`alice` 具有 `research:read`、`research:write`，可以调用所有工具。`bob` 具有 `research:read`，无法调用 `generate_report`。
- 固定描述清单：丢弃任何工具哈希发生变化的服务器。
- Rule of Two（双规则）审计：没有工具同时组合不受信任的输入、敏感数据和重大操作。

### 渲染

最终的 `generate_report` 任务返回内容块以及 `ui://report/current` 资源。客户端宿主（Claude Desktop 等）在沙箱 iframe 中渲染交互式仪表盘。仪表盘包含按排名排列的论文列表、引用次数，以及一个按钮——用户点击任意论文时会调用 `host.callTool('summarize_paper', {arxiv_id})`。

### 打包

整个系统以如下结构交付：

```
research-system/
  AGENTS.md                     # 项目约定
  skills/
    run-research/
      SKILL.md                  # 顶层工作流
  servers/
    research-mcp/               # MCP 服务器
      pyproject.toml
      src/
  agents/
    writer/                     # A2A 智能体
  gateway/
    config.yaml                 # RBAC + 固定清单
```

用户使用 `docker compose up` 部署。Claude Code、Cursor、Codex 和 opencode 用户可以通过调用 `run-research` 技能来驱动系统。

### 各 Phase 13 课程的贡献

| 课程 | 综合项目使用的内容 |
|------|-----------------|
| 01-05 | 工具接口、提供商可移植性、并行调用、schema、代码检查 |
| 06-10 | MCP 原语、服务器、客户端、传输、资源 + 提示词 |
| 11-14 | Sampling、根目录 + Elicitation、异步任务、`ui://` 应用 |
| 15-17 | 工具投毒、OAuth 2.1、网关 + 注册表 |
| 18 | A2A 子智能体委托 |
| 19 | OTel GenAI 追踪 |
| 20 | LLM 层的路由网关 |
| 21 | SKILL.md + AGENTS.md 打包 |

## 动手实践

`code/main.py` 将前述课程的模式缝合为一个可运行的演示。全部使用标准库，全部在进程内，你可以从头到尾阅读。它为研究与报告场景运行完整流程：与网关握手、模拟 OAuth 2.1、合并 tools/list、将 generate_report 作为任务运行、A2A 调用写作智能体、返回 ui:// 资源、发送 OTel span。

重点关注：

- 跨每个跳转的单一 trace id。
- 网关策略阻止第二个用户进行写操作。
- 任务生命周期经历 working → completed，同时返回文本和 ui:// 内容。
- A2A 调用的内部状态对编排器是不透明的。
- AGENTS.md 和 SKILL.md 是另一个智能体复现工作流所需的唯一文件。

## 产出技能

本课产出 `outputs/skill-ecosystem-blueprint.md`：给定一个产品需求（研究、摘要、自动化），该技能生成完整架构：哪些 MCP 原语、哪些网关控制、哪些 A2A 调用、哪些遥测、哪些打包方式。

## 练习

1. 运行 `code/main.py`。注意单一 trace id 以及 span 的嵌套方式。统计演示触及了 Phase 13 的多少个原语。

2. 扩展演示：添加第二个后端 MCP 服务器（例如 `bibliography`），确认网关将其工具合并到同一命名空间中。

3. 将假的 A2A 写作智能体替换为在子进程中运行的真实智能体。使用第 19 课的测试框架。

4. 在路由网关中的编排器和 LLM 之间添加 PII 脱敏步骤。确认用户查询中的电子邮件地址被脱敏处理。

5. 为将要维护此系统的团队成员编写 AGENTS.md。应能在五分钟内读完，并提供在 Cursor 或 Codex 中驱动综合项目所需的全部信息。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 综合项目（Capstone） | "Phase 13 集成演示" | 使用所有原语的端到端系统 |
| 研究与报告（Research and report） | "场景" | 搜索、摘要、渲染的组合模式 |
| 生态系统（Ecosystem） | "所有组件的组合" | 服务器 + 客户端 + 网关 + 子智能体 + 遥测 + 打包 |
| 追踪层级（Trace hierarchy） | "单一 trace id" | 每个跳转的 span 共享追踪；通过 span id 建立父子关系 |
| 网关颁发的令牌 | "传递式授权" | 客户端仅见网关令牌；网关持有上游凭证 |
| 合并命名空间 | "所有工具在一个扁平列表中" | 在网关处进行多服务器合并，冲突时添加前缀 |
| 不透明性边界 | "A2A 调用隐藏内部状态" | 子智能体的推理过程对编排器不可见 |
| 三层技术栈 | "AGENTS.md + SKILL.md + MCP" | 项目上下文 + 工作流 + 工具 |
| 纵深防御 | "多层安全" | 哈希固定 + OAuth + RBAC + 双规则 + 审计日志 |
| 规范合规矩阵 | "我们交付的规范要求内容" | 将交付物映射到 2025-11-25 版本要求的清单 |

## 延伸阅读

- [MCP——规范 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — 综合参考
- [MCP 博客——2026 年路线图](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — 协议的发展方向
- [a2a-protocol.org](https://a2a-protocol.org/latest/) — A2A v1.0 参考
- [OpenTelemetry——GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 权威追踪约定
- [Anthropic——Claude Agent SDK 概述](https://code.claude.com/docs/en/agent-sdk/overview) — 生产智能体运行时模式
