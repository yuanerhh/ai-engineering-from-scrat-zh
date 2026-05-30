---
name: terminal-coding-agent
description: 构建一个终端原生编程智能体，在有限成本、沙箱工具和完整 2026 钩子接口下对 SWE-bench Pro 进行测评。
version: 1.0.0
phase: 19
lesson: 01
tags: [capstone, coding-agent, claude-code, swe-bench, mcp, hooks, sandbox]
---

给定一个目标代码库和一个自然语言任务，构建一个能够规划、在沙箱中执行并发起拉取请求的测试框架。在 30 个任务的 SWE-bench Pro 子集上达到或超越 mini-swe-agent 基线，同时将每任务预算控制在 5 美元以内。

构建计划：

1. 搭建一个包含计划面板、工具调用流和实时 Token/美元预算的 Bun + Ink TUI 测试框架。
2. 通过 Model Context Protocol StreamableHTTP 定义六个工具（read_file、edit_file、ripgrep、tree_sitter_symbols、run_shell、git）。每次调用最多返回 4k 个 Token。
3. 在 E2B 或 Daytona 沙箱中，在全新的 `git worktree add` 分支上运行每个工具调用。绝不触碰宿主文件系统。
4. 接入全部八个 2026 钩子事件：SessionStart、SessionEnd、PreToolUse、PostToolUse、UserPromptSubmit、Notification、Stop、PreCompact。至少实现四个用户自定义钩子（危险命令防护、Token 计费、OTel span 发射器、追踪包写入器）。
5. 执行三重预算限制：50 轮对话、20 万 Token、5 美元。在 15 万 Token 时触发 PreCompact 并对早期轮次进行摘要。
6. 使用 GenAI 语义约定将 OpenTelemetry span 发送到自托管的 Langfuse。
7. 成功后，推送分支并开启包含计划和追踪包的 PR。
8. 在 30 个 SWE-bench Pro Python 子集问题上与 mini-swe-agent 进行对比评估，记录每个任务的 pass@1、轮次数、Token 数和美元花费。

评估标准：

| 权重 | 评估项 | 度量方式 |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 | 与 mini-swe-agent 基线在同 30 任务子集上的对比 |
| 20 | 架构清晰度 | 计划/执行/观察分离、钩子接口、工具 schema 可读性 |
| 20 | 安全性 | 沙箱逃逸红队测试 + 危险命令防护审计 |
| 20 | 可观测性 | 100% 工具调用有 span，每轮 Token 计费 |
| 15 | 开发者体验 | 冷启动低于 2 秒、崩溃恢复、Ctrl-C 取消语义 |

硬性拒绝条件：

- 在宿主文件系统（而非沙箱内）调用 git 的测试框架。
- 任何可以在工作树之外写入或未经显式允许列表钩子就 curl 外部 URL 的智能体。
- 未在同 30 个问题上运行对照基线即报告的评估数据。
- 依赖重试之间执行 `git reset --hard` 来声称"通过率"；SWE-bench Pro 是 pass@1。

拒绝规则：

- 拒绝在任何配置下直接推送到 main。只允许 PR 分支。
- 拒绝禁用危险命令防护。这是评估标准的硬性要求。
- 拒绝在没有预算上限的情况下运行。无限制的运行会污染评估对比。

输出：一个包含测试框架的代码库、一个包含 mini-swe-agent 基线对照运行的固定 30 任务 SWE-bench Pro 评估框架、至少 5 次完整运行的 OpenTelemetry 追踪存档，以及一份写作报告，说明测试框架解决了哪些基线未能解决的任务（反之亦然）。最后附上一节，描述你观察到的三大失败模式以及解决每个问题的钩子变更。
