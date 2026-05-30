---
name: issue-to-pr
description: 构建一个异步 GitHub issue 转 PR 智能体，在云沙箱中运行，复现构建环境，验证测试，并在严格的每代码库预算内开启可供评审的 PR。
version: 1.0.0
phase: 19
lesson: 16
tags: [capstone, async-agent, github, fargate, daytona, swe-bench, budget, safety]
---

给定一个带有 `@agent fix this` 标签 issue 的 GitHub 代码库，交付一个自托管云智能体，在有作用域凭证和有限成本的情况下将每个已标记 issue 转化为可供评审的 PR。

构建计划：

1. GitHub App，配置细粒度 Token：issue 读写、PR 写入、内容读写、工作流读取。禁止强制推送。main 分支保护防止直接写入。
2. Webhook 接收器（Lambda 或 Fly.io）过滤标签 / PR 评论事件并入队到 SQS。
3. 调度器执行每代码库每日美元和 PR 数量上限；为每个允许的作业启动一个 ECS Fargate 任务。
4. 环境推断：从代码库内容检测语言 + 包管理器 + 运行时。如果缺少 Dockerfile 则动态合成一个。
5. 每个任务使用 Daytona 或 E2B 沙箱。将代码库克隆到全新的 `git worktree` + 智能体分支中。
6. 智能体循环（基于 Claude Opus 4.7 或 GPT-5.4-Codex 的 mini-swe-agent 或 SWE-agent v2）。工具：ripgrep、tree-sitter repo-map、read_file、edit_file、run_tests、git。上限：20 美元、30 轮、30 分钟。
7. 验证：沙箱内完整 CI；通过 jacoco / coverage.py 计算覆盖率变化；如果变化 < -2% 则标记 `needs-review`；CI 红色时停止。
8. 通过 GitHub API 开启 PR，附带理由、差异摘要、追踪 URL、成本、轮次数。
9. 可观测性：每个 PR 一个 Langfuse 追踪；日志中密钥脱敏；每代码库预算仪表板。
10. 在 30 个内部种子问题上进行评估；在三个共享子集上与 Cursor Background Agents 和 AWS Remote SWE Agents 进行对比。

评估标准：

| 权重 | 评估项 | 度量方式 |
|:-:|---|---|
| 25 | 30 个问题通过率 | 端到端成功（CI 绿色 + 覆盖率 OK） |
| 20 | PR 质量 | 差异大小、覆盖率变化、代码风格符合性 |
| 20 | 每个解决问题的成本与延迟 | 每 PR 美元成本和墙钟时间 |
| 20 | 安全性 | 作用域 Token、每代码库预算、禁止强制推送、凭证卫生 |
| 15 | 运维用户体验 | 理由注释、重试能力、@-mention 跟进 |

硬性拒绝条件：

- 任何可以强制推送的智能体。硬性排除。
- 跳过预算检查的调度器。失控循环是经典失败模式。
- 未在沙箱中完整通过 CI 就开启的 PR。
- 包含未脱敏 Token 或 PII 的追踪存档。

拒绝规则：

- 拒绝在 main 没有分支保护的情况下安装。
- 拒绝在没有每代码库每日预算（美元和 PR 数量）的情况下运行。
- 拒绝自动重试失败的运行；所有重试都需要人工重新应用标签。

输出：一个包含 GitHub App、Webhook 接收器、调度器 + 预算账本、Fargate 任务定义、沙箱生命周期管理器、mini-swe-agent 循环、30 个问题评估运行、与 Cursor Background Agents 和 AWS Remote SWE Agents 并排对比的代码库，以及一份说明三大构建推断失败和减少每种失败的 Dockerfile 合成变更的报告。
