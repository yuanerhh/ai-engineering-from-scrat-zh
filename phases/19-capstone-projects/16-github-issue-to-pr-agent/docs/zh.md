# 顶点项目 16 — GitHub Issue 到 PR 自主智能体

> AWS Remote SWE Agents、Cursor Background Agents、OpenAI Codex Cloud 和 Google Jules 都交付了相同的 2026 年产品形态：标注一个 issue，得到一个 PR。在云沙箱中运行一个智能体，验证测试通过，并发布一个带理由说明的可审查 PR。难点在于：自动复现代码仓库的构建环境，防止凭证泄露，执行每仓库预算，以及确保智能体不能强制推送。这个顶点项目构建自托管版本，并在成本和通过率上与托管替代方案进行比较。

**类型：** 顶点项目  
**语言：** Python（智能体），TypeScript（GitHub App），YAML（Actions）  
**前置条件：** 第 11 阶段（LLM 工程），第 13 阶段（工具），第 14 阶段（智能体），第 15 阶段（自主系统），第 17 阶段（基础设施）  
**涵盖阶段：** P11 · P13 · P14 · P15 · P17  
**预计时间：** 30 小时

## 问题背景

异步云编程智能体是与交互式编程智能体（顶点项目 01）不同的产品类别。用户体验是一个 GitHub 标签。你给 issue 打上标签，一个 worker 在云沙箱中启动，克隆仓库，运行测试，编辑文件，验证，并开启一个在正文中包含智能体理由说明的 PR。没有交互循环，没有终端。AWS Remote SWE Agents、Cursor Background Agents、OpenAI Codex Cloud、Google Jules 和 Factory Droids 都汇聚到了这一点。

工程挑战是具体的：环境复现（智能体必须从零构建仓库，没有缓存的开发镜像）、测试抖动（必须重新运行或隔离）、凭证范围（带最小细粒度权限的 GitHub App）、每仓库每天的预算执行，以及禁止强制推送策略。这个顶点项目测量通过率、成本和安全性，并与托管替代方案进行比较。

## 核心概念

触发器是 GitHub webhook（issue 标签或 PR 评论）。调度器将工作排队到 ECS Fargate 或 Lambda。Worker 将仓库拉入一个 Daytona 或 E2B 沙箱，沙箱中有从仓库推断的通用 Dockerfile（语言、框架）。智能体对 Claude Opus 4.7 或 GPT-5.4-Codex 运行 mini-swe-agent 或 SWE-agent v2 循环。它迭代：读取代码，提出修复，应用补丁，运行测试。

验证是门控步骤。在 PR 开启之前，完整 CI 必须在沙箱中通过。计算覆盖率差值；如果超过阈值下降，PR 开启但被标记为 `needs-review`。智能体将理由说明发布为 PR 描述，加上审查者可以 ping 的 `@agent` 线程用于后续跟进。

安全性通过两个不同的 GitHub 接口来限定范围：App 提供带 `workflows: read` 和窄仓库内容/PR 作用域的短期安装 token；分支保护（而不是 App 权限）强制执行"禁止直接写入 `main`"和"禁止强制推送"——App 永远不被添加到绕过列表中。由于 GitHub App 权限不是路径范围的，对 `.github/workflows` 的路径范围只读访问需要在 worker 处通过对拟议差异的允许列表检查来执行。每仓库每天的预算上限在调度器处执行（例如，每仓库每天最多 5 个 PR，每 PR $20）。

## 架构图

```
GitHub issue labeled `@agent fix` or PR comment
            |
            v
    GitHub App webhook -> AWS Lambda dispatcher
            |
            v
    ECS Fargate task (or GitHub Actions self-hosted runner)
       - pull repo
       - infer Dockerfile (language, package manager)
       - Daytona / E2B sandbox with target runtime
       - clone -> git worktree -> agent branch
            |
            v
    mini-swe-agent / SWE-agent v2 loop
       Claude Opus 4.7 or GPT-5.4-Codex
       tools: ripgrep, tree-sitter, read/edit, run_tests, git
            |
            v
    verify CI passes in-sandbox + coverage delta check
            |
            v (verified)
    git push + open PR via GitHub App
       PR body = rationale + diff summary + trace URL
       label: needs-review
            |
            v
    operator reviews; can @-mention agent for follow-ups
```

## 技术栈

- 触发器：带细粒度 token 的 GitHub App；通过 Lambda 或 Fly.io 的 webhook 接收器
- Worker：ECS Fargate 任务（或 GitHub Actions 自托管运行器）
- 沙箱：每任务 Daytona devcontainer 或 E2B 沙箱
- 智能体循环：mini-swe-agent 基线或基于 Claude Opus 4.7/GPT-5.4-Codex 的 SWE-agent v2
- 检索：tree-sitter repo-map + ripgrep
- 验证：沙箱内完整 CI + 覆盖率差值门控
- 可观测性：Langfuse，带从 PR 正文链接的每 PR 追踪归档
- 预算：每仓库每日美元上限；每仓库每日最大 PR 数

## 构建步骤

1. **GitHub App。** 细粒度安装 token：issues 读写，pull_requests 写，contents 读写，workflows 读。分支保护（能做到这一点的唯一接口）强制执行"禁止直接推送到 `main`"和"禁止强制推送"；App 不在绕过列表中。Worker 对拟议差异执行"禁止写入 `.github/workflows` 下的内容"的允许列表检查，因为 GitHub App 权限不是路径范围的。

2. **Webhook 接收器。** Lambda 函数接受 issue 标签/PR 评论 webhook。按标签 `@agent fix this` 过滤。排队到 SQS。

3. **调度器。** 从 SQS 弹出任务。执行每仓库每日预算。启动一个带仓库 URL、issue 正文和全新 Daytona 沙箱的 ECS Fargate 任务。

4. **环境推断。** 检测语言（Python、Node、Go、Rust）和包管理器（uv、pnpm、go mod、cargo）。如果不存在 Dockerfile，则动态生成一个。

5. **智能体循环。** 带 Claude Opus 4.7 的 mini-swe-agent 或 SWE-agent v2。工具：ripgrep、tree-sitter repo-map、read_file、edit_file、run_tests、git。硬性限制：$20 成本，30 分钟墙上时钟，30 轮智能体轮次。

6. **验证。** 循环结束后，在沙箱内运行完整测试套件。通过 jacoco/coverage.py 计算覆盖率差值。如果 CI 红色：停止，不开启 PR。如果覆盖率下降超过 2%：开启带 `needs-review` 标签的 PR。

7. **PR 发布。** 推送智能体分支。通过 GitHub API 开启 PR，包含：标题、理由说明、差异摘要、追踪 URL、成本、轮次。

8. **凭证卫生。** Worker 使用短期 GitHub App 安装 token 运行。日志在归档前清洗掉密钥。

9. **评测。** 30 个不同难度的内部种子 issue。测量通过率、PR 质量（差异大小、风格、覆盖率）、成本、延迟。与 Cursor Background Agents 和 AWS Remote SWE Agents 在相同 issue 上进行比较。

## 使用示例

```
# on github.com
  - user labels issue #842 with `@agent fix this`
  - PR #1903 appears 14 minutes later
  - body:
    > Fixed NPE in widget.dedupe() caused by null comparator entry.
    > Added regression test widget_test.go::TestDedupeNullComparator.
    > Coverage delta: +0.12%
    > Turns: 7  Cost: $1.80  Trace: langfuse:...
    > Label: needs-review
```

## 交付物

`outputs/skill-issue-to-pr.md` 是交付物。一个 GitHub App + 异步云 worker，将带标签的 issue 转化为有预算限制和受限凭证的可审查 PR。

| 权重 | 评分标准 | 衡量方式 |
|:-:|---|---|
| 25 | 30 个 issue 上的通过率 | 端到端成功（CI 绿色 + 覆盖率通过） |
| 20 | PR 质量 | 差异大小、覆盖率差值、风格合规性 |
| 20 | 每个解决 issue 的成本和延迟 | 每 PR 的 $ 和墙上时钟 |
| 20 | 安全性 | 受限 token、每仓库预算、禁止强制推送、凭证卫生 |
| 15 | 运营者体验 | 理由说明评论、重试功能、@-mention 跟进 |
| **100** | | |

## 练习题

1. 添加"修复抖动测试"模式：标签 `@agent stabilize-flake TestX` 在沙箱中运行测试 50 次，并提出一个使其稳定的最小更改。

2. 在三个共享 issue 上与 Cursor Background Agents 比较成本。报告哪些工具在哪里胜出。

3. 实现预算仪表板：每仓库每日成本、每用户成本。对异常发出告警。

4. 构建"演练"模式：开启一个草稿 PR 而不运行 CI，以便审查者能以低成本检查计划。

5. 添加保留策略：7 天以上未合并的 PR 分支自动删除。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| GitHub App | "受限机器人身份" | 带细粒度权限 + 短期安装 token 的 App |
| 异步云智能体 | "后台智能体" | 在云沙箱中运行的非交互式 worker，而不是终端 |
| 环境推断 | "Dockerfile 合成" | 检测语言 + 包管理器，如果不存在则生成 Dockerfile |
| 验证 | "沙箱内 CI" | 在开启 PR 前在 worker 内运行完整测试套件 |
| 覆盖率差值 | "覆盖率保留" | 从基础到智能体分支的测试覆盖率百分比变化 |
| 每仓库预算 | "每日上限" | 调度器处执行的美元和 PR 数上限 |
| 理由说明 | "PR 正文解释" | 智能体对什么改变了以及为什么的摘要；PR 正文中必须包含 |

## 延伸阅读

- [AWS Remote SWE Agents](https://github.com/aws-samples/remote-swe-agents) — 标准异步云智能体参考
- [SWE-agent](https://github.com/SWE-agent/SWE-agent) — CLI 参考
- [Cursor Background Agents](https://docs.cursor.com/background-agent) — 商业替代方案
- [OpenAI Codex（云端）](https://openai.com/codex) — 托管竞品
- [Google Jules](https://jules.google) — Google 的托管版本
- [Factory Droids](https://www.factory.ai) — 备选商业参考
- [GitHub App 文档](https://docs.github.com/en/apps) — 受限机器人身份
- [Daytona 云沙箱](https://daytona.io) — 参考沙箱
