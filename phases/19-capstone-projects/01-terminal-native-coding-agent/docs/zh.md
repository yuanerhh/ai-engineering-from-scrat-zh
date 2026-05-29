# 顶点项目 01 — 终端原生编程智能体

> 2026 年，编程智能体的形态已趋于稳定：TUI 交互壳层、有状态执行计划、沙箱化工具表面、循环执行"规划—行动—观察—恢复"。Claude Code、Cursor 3 和 OpenCode 在宏观上看起来别无二致。本顶点项目要求你从零到一完整构建一个——CLI 输入，Pull Request 输出——并在 SWE-bench Pro 上与 mini-swe-agent 和 Live-SWE-agent 进行横向评测。你将亲身理解：真正的难点不是调用模型，而是工具循环、沙箱管理，以及 50 轮运行的成本上限控制。

**类型：** 顶点项目  
**语言：** TypeScript / Bun（交互壳层），Python（评测脚本）  
**前置条件：** 第 11 阶段（LLM 工程），第 13 阶段（工具与协议），第 14 阶段（智能体），第 15 阶段（自主系统），第 17 阶段（基础设施）  
**涵盖阶段：** P0 · P5 · P7 · P10 · P11 · P13 · P14 · P15 · P17 · P18  
**预计时间：** 35 小时

## 问题背景

2026 年，编程智能体成为 AI 应用的主导类别。Claude Code（Anthropic）、Cursor 3（配备 Composer 2 和 Agent Tabs）、Amp（Sourcegraph）、OpenCode（112k stars）、Factory Droids 以及 Google Jules，全都在交付同一架构的变体：终端交互壳层、受权限管控的工具表面、沙箱环境，以及围绕前沿模型构建的规划—行动—观察循环。前沿水位线很高——Live-SWE-agent 以 Opus 4.5 在 SWE-bench Verified 上达到 79.2%——但工程层面的挑战更为广阔。大多数失败案例并非模型出错，而是工具循环不稳定、上下文污染、token 成本失控以及破坏性文件系统操作。

你无法从外部仅靠推理就理解这些智能体。你必须亲手构建一个，亲眼看着循环在第 47 轮因 ripgrep 返回 8MB 匹配结果而崩溃，然后重建截断层。这就是本顶点项目的意义所在。

## 核心概念

交互壳层（harness）包含四个表面。**Plan（规划）** 维护一个 TodoWrite 风格的状态对象，模型在每轮中重写该对象。**Act（行动）** 分发工具调用（读取、编辑、运行、搜索、git）。**Observe（观察）** 捕获 stdout/stderr/退出码，截断并将摘要反馈回去。**Recover（恢复）** 处理工具错误，同时避免炸穿上下文窗口或陷入死循环。2026 年的形态还增加了一个东西：**hooks（钩子）**。`PreToolUse`、`PostToolUse`、`SessionStart`、`SessionEnd`、`UserPromptSubmit`、`Notification`、`Stop` 和 `PreCompact`——这些可配置的扩展点供运营方注入策略、遥测和护栏。

沙箱使用 E2B 或 Daytona。每个任务在一个全新的 devcontainer 中运行，挂载了读写权限的 git worktree。交互壳层永不触及宿主文件系统。worktree 在成功或失败后均被销毁。成本控制在三个层次强制执行：每轮 token 上限、每会话美元预算，以及硬性轮次限制（通常为 50 轮）。可观测性层使用 OpenTelemetry spans（配合 GenAI 语义约定），推送至自托管的 Langfuse。

## 架构图

```
  user CLI  ->  harness (Bun + Ink TUI)
                  |
                  v
           plan / act / observe loop  <--->  Claude Sonnet 4.7 / GPT-5.4-Codex / Gemini 3 Pro
                  |                          (via OpenRouter, model-agnostic)
                  v
           tool dispatcher (MCP StreamableHTTP client)
                  |
     +------------+------------+----------+
     v            v            v          v
  read/edit    ripgrep     tree-sitter   git/run
     |            |            |          |
     +------------+------------+----------+
                  |
                  v
           E2B / Daytona sandbox  (worktree isolated)
                  |
                  v
           hooks: Pre/Post, Session, Prompt, Compact
                  |
                  v
           OpenTelemetry -> Langfuse (spans, tokens, $)
                  |
                  v
           PR via GitHub app
```

## 技术栈

- 壳层运行时：Bun 1.2 + Ink 5（React in terminal）
- 模型访问：OpenRouter 统一 API，支持 Claude Sonnet 4.7、GPT-5.4-Codex、Gemini 3 Pro、Opus 4.5（最难任务）
- 工具传输：Model Context Protocol（MCP 2026 修订版）StreamableHTTP
- 沙箱：E2B sandboxes（JS SDK）或 Daytona devcontainers
- 代码搜索：ripgrep 子进程，17 种语言的 tree-sitter 解析器（预编译）
- 隔离：每任务 `git worktree add`，成功/失败后清理
- 评测框架：SWE-bench Pro（已验证子集）+ Terminal-Bench 2.0 + 自定义 30 任务保留集
- 可观测性：OpenTelemetry SDK（带 `gen_ai.*` 语义约定）→ 自托管 Langfuse
- PR 发布：GitHub App（细粒度 token，权限限定到目标仓库）

## 构建步骤

1. **TUI 与命令循环。** 使用 Ink 搭建 Bun 项目。接受 `agent run <repo> "<task>"` 命令。打印分栏视图：规划面板（上）、工具调用流（中）、token 预算（下）。添加 Ctrl-C 取消，退出前触发 `SessionEnd` 钩子。

2. **规划状态。** 定义带类型的 TodoWrite 模式（pending/in_progress/done 条目及备注）。模型在每轮以工具调用方式重写完整状态——不允许增量修改。将规划持久化至 `.agent/state.json`，以便崩溃后恢复。

3. **工具表面。** 定义六个工具：`read_file`、`edit_file`（带差异预览）、`ripgrep`、`tree_sitter_symbols`、`run_shell`（带超时）、`git`（status/diff/commit/push）。通过 MCP StreamableHTTP 暴露，使壳层与传输无关。每个工具返回截断输出（每次调用上限 4k token）。

4. **沙箱封装。** 每个任务生成一个 E2B 沙箱。`git worktree add -b agent/$TASK_ID` 创建一个全新分支。所有工具调用在沙箱内执行。宿主文件系统不可访问。

5. **钩子。** 实现 2026 年全部八种钩子类型。至少连接四个用户自定义钩子：(a) 拦截沙箱外 `rm -rf` 的 `PreToolUse` 破坏性命令守卫，(b) token 计账的 `PostToolUse`，(c) 预算初始化的 `SessionStart`，(d) 写入最终追踪包的 `Stop`。

6. **评测循环。** 克隆 SWE-bench Pro Python 的 30 个 issue 子集。针对每个运行你的壳层。与 mini-swe-agent（最小基线）在 pass@1、每任务轮次和每任务成本方面进行比较。将结果写入 `eval/results.jsonl`。

7. **成本控制。** 硬性截止：50 轮、200k 上下文、每任务 $5。`PreCompact` 钩子在达到 150k 标记时将旧轮次摘要化为先验状态块，为新观察腾出空间，同时不丢失规划。

8. **PR 发布。** 成功后，最后一步是 `git push` + GitHub API 调用，开启一个在正文中包含规划和差异摘要的 PR。

## 使用示例

```
$ agent run ./my-repo "Fix the race condition in worker.rs"
[plan]  1 locate worker.rs and enumerate mutex uses
        2 identify shared state under contention
        3 propose fix, verify tests
[tool]  ripgrep mutex.*lock -t rust           (44 matches, truncated)
[tool]  read_file src/worker.rs 120..180
[tool]  edit_file src/worker.rs (+8 -3)
[tool]  run_shell cargo test worker::          (passed)
[plan]  1 done · 2 done · 3 done
[done]  PR opened: #482   turns=9   tokens=38k   cost=$0.41
```

## 交付物

可交付技能存放于 `outputs/skill-terminal-coding-agent.md`。给定仓库路径和任务描述，它在沙箱中运行完整的规划—行动—观察循环，返回 PR URL 和追踪包。本顶点项目评分标准：

| 权重 | 评分标准 | 衡量方式 |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 vs 基线 | 你的壳层 vs mini-swe-agent，30 个匹配 Python 任务 |
| 20 | 架构清晰度 | 规划/行动/观察分离、钩子表面、工具模式——对照 Live-SWE-agent 布局审查 |
| 20 | 安全性 | 沙箱逃逸测试、权限提示、破坏性命令守卫通过红队测试 |
| 20 | 可观测性 | 追踪完整性（100% 工具调用已生成 span）、每轮 token 计账 |
| 15 | 开发者体验 | 冷启动 < 2s，崩溃恢复能恢复规划，Ctrl-C 干净取消进行中的工具 |
| **100** | | |

## 练习题

1. 将后端模型从 Claude Sonnet 4.7 替换为在 vLLM 上服务的 Qwen3-Coder-30B。比较 pass@1 和每任务成本。报告开源模型表现不如预期的地方。

2. 在 PR 发布前添加一个 `reviewer` 子智能体，它读取差异并可以请求修订循环。测量虚假 review 是否会使 SWE-bench pass rate 低于单智能体基线（提示：通常会）。

3. 对沙箱进行压力测试：编写一个尝试 `curl` 外部 URL 的任务，以及一个在 worktree 外写文件的任务。确认两者均被 PreToolUse 钩子阻止。记录这些尝试。

4. 用较小的模型（Haiku 4.5）实现 `PreCompact` 摘要化。测量经过 3 次压缩后规划保真度损失了多少。

5. 将 MCP StreamableHTTP 传输替换为 stdio。对冷启动和每次调用延迟进行基准测试。为纯本地使用场景选出胜者。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Harness（交互壳层） | "智能体循环" | 包裹模型的代码，负责分发工具、维护规划状态和执行预算 |
| Hook（钩子） | "智能体事件监听器" | 用户编写的脚本，由壳层在八个生命周期事件之一触发时运行 |
| Worktree | "Git 沙箱" | 在独立路径上的关联 git checkout；可抛弃，不影响主克隆 |
| TodoWrite | "规划状态" | 模型每轮重写的带类型的 pending/in-progress/done 条目列表 |
| StreamableHTTP | "MCP 传输" | 2026 年 MCP 修订版：长连接 HTTP 双向流；取代 SSE |
| Token ceiling（token 上限） | "上下文预算" | 每轮或每会话的输入+输出 token 上限；触发压缩或终止 |
| pass@1 | "单次通过率" | 在不重试或窥视测试集的情况下，首次运行解决 SWE-bench 任务的比例 |

## 延伸阅读

- [Claude Code 文档](https://docs.anthropic.com/en/docs/claude-code) — Anthropic 的参考壳层
- [Cursor 3 更新日志](https://cursor.com/changelog) — Agent Tabs 和 Composer 2 产品说明
- [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) — SWE-bench 壳层对比的最小基线
- [Live-SWE-agent](https://github.com/OpenAutoCoder/live-swe-agent) — 使用 Opus 4.5 在 SWE-bench Verified 上达到 79.2%
- [OpenCode](https://opencode.ai) — 开源壳层，112k stars
- [SWE-bench Pro 排行榜](https://www.swebench.com) — 本顶点项目的评测目标
- [Model Context Protocol 2026 路线图](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — StreamableHTTP、能力元数据
- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 工具调用和 token 使用的 span 模式
