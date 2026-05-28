# 智能体指令即可执行约束

> 以散文形式写就的指令是愿望。以约束形式写就的指令是测试。工作台将每条规则转化为智能体可以在运行时检查、审查者可以事后验证的东西。

**类型：** 构建
**编程语言：** Python（标准库）
**前置知识：** Phase 14 · 32（最小工作台）
**预计时间：** 约 50 分钟

## 学习目标

- 将路由散文与操作规则分离。
- 将启动规则、禁止操作、完成定义、不确定性处理和批准边界表达为机器可检查的约束。
- 实现一个根据规则集对运行评分的规则检查器。
- 使规则集对差异友好，让审查可以看到发生了什么变化。

## 问题背景

典型的 `AGENTS.md` 读起来像入职文档。它告诉智能体"要小心"、"彻底测试"和"不确定时请询问"。三天后，智能体发布了没有测试的变更，向禁止的目录写入，从不询问，因为它从不知道界线在哪里。

当指令是操作性的时候是有力的，当它们是理想性的时候是无力的。解决方案是写工作台可以解释、审查者可以评分的规则。

## 核心概念

规则属于 `docs/agent-rules.md`，与简短的根路由器分开。每条规则有名称、类别和检查。

```mermaid
flowchart LR
  Router[AGENTS.md] --> Rules[docs/agent-rules.md]
  Rules --> Checker[rule_checker.py]
  Checker --> Report[rule_report.json]
  Report --> Reviewer[审查者]
```

### 覆盖大多数规则的五个类别

| 类别 | 规则回答的问题 | 示例 |
|------|-------------|------|
| 启动 | 工作开始前必须为真的是什么？ | "状态文件存在且是最新的" |
| 禁止 | 什么永远不能发生？ | "不编辑 `scripts/release.sh`" |
| 完成定义 | 什么证明任务完成了？ | "pytest 以 0 退出且验收行通过" |
| 不确定性 | 智能体不确定时做什么？ | "创建问题笔记而不是猜测" |
| 批准 | 什么需要人类批准？ | "任何新依赖，任何生产写入" |

不符合这五类之一的规则通常想成为两条规则。强制拆分。

### 规则是机器可读的

每条规则有一个 slug、一个类别、一行描述，以及一个命名 `rule_checker.py` 中函数的 `check` 字段。添加规则意味着添加检查；检查器随工作台增长。

### 规则是差异友好的

规则每条标题对应一个 markdown 文件。重命名在差异中可见。新规则位于其类别的顶部。陈旧规则被删除，而不是注释掉，因为工作台是真实来源，而不是团队上个季度感受的聊天日志。

### 规则与框架防护栏

框架防护栏（OpenAI Agents SDK 防护栏、LangGraph 中断）在运行时级别执行规则。本课中的规则集是那些防护栏实现的人类可读、可审查的契约。两者都需要：运行时在轮次中捕获违规，规则集证明运行时在做正确的事情。

## 动手实践

`code/main.py` 包含：

- 将规则加载到数据类的 `agent-rules.md` 解析器。
- `rule_checker.py` 风格的检查函数，每个 `check` 引用一个。
- 一个违反两条规则的演示智能体运行，以及捕获它们的检查通过。

运行：

```
python3 code/main.py
```

输出：解析的规则集、运行追踪、每条规则的通过/失败，以及保存在脚本旁边的 `rule_report.json`。

## 生产中的模式

三种模式将持续一个季度的规则集与一周内衰退的规则集区分开来。

**写作时的严重性标注。** 每条规则携带 `severity`：`block`、`warn` 或 `info`。检查器报告所有三种；运行时只在 `block` 时拒绝。大多数团队早期高估严重性，然后在截止日期压力下悄悄降低；在写作时标注会强制提前校准。与验证门控（Phase 14 · 38）配合，它将任何对 `block` 规则的覆盖签署到 `overrides.jsonl` 审计日志中。

**规则过期作为强制函数。** 每条规则携带 `expires_at` 日期（默认从编写起 90 天）。当未过期规则连续 60 天没有违规时，检查器发出警告；下一个季度审查要么证明保留它的合理性，要么降级为 `info`，要么删除它。Cloudflare 的生产 AI 代码审查数据（2026 年 4 月，在 30 天内跨 5,169 个仓库进行 131,246 次审查运行）显示，有明确过期的规则集每个仓库保持在 30 条以下；没有的规则集增长到 80 条以上，大多数从未触发。

**Markdown 作为源，JSON 作为缓存。** `agent-rules.md` 是编写的文件；`agent-rules.lock.json` 是检查器在热路径上读取的缓存。锁由预提交钩子重新生成。Markdown 差异是可审查的；JSON 解析保持在每轮之外。与 `package.json` / `package-lock.json` 和 `Cargo.toml` / `Cargo.lock` 形态相同。

## 使用建议

在生产中：

- Claude Code、Codex、Cursor 在会话开始时读取规则，并在拒绝操作时引用它们。检查器在 CI 中重新运行它们以捕获静默漂移。
- OpenAI Agents SDK 防护栏将相同的检查注册为输入和输出防护栏。Markdown 是文档界面；SDK 是运行时界面。
- LangGraph 中断在正在进行的节点违反规则时触发。中断处理器读取规则，询问人类，然后恢复。

规则集在所有三者之间是可移植的，因为它只是 markdown 加函数名。

## 产出技能

`outputs/skill-rule-set-builder.md` 访谈项目所有者，将其现有的散文指令分类为五个类别，并生成版本化的 `agent-rules.md` 加检查器存根。

## 练习

1. 如果你的产品真正需要，添加第六个类别。为什么它不能归并到五个之一中。
2. 扩展检查器，使规则可以携带严重性（`block`、`warn`、`info`），报告相应地聚合。
3. 将检查器接入 CI：如果最新智能体运行中的 block 严重性规则失败，则使构建失败。
4. 每条规则添加"过期"字段。90 天没有检查失败后，规则进入审查。
5. 找一个真实的 `AGENTS.md` 并将其重写为五类别规则。其中有多少行是操作性的？有多少是理想性的？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 操作性规则 | "真正的指令" | 工作台可以在运行时检查的规则 |
| 理想性规则 | "要小心" | 没有检查的规则；要么删除要么升级 |
| 完成定义 | "验收" | 任务完成的客观、文件支持的证明 |
| Block 严重性 | "硬规则" | 违反时停止运行；不能在没有操作员的情况下消除 |
| 规则过期 | "陈旧规则清理" | N 天没有失败的规则进入退役 |

## 延伸阅读

- [OpenAI Agents SDK 防护栏](https://platform.openai.com/docs/guides/agents-sdk/guardrails)
- [LangGraph 中断](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/breakpoints/)
- [Anthropic，构建有效智能体](https://www.anthropic.com/research/building-effective-agents)
- [Rick Hightower，智能体规则：确定性策略引擎](https://medium.com/@richardhightower/agent-rulez-a-deterministic-policy-engine-for-ai-coding-agents-9489e0561edf) — 生产中的 block/warn/info 严重性
- [Cloudflare，大规模编排 AI 代码审查](https://blog.cloudflare.com/ai-code-review/) — 131k 次审查运行，规则组合经验
- [microservices.io，GenAI 开发平台——第 1 部分：防护栏](https://microservices.io/post/architecture/2026/03/09/genai-development-platform-part-1-development-guardrails.html) — 规则与 CI 之间的深度防御
- [类型检查合规性：确定性防护栏（arXiv 2604.01483）](https://arxiv.org/pdf/2604.01483) — Lean 4 作为规则即检查的上限
- [logi-cmd/agent-guardrails](https://github.com/logi-cmd/agent-guardrails) — 合并门控实现：范围、变异测试、违规预算
- Phase 14 · 32 — 此规则集嵌入的最小工作台
- Phase 14 · 38 — 消耗规则报告的验证门控
- Phase 14 · 39 — 对规则合规性评分的审查者智能体
