# 顶点项目 10 — 多智能体软件工程团队

> SWE-AF 的工厂架构、MetaGPT 的基于角色的提示词、AutoGen 0.4 的类型化 actor 图、Cognition 的 Devin 和 Factory 的 Droids 都汇聚到了同一个 2026 年形态：架构师规划，N 个编程者在并行 worktree 中工作，审查者把关，测试者验证。并行 worktree 将墙上时钟转化为吞吐量。共享状态和交接协议成为失败表面。这个顶点项目就是构建这个团队，在 SWE-bench Pro 上进行评测，并报告哪些交接会失败以及失败频率。

**类型：** 顶点项目  
**语言：** Python/TypeScript（智能体），Shell（worktree 脚本）  
**前置条件：** 第 11 阶段（LLM 工程），第 13 阶段（工具），第 14 阶段（智能体），第 15 阶段（自主系统），第 16 阶段（多智能体），第 17 阶段（基础设施）  
**涵盖阶段：** P11 · P13 · P14 · P15 · P16 · P17  
**预计时间：** 40 小时

## 问题背景

单智能体编程交互壳层在大型任务上碰到了上限。不是因为任何单个智能体能力不足，而是 20 万 token 的上下文无法同时容纳架构计划、四个并行代码库切片、审查者评论和测试输出。多智能体工厂将问题拆分：架构师负责计划，编程者在并行 worktree 中负责实现，审查者把关，测试者验证。SWE-AF 的"工厂"架构、MetaGPT 的角色、AutoGen 的类型化 actor 图——三种框架描述的是同一个形态。

失败表面是交接环节。架构师计划了编程者无法实现的东西。编程者产生了冲突的差异。审查者批准了一个幻觉修复。测试者与仍在编写代码的编程者产生竞争。你将构建这样一个团队，在 50 个 SWE-bench Pro 问题上运行，追踪每次交接，并发布事后分析。

## 核心概念

角色是类型化智能体。**架构师**（Claude Opus 4.7）读取问题，编写计划，并将其分解为带有明确接口的子任务。**编程者**（Claude Sonnet 4.7，N 个并行实例，每个在 `git worktree` + Daytona 沙箱中）独立实现子任务。**审查者**（GPT-5.4）读取合并后的差异，要么批准要么提出具体的修改请求。**测试者**（Gemini 2.5 Pro）在隔离环境中运行测试套件，报告通过/失败及附件。

通信通过共享任务板（文件支持或 Redis）进行。每个角色消费它被允许处理的任务。交接是 A2A 协议类型化消息。协调关注点：合并冲突解决（协调者角色或自动三方合并）、共享状态同步（一旦编程者开始，计划就被冻结；重新规划是独立事件），以及审查者门控（审查者不能批准自己的更改或自己提出的更改）。

token 放大是隐藏成本。每个角色边界都会增加摘要提示和交接上下文。单智能体的 40 轮运行在四个角色中变成了总计 160 轮。评分标准特别衡量 token 效率 vs 单智能体基线，因为问题不是"多智能体是否有效"，而是"它每美元是否划算"。

## 架构图

```
GitHub issue URL
      |
      v
Architect (Opus 4.7)
   reads issue, produces plan with subtasks + interfaces
      |
      v
Task board (file / Redis)
      |
   +-- subtask 1 ---+-- subtask 2 ---+-- subtask 3 ---+-- subtask 4 ---+
   v                v                v                v                v
Coder A          Coder B          Coder C          Coder D          (4 parallel)
 (Sonnet)         (Sonnet)         (Sonnet)         (Sonnet)
 worktree A       worktree B       worktree C       worktree D
 Daytona          Daytona          Daytona          Daytona
      |                |                |                |
      +--------+-------+-------+--------+
               v
           merge coordinator  (three-way merge + conflict resolution)
               |
               v
           Reviewer (GPT-5.4)
               |
               v
           Tester  (Gemini 2.5 Pro)  -> passes? -> open PR
                                     -> fails?  -> route back to coder
```

## 技术栈

- 编排：LangGraph，带共享状态 + 每智能体子图
- 消息传递：A2A 协议（Google 2025）用于类型化智能体间消息
- 模型：Opus 4.7（架构师），Sonnet 4.7（编程者），GPT-5.4（审查者），Gemini 2.5 Pro（测试者）
- Worktree 隔离：每个编程者用 `git worktree add` + Daytona 沙箱
- 合并协调者：自定义三方合并 + LLM 介导的冲突解决
- 评测：SWE-bench Pro（50 个问题），SWE-AF 场景，HumanEval++ 用于单元测试
- 可观测性：Langfuse 带角色标记的 span，每智能体 token 计账
- 部署：K8s，每个角色作为独立 Deployment + HPA 按积压扩缩容

## 构建步骤

1. **任务板。** 文件支持的 JSONL，带类型化消息：`plan_request`、`subtask`、`diff_ready`、`review_needed`、`test_needed`、`approved`、`rejected`、`replan_needed`。智能体按标签订阅。

2. **架构师。** 读取 GitHub 问题，用带计划模板的 Opus 4.7 运行，要求明确的子任务接口（涉及的文件、公开函数、测试影响）。输出一个带子任务 DAG 的 `plan_request`。

3. **编程者。** N 个并行 worker，每个从任务板认领一个子任务。每个生成一个新的 `git worktree add` 分支加 Daytona 沙箱。实现子任务。输出带补丁 + 测试差异的 `diff_ready`。

4. **合并协调者。** 所有编程者完成后，将 N 个分支三方合并到暂存分支。仅在文件级重叠存在时进行 LLM 介导的冲突解决。

5. **审查者。** GPT-5.4 读取合并后的差异。不能批准自己编写的差异。输出 `approved`（无操作）或带有路由到相关编程者的具体修改请求的 `review_feedback`。

6. **测试者。** Gemini 2.5 Pro 在干净沙箱中运行测试套件。捕获附件。输出 `test_passed` 或带堆栈追踪的 `test_failed`。失败测试循环回到拥有该失败子任务的编程者。

7. **交接计账。** 每条跨越角色边界的消息在 Langfuse 中获得一个带 payload 大小和所用模型的 span。计算每子任务 token 放大（编程者 tokens + 审查者 tokens + 测试者 tokens + 架构师份额 / 编程者 tokens）。

8. **评测。** 在 50 个 SWE-bench Pro 问题上运行。将 pass@1 和每解决问题的成本与单智能体基线（单个 worktree 中的单个 Sonnet 4.7）进行比较。

9. **事后分析。** 对每个失败问题，找出断裂的交接（计划过于模糊、合并冲突、审查者假批准、测试者抖动）。生成交接失败直方图。

## 使用示例

```
$ team run --issue https://github.com/acme/widget/issues/842
[architect] plan: 4 subtasks (parser, cache, api, migration)
[board]     dispatched to 4 coders in parallel worktrees
[coder-A]   subtask parser  -> 42 lines, tests pass locally
[coder-B]   subtask cache   -> 88 lines, tests pass locally
[coder-C]   subtask api     -> 31 lines, tests pass locally
[coder-D]   subtask migration -> 19 lines, tests pass locally
[merge]     3-way merge: 0 conflicts
[reviewer]  comments on cache (thread pool sizing); routed to coder-B
[coder-B]   revision: 92 lines; submits
[reviewer]  approved
[tester]    all 412 tests pass
[pr]        opened #3382   4 coders, 1 revision, $4.90, 18m
```

## 交付物

`outputs/skill-multi-agent-team.md` 是交付物。给定一个问题 URL 和并行度级别，团队产出一个带每角色 token 计账的可合并 PR。

| 权重 | 评分标准 | 衡量方式 |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 | 50 个问题匹配子集的 pass@1 |
| 20 | 并行加速 | 与单智能体基线的墙上时钟对比 |
| 20 | 审查质量 | 注入 bug 探测上的假批准率 |
| 20 | Token 效率 | 每个解决问题的总 token 数 vs 单智能体 |
| 15 | 协调工程 | 合并冲突解决、交接失败直方图 |
| **100** | | |

## 练习题

1. 在运行中途向差异中注入一个明显的 bug（在主体前多加一个 `return None`）。测量审查者的假批准率。调整审查者提示词直到假批准率低于 5%。

2. 减少到两个编程者（架构师 + 编程者 + 审查者 + 测试者，编程者顺序运行两个子任务）。比较墙上时钟和通过率。

3. 用单写入约束替换合并协调者（子任务接触不相交的文件集）。测量架构师的规划负担。

4. 将审查者从 GPT-5.4 替换为 Claude Opus 4.7。测量假批准率和 token 成本差值。

5. 添加第五个角色：文档编写者（Haiku 4.5）。审查后，它生成一个更新日志条目。测量文档质量是否能为额外的 token 支出提供正当理由。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 并行 worktree | "隔离分支" | `git worktree add` 为每个编程者生成一个新的工作目录 |
| 任务板 | "共享消息总线" | 智能体订阅的类型化消息的文件或 Redis 存储 |
| 交接 | "角色边界" | 任何从一个角色的上下文传递到另一个角色的消息 |
| Token 放大 | "多智能体开销" | 跨所有角色的总 token 数 / 相同任务的单智能体 token 数 |
| A2A 协议 | "智能体间通信" | Google 2025 年的类型化智能体间消息规范 |
| 合并协调者 | "集成者" | 运行三方合并并介导冲突的组件 |
| 假批准 | "审查者幻觉" | 审查者批准了一个有已知 bug 的差异 |

## 延伸阅读

- [SWE-AF 工厂架构](https://github.com/Agent-Field/SWE-AF) — 2026 年参考多智能体工厂
- [MetaGPT](https://github.com/FoundationAgents/MetaGPT) — 基于角色的多智能体框架
- [AutoGen v0.4](https://github.com/microsoft/autogen) — Microsoft 的类型化 actor 框架
- [Cognition AI（Devin）](https://cognition.ai) — 参考产品
- [Factory Droids](https://www.factory.ai) — 备选参考产品
- [Google A2A 协议](https://developers.google.com/agent-to-agent) — 智能体间消息规范
- [git worktree 文档](https://git-scm.com/docs/git-worktree) — 隔离基底
- [SWE-bench Pro](https://www.swebench.com) — 评测目标
