# 顶点项目 09 — 代码迁移智能体（仓库级语言/运行时升级）

> Amazon 的 MigrationBench（Java 8 升级到 17）和 Google 的 App Engine Py2-to-Py3 迁移器设定了 2026 年的标准。Moderne 的 OpenRewrite 在规模化场景下进行确定性 AST 重写。Grit 用 codemod 风格的 DSL 瞄准了同一问题。生产模式将两者结合：一个用于安全重写的确定性基底，加上一个用于模糊情况的智能体层，每个分支有沙箱进行构建，测试套件全部变绿后才开 PR。这个顶点项目要求迁移 50 个真实仓库，发布带失败分类的通过率报告。

**类型：** 顶点项目  
**语言：** Python（智能体），Java/Python（目标），TypeScript（仪表板）  
**前置条件：** 第 5 阶段（NLP），第 7 阶段（Transformer），第 11 阶段（LLM 工程），第 13 阶段（工具），第 14 阶段（智能体），第 15 阶段（自主系统），第 17 阶段（基础设施）  
**涵盖阶段：** P5 · P7 · P11 · P13 · P14 · P15 · P17  
**预计时间：** 30 小时

## 问题背景

大规模代码迁移是 2026 年编程智能体最干净的生产应用之一。评判标准显而易见（迁移后测试套件是否通过？），回报真实（Java 8 机群迁移是一个需要人力规模的项目），基准公开（MigrationBench 50 个仓库子集）。Moderne 的 OpenRewrite 处理确定性的部分。智能体层处理 OpenRewrite 规则无法覆盖的一切：模糊重写、构建系统漂移、长尾语法、传递依赖破损。

你将构建一个接收 Java 8 仓库（或 Python 2 仓库）并产出绿色 CI 迁移分支的智能体。你将测量通过率、测试覆盖率保留、每个仓库的成本，并构建失败分类体系。与纯确定性基线的并排对比会告诉你智能体的价值究竟在哪里。

## 核心概念

流水线有两层。**确定性基底**（Java 用 OpenRewrite，Python 用 libcst）安全地运行大部分机械式重写：导入、方法签名、空安全编辑、try-with-resources、废弃 API 替换。它快速且产生可审计的差异。**智能体层**（基于 Claude Opus 4.7 和 GPT-5.4-Codex 的 OpenAI Agents SDK 或 LangGraph）处理规则无法覆盖的情况：构建文件升级（Maven/Gradle/pyproject）、传递依赖冲突、测试抖动、自定义注解。

每个仓库在一个预装了目标运行时的 Daytona 沙箱中运行。智能体迭代：运行构建，分类失败，应用修复，重新运行。硬性限制：每个仓库 30 分钟，$8，20 轮智能体轮次。如果所有测试通过且覆盖率差值不为负，分支开启 PR。如果不通过，仓库被归入某个失败类别并附上证据。

失败分类体系是交付物。在 50 个仓库中，哪里出了问题？传递依赖？自定义注解？构建工具版本？与迁移无关的测试抖动？每个类别都有计数和示例差异。未来的规则作者可以瞄准前三名。

## 架构图

```
target repo
      |
      v
OpenRewrite / libcst deterministic recipes
   (safe, fast, auditable, ~70-80% of fixes)
      |
      v
Daytona sandbox per branch
      |
      v
agent loop (Claude Opus 4.7 / GPT-5.4-Codex):
   - run build -> capture failures
   - classify failures (build, test, lint)
   - apply fix (patch or retry recipe)
   - rerun
   - budget: 30 min, $8, 20 turns
      |
      v
test + coverage delta gate
      |
      v (passed)
open PR
      |
      v (failed)
file under failure class + attach repro
```

## 技术栈

- 确定性基底：OpenRewrite（Java）或 libcst（Python）
- 智能体：基于 Claude Opus 4.7 + GPT-5.4-Codex 的 OpenAI Agents SDK 或 LangGraph
- 沙箱：每分支 Daytona devcontainers，预装目标运行时（Java 17/Python 3.12）
- 构建系统：Maven、Gradle、uv（Python）
- 基准测试：Amazon MigrationBench 50 个仓库子集（Java 8 升级到 17），Google App Engine Py2-to-Py3 仓库
- 测试框架：并行运行器，Java 用 Jacoco 计算覆盖率，Python 用 coverage.py
- 可观测性：Langfuse + 每个仓库的追踪包（含每个差异块）
- 仪表板：带每类计数和示例差异的失败分类仪表板

## 构建步骤

1. **规则执行。** 首先运行 OpenRewrite（Java）或 libcst（Python）规则。处理 70-80% 的机械式迁移。作为"规则"提交。

2. **构建试运行。** Daytona 沙箱：安装目标运行时，运行构建。如果绿色，跳到测试。如果红色，移交给智能体。

3. **智能体循环。** 带工具的 LangGraph：`run_build`、`read_file`、`edit_file`、`run_test`、`git_diff`。智能体分类失败（依赖、语法、测试、构建工具）并应用有针对性的修复。重新运行。

4. **预算上限。** 每个仓库 30 分钟墙上时钟，$8 成本，20 轮智能体轮次。任何超限均暂停，并归入"budget_exhausted"类别，附上当前差异。

5. **测试 + 覆盖率门控。** 构建变绿后，运行测试套件。将覆盖率与基础仓库比较。如果覆盖率下降超过 2%，归入"coverage_regression"。

6. **PR 开启。** 成功后，推送分支，通过 GitHub API 开启 PR，内容包括：差异、规则应用摘要和智能体提交的摘要。

7. **失败分类。** 对每个失败仓库，标记类别：`dep_upgrade_required`、`build_tool_drift`、`custom_annotation`、`test_flake`、`syntax_edge_case`、`budget_exhausted`。构建仪表板。

8. **50 个仓库运行。** 在 MigrationBench 子集上执行。报告每类通过率、每个仓库成本、覆盖率保留，以及与纯确定性基线的对比。

## 使用示例

```
$ migrate legacy-java-service --target java17
[recipe]   27 rewrites applied (JUnit 4->5, HashMap initializer, try-with-resources)
[build]    FAIL: cannot find symbol sun.misc.BASE64Encoder
[agent]    turn 1 classify: removed_jdk_api
[agent]    turn 2 apply: sun.misc.BASE64Encoder -> java.util.Base64
[build]    OK
[tests]    412/412 passing; coverage 84.1% -> 84.3%
[pr]       opened #1841  cost=$3.20  turns=4
```

## 交付物

`outputs/skill-migration-agent.md` 是交付物。给定一个仓库，它先执行确定性规则，然后运行智能体循环，产出一个绿色迁移分支，或将仓库归入分类体系的某个类别。

| 权重 | 评分标准 | 衡量方式 |
|:-:|---|---|
| 25 | MigrationBench 通过率 | 50 个仓库子集的 pass@1 |
| 20 | 测试覆盖率保留 | 相比基础的平均覆盖率差值 |
| 20 | 每个迁移仓库的成本 | 通过运行的每个仓库成本（$） |
| 20 | 智能体/确定性工具集成 | OpenRewrite 处理的修复比例 vs 智能体自己提交的比例 |
| 15 | 失败分析报告 | 带示例的分类体系完整性 |
| **100** | | |

## 练习题

1. 仅用 OpenRewrite（不用智能体）运行迁移流水线。将通过率与完整流水线对比。找出智能体单独起决定作用的情况。

2. 实现"代码检查干净"检查：迁移后，运行代码风格检查器（Java 用 spotless，Python 用 ruff）。如果出现新的检查错误则 PR 失败。测量覆盖率保留但风格退化的比率。

3. 添加"最小差异"优化器：智能体分支通过测试后，进行第二轮裁剪不必要的更改。报告差异大小减少情况。

4. 扩展到第三种迁移：Node 18 升级到 Node 22。复用沙箱封装；将规则层替换为自定义 codemod。

5. 将首次绿色构建时间（TTFGB）作为用户体验指标。目标：p50 低于 10 分钟。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 确定性基底 | "规则引擎" | OpenRewrite/libcst：带安全保证的声明式 AST 重写 |
| Codemod | "代码修改程序" | 机械式修改源代码的重写规则 |
| 构建漂移 | "工具版本偏差" | 主版本之间 Maven/Gradle/uv 行为的细微变化 |
| 失败类别 | "分类桶" | 仓库未能迁移的标记原因：依赖、语法、测试、构建工具、预算 |
| 覆盖率差值 | "覆盖率保留" | 从基础到迁移分支的测试覆盖率百分比变化 |
| 智能体轮次 | "工具调用轮" | 智能体循环中的一次规划->行动->观察周期 |
| 预算耗尽 | "触碰上限" | 仓库在未通过的情况下消耗了 30 分钟/$8/20 轮限制 |

## 延伸阅读

- [Amazon MigrationBench](https://aws.amazon.com/blogs/devops/amazon-introduces-two-benchmark-datasets-for-evaluating-ai-agents-ability-on-code-migration/) — 2026 年的标准基准测试
- [Moderne.io OpenRewrite 平台](https://www.moderne.io) — 确定性基底参考
- [OpenRewrite 文档](https://docs.openrewrite.org) — 规则编写
- [Grit.io](https://www.grit.io) — 备选 codemod DSL
- [OpenAI 沙箱迁移手册](https://developers.openai.com/cookbook/examples/agents_sdk/sandboxed-code-migration/sandboxed_code_migration_agent) — Agents SDK 参考
- [Google App Engine Py2 to Py3 迁移器](https://cloud.google.com/appengine) — 备选迁移基准
- [libcst](https://github.com/Instagram/LibCST) — Python 确定性基底
- [Daytona sandboxes](https://daytona.io) — 参考每分支沙箱
