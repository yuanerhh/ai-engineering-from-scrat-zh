# 顶点项目 05 — 自主科研智能体（AI 科学家级别）

> Sakana 的 AI-Scientist-v2 发表了完整论文，Agent Laboratory 运行了实验，Allen AI 共享了追踪记录。2026 年的形态已趋于成熟：对实验进行规划—执行—验证的树搜索，预算成本，沙箱化代码执行，带视觉反馈的 LaTeX 写作器，以及自动化的 NeurIPS 风格评审集成。这个顶点项目要求你构建一个，在每篇论文 $30 以内端到端运行，并通过 Sakana 所记录的沙箱逃逸红队测试。

**类型：** 顶点项目  
**语言：** Python（智能体 + 沙箱），LaTeX（输出）  
**前置条件：** 第 2 阶段（机器学习），第 3 阶段（深度学习），第 7 阶段（Transformer），第 10 阶段（从零构建 LLM），第 14 阶段（智能体），第 15 阶段（自主系统），第 16 阶段（多智能体），第 18 阶段（安全）  
**涵盖阶段：** P0 · P2 · P3 · P7 · P10 · P14 · P15 · P16 · P18  
**预计时间：** 40 小时

## 问题背景

2026 年，自主科研智能体跨越了一个阈值。Sakana AI 的 AI-Scientist-v2 生成的论文通过了研讨会同行评审并发表于 Nature。ShinkaEvolve（ICLR 2026）将这条路线延伸到演化假说。AMD 的 Agent Laboratory 发布了可复现的追踪记录。这些智能体并不神奇——它们是一个规划—执行—验证循环，运行在候选实验树上，设有成本上限、随机种子绑定的沙箱和自动评审。精髓在于循环、预算和安全故事。

你通过在一个狭窄领域（例如，对 1 亿参数 Transformer 的注意力稀疏性进行消融研究）的种子想法上实现一个循环来学习这一切。价值不在于第一次运行就发现了什么新东西，而在于基础设施：树搜索、实验沙箱、写作—评审循环和红队报告。Sakana 团队记录了沙箱逃逸失败案例；你的智能体必须通过同样的红队测试。

## 核心概念

智能体是一个最优先树搜索（best-first tree search）。节点是实验规格：（假设、配置、代码、预期结果）。扩展步骤提出带有小幅编辑的子节点（换优化器、调整批量大小、消融某个组件）。每个子节点在一个全新的沙箱中运行，设有硬性资源上限。结果反馈进一个评分函数，按（新颖度 × 质量 × 剩余预算）对节点排序。树持续增长直到预算耗尽，然后对最优分支进行撰写。

写作器是多模态的。它生成 LaTeX 草稿，编译，渲染图表，并将渲染后的 PDF 反馈给 Claude Opus 4.7 的视觉模式，以评审布局、图表可读性和论断-证据对齐。由五个 LLM 评委组成的评审集成发出 NeurIPS 风格的评分（新颖度、严谨性、清晰度、可复现性、影响力）；如果平均分低于阈值，论文返回写作器并附上评审意见。

安全是核心约束。每个实验在 E2B 或 Daytona 沙箱中运行，无网络出口，有界限的墙上时钟，以及固定的资源限制。智能体的代码生成步骤通过一个阻止可逃脱沙箱的系统调用的策略层。红队报告复现了 Sakana 所记录的攻击面（fork 炸弹、文件系统逃逸、LLM 写的网络调用）。

## 架构图

```
seed idea + domain
      |
      v
  literature search (Semantic Scholar + OpenAlex + FAISS cache)
      |
      v
  LangGraph plan-execute-verify tree
      |
      v
  +--- expand node ----+      per-node sandbox
  |                    |      (E2B / Daytona)
  v                    v      resource caps
  child_1           child_k   no network egress
  |                    |      deterministic seeds
  v                    v
  run experiment       run experiment
  |                    |
  v                    v
  score nodes by (novelty, quality, budget)
      |
      v
  best branch -> LaTeX writer
      |
      v
  compile + vision critique (Opus 4.7 vision)
      |
      v
  reviewer ensemble (5 LLM judges, NeurIPS rubric)
      |
      v
  paper.pdf + review.md + trace.json
```

## 技术栈

- 编排：带检查点和人工审批门的 LangGraph
- 树搜索：实验节点上的自定义最优先搜索（Sakana v2 的 AB-MCTS 风格）
- 沙箱：每个实验用 E2B，Docker-in-Docker 备选；通过 cgroups 实施资源上限
- 文献：Semantic Scholar Graph API + OpenAlex + 摘要本地 FAISS 缓存
- 写作器：LaTeX 模板 + Claude Opus 4.7（视觉模式）用于图表评审和布局
- 评审器：5 个评委集成（Opus 4.7、GPT-5.4、Gemini 3 Pro、DeepSeek R1、Qwen3-Max），加权聚合
- 实验框架：PyTorch 2.5 用于物理实验，W&B 用于日志记录
- 可观测性：Langfuse 用于智能体追踪，每篇论文硬性预算 $30

## 构建步骤

1. **种子与领域范围划定。** 接受一个种子想法（例如，"研究亚 10 亿参数 Transformer 注意力图中的稀疏模式"）。定义搜索空间：模型、数据集、计算预算。

2. **文献综述。** 查询 Semantic Scholar + OpenAlex 获取 50 篇最多被引用的相关论文；在本地缓存摘要；生成一页领域摘要。

3. **树脚手架。** 用种子假说初始化根节点。实现 `expand(node) -> children`，每个子节点只做一处配置更改。实现 `score(node)` 作为加权的新颖度 × 质量 × 预算项。

4. **沙箱封装。** 每个实验运行 `docker run --network=none --memory=8g --cpus=2 --pids-limit=256 --read-only`（或等效的 E2B 策略）。种子写入沙箱；输出以只读方式挂载出来。

5. **规划—执行—验证循环。** `plan` 提出子节点。`execute` 运行沙箱，捕获日志和指标。`verify` 对指标运行单元检查（损失是否下降了？消融是否隔离了效果？）。失败节点将失败原因存储在树上。

6. **写作器。** 预算耗尽后，选择最优分支。用 matplotlib 渲染图表。用 Claude Opus 4.7 在上下文中包含分支追踪信息生成 LaTeX 草稿。编译。将编译好的 PDF 反馈给 Opus 4.7 视觉模式进行评审。迭代。

7. **评审集成。** 五个评委用 NeurIPS 风格的评分标准对草稿进行（新颖度、严谨性、清晰度、可复现性、影响力）评分。如果均值 < 4.0/5，附上评审意见返回写作器。3 次改写后强制停止。

8. **红队。** 构建或集成一套针对沙箱的对抗性任务：fork 炸弹、网络外泄尝试、文件系统逃逸、LLM 写的 shell 元字符。确认全部被阻止。撰写调查结果报告。

9. **可复现性。** 每篇论文随附其树搜索追踪 JSON、种子、W&B 运行链接、沙箱配置，以及端到端复现的 README。

## 使用示例

```
$ ai-scientist run --seed "attention sparsity in sub-1B transformers" --budget 30
[lit]    50 papers, digest in 12s
[tree]   expanded 8 nodes, budget 12/30
[exec]   node #3 sparsity=top-8, loss=2.83 (best so far)
[exec]   node #6 sparsity=top-4, loss=3.12 (worse)
[exec]   ...
[tree]   chose branch rooted at node #3 (novelty 0.62, quality 0.81)
[write]  LaTeX draft v1 complete
[vision] critique: figure 2 legend too small, claim-evidence ok
[write]  draft v2 after 3 edits
[review] mean 4.2/5 (novelty 3.9, rigor 4.3, clarity 4.1, repro 4.5, impact 4.2)
[done]   paper.pdf + review.md + trace.json     $28.40 spent
```

## 交付物

`outputs/skill-ai-scientist.md` 是交付物。给定一个种子想法 + 领域 + $30 预算，它运行完整流水线并输出一篇可评审的论文以及可复现性包。

| 权重 | 评分标准 | 衡量方式 |
|:-:|---|---|
| 25 | 论文质量 | 对照已发表研讨会论文的盲审评分 |
| 20 | 实验严谨性 | 基线、种子、消融实验；每个论断都有结果表格中的单元格支撑 |
| 20 | 成本与计算纪律 | 每篇论文 $30 上限执行，Langfuse 追踪 |
| 20 | 安全性 | 沙箱红队通过；网络策略和终止开关已验证 |
| 15 | 可复现性 | 相同种子的一键复现再现论文 |
| **100** | | |

## 练习题

1. 在同一领域针对三个不同的种子想法运行流水线。比较树搜索中哪些部分重叠。找出重复浪费的计算。

2. 在估计超过 $5 的节点执行前添加人在回路门控。测量总成本下降了多少。

3. 将评审集成替换为单个评委。在一组已知的劣质论文保留集上测量误接受率。

4. 引入网络外泄红队测试：智能体编写尝试 `curl` 外部地址的代码。确认 `--network=none` 策略阻止了它。记录该尝试。

5. 将你的树搜索与扁平随机基线（相同预算，无扩展策略）进行比较。报告新颖度 × 质量的提升。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Tree search（树搜索） | "AB-MCTS 风格扩展" | 用新颖度×质量×预算得分对实验节点进行最优先探索 |
| Sandbox（沙箱） | "实验隔离" | 无网络、有界 CPU/内存、固定种子、只读输入的容器 |
| Vision critique（视觉评审） | "渲染后阅读" | 将论文编译为 PDF，再反馈给 VLM 进行布局和论断-证据评审 |
| Reviewer ensemble（评审集成） | "自动化同行评审" | 多个 LLM 评委用 NeurIPS 评分标准对论文评分；加权聚合把关流水线 |
| Novelty score（新颖度得分） | "这是新东西吗？" | 惩罚与 50 篇文献缓存相近内容的启发式指标 |
| Cost ceiling（成本上限） | "$ 预算" | 每篇论文总支出的硬性上限；Langfuse 计数器 + 预估运行成本 |
| Red team（红队） | "沙箱逃逸审计" | 若策略有漏洞则会逃逸沙箱的对抗性任务 |

## 延伸阅读

- [Sakana AI-Scientist-v2 仓库](https://github.com/SakanaAI/AI-Scientist-v2) — 参考生产级科研智能体
- [Sakana AI-Scientist-v1 论文（arXiv:2408.06292）](https://arxiv.org/abs/2408.06292) — 原始方法论文
- [ShinkaEvolve（Sakana ICLR 2026）](https://sakana.ai) — 演化扩展
- [Agent Laboratory（AMD）](https://github.com/SamuelSchmidgall/AgentLaboratory) — 多角色科研实验室框架
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/) — 参考编排层
- [Semantic Scholar Graph API](https://api.semanticscholar.org/) — 文献搜索
- [E2B sandboxes](https://e2b.dev) — 参考实验隔离
- [NeurIPS 评审指南](https://neurips.cc/Conferences/2026/Reviewer-Guidelines) — 评审集成所编码的评分标准
