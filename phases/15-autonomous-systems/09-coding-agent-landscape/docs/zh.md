# 自主编码智能体全景（2026）

> SWE-bench Verified 在不到三年的时间里从 4% 上升到 80.9%。同样的 Claude Sonnet 4.5 在 SWE-agent v1 上得分 43.2%，在 Cline 自主模式下得分 59.8%——围绕模型的脚手架现在与模型本身同等重要。OpenHands（前身为 OpenDevin）是最活跃的 MIT 许可平台，其 CodeAct 循环直接在沙盒中执行 Python 操作，而不是 JSON 工具调用。标题数字隐藏了一个方法论问题：SWE-bench Verified 的 500 个任务中有 161 个只需要 1-2 行更改，而对于同样的前沿模型，SWE-bench Pro（10+ 行任务）的得分在 23-59% 之间。

**类型：** 学习
**编程语言：** Python（标准库，CodeAct 与 JSON 工具调用比较）
**前置知识：** Phase 14 · 07（工具使用）、Phase 15 · 01（长时间智能体）
**预计时间：** 约 45 分钟

## 问题背景

"哪个编码智能体最好"是个错误的问题。正确的问题是：在与我的工作匹配的任务分布上，使用我将在生产中运行的脚手架，我能得到什么样的端到端可靠性？

在 2022 年到 2026 年间，该领域认识到脚手架——检索层、规划器、沙盒、编辑-验证循环、反馈格式——是承重的。Claude Sonnet 4.5 在 SWE-agent v1 上的 SWE-bench Verified 得分为 43.2%；同一模型在 Cline 的自主脚手架内得分 59.8%。相同权重，16.6 个绝对百分点的差异。基础模型是一个组件；循环是产品。

伴随的问题是基准饱和掩盖了回归。SWE-bench Verified 接近饱和，简单任务尾部（500 个任务中有 161 个需要 ≤2 行）拉高了顶部分数。真实世界的质量在像 SWE-bench Pro（10+ 行更改）这样的分布上得到更好的测量，在那里同样的领先者仍然在 23-59% 之间。

## 核心概念

### SWE-bench，一段话概括

SWE-bench（Jimenez 等人）获取带有真实补丁的真实 GitHub Issue，并要求智能体生成使测试套件通过的补丁。SWE-bench Verified（OpenAI，2024 年）是一个人工策划的 500 任务子集，删除了模糊和破损的任务。SWE-bench Pro 是更难的后继者——需要 10+ 行更改的任务，当前前沿智能体在 23-59% 之间。

### 2022 → 2026 曲线实际上展示了什么

- **2022 年**：研究模型在原始 SWE-bench 上约 4%。
- **2024 年**：GPT-4 + Devin 风格脚手架约 14%；SWE-agent 约 12%。
- **2025 年**：Claude 3.5/3.7 Sonnet 在 Aider 和 SWE-agent 中进入 40-55% 范围。
- **2026 年**：Claude Sonnet 4.5 和前沿竞争者在 SWE-bench Verified 上达到 70-80%+。Epoch AI 的排行榜实时跟踪这一点。

斜率来自三个复合来源：更好的基础模型、更好的脚手架（CodeAct、反思、验证器循环）和更好的基准（Verified 消除噪声）。

### CodeAct 与 JSON 工具调用

OpenHands（All-Hands-AI，arXiv:2407.16741，前身为 OpenDevin）采取了一个特定的架构赌注：模型不是发出主机解码和执行的 JSON 工具调用，而是发出 Python 代码，Jupyter 风格的内核在沙盒中运行它。智能体可以在一次操作中循环文件、链接工具并捕获自己的异常。

权衡：

- **JSON 工具调用**：每次操作是一轮；易于审计；有限的可组合性；默认安全，因为每次调用都经过明确的验证器。
- **CodeAct**：一次操作可以是整个程序；可组合；需要加固的沙盒（OpenHands 使用 Docker 隔离）；故障模式包括沙盒运行时允许的任何内容。

两种架构都在生产中。CodeAct 在开放平台中占主导地位（OpenHands、smolagents）。JSON 工具调用在提供商控制执行器的托管服务中仍然占主导地位（Anthropic Managed Agents、OpenAI Assistants）。

### 2026 年格局中的脚手架

| 脚手架 | 许可证 | 执行模型 | 显著属性 |
|--------|--------|---------|---------|
| OpenHands（OpenDevin） | MIT | Docker 中的 CodeAct | 最活跃的开放平台；事件流可回放 |
| SWE-agent | MIT | 智能体-计算机接口（ACI） | 第一个端到端 SWE-bench 脚手架 |
| Aider | Apache-2 | 本地仓库中的编辑-差异 | 最小脚手架，强大的回归稳定性 |
| Cline | Apache-2 | 带工具策略的 VS Code 智能体 | Sonnet 4.5 上评分最高的开放脚手架 |
| Devin（Cognition） | 专有 | 托管 VM + 规划器 | 第一个"AI 软件工程师"产品类别 |
| Claude Code | 专有 | 权限模式 + 例程 | 第 10 课详细介绍智能体循环 |

### 为什么脚手架占主导地位

编码运行是一个长时间轨迹（第 1 课）。可靠性在各步骤中复合。脚手架获得分数的三个地方：

1. **检索**：找到正确的文件是静默的瓶颈。SWE-agent 的 ACI、OpenHands 的文件索引和 Aider 的仓库映射都在攻克这个问题。
2. **验证器循环**：运行测试、读取堆栈跟踪和重新尝试在 SWE-bench 上有 10+ 分的增量。
3. **失败遏制**：出错时回滚的沙盒防止复合损害。有和没有验证器循环的同一模型看起来像两个不同的产品。

### 基准饱和与真实分布

OpenHands 作者和 Epoch AI 都指出 SWE-bench Verified 有一个简单的尾部：500 个任务中有 161 个只需要 1-2 行更改。高分部分由这个尾部驱动。SWE-bench Pro 限制为 10+ 行更改，即使对于前沿系统也返回 23-59% 范围内的分数。你的生产分布几乎肯定比 Verified 更接近 Pro。

选择智能体的含义：在你自己的 bug 积压的类似 Pro 子集上运行。重要的分数是代表你实际发布内容的任务上的分数。

## 动手实践

`code/main.py` 在固定的小型任务分布上比较两个玩具智能体脚手架：

1. 每轮采取一个操作的 **JSON 工具调用**脚手架。
2. 每次操作可以发出小型 Python 片段的 **CodeAct** 脚手架。

两者都使用存根"模型"（确定性规则），因此比较将脚手架与模型质量分离开来。输出显示 CodeAct 脚手架以更少的轮次解决更多的任务，但代价是每次操作的爆炸半径更大。

## 产出技能

`outputs/skill-scaffold-audit.md` 帮助你在采用之前审计提议的编码智能体脚手架：检索质量、验证器存在、沙盒隔离和基准到分布的契合度。

## 练习

1. 运行 `code/main.py`。每个脚手架在同一任务集上需要多少轮？每个的每次操作爆炸半径是多少？

2. 阅读 OpenHands 论文（arXiv:2407.16741）。论文认为 CodeAct 在复杂任务上胜过 JSON 工具调用。找出论文承认的一个故障模式，并写一句话说明该模式何时会在生产中占主导地位。

3. 从你的 bug 积压中选择一个需要在两个文件上进行 10+ 行更改的任务。估计前沿模型在 (a) JSON 工具调用和 (b) CodeAct 下的端到端成功概率。为差距辩护。

4. SWE-bench Verified 有 161 个单文件、1-2 行任务。构建一个排除它们的分数。排行榜如何变化？

5. 阅读"介绍 SWE-bench Verified"（OpenAI）。解释用于删除模糊任务的具体方法，并命名策展会遗漏的一个类别。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| SWE-bench | "编码基准" | 带真实补丁和测试套件的真实 GitHub Issue |
| SWE-bench Verified | "已清理子集" | 500 个人工策划任务，存在简单尾部 |
| SWE-bench Pro | "更难子集" | 10+ 行更改；前沿在 23-59% |
| CodeAct | "代码即操作" | 智能体发出 Python；Jupyter 风格内核在沙盒中执行 |
| JSON 工具调用 | "函数调用" | 每次操作是执行前经过验证的结构化 JSON 有效负载 |
| 脚手架 | "智能体框架" | 基础模型周围的检索 + 规划器 + 执行器 + 验证器循环 |
| ACI（智能体-计算机接口） | "SWE-agent 的格式" | 为 LLM 人机工程学设计的命令集，而不是人类 shell |
| 验证器循环 | "测试和重试" | 运行测试、读取输出、修改补丁；最大的非模型可靠性增益 |

## 延伸阅读

- [Jimenez 等人 — SWE-bench](https://www.swebench.com/) — 原始基准和方法论
- [OpenAI — 介绍 SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — 如何构建策划子集
- [Wang 等人 — OpenHands: An Open Platform for AI Software Developers](https://arxiv.org/abs/2407.16741) — CodeAct 架构和事件流设计
- [Epoch AI — SWE-bench 排行榜](https://epoch.ai/benchmarks) — 实时跟踪的分数
- [Anthropic — 测量智能体自主性](https://www.anthropic.com/research/measuring-agent-autonomy) — 长时间编码智能体可靠性框架
