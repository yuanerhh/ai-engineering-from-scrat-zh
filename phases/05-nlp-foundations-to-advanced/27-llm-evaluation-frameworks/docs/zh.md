# LLM 评估——RAGAS、DeepEval、G-Eval

> 精确匹配（Exact Match）和 F1 无法识别语义等价。人工审阅无法规模化。LLM 即评判（LLM-as-judge）是生产环境的答案——但前提是有足够的校准让你信任这个数字。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 5 · 13（问答系统）、Phase 5 · 14（信息检索）
**时长：** 约 75 分钟

## 问题背景

你的 RAG 系统回答："June 29th, 2007."
黄金参考答案是："June 29, 2007."
精确匹配得 0 分。F1 得约 75%。人类会给 100 分。

现在乘以 10,000 个测试案例。再乘以每一次对检索器、分块、提示或模型的更改。你需要一个能理解语义、可以低成本规模化运行、不会对回归撒谎、并能暴露正确失败模式的评估器。

2026 年有三个框架解决这个问题：

- **RAGAS。** 检索增强生成评估（Retrieval-Augmented Generation ASsessment）。四个 RAG 指标（忠实度、答案相关性、上下文精确率、上下文召回率），后端为 NLI + LLM 评判。有研究支撑，轻量。
- **DeepEval。** 面向 LLM 的 pytest。G-Eval、任务完成度、幻觉、偏见指标。原生支持 CI/CD。
- **G-Eval。** 一种方法（也是 DeepEval 的一个指标）：带思维链的 LLM 即评判，自定义标准，0-1 评分。

三者都依赖 LLM 即评判。本课建立对这种方法及其周边信任层的直觉。

## 核心概念

![四个评估维度，LLM 即评判架构](../assets/llm-evaluation.svg)

**LLM 即评判（LLM-as-judge）。** 用一个 LLM 替代静态指标，根据评分标准对输出打分。给定 `(查询, 上下文, 答案)`，提示一个评判 LLM："对忠实度打 0-1 分。"返回分数。

为什么有效：LLM 以极低成本近似人类判断。GPT-4o-mini 每个评分案例约 0.003 美元，每次 1000 个样本的回归评估不到 5 美元。

为什么会悄无声息地失效：

1. **评判偏差（Judge bias）。** 评判者偏爱更长的答案、来自同一模型家族的答案、与提示风格匹配的答案。
2. **JSON 解析失败。** 格式错误的 JSON → NaN 分数 → 静默地从聚合中排除。RAGAS 用户深知此痛。用 try/except + 显式失败模式来防守。
3. **跨模型版本漂移。** 升级评判模型会改变所有指标。锁定评判模型 + 版本号。

**RAG 四指标。**

| 指标 | 问的问题 | 后端 |
|------|---------|------|
| 忠实度（Faithfulness） | 答案中的每个声明都来自检索到的上下文吗？ | 基于 NLI 的蕴含 |
| 答案相关性（Answer relevance） | 答案是否回应了问题？ | 从答案生成假设问题；与真实问题比较 |
| 上下文精确率（Context precision） | 检索到的块中有多少比例是相关的？ | LLM 评判 |
| 上下文召回率（Context recall） | 检索是否返回了所需的所有内容？ | 根据黄金答案做 LLM 评判 |

**G-Eval。** 定义自定义标准："答案是否引用了正确的来源？"框架自动展开为思维链评估步骤，然后打 0-1 分。适合 RAGAS 不覆盖的领域专属质量维度。

**校准（Calibration）。** 在有人工标签的相关性分析之前，永远不要信任原始评判分数。运行 100 个人工标注的样本，绘制评判分数 vs 人工分数，计算 Spearman ρ。如果 ρ < 0.7，你的评分标准需要改进。

## 动手实现

### 步骤一：用 NLI 计算忠实度（RAGAS 风格）

```python
from typing import Callable
from transformers import pipeline

nli = pipeline("text-classification",
               model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
               top_k=None)

# `llm` 是任意可调用对象：提示字符串 -> 生成字符串
# 例如：llm = lambda p: client.messages.create(model="claude-haiku-4-5", ...).content[0].text
LLM = Callable[[str], str]


def atomic_claims(answer: str, llm: LLM) -> list[str]:
    prompt = f"""Break this answer into simple factual claims (one per line):
{answer}
"""
    return llm(prompt).splitlines()


def faithfulness(answer: str, context: str, llm: LLM) -> float:
    claims = atomic_claims(answer, llm)
    if not claims:
        return 0.0
    supported = 0
    for claim in claims:
        result = nli({"text": context, "text_pair": claim})[0]
        entail = next((s for s in result if s["label"] == "entailment"), None)
        if entail and entail["score"] > 0.5:
            supported += 1
    return supported / len(claims)
```

将答案分解为原子声明。用 NLI 对每个声明与检索到的上下文进行蕴含检验。忠实度 = 被支持的比例。

### 步骤二：答案相关性

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# encoder: 任何实现了 .encode(texts, normalize_embeddings=True) -> ndarray 的模型
# 例如：encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")

def answer_relevance(question: str, answer: str, encoder, llm: LLM, n: int = 3) -> float:
    prompt = f"Write {n} questions this answer could be the answer to:\n{answer}"
    generated = [line for line in llm(prompt).splitlines() if line.strip()][:n]
    if not generated:
        return 0.0
    q_emb = np.asarray(encoder.encode([question], normalize_embeddings=True)[0])
    g_embs = np.asarray(encoder.encode(generated, normalize_embeddings=True))
    sims = [float(q_emb @ g_emb) for g_emb in g_embs]
    return sum(sims) / len(sims)
```

如果答案暗示的问题与被问的问题不同，相关性就会下降。

### 步骤三：G-Eval 自定义指标

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

metric = GEval(
    name="Correctness",
    criteria="The answer should be factually accurate and match the expected output.",
    evaluation_steps=[
        "Read the expected output.",
        "Read the actual output.",
        "List factual claims in the actual output.",
        "For each claim, mark supported or unsupported by the expected output.",
        "Return score = fraction supported.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
)

test = LLMTestCase(input="When was the first iPhone released?",
                   actual_output="June 29th, 2007.",
                   expected_output="June 29, 2007.")
metric.measure(test)
print(metric.score, metric.reason)
```

评估步骤就是评分标准。显式步骤比隐式的"打 0-1 分"提示更稳定。

### 步骤四：CI 门控

```python
import deepeval
from deepeval.metrics import FaithfulnessMetric, ContextualRelevancyMetric


def test_rag_system():
    cases = load_regression_cases()
    faith = FaithfulnessMetric(threshold=0.85)
    rel = ContextualRelevancyMetric(threshold=0.7)
    for case in cases:
        faith.measure(case)
        assert faith.score >= 0.85, f"faithfulness regression on {case.id}"
        rel.measure(case)
        assert rel.score >= 0.7, f"relevancy regression on {case.id}"
```

作为 pytest 文件发布。在每个 PR 上运行。在回归时阻止合并。

### 步骤五：从零构建简单评估器

见 `code/main.py`。仅用标准库实现忠实度（答案声明与上下文的重叠）和相关性（答案 token 与问题 token 的重叠）的近似版本。不适合生产，但展示了任务的形态。

## 常见坑

- **没有校准。** 与人工标签相关性为 0.3 的评判器不过是噪声。上线前必须进行校准。
- **自我评估（Self-evaluation）。** 使用同一个 LLM 既生成又评判会使分数虚高 10-20%。使用不同模型家族作为评判器。
- **成对评判中的位置偏差。** 评判者偏爱呈现在前面的选项。始终随机化顺序并两次运行。
- **整体聚合掩盖失败。** 平均分 0.85 经常掩盖 5% 的灾难性失败。始终检查底部分位数。
- **黄金数据集腐化（Golden dataset rot）。** 未版本化的评估集随时间漂移，会破坏纵向比较。对数据集的每次变更都打标签。
- **LLM 成本。** 规模化时，评判调用主导成本。使用满足校准阈值的最廉价模型：GPT-4o-mini、Claude Haiku、Mistral-small。

## 生产使用

2026 年技术栈：

| 使用场景 | 框架 |
|---------|------|
| RAG 质量监控 | RAGAS（4 个指标） |
| CI/CD 回归门控 | DeepEval + pytest |
| 自定义领域标准 | DeepEval 内的 G-Eval |
| 线上流量实时监控 | RAGAS 无参考模式 |
| 人在回路抽查 | LangSmith 或 Phoenix 配合注释 UI |
| 红队测试 / 安全评估 | Promptfoo + DeepEval |

典型技术栈：RAGAS 用于监控，DeepEval 用于 CI，G-Eval 用于新维度。同时运行三者，它们会有有价值的分歧。

## 上手实践

将以下内容保存为 `outputs/skill-eval-architect.md`：

```markdown
---
name: eval-architect
description: Design an LLM evaluation plan with calibrated judge and CI gates.
version: 1.0.0
phase: 5
lesson: 27
tags: [nlp, evaluation, rag]
---

Given a use case (RAG / agent / generative task), output:

1. Metrics. Faithfulness / relevance / context-precision / context-recall + any custom G-Eval metrics with criteria.
2. Judge model. Named model + version, rationale for cost vs accuracy.
3. Calibration. Hand-labeled set size, target Spearman rho vs human > 0.7.
4. Dataset versioning. Tag strategy, change log, stratification.
5. CI gate. Thresholds per metric, regression-window logic, bottom-quantile alert.

Refuse to rely on a judge untested against ≥50 human-labeled examples. Refuse self-evaluation (same model generates + judges). Refuse aggregate-only reporting without bottom-10% surfacing. Flag any pipeline where judge upgrade lands without parallel baseline eval.
```

## 练习

1. **简单。** 在 10 个已知含有幻觉的 RAG 样本上使用 RAGAS。验证忠实度指标能捕获每一个幻觉。
2. **中等。** 人工标注 50 个问答答案的正确性（0-1）。用 G-Eval 打分。测量评判分数与人工分数之间的 Spearman ρ。
3. **困难。** 用 DeepEval 构建 pytest CI 门控。故意让检索器退化。验证门控触发失败。对最低 10% 的分数添加底部分位数警报。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| LLM 即评判（LLM-as-judge） | 用 LLM 打分 | 提示一个评判模型根据评分标准给输出打 0-1 分。 |
| RAGAS | RAG 指标库 | 开源评估框架，包含 4 个无参考 RAG 指标。 |
| 忠实度（Faithfulness） | 答案有根据吗？ | 检索上下文蕴含的答案声明比例。 |
| 上下文精确率（Context precision） | 检索到的块是否相关？ | Top-K 块中实际有用的比例。 |
| 上下文召回率（Context recall） | 检索找到所需内容了吗？ | 检索块支持的黄金答案声明比例。 |
| G-Eval | 自定义 LLM 评判 | 评分标准 + 思维链评估步骤 + 0-1 分数。 |
| 校准（Calibration） | 信任但需验证 | 评判分数与人工分数之间的 Spearman 相关性。 |

## 延伸阅读

- [Es et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) — RAGAS 论文
- [Liu et al. (2023). G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://arxiv.org/abs/2303.16634) — G-Eval 论文
- [DeepEval docs](https://deepeval.com/docs/metrics-introduction) — 开源生产技术栈
- [Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — 偏差、校准、局限性
- [MLflow GenAI Scorer](https://mlflow.org/blog/third-party-scorers) — 集成 RAGAS、DeepEval、Phoenix 的统一框架
