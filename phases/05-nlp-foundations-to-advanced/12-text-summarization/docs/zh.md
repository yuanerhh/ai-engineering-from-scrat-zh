# 文本摘要

> 抽取式系统告诉你文档说了什么，生成式系统告诉你作者意图什么。不同的任务，不同的陷阱。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 5 · 02（词袋模型 + TF-IDF）、Phase 5 · 11（机器翻译）
**时长：** 约 75 分钟

## 问题背景

一篇 2000 字的新闻出现在你的订阅中。你需要 120 字来概括它。你可以从文章中挑选三个最重要的句子（抽取式），也可以用自己的话重写内容（生成式）。两者都叫摘要，但完全是不同的问题。

抽取式摘要是一个排序问题。为每个句子打分，返回前 `k` 个。输出始终符合语法，因为是逐字摘录的。风险是遗漏分散在文章各处的内容。

生成式摘要是一个生成问题。Transformer 根据输入生成新文本。输出流畅且压缩，但可能产生源文本中没有的幻觉事实。风险是自信地捏造。

本课构建两者，各自承担其失败模式。

## 核心概念

**抽取式。** 将文章视为图，节点是句子，边是相似度。在图上运行 PageRank（或类似方法）来根据句子与其他一切的连接程度打分。得分最高的句子就是摘要。典型实现是 **TextRank**（Mihalcea 和 Tarau，2004）。

**生成式。** 在文档-摘要对上微调 Transformer 编码器-解码器（BART、T5、Pegasus）。推理时，模型读取文档并通过交叉注意力逐 token 生成摘要。Pegasus 特别使用了间隔句子预训练目标，使其在没有太多微调的情况下就能很好地进行摘要。

用 **ROUGE**（Recall-Oriented Understudy for Gisting Evaluation，面向召回的摘要评估）进行评估。ROUGE-1 和 ROUGE-2 评分一元组和二元组重叠。ROUGE-L 评分最长公共子序列。越高越好，40 ROUGE-L 是"好"，50 是"出色"。每篇论文都报告全部三个。使用 `rouge-score` 包。

## 动手实现

### 步骤一：TextRank（抽取式）

```python
import math
import re
from collections import Counter


def sentence_split(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())


def similarity(s1, s2):
    w1 = Counter(s1.lower().split())
    w2 = Counter(s2.lower().split())
    intersection = sum((w1 & w2).values())
    denom = math.log(len(w1) + 1) + math.log(len(w2) + 1)
    if denom == 0:
        return 0.0
    return intersection / denom


def textrank(text, top_k=3, damping=0.85, iterations=50, epsilon=1e-4):
    sentences = sentence_split(text)
    n = len(sentences)
    if n <= top_k:
        return sentences

    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                sim[i][j] = similarity(sentences[i], sentences[j])

    scores = [1.0] * n
    for _ in range(iterations):
        new_scores = [1 - damping] * n
        for i in range(n):
            total_out = sum(sim[i]) or 1e-9
            for j in range(n):
                if sim[i][j] > 0:
                    new_scores[j] += damping * sim[i][j] / total_out * scores[i]
        if max(abs(s - ns) for s, ns in zip(scores, new_scores)) < epsilon:
            scores = new_scores
            break
        scores = new_scores

    ranked = sorted(range(n), key=lambda k: scores[k], reverse=True)[:top_k]
    ranked.sort()
    return [sentences[i] for i in ranked]
```

两点值得说明：相似度函数使用对数归一化词语重叠，这是原始 TextRank 变体。TF-IDF 向量的余弦相似度也可以用。阻尼因子 0.85 和迭代次数是 PageRank 的默认值。

### 步骤二：使用 BART 进行生成式摘要

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """（长篇新闻文章文本）"""

summary = summarizer(article, max_length=120, min_length=60, do_sample=False)
print(summary[0]["summary_text"])
```

BART-large-CNN 在 CNN/DailyMail 语料库上微调。开箱即用产生新闻风格的摘要。对于其他领域（科学论文、对话、法律），使用相应的 Pegasus 检查点或在你的目标数据上微调。

### 步骤三：ROUGE 评估

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
scores = scorer.score(reference_summary, generated_summary)
print({k: round(v.fmeasure, 3) for k, v in scores.items()})
```

始终使用词干提取。否则，"running"和"run"会被视为不同的词，ROUGE 会低估匹配。

### 超越 ROUGE（2026 年摘要评估）

ROUGE 主导摘要指标已经二十年，到 2026 年单独使用已不足够。一项大规模的 NLG 论文元分析显示：

- **BERTScore**（上下文嵌入相似度）在 2023 年前获得了关注，现在在大多数摘要论文中与 ROUGE 一起报告。
- **BARTScore** 将评估视为生成：根据预训练 BART 在给定源文本下生成摘要的可能性打分。
- **MoverScore**（上下文嵌入上的地球搬运距离）在 2025 年摘要基准中名列前茅，因为它比 ROUGE 更好地捕捉语义重叠。
- **FactCC** 和**基于 QA 的忠实度**在 2021-2023 年很常见，现在经常被 **G-Eval** 替代（一个 GPT-4 提示链，用思维链推理评分连贯性、一致性、流畅性和相关性）。
- **G-Eval** 和类似的 LLM 评审方法在评分标准设计良好时与人工判断约 80% 一致。

生产建议：为历史比较报告 ROUGE-L，为语义重叠报告 BERTScore，为连贯性和真实性报告 G-Eval。在信任用于生产数据之前，用 50-100 个人工标注摘要进行校准。

### 步骤四：真实性问题

生成式摘要容易产生幻觉。抽取式摘要幻觉风险低得多，因为输出是逐字摘录的，尽管如果源句子被断章取义、过时或顺序错误，也可能产生误导。这是生产系统对合规相关内容仍然倾向于抽取式方法的最主要原因。

需要命名的幻觉类型：

- **实体替换（Entity swap）。** 源文说"John Smith"，摘要说"John Brown"。
- **数字漂移（Number drift）。** 源文说"25,000"，摘要说"25 million"。
- **极性翻转（Polarity flip）。** 源文说"拒绝了提案"，摘要说"接受了提案"。
- **事实发明（Fact invention）。** 源文未提及 CEO，摘要说 CEO 批准了。

有效的评估方法：

- **FactCC。** 在源句子和摘要句子之间的蕴含关系上训练的二元分类器。预测事实性/非事实性。
- **基于 QA 的真实性。** 对 QA 模型提问，答案在源文中。如果摘要支持不同的答案，则标记。
- **实体级 F1。** 比较源文和摘要中的命名实体。仅出现在摘要中的实体是可疑的。

对于任何真实性至关重要的面向用户内容（新闻、医学、法律、金融），抽取式是更安全的默认选择。生成式需要在循环中进行真实性检查。

## 生产使用

2026 年的技术栈：

| 使用场景 | 推荐 |
|---------|------|
| 新闻，3-5 句摘要，英语 | `facebook/bart-large-cnn` |
| 科学论文 | `google/pegasus-pubmed` 或调优的 T5 |
| 多文档、长篇 | 任何具有 32k+ 上下文的 LLM，使用提示 |
| 对话摘要 | `philschmid/bart-large-cnn-samsum` |
| 抽取式，低幻觉风险 | TextRank 或 `sumy` 的 LSA / LexRank |

当计算不是限制时，具有长上下文的 LLM 在 2026 年通常优于专业模型。权衡是成本和可重现性；专业模型给出更一致的输出。

## 上手实践

将以下内容保存为 `outputs/skill-summary-picker.md`：

```markdown
---
name: summary-picker
description: Pick extractive or abstractive, named library, factuality check.
version: 1.0.0
phase: 5
lesson: 12
tags: [nlp, summarization]
---

Given a task (document type, compliance requirement, length, compute budget), output:

1. Approach. Extractive or abstractive. Explain in one sentence why.
2. Starting model / library. Name it. `sumy.TextRankSummarizer`, `facebook/bart-large-cnn`, `google/pegasus-pubmed`, or an LLM prompt.
3. Evaluation plan. ROUGE-1, ROUGE-2, ROUGE-L (use rouge-score with stemming). Plus factuality check if abstractive.
4. One failure mode to probe. Entity swap is the most common in abstractive news summarization; flag samples where source entities do not appear in summary.

Refuse abstractive summarization for medical, legal, financial, or regulated content without a factuality gate. Flag input over the model's context window as needing chunked map-reduce summarization (not just truncation).
```

## 练习

1. **简单。** 对 5 篇新闻文章运行 TextRank。将前 3 个句子与参考摘要比较。测量 ROUGE-L。你应该在 CNN/DailyMail 风格文章上看到 30-45 的 ROUGE-L。
2. **中等。** 实现实体级真实性检查：用 spaCy 从源文和摘要中提取命名实体，计算源文实体在摘要中的召回率和摘要实体对源文的精确率。高精确率低召回率意味着安全但简洁；低精确率意味着有幻觉实体。
3. **困难。** 在 50 篇 CNN/DailyMail 文章上比较 BART-large-CNN 和 LLM（Claude 或 GPT-4）。报告 ROUGE-L、真实性（按实体 F1）和每篇摘要的成本。记录每种方法在哪里胜出。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 抽取式（Extractive） | 挑选句子 | 逐字摘录源文中的句子，不会产生幻觉。 |
| 生成式（Abstractive） | 改写 | 根据源文生成新文本，可能产生幻觉。 |
| ROUGE | 摘要指标 | 系统输出与参考之间的 n-gram / LCS 重叠。 |
| TextRank | 基于图的抽取式 | 句子相似度图上的 PageRank。 |
| 真实性（Factuality） | 是否正确 | 摘要声明是否得到源文支持。 |
| 幻觉（Hallucination） | 编造内容 | 摘要中源文不支持的内容。 |

## 延伸阅读

- [Mihalcea and Tarau (2004). TextRank: Bringing Order into Texts](https://aclanthology.org/W04-3252/) — 抽取式经典论文
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461) — BART 论文
- [Zhang et al. (2019). PEGASUS: Pre-training with Extracted Gap-sentences](https://arxiv.org/abs/1912.08777) — Pegasus 和间隔句子目标
- [Lin (2004). ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/) — ROUGE 论文
- [Maynez et al. (2020). On Faithfulness and Factuality in Abstractive Summarization](https://arxiv.org/abs/2005.00661) — 真实性全景论文
