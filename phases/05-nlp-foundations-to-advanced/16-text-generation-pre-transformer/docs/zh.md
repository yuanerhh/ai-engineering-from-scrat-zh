# Transformer 之前的文本生成——N-gram 语言模型

> 如果一个词让模型感到意外，说明模型不好。困惑度将意外程度量化为数字。平滑使其保持有限。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 5 · 01（文本处理）、Phase 2 · 14（朴素贝叶斯）
**时长：** 约 45 分钟

## 问题背景

在 Transformer 出现之前，在 RNN 出现之前，在词嵌入出现之前，语言模型通过统计某个词跟在前 `n-1` 个词后面的频率来预测下一个词。统计"the cat"→"sat"47次、"the cat"→"jumped"12次、"the cat"→"refrigerator"0次，归一化后得到概率分布。

这就是 N-gram 语言模型。从 1980 年到 2015 年，它运行在每一个语音识别系统、每一个拼写检查器和每一个基于短语的机器翻译系统中。当你需要廉价的设备端语言建模时，它至今仍在运行。

有趣的问题是如何处理未见过的 N-gram。基于原始计数的模型会对它没有见过的任何内容赋予零概率，这是灾难性的，因为句子很长，几乎每个长句子都包含至少一个未见过的序列。五十年的平滑技术研究解决了这个问题。Kneser-Ney 平滑是其成果，现代深度学习继承了其经验传统。

## 核心概念

![N-gram 模型：计数、平滑、生成](../assets/ngram.svg)

**N-gram 概率：** `P(w_i | w_{i-n+1}, ..., w_{i-1})`。固定 `n`（通常三元组取 3，四元组取 4），从计数中计算：

```text
P(w | context) = count(context, w) / count(context)
```

**零计数问题。** 任何在训练中未见过的 N-gram 概率为零。2007 年对 Brown 语料库的研究发现，即使是四元组模型，保留集中 30% 的四元组在训练中未出现过。不使用平滑就无法在任何真实文本上评估。

**平滑方案，按复杂度排序：**

1. **Laplace（加一）。** 对每个计数加 1。简单，但在罕见事件上效果很差。
2. **Good-Turing。** 根据频率的频率，将概率质量从高频事件重新分配给未见事件。
3. **插值（Interpolation）。** 用可调权重组合 N-gram、(N-1)-gram 等的估计。
4. **回退（Backoff）。** 如果 N-gram 计数为零，回退到 (N-1)-gram。Katz 回退对此进行了归一化。
5. **绝对折扣（Absolute discounting）。** 从所有计数中减去固定折扣 `D`，重新分配给未见事件。
6. **Kneser-Ney。** 绝对折扣加上聪明的低阶模型选择：使用**延续概率**（一个词出现在多少个上下文中）而非原始频率。

Kneser-Ney 的洞见很深刻。"San Francisco"是常见的二元组。单个词"Francisco"大多出现在"San"之后。朴素绝对折扣给"Francisco"很高的一元组概率（因为计数很高）。而 Kneser-Ney 注意到"Francisco"只出现在一个上下文中，因此相应地降低其延续概率。结果：以"Francisco"结尾的新二元组获得了恰当的低概率。

**评估：困惑度（Perplexity）。** 在保留测试集上每个词的平均负对数似然的指数。越低越好。困惑度为 100 意味着模型的困惑程度相当于从 100 个词中均匀随机选择。

```text
perplexity = exp(- (1/N) * Σ log P(w_i | context_i))
```

## 动手实现

### 步骤一：三元组计数

```python
from collections import Counter, defaultdict


def train_ngram(corpus_tokens, n=3):
    ngrams = Counter()
    contexts = Counter()
    for sentence in corpus_tokens:
        padded = ["<s>"] * (n - 1) + sentence + ["</s>"]
        for i in range(len(padded) - n + 1):
            ctx = tuple(padded[i:i + n - 1])
            word = padded[i + n - 1]
            ngrams[ctx + (word,)] += 1
            contexts[ctx] += 1
    return ngrams, contexts


def raw_probability(ngrams, contexts, context, word):
    ctx = tuple(context)
    if contexts.get(ctx, 0) == 0:
        return 0.0
    return ngrams.get(ctx + (word,), 0) / contexts[ctx]
```

输入是分词后的句子列表，输出是 N-gram 计数和上下文计数。`<s>` 和 `</s>` 是句子边界标记。

### 步骤二：Laplace 平滑

```python
def laplace_probability(ngrams, contexts, vocab_size, context, word):
    ctx = tuple(context)
    numerator = ngrams.get(ctx + (word,), 0) + 1
    denominator = contexts.get(ctx, 0) + vocab_size
    return numerator / denominator
```

对每个计数加 1，能进行平滑，但过度分配了质量给未见事件，也损害了罕见已知事件。

### 步骤三：Kneser-Ney（二元组，插值版）

```python
def kneser_ney_bigram_model(corpus_tokens, discount=0.75):
    unigrams = Counter()
    bigrams = Counter()
    unigram_contexts = defaultdict(set)

    for sentence in corpus_tokens:
        padded = ["<s>"] + sentence + ["</s>"]
        for i, w in enumerate(padded):
            unigrams[w] += 1
            if i > 0:
                prev = padded[i - 1]
                bigrams[(prev, w)] += 1
                unigram_contexts[w].add(prev)

    total_unique_bigrams = sum(len(ctx_set) for ctx_set in unigram_contexts.values())
    continuation_prob = {
        w: len(ctx_set) / total_unique_bigrams for w, ctx_set in unigram_contexts.items()
    }

    context_totals = Counter()
    for (prev, w), count in bigrams.items():
        context_totals[prev] += count

    unique_follow = defaultdict(set)
    for (prev, w) in bigrams:
        unique_follow[prev].add(w)

    def prob(prev, w):
        count = bigrams.get((prev, w), 0)
        denom = context_totals.get(prev, 0)
        if denom == 0:
            return continuation_prob.get(w, 1e-9)
        first_term = max(count - discount, 0) / denom
        lambda_prev = discount * len(unique_follow[prev]) / denom
        return first_term + lambda_prev * continuation_prob.get(w, 1e-9)

    return prob
```

三个关键部分：`continuation_prob` 捕获"这个词出现在多少个不同的上下文中？"（Kneser-Ney 的创新）；`lambda_prev` 是折扣释放的质量，用于加权回退；最终概率是折扣后的主项加上加权的延续项。

### 步骤四：采样生成文本

```python
import random


def generate(prob_fn, vocab, prefix, max_len=30, seed=0):
    rng = random.Random(seed)
    tokens = list(prefix)
    for _ in range(max_len):
        candidates = [(w, prob_fn(tokens[-1], w)) for w in vocab]
        total = sum(p for _, p in candidates)
        r = rng.random() * total
        acc = 0.0
        for w, p in candidates:
            acc += p
            if r <= acc:
                tokens.append(w)
                break
        if tokens[-1] == "</s>":
            break
    return tokens
```

按概率比例采样，每个随机种子产生不同输出。如果需要类似束搜索的输出，在每步取 argmax（贪婪），并加入小的随机性旋钮（温度）。

### 步骤五：困惑度

```python
import math


def perplexity(prob_fn, sentences):
    total_log_prob = 0.0
    total_tokens = 0
    for sentence in sentences:
        padded = ["<s>"] + sentence + ["</s>"]
        for i in range(1, len(padded)):
            p = prob_fn(padded[i - 1], padded[i])
            total_log_prob += math.log(max(p, 1e-12))
            total_tokens += 1
    return math.exp(-total_log_prob / total_tokens)
```

越低越好。Brown 语料库上，调优好的四元组 KN 模型的困惑度约为 140，Transformer 语言模型在同一测试集上达到 15-30。差距约为 10 倍，这就是该领域为何向前演进的原因。

## 生产使用

- **经典 NLP 教学。** 接触平滑、MLE 和困惑度最清晰的方式。
- **KenLM。** 生产级 N-gram 库，在低延迟至关重要的语音和机器翻译系统中用作重评分器。
- **设备端自动补全。** 键盘中的三元组模型，至今仍在使用。
- **基准线。** 在宣称你的神经语言模型效果好之前，先计算 N-gram LM 的困惑度。如果你的 Transformer 未能大幅超越 KN 基准，说明出了问题。

## 上手实践

将以下内容保存为 `outputs/prompt-lm-baseline.md`：

```markdown
---
name: lm-baseline
description: Build a reproducible n-gram language model baseline before training a neural LM.
phase: 5
lesson: 16
---

Given a corpus and target use (next-word prediction, rescoring, perplexity baseline), output:

1. N-gram order. Trigram for general English, 4-gram if corpus is large, 5-gram for speech rescoring.
2. Smoothing. Modified Kneser-Ney is the default; Laplace only for teaching.
3. Library. `kenlm` for production, `nltk.lm` for teaching, roll your own only to learn.
4. Evaluation. Held-out perplexity with consistent tokenization between train and test sets.

Refuse to report perplexity computed with different tokenization between systems being compared — perplexity numbers are comparable only under identical tokenization. Flag OOV rate in test set; KN handles OOV poorly unless you reserve a special <UNK> token during training.
```

## 练习

1. **简单。** 在 1,000 句莎士比亚语料上训练三元组语言模型，生成 20 个句子。它们局部上合理，但整体上不连贯。这是经典演示。
2. **中等。** 在保留的莎士比亚测试集上计算 KN 模型的困惑度，与 Laplace 比较。KN 应该将困惑度降低 30-50%。
3. **困难。** 构建三元组拼写纠正器：给定一个拼写错误的词及其上下文，生成纠正候选，并按语言模型下的上下文概率排序。在 Birkbeck 拼写语料库（公开）上评估。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| N-gram | 词序列 | 由 `n` 个连续 token 组成的序列。 |
| 平滑（Smoothing） | 避免零概率 | 重新分配概率质量，使未见事件获得非零概率。 |
| 困惑度（Perplexity） | 语言模型质量指标 | 保留数据上的 `exp(-平均对数概率)`，越低越好。 |
| 回退（Backoff） | 退回到更短的上下文 | 如果三元组计数为零，使用二元组。Katz 回退将此形式化。 |
| Kneser-Ney | N-gram 最佳平滑方案 | 绝对折扣 + 低阶模型使用延续概率。 |
| 延续概率（Continuation probability） | KN 特有概念 | `P(w)` 按词 `w` 出现的上下文数量加权，而非原始计数。 |

## 延伸阅读

- [Jurafsky and Martin — Speech and Language Processing, Chapter 3 (2026 draft)](https://web.stanford.edu/~jurafsky/slp3/3.pdf) — N-gram 语言模型和平滑的权威论述
- [Chen and Goodman (1998). An Empirical Study of Smoothing Techniques for Language Modeling](https://dash.harvard.edu/handle/1/25104739) — 确定 Kneser-Ney 为最优 N-gram 平滑方案的论文
- [Kneser and Ney (1995). Improved Backing-off for M-gram Language Modeling](https://ieeexplore.ieee.org/document/479394) — KN 原始论文
- [KenLM](https://kheafield.com/code/kenlm/) — 快速生产级 N-gram 语言模型，在 2026 年仍用于延迟敏感的应用
