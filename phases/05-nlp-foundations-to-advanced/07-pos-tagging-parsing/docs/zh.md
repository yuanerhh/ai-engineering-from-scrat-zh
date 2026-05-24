# 词性标注与句法分析

> 语法曾经不受欢迎了一段时间。然后每个 LLM 管线都需要验证结构化提取，它又回来了。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 5 · 01（文本处理）、Phase 2 · 14（朴素贝叶斯）
**时长：** 约 45 分钟

## 问题背景

第 01 课承诺词形还原需要词性标签。不知道 `running` 是动词，词形还原器就无法将其还原为 `run`；不知道 `better` 是形容词，就无法还原为 `good`。

这个承诺背后隐藏着整个子领域。词性标注（POS tagging）为词语分配语法类别。句法分析恢复句子的树状结构：哪个词修饰哪个，哪个动词支配哪些论元。经典 NLP 花了二十年精炼这两者。然后深度学习将它们压缩为预训练 Transformer 之上的 token 分类任务，研究社区继续前进了。

但应用社区没有。每个结构化提取管线仍在底层使用词性和依存树。LLM 生成的 JSON 根据语法约束进行验证。问答系统使用依存分析分解查询。机器翻译质量评估器检查分析树的对齐情况。

值得了解。本课介绍标注集、基线，以及你停止从零实现转而调用 spaCy 的时刻。

## 核心概念

**词性标注**为每个 token 标注语法类别。**Penn Treebank（PTB）**标注集是英语默认标准。36 个标签，区别之细令一般读者感到繁琐：`NN` 单数名词、`NNS` 复数名词、`NNP` 专有名词单数、`VBD` 动词过去时、`VBZ` 动词第三人称单数现在时等等。**通用依存（Universal Dependencies，UD）**标注集更粗粒度（17 个标签）且与语言无关；它成为跨语言工作的默认标准。

```
The/DET cats/NOUN were/AUX running/VERB at/ADP 3pm/NOUN ./PUNCT
```

**句法分析**产生树状结构。两种主要风格：

- **成分分析（Constituency parsing）。** 名词短语、动词短语、介词短语相互嵌套。输出是以非终端类别（NP、VP、PP）为节点、词语为叶子的树。
- **依存分析（Dependency parsing）。** 每个词有一个它依附的中心词，并带有语法关系标签。输出是一棵树，每条边是（中心词，依附词，关系）三元组。

依存分析在 2010 年代胜出，因为它在不同语言间（尤其是自由词序语言）泛化得更好。

```
running 是 ROOT
cats 是 running 的 nsubj（名词性主语）
were 是 running 的 aux（助动词）
at 是 running 的 prep（介词修饰）
3pm 是 at 的 pobj（介词宾语）
```

## 动手实现

### 步骤一：最高频标签基线

最笨但管用的词性标注器。对每个词，预测它在训练中出现最频繁的标签。

```python
from collections import Counter, defaultdict


def train_mft(train_examples):
    word_tag_counts = defaultdict(Counter)
    all_tags = Counter()
    for tokens, tags in train_examples:
        for token, tag in zip(tokens, tags):
            word_tag_counts[token.lower()][tag] += 1
            all_tags[tag] += 1
    word_best = {w: c.most_common(1)[0][0] for w, c in word_tag_counts.items()}
    default_tag = all_tags.most_common(1)[0][0]
    return word_best, default_tag


def predict_mft(tokens, word_best, default_tag):
    return [word_best.get(t.lower(), default_tag) for t in tokens]
```

在 Brown 语料库上，这个基线能达到 ~85% 准确率。不高，但这是任何严肃模型不应低于的下限。

### 步骤二：二元组 HMM 标注器

建模序列的联合概率：

```
P(tags, words) = prod P(tag_i | tag_{i-1}) * P(word_i | tag_i)
```

两个表：转移概率（给定前一个标签时的当前标签）和发射概率（给定标签时的词语）。两者都从带 Laplace 平滑的计数中估计。用 Viterbi（在标签格上的动态规划）解码。

```python
import math


def train_hmm(train_examples, alpha=0.01):
    transitions = defaultdict(Counter)
    emissions = defaultdict(Counter)
    tags = set()
    vocab = set()

    for tokens, ts in train_examples:
        prev = "<BOS>"
        for token, tag in zip(tokens, ts):
            transitions[prev][tag] += 1
            emissions[tag][token.lower()] += 1
            tags.add(tag)
            vocab.add(token.lower())
            prev = tag
        transitions[prev]["<EOS>"] += 1

    return transitions, emissions, tags, vocab


def log_prob(table, given, key, smooth_denom, alpha):
    return math.log((table[given].get(key, 0) + alpha) / smooth_denom)


def viterbi(tokens, transitions, emissions, tags, vocab, alpha=0.01):
    tags_list = list(tags)
    n = len(tokens)
    V = [[0.0] * len(tags_list) for _ in range(n)]
    back = [[0] * len(tags_list) for _ in range(n)]

    for j, tag in enumerate(tags_list):
        em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
        tr_denom = sum(transitions["<BOS>"].values()) + alpha * (len(tags_list) + 1)
        tr = log_prob(transitions, "<BOS>", tag, tr_denom, alpha)
        em = log_prob(emissions, tag, tokens[0].lower(), em_denom, alpha)
        V[0][j] = tr + em
        back[0][j] = 0

    for i in range(1, n):
        for j, tag in enumerate(tags_list):
            em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
            em = log_prob(emissions, tag, tokens[i].lower(), em_denom, alpha)
            best_prev = 0
            best_score = -1e30
            for k, prev_tag in enumerate(tags_list):
                tr_denom = sum(transitions[prev_tag].values()) + alpha * (len(tags_list) + 1)
                tr = log_prob(transitions, prev_tag, tag, tr_denom, alpha)
                score = V[i - 1][k] + tr + em
                if score > best_score:
                    best_score = score
                    best_prev = k
            V[i][j] = best_score
            back[i][j] = best_prev

    last_best = max(range(len(tags_list)), key=lambda j: V[n - 1][j])
    path = [last_best]
    for i in range(n - 1, 0, -1):
        path.append(back[i][path[-1]])
    return [tags_list[j] for j in reversed(path)]
```

Brown 语料库上的二元组 HMM 达到 ~93% 准确率。从 85% 到 93% 的飞跃主要来自转移概率——模型学会了 `DET NOUN` 很常见，`NOUN DET` 很少见。

### 步骤三：为什么现代标注器胜过这个

转移和发射概率是局部的。它们无法捕捉 `saw` 在"I bought a saw"中是名词，而在"I saw the movie"中是动词。带任意特征（后缀、词形、前后词、词本身）的 CRF 能达到 ~97%。BiLSTM-CRF 或 Transformer 能达到 ~98%+。

这个任务的准确率上限由标注者分歧决定。人工标注者在 Penn Treebank 上约 97% 的时间达成一致。超过 98% 的模型可能是在过拟合测试集。

### 步骤四：依存分析概述

从零完整实现依存分析超出了本课范围；权威教科书处理在 Jurafsky 和 Martin 的书中。两个经典流派需要了解：

- **基于转移的**分析器（弧急迫、弧标准）像移位-规约分析器一样工作：读取 token，将其压入栈，应用创建弧的规约动作。贪婪解码很快。经典实现是 MaltParser，现代神经版本是 Chen 和 Manning 的基于转移的分析器。
- **基于图的**分析器（Eisner 算法、Dozat-Manning 双仿射）为每个可能的中心词-依附词边打分，选择最大生成树。较慢但更准确。

对于大多数应用工作，调用 spaCy：

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running at 3pm.")
for token in doc:
    print(f"{token.text:10s} tag={token.tag_:5s} pos={token.pos_:6s} dep={token.dep_:10s} head={token.head.text}")
```

```
The        tag=DT    pos=DET    dep=det        head=cats
cats       tag=NNS   pos=NOUN   dep=nsubj      head=running
were       tag=VBD   pos=AUX    dep=aux        head=running
running    tag=VBG   pos=VERB   dep=ROOT       head=running
at         tag=IN    pos=ADP    dep=prep       head=running
3pm        tag=NN    pos=NOUN   dep=pobj       head=at
.          tag=.     pos=PUNCT  dep=punct      head=running
```

从下到上读 `dep` 列，句子的语法结构便清晰呈现。

## 生产使用

每个生产 NLP 库都将词性和依存分析器作为标准管线的一部分提供。

- **spaCy**（`en_core_web_sm`/`md`/`lg`/`trf`）。快速、准确，与分词、NER、词形还原集成。`token.tag_`（Penn）、`token.pos_`（UD）、`token.dep_`（依存关系）。
- **Stanford NLP（stanza）**。斯坦福 CoreNLP 的继任者。60+ 种语言的最先进水平。
- **trankit**。基于 Transformer，UD 准确率高。
- **NLTK**。`pos_tag`。可用，但慢且较旧。适合教学。

### 2026 年仍然重要的场景

- **词形还原。** 第 01 课需要词性才能正确词形还原。永远如此。
- **从 LLM 输出中进行结构化提取。** 验证生成句子是否遵守语法约束（如主谓一致、必需修饰语）。
- **基于方面的情感分析。** 依存分析告诉你哪个形容词修饰哪个名词。
- **查询理解。** "movies directed by Wes Anderson starring Bill Murray"通过分析分解为结构化约束。
- **跨语言迁移。** UD 标签和依存关系与语言无关，使新语言的零样本结构化分析成为可能。
- **低计算量管线。** 如果无法部署 Transformer，词性 + 依存分析 + 词典出人意料地有效。

## 上手实践

将以下内容保存为 `outputs/skill-grammar-pipeline.md`：

```markdown
---
name: grammar-pipeline
description: Design a classical POS + dependency pipeline for a downstream NLP task.
version: 1.0.0
phase: 5
lesson: 07
tags: [nlp, pos, parsing]
---

Given a downstream task (information extraction, rewrite validation, query decomposition, lemmatization), you output:

1. Tagset to use. Penn Treebank for English-only legacy pipelines, Universal Dependencies for multilingual or cross-lingual.
2. Library. spaCy for most production, stanza for academic-grade multilingual, trankit for highest UD accuracy. Name the specific model ID.
3. Integration pattern. Show the 3-5 lines that call the library and consume the needed attributes (`.pos_`, `.dep_`, `.head`).
4. Failure mode to test. Noun-verb ambiguity (`saw`, `book`, `can`) and PP-attachment ambiguity are the classical traps. Sample 20 outputs and eyeball.

Refuse to recommend rolling your own parser. Building parsers from scratch is a research project, not an application task. Flag any pipeline that consumes POS tags without handling lowercase/uppercase variants as fragile.
```

## 练习

1. **简单。** 使用最高频标签基线在小型标注语料库（如 NLTK 的 Brown 子集）上，测量保留句子的准确率。验证约 85% 的结果。
2. **中等。** 训练上述二元组 HMM 并报告每个标签的精确率/召回率。HMM 最容易混淆哪些标签？
3. **困难。** 使用 spaCy 的依存分析从 1000 句样本中提取主谓宾三元组。在 50 个手工标注的三元组上评估。记录提取失败的地方（通常是被动句、并列结构和省略主语）。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 词性标签（POS tag） | 词的类型 | 语法类别。PTB 有 36 个，UD 有 17 个。 |
| Penn Treebank | 标准标注集 | 英语专用，动词时态和名词数的细粒度区分。 |
| 通用依存（Universal Dependencies） | 多语言标注集 | 比 PTB 更粗粒度，与语言无关，是跨语言工作的默认标准。 |
| 依存分析（Dependency parse） | 句子树 | 每个词有一个中心词，每条边有一个语法关系标签。 |
| Viterbi | 动态规划 | 给定发射和转移概率，找到概率最高的标签序列。 |

## 延伸阅读

- [Jurafsky and Martin — Speech and Language Processing, chapters 8 and 18](https://web.stanford.edu/~jurafsky/slp3/) — 词性和分析的权威教科书处理
- [Universal Dependencies project](https://universaldependencies.org/) — 每个多语言分析器使用的跨语言标注集和树库集合
- [spaCy linguistic features guide](https://spacy.io/usage/linguistic-features) — `Token` 上每个属性的实用参考
- [Chen and Manning (2014). A Fast and Accurate Dependency Parser using Neural Networks](https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf) — 将神经分析器带入主流的论文
