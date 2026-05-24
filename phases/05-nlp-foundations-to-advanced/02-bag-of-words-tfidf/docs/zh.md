# 词袋模型、TF-IDF 与文本表示

> 先计数，再思考。2026 年 TF-IDF 在定义明确的任务上仍然胜过嵌入。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 5 · 01（文本处理）、Phase 2 · 02（线性回归从零实现）
**时长：** 约 75 分钟

## 问题背景

模型需要数字，你有的是字符串。

每个 NLP 管线都必须回答同一个问题：如何将长度可变的 token 流转化为分类器可以消费的固定大小向量。这个领域最早的答案也是最笨但管用的那个：数词，然后造向量。

这个向量所承载的生产 NLP 比任何嵌入模型都多。垃圾邮件过滤、主题分类、日志异常检测、搜索排序（BM25 之前）、第一波情感分析、学术 NLP 基准测试的第一个十年。2026 年的从业者在窄分类任务上仍然优先使用它。它速度快、可解释，在词语出现与否才是关键的任务上，往往与 4 亿参数嵌入模型无法区分。

本课从零构建词袋模型，然后是 TF-IDF，再展示 scikit-learn 如何用三行代码完成同样的事，最后指出让你转向嵌入的失败模式。

## 核心概念

**词袋模型（Bag of Words，BoW）** 丢弃顺序。对每个文档，统计词汇表中每个词出现的次数。向量长度等于词汇量大小。位置 `i` 是词 `i` 的计数。

**TF-IDF** 对 BoW 重新加权。出现在每个文档中的词没有信息量，因此降低权重。在语料库中罕见但在单个文档中频繁的词是信号，因此提高权重。

```
TF-IDF(w, d) = TF(w, d) * IDF(w)
             = count(w in d) / |d| * log(N / df(w))
```

其中 `TF` 是词在文档中的词频，`df` 是文档频率（包含该词的文档数），`N` 是文档总数。`log` 使无处不在的词的权重有界。

关键特性：两者都产生具有可解释轴的稀疏向量。你可以查看训练好的分类器的权重，看出哪些词将文档推向每个类别。而 768 维的 BERT 嵌入则无法做到这点。

## 动手实现

### 步骤一：构建词汇表

```python
def build_vocab(docs):
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab
```

输入：分词后的文档列表（任何词级分词器都可以；本课的 `code/main.py` 使用简化的小写变体）。输出：`{词: 索引}` 字典。稳定的插入顺序意味着词索引 0 是第一个文档中首先见到的词。约定因人而异；scikit-learn 按字母顺序排序。

### 步骤二：词袋模型

```python
def bag_of_words(docs, vocab):
    matrix = [[0] * len(vocab) for _ in docs]
    for i, doc in enumerate(docs):
        for token in doc:
            if token in vocab:
                matrix[i][vocab[token]] += 1
    return matrix
```

```python
>>> docs = [["cat", "sat", "on", "mat"], ["cat", "cat", "ran"]]
>>> vocab = build_vocab(docs)
>>> bag_of_words(docs, vocab)
[[1, 1, 1, 1, 0], [2, 0, 0, 0, 1]]
```

行是文档，列是词汇索引。条目 `[i][j]` 是"词 `j` 在文档 `i` 中出现了多少次"。文档 1 中 `cat` 出现两次，因为它确实出现了两次。文档 0 中 `ran` 出现零次，因为它没有出现。

### 步骤三：词频与文档频率

```python
import math


def term_frequency(doc_bow, doc_length):
    return [c / doc_length if doc_length else 0 for c in doc_bow]


def document_frequency(bow_matrix):
    df = [0] * len(bow_matrix[0])
    for row in bow_matrix:
        for j, count in enumerate(row):
            if count > 0:
                df[j] += 1
    return df


def inverse_document_frequency(df, n_docs):
    return [math.log((n_docs + 1) / (d + 1)) + 1 for d in df]
```

两个值得命名的平滑技巧：`(n+1)/(d+1)` 避免 `log(x/0)`；末尾的 `+1` 确保出现在所有文档中的词 IDF 为 1（而非 0），与 scikit-learn 的默认行为一致。其他实现使用原始 `log(N/df)`，两者都能工作；平滑版本更友好。

### 步骤四：TF-IDF

```python
def tfidf(bow_matrix):
    n_docs = len(bow_matrix)
    df = document_frequency(bow_matrix)
    idf = inverse_document_frequency(df, n_docs)
    out = []
    for row in bow_matrix:
        length = sum(row)
        tf = term_frequency(row, length)
        out.append([tf_j * idf_j for tf_j, idf_j in zip(tf, idf)])
    return out
```

```python
>>> docs = [
...     ["the", "cat", "sat"],
...     ["the", "dog", "sat"],
...     ["the", "cat", "ran"],
... ]
>>> vocab = build_vocab(docs)
>>> bow = bag_of_words(docs, vocab)
>>> tfidf(bow)
```

三个文档，五个词汇词（`the`、`cat`、`sat`、`dog`、`ran`）。`the` 出现在所有三个文档中，因此 IDF 低。`dog` 只出现在一个文档中，因此 IDF 高。向量是稀疏的（大多数条目较小），区分性词语突出。

### 步骤五：L2 归一化行

```python
def l2_normalize(matrix):
    out = []
    for row in matrix:
        norm = math.sqrt(sum(x * x for x in row))
        out.append([x / norm if norm else 0 for x in row])
    return out
```

没有归一化，较长的文档会得到较大的向量并主导相似度分数。L2 归一化将每个文档置于单位超球面上。行之间的余弦相似度现在只需做点积。

## 生产使用

scikit-learn 提供了生产版本。

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

docs = ["the cat sat on the mat", "the dog sat on the mat", "the cat ran"]

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(docs)
print(bow_vectorizer.get_feature_names_out())
print(bow.toarray())

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(docs)
print(tfidf.toarray().round(3))
```

`CountVectorizer` 在一次调用中完成分词、构建词汇表和词袋模型。`TfidfVectorizer` 添加 IDF 加权和 L2 归一化。两者都返回稀疏矩阵。对于 10 万个文档，密集版本装不进内存；在分类器要求密集矩阵之前，保持稀疏格式。

改变一切的参数：

| 参数 | 效果 |
|-----|------|
| `ngram_range=(1, 2)` | 包含二元组。通常提升分类效果。 |
| `min_df=2` | 丢弃出现在少于 2 个文档中的词。在嘈杂数据上裁减词汇表。 |
| `max_df=0.95` | 丢弃出现在 95% 以上文档中的词。近似停用词移除，无需硬编码列表。 |
| `stop_words="english"` | scikit-learn 的内置停用词列表。取决于任务——情感分析**不应**删除否定词。 |
| `sublinear_tf=True` | 使用 `1 + log(tf)` 代替原始 `tf`。当一个词在单个文档中重复多次时有帮助。 |

### TF-IDF 仍然胜出的场景（截至 2026 年）

- 垃圾邮件检测、主题标注、日志异常标记。词语出现与否才是关键；语义细微差别无关紧要。
- 低数据场景（数百个标注样本）。TF-IDF 加逻辑回归没有预训练成本。
- 任何延迟敏感的场景。TF-IDF 加线性模型在微秒内响应。通过 Transformer 对文档进行嵌入需要 10-100 毫秒。
- 必须解释预测的系统。检查分类器的系数。权重最高的正向词就是原因。

### TF-IDF 失败的场景

语义盲目性失败。考虑这两个文档：

- "The movie was not good at all."（这部电影一点都不好。）
- "The movie was excellent."（这部电影非常好。）

一个是负面评价，一个是正面评价。它们的 TF-IDF 重叠恰好是 `{the, movie, was}`。词袋分类器必须记住 `not` 在 `good` 附近会翻转标签。在足够的数据上它能学到这一点，但远不如理解句法的模型优雅。

另一个失败：推理时的词汇表外词语（out-of-vocabulary，OOV）。在 IMDb 评论上训练的词袋模型不知道如何处理 `Zoomer-approved`，如果该 token 从未出现在训练数据中。子词嵌入（第 04 课）可以处理这个问题，TF-IDF 无法。

### 混合方案：TF-IDF 加权嵌入

2026 年中等数据分类的务实默认方案：使用 TF-IDF 权重作为词嵌入的注意力权重。

```python
def tfidf_weighted_embedding(doc, tfidf_scores, embedding_table, dim):
    vec = [0.0] * dim
    total_weight = 0.0
    for token in doc:
        if token not in embedding_table or token not in tfidf_scores:
            continue
        weight = tfidf_scores[token]
        emb = embedding_table[token]
        for i in range(dim):
            vec[i] += weight * emb[i]
        total_weight += weight
    if total_weight == 0:
        return vec
    return [v / total_weight for v in vec]
```

你从嵌入中获得语义能力，从 TF-IDF 中获得对稀有词的强调。分类器在池化后的向量上训练。在约 5 万个标注样本以下的情感、主题和意图分类任务上，这比单独使用任何一种方法表现更好。

## 上手实践

将以下内容保存为 `outputs/prompt-vectorization-picker.md`：

```markdown
---
name: vectorization-picker
description: Given a text-classification task, recommend BoW, TF-IDF, embeddings, or a hybrid.
phase: 5
lesson: 02
---

You recommend a text-vectorization strategy. Given a task description, output:

1. Representation (BoW, TF-IDF, transformer embeddings, or a hybrid). Explain why in one sentence.
2. Specific vectorizer configuration. Name the library. Quote the arguments (`ngram_range`, `min_df`, `max_df`, `sublinear_tf`, `stop_words`).
3. One failure mode to test before shipping.

Refuse to recommend embeddings when the user has under 500 labeled examples unless they show evidence of semantic failure in a TF-IDF baseline. Refuse to remove stopwords for sentiment analysis (negations carry signal). Flag class imbalance as needing more than a vectorizer change.

Example input: "Classifying 30k customer support tickets into 12 categories. Most tickets are 2-3 sentences. English only. Need explainability for audit logs."

Example output:

- Representation: TF-IDF. 30k examples is not small; explainability requirement rules out dense embeddings.
- Config: `TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True, stop_words=None)`. Keep stopwords because category keywords sometimes are stopwords ("not working" vs "working").
- Failure to test: verify `min_df=3` does not drop rare category keywords. Run `get_feature_names_out` filtered by class and eyeball.
```

## 练习

1. **简单。** 在 L2 归一化的 TF-IDF 输出上实现 `cosine_similarity(doc_vec_a, doc_vec_b)`。验证相同文档得分为 1.0，词汇表不相交的文档得分为 0.0。
2. **中等。** 为 `bag_of_words` 添加 n-gram 支持。参数 `n` 产生 n-gram 的计数。测试 `n=2` 对 `["the", "cat", "sat"]` 产生 `["the cat", "cat sat"]` 的二元组计数。
3. **困难。** 使用 GloVe 100 维向量（下载一次，缓存）构建上述 TF-IDF 加权嵌入混合方案。在 20 个新闻组数据集上，将分类准确率与纯 TF-IDF 和纯均值池化嵌入进行比较。报告哪种方法在哪里胜出。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| BoW（词袋） | 词频向量 | 一个文档中词汇表词语的计数。丢弃顺序。 |
| TF（词频） | 词频 | 词在文档中的计数，可选地按文档长度归一化。 |
| DF（文档频率） | 文档频率 | 至少包含该词一次的文档数。 |
| IDF（逆文档频率） | 逆文档频率 | `log(N / df)` 平滑版本。对无处不在的词降权。 |
| 稀疏向量 | 大部分为零 | 词汇表通常有 1 万到 10 万词；任何给定文档中大多数词缺席。 |
| 余弦相似度 | 向量夹角 | L2 归一化向量的点积。1 表示相同，0 表示正交。 |

## 延伸阅读

- [scikit-learn — feature extraction from text](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — 权威 API 参考，附每个参数的说明
- [Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval](https://www.sciencedirect.com/science/article/pii/0306457388900210) — 使 TF-IDF 成为十年默认方案的论文
- ["Why TF-IDF Still Beats Embeddings" — Ashfaque Thonikkadavan (Medium)](https://medium.com/@cmtwskb/why-tf-idf-still-beats-embeddings-ad85c123e1b2) — 2026 年关于旧方法何时胜出及其原因的分析
