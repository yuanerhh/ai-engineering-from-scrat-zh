# 词嵌入——从零实现 Word2Vec

> 词语因其所处的语境而被认知。基于这一思想训练一个浅层网络，几何结构便随之浮现。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 5 · 02（词袋模型 + TF-IDF）、Phase 3 · 03（反向传播从零实现）
**时长：** 约 75 分钟

## 问题背景

TF-IDF 知道 `dog` 和 `puppy` 是不同的词，但不知道它们意思几乎相同。在 `dog` 上训练的分类器无法泛化到关于 `puppy` 的评论。你可以通过列出同义词来打补丁，但这在罕见术语、领域术语以及你没有预料到的所有语言上都会失效。

你想要一种表示，使 `dog` 和 `puppy` 在空间中靠近。使 `king - man + woman` 落在 `queen` 附近。使在 `dog` 上训练的模型能够免费将一些信号迁移到 `puppy`。

Word2Vec 给了我们这个空间。两层神经网络，万亿 token 的训练运行，发表于 2013 年。架构几乎简单得令人尴尬，但结果重塑了 NLP 长达十年。

## 核心概念

**分布假设**（Firth，1957）："你将通过一个词的同伴认识它。"如果两个词出现在相似的上下文中，它们可能意思相近。

Word2Vec 有两种风格，都利用了这个思想。

- **Skip-gram：** 给定中心词，预测周围的词。窗口大小为 2 时，`cat -> (the, sat, on)`。
- **CBOW（连续词袋）：** 给定周围的词，预测中心词。`(the, sat, on) -> cat`。

Skip-gram 训练更慢，但对罕见词处理得更好，成为了默认选择。

网络有一个无非线性的隐藏层。输入是词汇表上的 one-hot 向量，输出是词汇表上的 softmax。训练后，丢弃输出层。隐藏层权重就是嵌入。

```
one-hot(center) ── W ──▶ hidden (d-dim) ── W' ──▶ softmax(vocab)
                          ^
                          这就是嵌入
```

技巧：对 10 万个词的 softmax 代价极高。Word2Vec 使用**负采样（negative sampling）**将其转化为二元分类任务。预测"这个上下文词是否出现在这个中心词附近，是或否"。每个训练对采样少量负样本（非共现词），而不是对整个词汇表计算 softmax。

## 动手实现

### 步骤一：从语料库生成训练对

```python
def skipgram_pairs(docs, window=2):
    pairs = []
    for doc in docs:
        for i, center in enumerate(doc):
            for j in range(max(0, i - window), min(len(doc), i + window + 1)):
                if i == j:
                    continue
                pairs.append((center, doc[j]))
    return pairs
```

```python
>>> skipgram_pairs([["the", "cat", "sat", "on", "mat"]], window=2)
[('the', 'cat'), ('the', 'sat'),
 ('cat', 'the'), ('cat', 'sat'), ('cat', 'on'),
 ('sat', 'the'), ('sat', 'cat'), ('sat', 'on'), ('sat', 'mat'),
 ...]
```

窗口内的每个（中心词，上下文词）对都是一个正训练样本。

### 步骤二：嵌入表

两个矩阵。`W` 是中心词嵌入表（保留的那个）。`W'` 是上下文词表（通常丢弃，有时与 `W` 取平均）。

```python
import numpy as np


def init_embeddings(vocab_size, dim, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime
```

小随机初始化。词汇量 1 万维度 100 维是实际的；用于教学时，50 词汇 × 16 维就足以看到几何结构。

### 步骤三：负采样目标函数

对每个正样本对 `(center, context)`，从词汇表中随机采样 `k` 个词作为负样本。训练模型使正样本的点积 `W[center] · W'[context]` 高，负样本的点积低。

```python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_pair(W, W_prime, center_idx, context_idx, negative_indices, lr):
    v_c = W[center_idx]
    u_pos = W_prime[context_idx]
    u_negs = W_prime[negative_indices]

    pos_score = sigmoid(v_c @ u_pos)
    neg_scores = sigmoid(u_negs @ v_c)

    grad_center = (pos_score - 1) * u_pos
    for i, u in enumerate(u_negs):
        grad_center += neg_scores[i] * u

    W[context_idx] = W[context_idx]
    W_prime[context_idx] -= lr * (pos_score - 1) * v_c
    for i, neg_idx in enumerate(negative_indices):
        W_prime[neg_idx] -= lr * neg_scores[i] * v_c
    W[center_idx] -= lr * grad_center
```

核心公式：正样本对的逻辑损失（希望 sigmoid 接近 1）加上负样本对的逻辑损失（希望 sigmoid 接近 0）。梯度流向两个表。完整推导在原始论文中；如果想记牢，用纸笔过一遍。

### 步骤四：在玩具语料库上训练

```python
def train(docs, dim=16, window=2, k_neg=5, epochs=100, lr=0.05, seed=0):
    vocab = build_vocab(docs)
    vocab_size = len(vocab)
    rng = np.random.default_rng(seed)
    W, W_prime = init_embeddings(vocab_size, dim, seed=seed)
    pairs = skipgram_pairs(docs, window=window)

    for epoch in range(epochs):
        rng.shuffle(pairs)
        for center, context in pairs:
            c_idx = vocab[center]
            ctx_idx = vocab[context]
            negs = rng.integers(0, vocab_size, size=k_neg)
            negs = [n for n in negs if n != ctx_idx and n != c_idx]
            train_pair(W, W_prime, c_idx, ctx_idx, negs, lr)
    return vocab, W
```

在大型语料库上经过足够多的 epoch 后，共享上下文的词具有相似的中心嵌入。在玩具语料库上，你能隐约看到这个效果；在数十亿 token 上，效果会非常显著。

### 步骤五：类比技巧

```python
def nearest(vocab, W, target_vec, topk=5, exclude=None):
    exclude = exclude or set()
    inv_vocab = {i: w for w, i in vocab.items()}
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    W_norm = W / norms
    target = target_vec / (np.linalg.norm(target_vec) + 1e-9)
    sims = W_norm @ target
    order = np.argsort(-sims)
    out = []
    for i in order:
        if i in exclude:
            continue
        out.append((inv_vocab[i], float(sims[i])))
        if len(out) == topk:
            break
    return out


def analogy(vocab, W, a, b, c, topk=5):
    v = W[vocab[b]] - W[vocab[a]] + W[vocab[c]]
    return nearest(vocab, W, v, topk=topk, exclude={vocab[a], vocab[b], vocab[c]})
```

在预训练的 300 维 Google News 向量上：

```python
>>> analogy(vocab, W, "man", "king", "woman")
[('queen', 0.71), ('monarch', 0.62), ('princess', 0.59), ...]
```

`king - man + woman = queen`。不是因为模型知道什么是皇室，而是因为向量 `(king - man)` 捕捉到了类似"皇家"的概念，将其加到 `woman` 上就落到了皇家女性区域附近。

## 生产使用

从零实现 Word2Vec 是为了学习。生产 NLP 使用 `gensim`。

```python
from gensim.models import Word2Vec

sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "across", "the", "room"],
]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    negative=5,
    workers=4,
    epochs=30,
)

print(model.wv["cat"])
print(model.wv.most_similar("cat", topn=3))
```

实际工作中，你几乎不会自己训练 Word2Vec，而是下载预训练向量。

- **GloVe** — 斯坦福的共现矩阵分解方法。提供 50 维、100 维、200 维、300 维检查点。覆盖范围广。第 04 课专门介绍 GloVe。
- **fastText** — Facebook 的 Word2Vec 扩展，嵌入字符 n-gram。通过子词组合处理词汇表外的词。第 04 课介绍。
- **Google News 预训练 Word2Vec** — 300 维，300 万词汇量，发布于 2013 年。至今每天仍被下载。

### 2026 年 Word2Vec 仍然胜出的场景

- 轻量级领域专属检索。在笔记本电脑上对医学摘要训练一小时，获得通用模型无法捕捉的专业向量。
- 类比风格的特征工程。`gender_vector = mean(man - woman pairs)`。从其他词中减去它，得到一个中性的性别轴。仍被公平性研究使用。
- 可解释性。100 维足够小，可以通过 PCA 或 t-SNE 绘图，实际看到聚类的形成。
- 任何需要在无 GPU 的设备端运行推理的场景。Word2Vec 查找只是一次行获取。

### Word2Vec 失败的场景

多义词障壁（polysemy wall）。`bank` 只有一个向量。`river bank`（河岸）和 `financial bank`（银行）共享同一个向量。`table`（电子表格和家具）也共享同一个向量。下游分类器无法从向量中区分不同含义。

上下文嵌入（ELMo、BERT 以及此后的所有 Transformer）通过根据周围上下文为词的每次出现产生不同的向量解决了这个问题。这就是从 Word2Vec 到 BERT 的跃变：从静态嵌入到上下文嵌入。Phase 7 介绍 Transformer 部分。

另一个失败是词汇表外问题（out-of-vocabulary，OOV）。如果 `Zoomer-approved` 没有出现在训练数据中，Word2Vec 就无从知晓。fastText 通过子词组合解决了这个问题（第 04 课）。

## 上手实践

将以下内容保存为 `outputs/skill-embedding-probe.md`：

```markdown
---
name: embedding-probe
description: Inspect a word2vec model. Run analogies, find neighbors, diagnose quality.
version: 1.0.0
phase: 5
lesson: 03
tags: [nlp, embeddings, debugging]
---

You probe trained word embeddings to verify they are working. Given a `gensim.models.KeyedVectors` object and a vocabulary, you run:

1. Three canonical analogy tests. `king : man :: queen : woman`. `paris : france :: tokyo : japan`. `walking : walked :: swimming : ?`. Report the top-1 result and its cosine.
2. Five nearest-neighbor tests on domain-specific words the user supplies. Print top-5 neighbors with cosines.
3. One symmetry check. `similarity(a, b) == similarity(b, a)` to within float precision.
4. One degenerate check. If any embedding has a norm below 0.01 or above 100, the model has a training bug. Flag it.

Refuse to declare a model good on analogy accuracy alone. Analogy benchmarks are gameable and do not transfer to downstream tasks. Recommend intrinsic + downstream evaluation together.
```

## 练习

1. **简单。** 在一个很小的语料库（20 个关于猫和狗的句子）上运行训练循环。经过 200 个 epoch 后，验证 `nearest(vocab, W, W[vocab["cat"]])` 在其前 3 名中返回 `dog`。如果不是，增加 epoch 数或词汇量。
2. **中等。** 添加对高频词的降采样。频率超过 `10^-5` 的词以与其频率成比例的概率从训练对中丢弃。测量这对罕见词相似度的影响。
3. **困难。** 在 20 个新闻组语料库上训练模型。计算两个偏差轴：`he - she` 和 `doctor - nurse`。将职业词投影到两个轴上。报告哪些职业的偏差差距最大。这是公平性研究者使用的那种探针。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 词嵌入（Word embedding） | 词作为向量 | 从上下文中学习的密集低维（通常 100-300 维）表示。 |
| Skip-gram | Word2Vec 技巧 | 从中心词预测上下文词。比 CBOW 慢，但对罕见词更好。 |
| 负采样（Negative sampling） | 训练捷径 | 用对 `k` 个随机词的二元分类替代对全词汇表的 softmax。 |
| 静态嵌入（Static embedding） | 每词一个向量 | 无论上下文如何，同一个向量。在多义词上失败。 |
| 上下文嵌入（Contextual embedding） | 上下文敏感向量 | 基于周围词为每次出现产生不同向量。Transformer 产生的就是这种。 |
| OOV | 词汇表外词 | 训练中未见过的词。Word2Vec 无法为其产生向量。 |

## 延伸阅读

- [Mikolov et al. (2013). Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) — 负采样论文，简短易读
- [Rong, X. (2014). word2vec Parameter Learning Explained](https://arxiv.org/abs/1411.2738) — 最清晰的梯度推导，如果原始论文的数学感觉密集
- [gensim Word2Vec tutorial](https://radimrehurek.com/gensim/models/word2vec.html) — 真正可用的生产训练设置
