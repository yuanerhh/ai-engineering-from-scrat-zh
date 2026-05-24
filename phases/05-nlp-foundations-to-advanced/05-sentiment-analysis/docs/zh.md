# 情感分析

> NLP 的经典任务。关于经典文本分类你需要知道的大部分内容都在这里。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 5 · 02（词袋模型 + TF-IDF）、Phase 2 · 14（朴素贝叶斯）
**时长：** 约 75 分钟

## 问题背景

"The food was not great."（这道菜不怎么好。）是正面还是负面？

情感分析听起来简单：评论者说他们喜欢或不喜欢某件事，给句子打标签就好。它成为 NLP 经典任务的原因在于每个看起来简单的情况背后都隐藏着困难的案例。否定词翻转含义，讽刺将其颠倒。"Not bad at all"尽管包含两个负面编码的词却是正面的。表情符号比周围文本携带更多信号。领域词汇很重要（音乐评论中的 `tight` 与时尚评论中的 `tight` 含义不同）。

情感分析是经典 NLP 的实验室。如果你理解了为什么每个朴素基线都有特定的失败模式，你就理解了为什么每个更丰富的模型被发明出来。本课从零构建朴素贝叶斯基线，添加逻辑回归，并指出使生产情感分析成为合规级问题的陷阱。

## 核心概念

经典情感分析是两步配方：

1. **表示。** 将文本转化为特征向量。词袋模型、TF-IDF 或 n-gram。
2. **分类。** 在标注样本上拟合线性模型（朴素贝叶斯、逻辑回归、SVM）。

朴素贝叶斯是最笨但管用的模型。假设给定标签后每个特征都是独立的。从计数中估计 `P(词 | 正面)` 和 `P(词 | 负面)`。推理时，将概率相乘。"朴素"的独立假设在逻辑上荒谬，但结果却出人意料地强。原因在于：对于稀疏文本特征和中等数量的数据，分类器关心每个词倾向于哪一侧，而不是精确的联合概率。

逻辑回归修正了独立假设。它为每个特征学习一个权重，包括负权重。`not good` 作为二元组特征会得到负权重。朴素贝叶斯无法对从未标注过的二元组做到这一点。

## 动手实现

### 步骤一：一个真实的小数据集

```python
POSITIVE = [
    "absolutely loved this movie",
    "beautiful cinematography and a great story",
    "one of the best films of the year",
    "brilliant acting from the lead",
    "heartwarming and funny",
]

NEGATIVE = [
    "boring and far too long",
    "not worth your time",
    "the plot made no sense",
    "terrible acting, awful script",
    "i want my two hours back",
]
```

故意保持很小。实际工作使用数万个样本（IMDb、SST-2、Yelp 极性数据集）。数学完全相同。

### 步骤二：从零实现多项式朴素贝叶斯

```python
import math
from collections import Counter


def train_nb(docs_by_class, vocab, alpha=1.0):
    class_priors = {}
    class_word_probs = {}
    total_docs = sum(len(d) for d in docs_by_class.values())

    for cls, docs in docs_by_class.items():
        class_priors[cls] = len(docs) / total_docs
        counts = Counter()
        for doc in docs:
            for token in doc:
                counts[token] += 1
        total = sum(counts.values()) + alpha * len(vocab)
        class_word_probs[cls] = {
            w: (counts[w] + alpha) / total for w in vocab
        }
    return class_priors, class_word_probs


def predict_nb(doc, class_priors, class_word_probs):
    scores = {}
    for cls in class_priors:
        s = math.log(class_priors[cls])
        for token in doc:
            if token in class_word_probs[cls]:
                s += math.log(class_word_probs[cls][token])
        scores[cls] = s
    return max(scores, key=scores.get)
```

加法平滑（alpha=1.0）即 Laplace 平滑。没有它，在某个类别中未见过的词的概率为零，对数就会爆炸。实际中 `alpha=0.01` 很常见，`alpha=1.0` 是教学默认值。

### 步骤三：从零实现逻辑回归

```python
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_lr(X, y, epochs=500, lr=0.05, l2=0.01):
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0.0
    for _ in range(epochs):
        logits = X @ w + b
        preds = sigmoid(logits)
        err = preds - y
        grad_w = X.T @ err / len(y) + l2 * w
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def predict_lr(X, w, b):
    return (sigmoid(X @ w + b) >= 0.5).astype(int)
```

L2 正则化在这里很重要。文本特征是稀疏的；没有 L2，模型会记忆训练样本。从 `0.01` 开始调整。

### 步骤四：处理否定（失败模式）

考虑"not good"和"not bad"。词袋分类器看到 `{not, good}` 和 `{not, bad}`，并从训练中出现频率更高的那个学习。二元组分类器看到 `not_good` 和 `not_bad`，并将它们视为不同特征，这通常足够了。

在没有二元组时有效的粗略修复：**否定范围（negation scoping）**。在否定词之后、下一个标点之前，为 token 加上 `NOT_` 前缀。

```python
NEGATION_WORDS = {"not", "no", "never", "nor", "none", "nothing", "neither"}
NEGATION_TERMINATORS = {".", "!", "?", ",", ";"}


def apply_negation(tokens):
    out = []
    negate = False
    for token in tokens:
        if token in NEGATION_TERMINATORS:
            negate = False
            out.append(token)
            continue
        if token in NEGATION_WORDS:
            negate = True
            out.append(token)
            continue
        out.append(f"NOT_{token}" if negate else token)
    return out
```

```python
>>> apply_negation(["not", "good", "at", "all", ".", "but", "funny"])
['not', 'NOT_good', 'NOT_at', 'NOT_all', '.', 'but', 'funny']
```

现在 `good` 和 `NOT_good` 是不同的特征。分类器可以给它们相反的权重。三行预处理，情感基准测试上可测量的准确率提升。

### 步骤五：重要的评估指标

如果类别不平衡，准确率单独来看会产生误导。真实情感语料库通常有 70-80% 正面或 70-80% 负面；一个总预测多数类的分类器能达到 80% 准确率，却毫无价值。报告以下全部指标：

- **每类精确率和召回率。** 每类一对。取宏平均（macro-average）得到一个尊重类别平衡的单一数字。
- **宏 F1（不平衡数据的主要指标）。** 每类 F1 分数的平均，等权重。类别不平衡时用这个代替准确率。
- **加权 F1（替代方案）。** 与宏 F1 相同，但按类别频率加权。当不平衡本身有业务含义时，与宏 F1 一起报告。
- **混淆矩阵。** 原始计数。在信任任何标量指标之前始终检查；它揭示模型混淆的类别对。
- **每类错误样本。** 每类抽取 5 个错误预测并阅读。没有什么能替代阅读实际错误。

对于严重不平衡的数据（> 95:5 比例），报告 **AUROC** 和 **AUPRC** 而非准确率。AUPRC 对少数类更敏感，而这通常是你真正关心的（垃圾邮件、欺诈、罕见情感）。

**常见 bug。** 在不平衡数据上报告微 F1 而非宏 F1，会得到看起来高的数字，因为它被多数类主导。宏 F1 迫使你看到少数类的性能。

```python
def evaluate(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
```

## 生产使用

scikit-learn 用六行代码正确实现。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words=None)),
    ("clf", LogisticRegression(C=1.0, max_iter=1000)),
])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
```

三点值得注意：`stop_words=None` 保留了否定词；`ngram_range=(1, 2)` 添加了二元组，使 `not_good` 成为一个特征；`sublinear_tf=True` 抑制重复词。这三个参数是 SST-2 上 75% 准确率基线与 85% 准确率基线之间的差距。

### 何时选择 Transformer

- 讽刺检测。经典模型在这里会失败，毫无例外。
- 情感在文档中途发生转变的长评论。
- 基于方面的情感分析（Aspect-based sentiment）。"Camera was great but battery was terrible."你需要将情感归因于特定方面。只能用 Transformer 或结构化输出模型。
- 非英语的低资源语言。多语言 BERT 免费给你一个零样本基线。

如果你需要上述任何功能，跳到 Phase 7（Transformer 深入解析）。否则，TF-IDF 加二元组加否定处理的朴素贝叶斯或逻辑回归就是你的 2026 年生产基线。

### 可重现性陷阱（再次）

重新训练情感模型是家常便饭，重新评估它们却不是。论文中报告的准确率数字使用特定的数据划分、特定的预处理、特定的分词器。如果你在不使用完全相同管线的情况下将你的新模型与基线进行比较，你会得到误导性的差距。始终在你的管线上重新生成基线，而非使用论文的数字。

## 上手实践

将以下内容保存为 `outputs/prompt-sentiment-baseline.md`：

```markdown
---
name: sentiment-baseline
description: Design a sentiment analysis baseline for a new dataset.
phase: 5
lesson: 05
---

Given a dataset description (domain, language, size, label granularity, latency budget), you output:

1. Feature extraction recipe. Specify tokenizer, n-gram range, stopword policy (usually keep), negation handling (scoped prefix or bigrams).
2. Classifier. Naive Bayes for baseline, logistic regression for production, transformer only if the domain needs sarcasm / aspects / cross-lingual.
3. Evaluation plan. Report precision, recall, F1, confusion matrix, and per-class error samples (not just scalars).
4. One failure mode to monitor post-deployment. Domain drift and sarcasm are the top two.

Refuse to recommend dropping stopwords for sentiment tasks. Refuse to report accuracy as the sole metric when classes are imbalanced (e.g., 90% positive). Flag subword-rich languages as needing FastText or transformer embeddings over word-level TF-IDF.
```

## 练习

1. **简单。** 在 scikit-learn 管线中添加 `apply_negation` 作为预处理步骤，在小型情感数据集上测量 F1 差值。
2. **中等。** 实现类权重逻辑回归（向 scikit-learn 传入 `class_weight="balanced"`，或自己推导梯度）。在合成的 90:10 类别不平衡上测量效果。
3. **困难。** 通过在情感模型的残差上训练第二个分类器来构建讽刺检测器。记录你的实验设置。当你的准确率低于随机水平时警告读者（二分类讽刺的随机水平约为 50%，大多数第一次尝试都在这个水平左右）。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 极性（Polarity） | 正面或负面 | 二元标签；有时扩展为中性或细粒度（5 星）。 |
| 基于方面的情感（Aspect-based sentiment） | 每方面极性 | 将情感归因于文本中提到的特定实体或属性。 |
| 否定范围（Negation scoping） | 翻转附近 token | 在"not"之后到标点之间的 token 加上 `NOT_` 前缀。 |
| Laplace 平滑 | 计数加 1 | 防止朴素贝叶斯中出现零概率特征。 |
| L2 正则化 | 权重收缩 | 在损失中加入 `lambda * sum(w^2)`。对稀疏文本特征至关重要。 |

## 延伸阅读

- [Pang and Lee (2008). Opinion Mining and Sentiment Analysis](https://www.cs.cornell.edu/home/llee/opinion-mining-sentiment-analysis-survey.html) — 基础综述，前四节涵盖所有经典内容
- [Wang and Manning (2012). Baselines and Bigrams: Simple, Good Sentiment and Topic Classification](https://aclanthology.org/P12-2018/) — 证明二元组加朴素贝叶斯在短文本上难以超越的论文
- [scikit-learn text feature extraction docs](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — `CountVectorizer`、`TfidfVectorizer` 及你需要调整的每个参数的参考
