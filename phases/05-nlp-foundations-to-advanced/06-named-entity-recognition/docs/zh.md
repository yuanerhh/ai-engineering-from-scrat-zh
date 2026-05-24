# 命名实体识别

> 把名字提取出来。听起来简单，直到你碰到模糊边界、嵌套实体和领域专业术语。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 5 · 02（词袋模型 + TF-IDF）、Phase 5 · 03（词嵌入）
**时长：** 约 75 分钟

## 问题背景

"Apple sued Google over its iPhone search deal in the US."（苹果公司在美国就 iPhone 搜索协议起诉谷歌。）五个实体：Apple（ORG，组织）、Google（ORG）、iPhone（PRODUCT，产品）、search deal（也许）、US（GPE，地缘政治实体）。一个好的命名实体识别（NER）系统能提取所有这些并给出正确类型。一个差的系统会漏掉 iPhone，混淆水果苹果和苹果公司，并将"US"标注为 PERSON（人名）。

NER 是每个结构化提取管线背后的主力。简历解析、合规日志扫描、医疗记录匿名化、搜索查询理解、聊天机器人响应的基础、法律合同提取。你从来看不到它，却始终依赖它。

本课从经典路径（规则系统、HMM、CRF）走向现代路径（BiLSTM-CRF，然后是 Transformer）。每一步都解决了前一步的特定局限。这个模式本身就是课程。

## 核心概念

**BIO 标注**（或 BILOU）将实体提取转化为序列标注问题。用 `B-TYPE`（实体开始）、`I-TYPE`（实体内部）或 `O`（任何实体之外）标注每个 token。

```
Apple    B-ORG
sued     O
Google   B-ORG
over     O
its      O
iPhone   B-PRODUCT
search   O
deal     O
in       O
the      O
US       B-GPE
.        O
```

多 token 实体链式连接：`New B-GPE`、`York I-GPE`、`City I-GPE`。理解 BIO 的模型可以提取任意跨度。

架构演进：

- **规则系统。** 正则表达式 + 词典查找。对已知实体精确率高，对新实体覆盖率为零。
- **HMM（隐马尔可夫模型）。** 给定标签下 token 的发射概率，标签到标签的转移概率，Viterbi 解码。在标注数据上训练。
- **CRF（条件随机场）。** 类似 HMM 但是判别式的，因此可以混合任意特征（词形、大小写、相邻词）。2026 年低资源部署的经典生产主力。
- **BiLSTM-CRF。** 用神经特征替代手工特征。LSTM 双向读取句子，顶部的 CRF 层强制一致的标签序列。
- **基于 Transformer。** 用 token 分类头微调 BERT。准确率最高，计算量最大。

## 动手实现

### 步骤一：BIO 标注辅助函数

```python
def spans_to_bio(tokens, spans):
    labels = ["O"] * len(tokens)
    for start, end, label in spans:
        labels[start] = f"B-{label}"
        for i in range(start + 1, end):
            labels[i] = f"I-{label}"
    return labels


def bio_to_spans(tokens, labels):
    spans = []
    current = None
    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if current:
                spans.append(current)
            current = (i, i + 1, label[2:])
        elif label.startswith("I-") and current and current[2] == label[2:]:
            current = (current[0], i + 1, current[2])
        else:
            if current:
                spans.append(current)
                current = None
    if current:
        spans.append(current)
    return spans
```

```python
>>> tokens = ["Apple", "sued", "Google", "over", "iPhone", "sales", "."]
>>> labels = ["B-ORG", "O", "B-ORG", "O", "B-PRODUCT", "O", "O"]
>>> bio_to_spans(tokens, labels)
[(0, 1, 'ORG'), (2, 3, 'ORG'), (4, 5, 'PRODUCT')]
```

### 步骤二：手工特征

对于经典（非神经）NER，特征是关键。有用的特征：

```python
def token_features(token, prev_token, next_token):
    return {
        "lower": token.lower(),
        "is_upper": token.isupper(),
        "is_title": token.istitle(),
        "has_digit": any(c.isdigit() for c in token),
        "suffix_3": token[-3:].lower(),
        "shape": word_shape(token),
        "prev_lower": prev_token.lower() if prev_token else "<BOS>",
        "next_lower": next_token.lower() if next_token else "<EOS>",
    }


def word_shape(word):
    out = []
    for c in word:
        if c.isupper():
            out.append("X")
        elif c.islower():
            out.append("x")
        elif c.isdigit():
            out.append("d")
        else:
            out.append(c)
    return "".join(out)
```

`word_shape("iPhone")` 返回 `xXxxxx`。`word_shape("USA-2024")` 返回 `XXX-dddd`。大小写模式对专有名词是高信号特征。

### 步骤三：简单的规则加词典基线

```python
ORG_GAZETTEER = {"Apple", "Google", "Microsoft", "OpenAI", "Meta", "Amazon", "Netflix"}
GPE_GAZETTEER = {"US", "USA", "UK", "India", "Germany", "France"}
PRODUCT_GAZETTEER = {"iPhone", "Android", "Windows", "ChatGPT", "Claude"}


def rule_based_ner(tokens):
    labels = []
    for token in tokens:
        if token in ORG_GAZETTEER:
            labels.append("B-ORG")
        elif token in GPE_GAZETTEER:
            labels.append("B-GPE")
        elif token in PRODUCT_GAZETTEER:
            labels.append("B-PRODUCT")
        else:
            labels.append("O")
    return labels
```

生产词典有数百万条从维基百科和 DBpedia 抓取的条目。覆盖率很好，但消歧（苹果公司 vs. 水果苹果）很差。这就是统计模型获胜的原因。

### 步骤四：CRF 步骤（概述，非完整实现）

从零实现 50 行的完整 CRF 在没有概率论基础的情况下很难理解，使用 `sklearn-crfsuite` 代替：

```python
import sklearn_crfsuite

def to_features(tokens):
    out = []
    for i, tok in enumerate(tokens):
        prev = tokens[i - 1] if i > 0 else ""
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        out.append({
            "word.lower()": tok.lower(),
            "word.isupper()": tok.isupper(),
            "word.istitle()": tok.istitle(),
            "word.isdigit()": tok.isdigit(),
            "word.suffix3": tok[-3:].lower(),
            "word.shape": word_shape(tok),
            "prev.word.lower()": prev.lower(),
            "next.word.lower()": nxt.lower(),
            "BOS": i == 0,
            "EOS": i == len(tokens) - 1,
        })
    return out


crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
X_train = [to_features(s) for s in sentences_tokenized]
crf.fit(X_train, bio_labels_train)
```

`c1` 和 `c2` 是 L1 和 L2 正则化。`all_possible_transitions=True` 让模型学习到非法序列（如 `O` 之后的 `I-ORG`）不可能出现，这就是 CRF 无需你明确写约束就能强制 BIO 一致性的方式。

### 步骤五：BiLSTM-CRF 的贡献

特征变为可学习的。输入：token 嵌入（GloVe 或 fastText）。LSTM 双向读取句子。拼接后的隐藏状态经过 CRF 输出层。CRF 仍然强制标签序列一致性；LSTM 用学习到的特征替代手工特征。

```python
import torch
import torch.nn as nn


class BiLSTM_CRF_Head(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_labels):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, n_labels)

    def forward(self, token_ids):
        e = self.embed(token_ids)
        h, _ = self.lstm(e)
        emissions = self.fc(h)
        return emissions
```

对于 CRF 层，使用 `torchcrf.CRF`（`pip install pytorch-crf`）。相比手工 CRF 的提升是可以测量的，但除非你有数万个标注句子，否则提升没有你预期的那么大。

## 生产使用

spaCy 开箱即用提供生产级 NER。

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple sued Google over its iPhone search deal in the US.")
for ent in doc.ents:
    print(f"{ent.text:20s} {ent.label_}")
```

```
Apple                ORG
Google               ORG
iPhone               ORG
US                   GPE
```

注意 `iPhone` 被标注为 `ORG` 而非 `PRODUCT`——spaCy 的小模型产品实体覆盖较弱。大模型（`en_core_web_lg`）更好，Transformer 模型（`en_core_web_trf`）更好。

Hugging Face 的 BERT 级 NER：

```python
from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
print(ner("Apple sued Google over its iPhone in the US."))
```

```
[{'entity_group': 'ORG', 'word': 'Apple', ...},
 {'entity_group': 'ORG', 'word': 'Google', ...},
 {'entity_group': 'MISC', 'word': 'iPhone', ...},
 {'entity_group': 'LOC', 'word': 'US', ...}]
```

`aggregation_strategy="simple"` 将连续的 B-X、I-X token 合并为一个跨度。不设置时，你会得到 token 级标签，需要自己合并。

### 基于 LLM 的 NER（2026 年的选项）

零样本和少样本 LLM NER 现在在许多领域与微调模型相当，在标注数据稀少时表现显著更好。

- **零样本提示。** 给 LLM 一个实体类型列表和示例结构，要求 JSON 输出。开箱即用；在新领域上准确率中等。
- **ZeroTuneBio 风格提示。** 将任务分解为候选提取→含义解释→判断→复查。多阶段提示（而非一次性）在生物医学 NER 上显著提升准确率，同样的模式适用于法律、金融和科学领域。
- **RAG 动态提示。** 对每次推理从小型标注种子集中检索最相似的标注样本，动态构建少样本提示。在 2026 年基准测试中，这使 GPT-4 生物医学 NER 的 F1 比静态提示高出 11-12%。
- **按实体类型分解。** 对于长文档，一次提取所有实体类型的单次调用会随长度增加丢失召回率。每种实体类型运行一次提取。推理成本更高，但准确率显著提升。这是临床记录和法律合同的标准模式。

2026 年的生产建议：在收集训练数据之前，先建立 LLM 零样本基线。通常 F1 足够好，你永远不需要微调。

### 经典 NER 仍然胜出的场景

即使 LLM 可用，经典 NER 在以下情况仍然胜出：

- 延迟预算低于 50ms。
- 你有数千个标注样本并需要 98%+ F1。
- 领域有稳定的本体，预训练的 CRF 或 BiLSTM 可以很好地迁移。
- 监管约束要求本地部署的非生成式模型。

### 经典 NER 失败的场景

- **领域偏移。** 在 CoNLL 上训练的 NER 用于法律合同，性能比词典还差。在你的领域上微调。
- **嵌套实体。** "Bank of America Tower"同时是 ORG（组织）和 FACILITY（设施）。标准 BIO 无法表示重叠跨度。你需要嵌套 NER（多遍或基于跨度的模型）。
- **长实体。** "United States Federal Deposit Insurance Corporation."Token 级模型有时会分割这个。使用 `aggregation_strategy` 或后处理。
- **稀疏类型。** DRUG_BRAND、ADVERSE_EVENT、DOSE 等医学 NER 标签。通用模型毫无概念。Scispacy 和 BioBERT 是那里的起点。

## 上手实践

将以下内容保存为 `outputs/skill-ner-picker.md`：

```markdown
---
name: ner-picker
description: Pick the right NER approach for a given extraction task.
version: 1.0.0
phase: 5
lesson: 06
tags: [nlp, ner, extraction]
---

Given a task description (domain, label set, language, latency, data volume), output:

1. Approach. Rule-based + gazetteer, CRF, BiLSTM-CRF, or transformer fine-tune.
2. Starting model. Name it (spaCy model ID, Hugging Face checkpoint ID, or "custom, trained from scratch").
3. Labeling strategy. BIO, BILOU, or span-based. Justify in one sentence.
4. Evaluation. Use `seqeval`. Always report entity-level F1 (not token-level).

Refuse to recommend fine-tuning a transformer for under 500 labeled examples unless the user already has a pretrained domain model. Flag nested entities as needing span-based or multi-pass models. Require a gazetteer audit if the user mentions "production scale" and labels are unchanged from CoNLL-2003.
```

## 练习

1. **简单。** 实现 `bio_to_spans`（`spans_to_bio` 的逆函数），并在 10 个句子上验证往返一致性。
2. **中等。** 在 CoNLL-2003 英语 NER 数据集上训练上述 sklearn-crfsuite CRF。使用 `seqeval` 报告每类实体 F1。典型结果：~84 F1。
3. **困难。** 在领域专属 NER 数据集（医学、法律或金融）上微调 `distilbert-base-cased`，与 spaCy 小模型进行比较。记录数据泄露检查，写下令你惊讶的发现。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| NER | 提取名字 | 用类型（PERSON、ORG、GPE、DATE……）标注 token 跨度。 |
| BIO | 标注方案 | `B-X` 开始，`I-X` 继续，`O` 在外。 |
| BILOU | 更好的 BIO | 添加 `L-X`（最后）、`U-X`（单元）以获得更清晰的边界。 |
| CRF | 结构化分类器 | 建模标签之间的转移，而非仅发射。强制有效序列。 |
| 嵌套 NER | 重叠实体 | 一个跨度与其子跨度是不同类型的实体。BIO 无法表达这种情况。 |
| 实体级 F1 | 正确的 NER 指标 | 预测跨度必须与真实跨度精确匹配。Token 级 F1 会高估准确率。 |

## 延伸阅读

- [Lample et al. (2016). Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360) — BiLSTM-CRF 论文，经典
- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — 引入了成为标准的 token 分类模式
- [spaCy linguistic features — named entities](https://spacy.io/usage/linguistic-features#named-entities) — `Doc.ents` 和 `Span` 上每个属性的实用参考
- [seqeval](https://github.com/chakki-works/seqeval) — 正确的指标库，始终使用它
