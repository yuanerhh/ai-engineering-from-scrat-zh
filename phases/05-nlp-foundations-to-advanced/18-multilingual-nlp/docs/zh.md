# 多语言 NLP

> 一个模型，100+ 种语言，其中大多数语言没有训练数据。跨语言迁移是 2020 年代的实用奇迹。

**类型：** 学习
**语言：** Python
**前置条件：** Phase 5 · 04（GloVe、FastText、子词）、Phase 5 · 11（机器翻译）
**时长：** 约 45 分钟

## 问题背景

英语有数十亿个标注样本，乌尔都语有数千个，迈蒂利语几乎没有。任何服务全球受众的实用 NLP 系统都必须在没有任务专属训练数据的长尾语言上工作。

多语言模型通过同时在多种语言上训练一个模型来解决这个问题。共享的表示使模型能够将在高资源语言中学到的技能迁移到低资源语言。在英语情感分析上微调模型，它开箱即可在乌尔都语上产生出奇好的情感预测。这就是**零样本跨语言迁移（Zero-shot cross-lingual transfer）**，它重新塑造了 NLP 在全球部署的方式。

本课指出权衡点、标准模型，以及新接触多语言工作的团队常踩的坑：选择迁移的源语言。

## 核心概念

![通过共享多语言嵌入空间的跨语言迁移](../assets/multilingual.svg)

**共享词汇表。** 多语言模型使用在所有目标语言文本上训练的 SentencePiece 或 WordPiece 分词器。词汇表是共享的：同一子词单元在相关语言中代表相同的词素。英语和意大利语中的 `anti-` 得到相同的 token。

**共享表示。** 在多种语言上进行掩码语言建模预训练的 Transformer 学会了：不同语言中语义相似的句子产生相似的隐藏状态。mBERT、XLM-R 和 NLLB 都表现出这一特性。英语"cat"的嵌入与法语"chat"和西班牙语"gato"相近，完整句子的嵌入也是如此。

**零样本迁移。** 在一种语言（通常是英语）的标注数据上微调模型，推断时在模型支持的任何其他语言上运行，无需目标语言标签。对于类型学相近的语言效果强，对于距离远的语言效果较弱。

**少样本微调。** 加入 100-500 个目标语言标注样本，分类任务的准确率跳升至英语基准的 95-98%。这是多语言 NLP 中单一最具成本效益的杠杆。

## 模型概览

| 模型 | 年份 | 覆盖范围 | 备注 |
|------|------|---------|------|
| mBERT | 2018 | 104 种语言 | 在维基百科上训练，第一个实用的多语言语言模型，低资源效果弱 |
| XLM-R | 2019 | 100 种语言 | 在 CommonCrawl 上训练（远大于维基百科），设定跨语言基准。Base 270M，Large 550M |
| XLM-V | 2023 | 100 种语言 | XLM-R 配备 100 万 token 词汇表（vs 25 万），低资源效果更好 |
| mT5 | 2020 | 101 种语言 | 用于多语言生成的 T5 架构 |
| NLLB-200 | 2022 | 200 种语言 | Meta 的翻译模型，包含 55 种低资源语言 |
| BLOOM | 2022 | 46 种语言 + 13 种编程语言 | 开源 176B 多语言 LLM |
| Aya-23 | 2024 | 23 种语言 | Cohere 的多语言 LLM，在阿拉伯语、印地语、斯瓦希里语上表现强 |

按用例选择：分类任务用 XLM-R-base 作为合理默认。生成任务根据翻译与开放式生成选择 mT5 或 NLLB。LLM 风格的工作配合 Aya-23 或带明确多语言提示的 Claude。

## 源语言决策（2026 年研究进展）

大多数团队默认以英语作为微调源语言。近期研究（2026 年）表明这往往是错误的。

语言相似度比原始语料库大小更能预测迁移质量。对于斯拉夫语系目标语言，德语或俄语通常优于英语；对于印度语系目标语言，印地语通常优于英语。**qWALS** 相似度指标（2026 年，基于世界语言结构图集特征）对此进行了量化。**LANGRANK**（Lin 等人，ACL 2019）是一种较早的独立方法，通过语言相似度、语料库大小和遗传相关性的综合指标对候选源语言排序。

实用规则：如果目标语言有类型学上接近的高资源近亲，先尝试在那种语言上微调，然后与英语微调进行比较。

## 动手实现

### 步骤一：零样本跨语言分类

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("joeddav/xlm-roberta-large-xnli")
model = AutoModelForSequenceClassification.from_pretrained("joeddav/xlm-roberta-large-xnli")


def classify(text, candidate_labels, hypothesis_template="This text is about {}."):
    scores = {}
    for label in candidate_labels:
        hypothesis = hypothesis_template.format(label)
        inputs = tok(text, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        entail_score = torch.softmax(logits, dim=-1)[2].item()
        scores[label] = entail_score
    return dict(sorted(scores.items(), key=lambda x: -x[1]))


print(classify("I love this product!", ["positive", "negative", "neutral"]))
print(classify("मुझे यह उत्पाद पसंद है!", ["positive", "negative", "neutral"]))
print(classify("J'adore ce produit !", ["positive", "negative", "neutral"]))
```

一个模型，三种语言，相同的 API。在 NLI 数据上训练的 XLM-R 通过蕴含技巧很好地迁移到分类任务。

### 步骤二：多语言嵌入空间

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

pairs = [
    ("The cat is sleeping.", "Le chat dort."),
    ("The cat is sleeping.", "El gato está durmiendo."),
    ("The cat is sleeping.", "Die Katze schläft."),
    ("The cat is sleeping.", "The dog is barking."),
]

for eng, other in pairs:
    emb_eng = model.encode([eng], normalize_embeddings=True)[0]
    emb_other = model.encode([other], normalize_embeddings=True)[0]
    sim = float(np.dot(emb_eng, emb_other))
    print(f"  {eng!r} <-> {other!r}: cos={sim:.3f}")
```

翻译在嵌入空间中位置相近，而不同含义的英语句子则距离更远。这就是跨语言检索、聚类和相似度计算的基础。

### 步骤三：少样本微调策略

```python
from transformers import TrainingArguments, Trainer
from datasets import Dataset


def few_shot_finetune(base_model, base_tokenizer, examples):
    ds = Dataset.from_list(examples)

    def tokenize_fn(ex):
        out = base_tokenizer(ex["text"], truncation=True, max_length=128)
        out["labels"] = ex["label"]
        return out

    ds = ds.map(tokenize_fn)
    args = TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=8,
        num_train_epochs=5,
        learning_rate=2e-5,
        save_strategy="no",
    )
    trainer = Trainer(model=base_model, args=args, train_dataset=ds)
    trainer.train()
    return base_model
```

对于 100-500 个目标语言样本，`num_train_epochs=5` 和 `learning_rate=2e-5` 是安全的默认值。更高的学习率会导致多语言对齐崩溃，模型退化为仅英语模型。

## 真正有效的评估

- **每种语言的保留集准确率。** 不要聚合。聚合数字会掩盖长尾问题。
- **与单语基准对比。** 对于有足够数据的语言，从头训练的单语模型有时会击败多语言模型，需要测试。
- **实体级测试。** 目标语言中的命名实体。多语言模型对于拉丁字母以外的文字往往分词较弱。
- **跨语言一致性。** 两种语言中相同含义的内容应产生相同的预测，测量两者之间的差距。

## 生产使用

2026 年技术栈：

| 任务 | 推荐 |
|------|------|
| 分类，100 种语言 | 微调的 XLM-R-base（约 270M） |
| 零样本文本分类 | `joeddav/xlm-roberta-large-xnli` |
| 多语言句子嵌入 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| 翻译，200 种语言 | `facebook/nllb-200-distilled-600M`（见第 11 课） |
| 生成式多语言 | Claude、GPT-4、Aya-23、mT5-XXL |
| 低资源语言 NLP | XLM-V 或在相关高资源语言上的领域专属微调 |

如果性能重要，始终预算用于目标语言的微调。零样本只是起点，不是最终答案。

### 分词代价（低资源语言出错的地方）

多语言模型在所有语言间共享一个分词器，而该词汇表是在以英语、法语、西班牙语、中文、德语为主的语料库上训练的。对于主导集之外的任何语言，三种代价会悄悄叠加：

- **生育率代价（Fertility tax）。** 低资源语言文本每个词需要的 token 远多于英语。一个印地语句子可能需要等效英语句子 3-5 倍的 token，直接消耗上下文窗口、训练效率和延迟。
- **变体恢复代价。** 每一个拼写错误、变音符号变体、Unicode 规范化不匹配或大小写变化在嵌入空间中都变成一个冷启动的无关序列。模型无法学习母语者认为显而易见的正字法对应关系。
- **容量溢出代价。** 前两种代价消耗上下文位置、层深度和嵌入维度。留给实际推理的资源比高资源语言从同一模型获得的系统性更少。

实际症状：模型在印地语上训练正常，损失曲线看起来对，评估困惑度看起来合理，但生产输出微妙地出错。形态在句子中间崩溃，罕见的屈折变化无法恢复。**你无法通过扩大数据规模来解决一个有问题的分词器。**

缓解措施：选择对目标语言有良好覆盖的分词器（XLM-V 的 100 万 token 词汇表是直接修复方案）；在训练前验证目标文本上的分词生育率；使用字节级回退（SentencePiece `byte_fallback=True`，或类似 GPT-2 风格的字节级 BPE），确保真正长尾的文字永远不会出现 OOV。

## 上手实践

将以下内容保存为 `outputs/skill-multilingual-picker.md`：

```markdown
---
name: multilingual-picker
description: Pick source language, target model, and evaluation plan for a multilingual NLP task.
version: 1.0.0
phase: 5
lesson: 18
tags: [nlp, multilingual, cross-lingual]
---

Given requirements (target languages, task type, available labeled data per language), output:

1. Source language for fine-tuning. Default English; check LANGRANK or qWALS if target language has a typologically close high-resource language.
2. Base model. XLM-R (classification), mT5 (generation), NLLB (translation), Aya-23 (generative LLM).
3. Few-shot budget. Start with 100-500 target-language examples if available. Zero-shot only if labeling is infeasible.
4. Evaluation plan. Per-language accuracy (not aggregate), cross-lingual consistency, entity-level F1 on non-Latin scripts.

Refuse to ship a multilingual model without per-language evaluation — aggregate metrics hide long-tail failures. Flag scripts with low tokenization coverage (Amharic, Tigrinya, many African languages) as needing a model with byte-fallback (SentencePiece with byte_fallback=True, or byte-level tokenizer like GPT-2).
```

## 练习

1. **简单。** 对英语、法语、印地语和阿拉伯语各 10 个句子运行零样本分类管线，报告每种语言的准确率。你应该看到法语效果强，印地语尚可，阿拉伯语参差不齐。
2. **中等。** 用 `paraphrase-multilingual-MiniLM-L12-v2` 在小型混合语言语料库上构建跨语言检索器。用英语查询，检索任意语言的文档，测量 Recall@5。
3. **困难。** 比较英语源语言和印地语源语言在印地语分类任务上的微调效果。在两种方案下各用 500 个目标语言样本进行少样本微调，报告哪种源语言产生更好的印地语准确率及差距。这是 LANGRANK 论点的小型复现。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 多语言模型（Multilingual model） | 一个模型，多种语言 | 跨语言共享词汇表和参数。 |
| 跨语言迁移（Cross-lingual transfer） | 一种语言训练，另一种语言运行 | 在源语言上微调，在目标语言上评估，无需目标语言标注。 |
| 零样本（Zero-shot） | 无目标语言标注 | 无需在目标语言上微调的迁移。 |
| 少样本（Few-shot） | 少量目标语言标注 | 用于微调的 100-500 个目标语言样本。 |
| mBERT | 第一个多语言语言模型 | 在维基百科上预训练的 104 语言 BERT。 |
| XLM-R | 标准跨语言基准 | 在 CommonCrawl 上预训练的 100 语言 RoBERTa。 |
| NLLB | Meta 的 200 语言机器翻译 | No Language Left Behind，包含 55 种低资源语言。 |

## 延伸阅读

- [Conneau et al. (2019). Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116) — XLM-R 论文
- [Pires, Schlinger, Garrette (2019). How Multilingual is Multilingual BERT?](https://arxiv.org/abs/1906.01502) — 开启跨语言迁移研究线的分析论文
- [Costa-jussà et al. (2022). No Language Left Behind](https://arxiv.org/abs/2207.04672) — NLLB-200 论文
- [Üstün et al. (2024). Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model](https://arxiv.org/abs/2402.07827) — Aya，Cohere 的多语言 LLM
- [Language Similarity Predicts Cross-Lingual Transfer Learning Performance (2026)](https://www.mdpi.com/2504-4990/8/3/65) — qWALS / LANGRANK 源语言选择论文
