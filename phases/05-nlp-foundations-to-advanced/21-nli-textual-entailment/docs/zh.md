# 自然语言推断——文本蕴含

> "t 蕴含 h" 意味着读了 t 的人会得出 h 为真的结论。NLI 是预测蕴含/矛盾/中性的任务。表面看起来枯燥，在生产中却是承重墙。

**类型：** 学习
**语言：** Python
**前置条件：** Phase 5 · 05（情感分析）、Phase 5 · 13（问答系统）
**时长：** 约 60 分钟

## 问题背景

你构建了一个摘要系统，它生成了一段摘要。你怎么知道摘要中没有幻觉？

你构建了一个聊天机器人，它回答了"是"。你怎么知道这个答案有检索到的段落支撑？

你需要将 10,000 篇新闻文章按话题分类，但没有训练标签，能复用一个模型吗？

这三个问题都归结为**自然语言推断（NLI，Natural Language Inference）**。NLI 提问：给定前提 `t` 和假设 `h`，`h` 是被 `t` 蕴含、矛盾，还是中性（无关）？

- **幻觉检查：** `t` = 源文档，`h` = 摘要声明。非蕴含 = 幻觉。
- **基于依据的问答：** `t` = 检索到的段落，`h` = 生成的答案。非蕴含 = 捏造。
- **零样本分类：** `t` = 文档，`h` = 语言化的标签（"这段文字关于体育"）。蕴含 = 预测标签。

一个任务，三种生产用途。这就是为什么每个 RAG 评估框架都在底层内置了 NLI 模型。

## 核心概念

![NLI：三分类，前提 vs 假设](../assets/nli.svg)

**三个标签：**

- **蕴含（Entailment）。** `t` → `h`。"猫在垫子上"蕴含"有一只猫"。
- **矛盾（Contradiction）。** `t` → ¬`h`。"猫在垫子上"与"没有猫"矛盾。
- **中性（Neutral）。** 双向均无推断。"猫在垫子上"对于"猫饿了"是中性的。

**不是逻辑蕴含。** NLI 是**自然**语言推断——典型人类读者会推断的内容，而非严格逻辑。"John 遛了他的狗"在 NLI 中蕴含"John 有一只狗"，但严格的一阶逻辑只有在你将"拥有"公理化后才会承认这点。

**数据集：**

- **SNLI**（2015）。57 万人工标注对，以图像描述为前提，领域较窄。
- **MultiNLI**（2017）。跨 10 种体裁的 43.3 万对，2026 年的标准训练语料库。
- **ANLI**（2019）。对抗性 NLI，人类专门编写用于突破现有模型的样本，难度更高。
- **DocNLI、ConTRoL**（2020-21）。文档级别前提，测试多跳和长距离推断。

**架构。** Transformer 编码器（BERT、RoBERTa、DeBERTa）读取 `[CLS] 前提 [SEP] 假设 [SEP]`。`[CLS]` 表示输入三路 softmax，在 MNLI 上训练后，对于分布内样本准确率超过 90%。

**通过 NLI 实现零样本分类。** 给定文档和候选标签，将每个标签转化为假设（"这段文字关于体育"），计算每个标签的蕴含概率，选取最大值。这是 Hugging Face `zero-shot-classification` pipeline 背后的机制。

## 动手实现

### 步骤一：运行预训练 NLI 模型

```python
from transformers import pipeline

nli = pipeline("text-classification",
               model="facebook/bart-large-mnli",
               top_k=None)  # 返回所有标签；替代已废弃的 return_all_scores=True

premise = "The cat is sleeping on the couch."
hypothesis = "There is a cat in the room."

result = nli({"text": premise, "text_pair": hypothesis})[0]
print(result)
# [{'label': 'entailment', 'score': 0.97},
#  {'label': 'neutral', 'score': 0.02},
#  {'label': 'contradiction', 'score': 0.01}]
```

用于生产 NLI 的开源默认模型：`facebook/bart-large-mnli` 和 `microsoft/deberta-v3-large-mnli`。DeBERTa-v3 在排行榜上位居前列。

### 步骤二：零样本分类

```python
zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "The stock market rallied after the central bank cut interest rates."
labels = ["finance", "sports", "politics", "technology"]

result = zs(text, candidate_labels=labels)
print(result)
# {'labels': ['finance', 'politics', 'technology', 'sports'],
#  'scores': [0.92, 0.05, 0.02, 0.01]}
```

默认模板是"This example is about {label}."，可用 `hypothesis_template` 自定义。无需训练数据，无需微调，开箱即用。

### 步骤三：RAG 忠实度检查

```python
def is_faithful(answer, context, threshold=0.5):
    result = nli({"text": context, "text_pair": answer})[0]
    entail = next(s for s in result if s["label"] == "entailment")
    return entail["score"] > threshold
```

这是 RAGAS 忠实度的核心。将生成的答案拆分为原子声明，对每条声明与检索到的上下文进行检查，报告蕴含的比例。

### 步骤四：手写 NLI 分类器（概念性）

见 `code/main.py` 的仅用标准库的玩具实现：通过词汇重叠和否定检测来比较前提和假设。竞争力不如 Transformer 模型，但展示了任务的形状：两段文本输入，三路标签输出，损失为对 `{蕴含, 矛盾, 中性}` 的交叉熵。

## 常见坑

- **仅假设的捷径。** 模型仅从假设就能以约 60% 的准确率预测 SNLI 标签，因为"not"、"nobody"、"never"与矛盾标签相关。这是检测标签泄露的强基准。
- **词汇重叠启发式。** "每个子序列都被蕴含"的启发式通过了 SNLI，但在 HANS/ANLI 上失败。使用对抗性基准测试。
- **文档级退化。** 句子级 NLI 模型在文档级前提上 F1 下降 20+ 分。对于长上下文使用 DocNLI 训练的模型。
- **零样本模板敏感性。** "This example is about {label}"与"{label}"与"The topic is {label}"之间的准确率差异可超过 10 分。调整模板。
- **领域不匹配。** MNLI 在通用英语上训练。法律、医学和科学文本需要领域专属 NLI 模型（如 SciNLI、MedNLI）。

## 生产使用

2026 年技术栈：

| 使用场景 | 模型 |
|---------|------|
| 通用 NLI | `microsoft/deberta-v3-large-mnli` |
| 快速/边缘 | `cross-encoder/nli-deberta-v3-base` |
| 零样本分类（轻量） | `facebook/bart-large-mnli` |
| 文档级 NLI | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |
| 多语言 | `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` |
| RAG 幻觉检测 | RAGAS / DeepEval 内部的 NLI 层 |

2026 年的元模式：NLI 是文本理解的"万能胶"。任何时候你需要"A 是否支持 B？"或"A 是否与 B 矛盾？"——在考虑额外的 LLM 调用之前先考虑 NLI。

## 上手实践

将以下内容保存为 `outputs/skill-nli-picker.md`：

```markdown
---
name: nli-picker
description: Pick an NLI model, label template, and evaluation setup for a classification / faithfulness / zero-shot task.
version: 1.0.0
phase: 5
lesson: 21
tags: [nlp, nli, zero-shot]
---

Given a use case (faithfulness check, zero-shot classification, document-level inference), output:

1. Model. Named NLI checkpoint. Reason tied to domain, length, language.
2. Template (if zero-shot). Verbalization pattern. Example.
3. Threshold. Entailment cutoff for the decision rule. Reason based on calibration.
4. Evaluation. Accuracy on held-out labeled set, hypothesis-only baseline, adversarial subset.

Refuse to ship zero-shot classification without a 100-example labeled sanity check. Refuse to use a sentence-level NLI model on document-length premises. Flag any claim that NLI solves hallucination — it reduces it; it does not eliminate it.
```

## 练习

1. **简单。** 对涵盖全部三类的 20 个手工构建的（前提、假设、标签）三元组运行 `facebook/bart-large-mnli`，测量准确率，并加入对抗性"子序列启发式"陷阱（"I did not eat the cake" vs "I ate the cake"），看是否会出错。
2. **中等。** 在 100 个 AG News 标题上比较零样本模板 `"This text is about {label}"`、`"The topic is {label}"` 和 `"{label}"` 的准确率差异。
3. **困难。** 构建 RAG 忠实度检查器：原子声明分解 + 逐条 NLI 检查。在 50 个带金标上下文的 RAG 生成答案上评估，测量假阳性率和假阴性率与人工标注的对比。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| NLI | 自然语言推断 | 前提-假设关系的三路分类。 |
| RTE | 识别文本蕴含 | NLI 的旧名称；相同任务。 |
| 蕴含（Entailment） | "t 蕴含 h" | 给定 t，典型读者会得出 h 为真。 |
| 矛盾（Contradiction） | "t 排除 h" | 给定 t，典型读者会得出 h 为假。 |
| 中性（Neutral） | "悬而未决" | t 到 h 双向均无推断。 |
| 零样本分类 | NLI 作为分类器 | 将标签语言化为假设，选取最大蕴含分。 |
| 忠实度（Faithfulness） | 答案是否有依据 | 对（检索上下文、生成答案）进行 NLI。 |

## 延伸阅读

- [Bowman et al. (2015). A large annotated corpus for learning natural language inference](https://arxiv.org/abs/1508.05326) — SNLI 论文
- [Williams, Nangia, Bowman (2017). A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference](https://arxiv.org/abs/1704.05426) — MultiNLI 论文
- [Nie et al. (2019). Adversarial NLI](https://arxiv.org/abs/1910.14599) — ANLI 基准
- [Yin, Hay, Roth (2019). Benchmarking Zero-shot Text Classification](https://arxiv.org/abs/1909.00161) — NLI 作为分类器
- [He et al. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654) — 2026 年 NLI 主力模型
