# 问答系统

> 三种系统塑造了现代问答。抽取式找到文本片段。检索增强将答案锚定在文档中。生成式产生答案。每一个现代 AI 助手都是这三者的混合。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 5 · 11（机器翻译）、Phase 5 · 10（注意力机制）
**时长：** 约 75 分钟

## 问题背景

用户输入"第一部 iPhone 是什么时候发布的？"，期望得到"2007 年 6 月 29 日"。不是"苹果的历史悠久而多彩"，不是孤零零的"2007"，而是一个直接、有据可查、正确的答案。

过去十年，三种架构主导了问答领域。

- **抽取式问答（Extractive QA）。** 给定问题和一个已知包含答案的段落，在段落中找到答案片段的起始和结束索引。SQuAD 是标准基准。
- **开放域问答（Open-domain QA）。** 不给定段落。先检索相关段落，再抽取或生成答案。这是当今所有 RAG 管线的基础。
- **生成式/闭卷问答（Generative/Closed-book QA）。** 大型语言模型从其参数记忆中作答。无需检索。推理最快，事实可靠性最低。

2026 年的趋势是混合式：检索最佳的几个段落，然后提示生成模型以这些段落为依据作答。这就是 RAG（检索增强生成），第 14 课深入讲解检索那一半。本课构建问答那一半。

## 核心概念

![问答架构：抽取式、检索增强式、生成式](../assets/qa.svg)

**抽取式。** 用 Transformer（BERT 族）联合编码问题和段落。训练两个分类头，预测答案的起始和结束 token 索引。损失是对有效位置的交叉熵。输出是段落中的一个文本片段。从构造上看，永远不会产生幻觉，也永远无法处理段落无法回答的问题。

**检索增强（RAG）。** 两个阶段。首先，检索器从语料库中找出前 `k` 个段落。其次，阅读器（抽取式或生成式）利用这些段落生成答案。检索器-阅读器的分离使得两者可以独立训练和评估。现代 RAG 通常在两者之间加入重排器。

**生成式。** 仅解码器的 LLM（GPT、Claude、Llama）从学习到的权重中作答。无检索步骤。在常识性知识上表现出色，在罕见或最新事实上表现灾难性。幻觉率与预训练数据中的事实频率成反比。

## 动手实现

### 步骤一：用预训练模型做抽取式问答

```python
from transformers import pipeline

qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

passage = (
    "Apple Inc. released the first iPhone on June 29, 2007. "
    "The device was announced by Steve Jobs at Macworld in January 2007."
)
question = "When was the first iPhone released?"

answer = qa(question=question, context=passage)
print(answer)
```

```python
{'score': 0.98, 'start': 57, 'end': 70, 'answer': 'June 29, 2007'}
```

`deepset/roberta-base-squad2` 在 SQuAD 2.0 上训练，后者包含无法回答的问题。默认情况下，`question-answering` pipeline 会返回得分最高的文本片段，即使模型的空答案得分更高——它不会自动返回空答案。要获得明确的"无答案"行为，在 pipeline 调用中传入 `handle_impossible_answer=True`：只有当空答案得分超过所有片段得分时，pipeline 才返回空答案。无论如何都要检查 `score` 字段。

### 步骤二：检索增强管线（草图）

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

corpus = [
    "Apple Inc. released the first iPhone on June 29, 2007.",
    "Macworld 2007 featured the iPhone announcement by Steve Jobs.",
    "Android launched in 2008 as Google's mobile operating system.",
    "The first iPod was released in 2001.",
]
corpus_embeddings = encoder.encode(corpus, normalize_embeddings=True)


def retrieve(question, top_k=2):
    q_emb = encoder.encode([question], normalize_embeddings=True)
    sims = (corpus_embeddings @ q_emb.T).squeeze()
    order = np.argsort(-sims)[:top_k]
    return [corpus[i] for i in order]


def answer(question):
    passages = retrieve(question, top_k=2)
    combined = " ".join(passages)
    return qa(question=question, context=combined)


print(answer("When was the first iPhone released?"))
```

两阶段管线。密集检索器（Sentence-BERT）通过语义相似度找到相关段落。抽取式阅读器（RoBERTa-SQuAD）从合并的顶级段落中提取答案片段。适用于小型语料库。对于百万级文档语料库，使用 FAISS 或向量数据库。

### 步骤三：带 RAG 的生成式问答

```python
def rag_generate(question, llm):
    passages = retrieve(question, top_k=3)
    prompt = f"""Context:
{chr(10).join('- ' + p for p in passages)}

Question: {question}

Answer using only the context above. If the context does not contain the answer, say "I don't know."
"""
    return llm(prompt)
```

提示模式至关重要。明确告诉模型以上下文为依据，并在上下文不足时返回"I don't know"，与朴素提示相比，幻觉率降低 40-60%。更复杂的模式还会添加引用、置信度评分和结构化抽取。

### 步骤四：反映真实世界的评估

SQuAD 使用**精确匹配（EM，Exact Match）**和**token 级 F1**。EM 是归一化后的严格匹配（小写化、去标点、去冠词）——要么预测与参考完全匹配，要么得 0 分。F1 基于预测和参考的 token 重叠计算，给予部分分数。两者都低估了改写：比如"June 29, 2007"与"June 29th, 2007"通常得到 0 的 EM（序数词打破了归一化），但因重叠 token 仍能获得较高的 F1。

面向生产的 QA 评估：

- **答案准确性**（LLM 评审或人工评审，因为指标无法捕捉语义等价性）。
- **引用准确性。** 引用的段落是否真正支持答案？通过生成的引用与检索段落之间的字符串匹配即可自动检查。
- **拒绝校准（Refusal calibration）。** 当答案不在检索段落中时，系统是否正确说"I don't know"？衡量错误置信率。
- **检索召回率。** 在评估阅读器之前，先衡量检索器是否将正确段落纳入前 `k` 名。阅读器无法修复缺失的段落。

### RAGAS：2026 年的生产评估框架

`RAGAS` 专为 RAG 系统而设计，是 2026 年的默认生产评估方案。无需金标准参考即可对四个维度打分：

- **忠实度（Faithfulness）。** 答案中的每条声明是否来自检索到的上下文？通过基于 NLI 的蕴含关系衡量。这是你的主要幻觉指标。
- **答案相关性（Answer relevance）。** 答案是否回应了问题？通过从答案生成假设问题并与真实问题对比来衡量。
- **上下文精确率（Context precision）。** 检索到的块中，真正相关的占多少？精确率低 = 提示中存在噪声。
- **上下文召回率（Context recall）。** 检索集合是否包含了所有必要信息？召回率低 = 阅读器无法成功作答。

无参考评分让你在没有精心策划的金标准答案的情况下，对线上生产流量进行评估。在开放式问题上叠加 LLM 作为评审，处理精确匹配指标无用的场景。

`pip install ragas`。接入你的检索器和阅读器，每个查询获得四个标量，对回归发出警报。

## 生产使用

2026 年技术栈：

| 使用场景 | 推荐 |
|---------|------|
| 给定段落，找答案片段 | `deepset/roberta-base-squad2` |
| 固定语料库，不接受闭卷 | RAG：密集检索器 + LLM 阅读器 |
| 文档存储上的实时问答 | RAG，带混合检索（BM25 + 密集）+ 重排器（第 14 课） |
| 对话式问答（追问） | LLM + 对话历史，每轮带 RAG |
| 高度事实性、受监管领域 | 对权威语料库做抽取式；绝不单独使用生成式 |

2026 年抽取式问答已不再流行，因为带 LLM 的 RAG 能处理更多场景。但在需要逐字引用的场景中它仍在部署：法律研究、合规监管、审计工具。

## 上手实践

将以下内容保存为 `outputs/skill-qa-architect.md`：

```markdown
---
name: qa-architect
description: Choose QA architecture, retrieval strategy, and evaluation plan.
version: 1.0.0
phase: 5
lesson: 13
tags: [nlp, qa, rag]
---

Given requirements (corpus size, question type, factuality constraint, latency budget), output:

1. Architecture. Extractive, RAG with extractive reader, RAG with generative reader, or closed-book LLM. One-sentence reason.
2. Retriever. None, BM25, dense (name the encoder), or hybrid.
3. Reader. SQuAD-tuned model, LLM by name, or "domain-fine-tuned DistilBERT."
4. Evaluation. EM + F1 for extractive benchmarks; answer accuracy + citation accuracy + refusal calibration for production. Name what you are measuring and how you are measuring it.

Refuse closed-book LLM answers for regulatory or compliance-sensitive questions. Refuse any QA system without a retrieval-recall baseline (you cannot evaluate the reader without knowing the retriever surfaced the right passage). Flag questions that require multi-hop reasoning as needing specialized multi-hop retrievers like HotpotQA-trained systems.
```

## 练习

1. **简单。** 在 10 篇维基百科段落上运行上述 SQuAD 抽取式管线。手工构造 10 个问题，测量答案正确的频率。如果段落和问题干净，你应该看到 7-9 个正确。
2. **中等。** 添加拒绝分类器：当检索最高分低于阈值（如余弦相似度 0.3）时，返回"I don't know"而不调用阅读器。在保留集上调整阈值。
3. **困难。** 在你选择的 10,000 篇文档语料库上构建 RAG 管线。实现混合检索（BM25 + 密集）并使用 RRF 融合（见第 14 课）。衡量有无混合检索步骤时的答案准确性，记录哪类问题受益最多。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 抽取式问答（Extractive QA） | 找答案片段 | 预测给定段落中答案的起始和结束索引。 |
| 开放域问答（Open-domain QA） | 语料库上的问答 | 无给定段落；必须先检索再作答。 |
| RAG | 检索后生成 | 检索增强生成。检索器 + 阅读器管线。 |
| SQuAD | 标准基准 | 斯坦福问答数据集，使用 EM + F1 指标。 |
| 幻觉（Hallucination） | 捏造答案 | 阅读器输出中检索上下文不支持的内容。 |
| 拒绝校准（Refusal calibration） | 知道何时沉默 | 系统在无法作答时正确说"I don't know"。 |

## 延伸阅读

- [Rajpurkar et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250) — 基准论文
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — DPR，问答领域标准密集检索器
- [Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — 命名 RAG 的论文
- [Gao et al. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) — 全面的 RAG 综述
