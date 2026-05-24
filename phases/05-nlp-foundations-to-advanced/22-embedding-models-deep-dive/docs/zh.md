# 嵌入模型——2026 年深度解析

> Word2Vec 给你每个词一个向量。现代嵌入模型给你每个段落一个向量，跨语言，具有稀疏、密集和多向量视图，大小适合你的索引。选错了，你的 RAG 就会检索到错误的内容。

**类型：** 学习
**语言：** Python
**前置条件：** Phase 5 · 03（Word2Vec）、Phase 5 · 14（信息检索）
**时长：** 约 60 分钟

## 问题背景

你的 RAG 系统有 40% 的时间检索到错误的段落，罪魁祸首很少是向量数据库或提示，而是嵌入模型。

2026 年选择嵌入意味着在五个维度上权衡：

1. **密集 vs 稀疏 vs 多向量。** 每个段落一个向量，还是每个 token 一个，还是稀疏加权词袋。
2. **语言覆盖。** 单语英语模型在纯英语任务上仍然胜出；语料库混合时多语言模型胜出。
3. **上下文长度。** 512 token vs 8,192 vs 32,768——实际有效容量通常是宣传最大值的 60-70%。
4. **维度预算。** 全精度 3,072 个浮点数 = 每个向量 12 KB。1 亿个向量，存储每月约 1,300 美元。Matryoshka 截断可以节省 4 倍。
5. **开源 vs 托管。** 开源权重意味着你掌控技术栈和数据；托管意味着以控制权换取始终最新。

本课列举权衡点，使你能基于证据而非上季度流行的东西做选择。

## 核心概念

![密集、稀疏和多向量嵌入](../assets/embedding-modes.svg)

**密集嵌入。** 每个段落一个向量（通常 384-3,072 维）。余弦相似度按语义接近程度对段落排序。OpenAI `text-embedding-3-large`、BGE-M3 密集模式、Voyage-3，是默认选择。

**稀疏嵌入。** SPLADE 风格。Transformer 为每个词汇 token 预测权重，然后将大多数置零。结果是一个大小为 |词汇表| 的稀疏向量。捕获词汇匹配（类似 BM25），但具有学习到的词项权重，在关键词密集的查询上表现强。

**多向量（后期交互）。** ColBERTv2、Jina-ColBERT。每个 token 一个向量。用 MaxSim 评分：对每个查询 token，找到最相似的文档 token，对分数求和。存储和评分成本更高，但在长查询和领域专属语料库上胜出。

**BGE-M3：三者合一。** 单一模型同时输出密集、稀疏和多向量表示。每种都可以独立查询；分数通过加权求和融合。2026 年在希望从一个检查点获得灵活性时的默认选择。

**Matryoshka 表示学习。** 经过训练使向量的前 N 维形成有用的独立嵌入。将 1,536 维向量截断到 256 维，付出约 1% 的精度换取 6 倍的存储节省。OpenAI text-3、Cohere v4、Voyage-4、Jina v5、Gemini Embedding 2、Nomic v1.5+ 支持。

### MTEB 排行榜讲述的是部分故事

大规模文本嵌入基准——发布时包含 8 种任务类型的 56 个任务（2022 年），在 MTEB v2 中扩展到 100+ 个任务。2026 年初，Gemini Embedding 2 在检索中领先（67.71 MTEB-R），Cohere embed-v4 在综合排名领先（65.2 MTEB），BGE-M3 领先开源多语言（63.0）。排行榜是必要的，但不充分——始终在你的领域上做基准测试。

### 三层模式

| 使用场景 | 模式 |
|---------|------|
| 快速首轮筛选 | 密集双编码器（BGE-M3，text-3-small） |
| 提升召回率 | 稀疏（SPLADE，BGE-M3 稀疏）+ RRF 融合 |
| 前 50 精确化 | 多向量（ColBERTv2）或交叉编码器重排 |

大多数生产技术栈使用全部三层。

## 动手实现

### 步骤一：基准——用 Sentence-BERT 做密集嵌入

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
corpus = [
    "The first iPhone launched in 2007.",
    "Apple released the iPod in 2001.",
    "Android is an operating system from Google.",
]
emb = encoder.encode(corpus, normalize_embeddings=True)

query = "When was the iPhone released?"
q_emb = encoder.encode([query], normalize_embeddings=True)[0]
scores = emb @ q_emb
print(sorted(enumerate(scores), key=lambda x: -x[1]))
```

`normalize_embeddings=True` 使点积等于余弦相似度，始终设置它。

### 步骤二：Matryoshka 截断

```python
def truncate(vectors, dim):
    out = vectors[:, :dim]
    return out / np.linalg.norm(out, axis=1, keepdims=True)

emb_256 = truncate(emb, 256)
emb_128 = truncate(emb, 128)
```

截断后重新归一化。Nomic v1.5、OpenAI text-3 和 Voyage-4 经过训练使前几个维度级别无损。非 Matryoshka 模型（原始 Sentence-BERT）在截断时急剧退化。

### 步骤三：BGE-M3 多功能

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

output = model.encode(
    corpus,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)
# output["dense_vecs"]:    (n_docs, 1024)
# output["lexical_weights"]: list of dict {token_id: weight}
# output["colbert_vecs"]:  list of (n_tokens, 1024) arrays
```

三个索引，一次推断调用。分数融合：

```python
dense_score = ... # 对 dense_vecs 做余弦
sparse_score = model.compute_lexical_matching_score(q_lex, d_lex)
colbert_score = model.colbert_score(q_col, d_col)
final = 0.4 * dense_score + 0.2 * sparse_score + 0.4 * colbert_score
```

在你的领域上调整权重。

### 步骤四：在自定义任务上进行 MTEB 评估

```python
from mteb import MTEB

tasks = ["ArguAna", "SciFact", "NFCorpus"]
evaluation = MTEB(tasks=tasks)
results = evaluation.run(encoder, output_folder="./mteb-results")
```

在**有代表性的**子集上运行候选模型，不要只信任排行榜排名——你的领域很重要。

### 步骤五：从零手写余弦相似度

见 `code/main.py`。仅用标准库的平均哈希技巧嵌入，与 Transformer 嵌入竞争力不强，但展示了任务形状：分词 → 向量 → 归一化 → 点积。

## 常见坑

- **查询和文档使用相同模型路径。** 有些模型（Voyage、Jina-ColBERT）使用非对称编码——查询和文档通过不同路径。始终查看模型卡。
- **缺少前缀。** `bge-*` 模型的查询需要前置 `"Represent this sentence for searching relevant passages: "`。忘记会损失 3-5 个召回点。
- **过度截断 Matryoshka。** 1,536 → 256 通常安全，1,536 → 64 不安全。在评估集上验证。
- **上下文截断。** 大多数模型静默截断超过最大长度的输入。长文档需要分块（见第 23 课）。
- **忽略延迟尾部。** MTEB 分数隐藏了 p99 延迟，一个 6 亿参数的模型可能比 3.35 亿参数的模型高 2 分，但每次查询成本高 3 倍。

## 生产使用

2026 年技术栈：

| 场景 | 选择 |
|------|------|
| 纯英语，快速，API | `text-embedding-3-large` 或 `voyage-3-large` |
| 开源权重，英语 | `BAAI/bge-large-en-v1.5` |
| 开源权重，多语言 | `BAAI/bge-m3` 或 `Qwen3-Embedding-8B` |
| 长上下文（32k+） | Voyage-3-large、Cohere embed-v4、Qwen3-Embedding-8B |
| 仅 CPU 部署 | Nomic Embed v2（1.37 亿参数，MoE） |
| 存储受限 | Matryoshka 截断 + int8 量化 |
| 关键词密集的查询 | 添加 SPLADE 稀疏，与密集 RRF 融合 |

2026 年模式：从 BGE-M3 或 text-3-large 开始，用 MTEB 在你的领域评估，如果领域专属模型赢了 3 分以上则替换。

## 上手实践

将以下内容保存为 `outputs/skill-embedding-picker.md`：

```markdown
---
name: embedding-picker
description: Pick embedding model, dimension, and retrieval mode for a given corpus and deployment.
version: 1.0.0
phase: 5
lesson: 22
tags: [nlp, embeddings, retrieval]
---

Given a corpus (size, languages, domain, avg length), deployment target (cloud / edge / on-prem), latency budget, and storage budget, output:

1. Model. Named checkpoint or API. One-sentence reason.
2. Dimension. Full / Matryoshka-truncated / int8-quantized. Reason tied to storage budget.
3. Mode. Dense / sparse / multi-vector / hybrid. Reason.
4. Query prefix / template if required by the model card.
5. Evaluation plan. MTEB tasks relevant to domain + held-out domain eval with nDCG@10.

Refuse recommendations that truncate Matryoshka to <64 dims without domain validation. Refuse ColBERTv2 for corpora under 10k passages (overhead not justified). Flag long-document corpora (>8k tokens) routed to models with 512-token windows.
```

## 练习

1. **简单。** 用 `bge-small-en-v1.5` 以全维度（384）和 Matryoshka 128 编码 100 个句子，在 10 个查询上测量 MRR 下降。
2. **中等。** 在你领域的 500 个段落上比较 BGE-M3 密集、稀疏和 ColBERT，哪种在 Recall@10 上胜出？RRF 融合是否优于最好的单一模式？
3. **困难。** 对三个候选模型在你的前 2 个领域任务上运行 MTEB，报告 MTEB 分数、100 次查询批次上的 p99 延迟以及每百万次查询的成本，选出帕累托最优方案。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 密集嵌入（Dense embedding） | 那个向量 | 每段文本一个固定大小的向量，余弦相似度用于排序。 |
| 稀疏嵌入（Sparse embedding） | 学习的 BM25 | 每个词汇 token 一个权重，大多数为零，端到端训练。 |
| 多向量（Multi-vector） | ColBERT 风格 | 每个 token 一个向量；MaxSim 评分；更大索引，更好召回。 |
| Matryoshka | 俄罗斯套娃技巧 | 向量的前 N 维自身就是有效的较小嵌入。 |
| MTEB | 那个基准 | 大规模文本嵌入基准——发布时 56 个任务，v2 中 100+。 |
| BEIR | 检索基准 | 18 个零样本检索任务；常用于跨领域鲁棒性引用。 |
| 非对称编码（Asymmetric encoding） | 查询 ≠ 文档路径 | 模型对查询和文档使用不同的投影。 |

## 延伸阅读

- [Reimers, Gurevych (2019). Sentence-BERT](https://arxiv.org/abs/1908.10084) — 双编码器论文
- [Muennighoff et al. (2022). MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316) — 排行榜论文
- [Chen et al. (2024). BGE-M3: Multi-lingual, Multi-functionality, Multi-granularity](https://arxiv.org/abs/2402.03216) — 三模式统一模型
- [Kusupati et al. (2022). Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) — 维度梯度训练目标
- [Santhanam et al. (2022). ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/abs/2112.01488) — 生产中的后期交互
- [MTEB leaderboard on Hugging Face](https://huggingface.co/spaces/mteb/leaderboard) — 实时排名
