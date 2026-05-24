# 信息检索与搜索

> BM25 精准但脆弱。密集检索覆盖面广但遗漏关键词。混合检索是 2026 年的默认方案。其余都是调参。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 5 · 02（词袋模型 + TF-IDF）、Phase 5 · 04（GloVe、FastText、子词）
**时长：** 约 75 分钟

## 问题背景

用户输入"如果有人撒谎获取钱财会怎样"，期望找到真正涵盖该情况的法规："第 420 条 IPC"。关键词搜索完全遗漏（没有共同词汇）。语义搜索在嵌入未基于法律文本训练的情况下也会遗漏。真实搜索必须两者兼顾。

IR（信息检索）是每个 RAG 系统、每个搜索栏、每个文档站点模糊查找的底层管线。2026 年在生产中有效的架构不是单一方法，而是一系列互补方法的链条，每一层捕获上一层的失败。

本课构建每个组件，并指出每个组件能捕获哪些失败。

## 核心概念

![混合检索：BM25 + 密集 + RRF + 交叉编码器重排](../assets/retrieval.svg)

四层架构，按需选用。

1. **稀疏检索（BM25）。** 快速、精准匹配精确词，语义理解差。基于倒排索引运行。在百万级文档上每次查询低于 10ms。能正确处理法规引用、产品代码、错误信息、命名实体。
2. **密集检索。** 将查询和文档编码为向量，进行最近邻搜索。能捕获改写和语义相似性，但会遗漏只相差一个字符的精确关键词匹配。使用 FAISS 或向量数据库时每次查询 50-200ms。
3. **融合（Fusion）。** 合并稀疏和密集的排名列表。**倒数排名融合（RRF，Reciprocal Rank Fusion）**是简单的默认方案，因为它忽略原始分数（两者处于不同量级），只使用排名位置。当你知道某个信号在特定领域占主导时，可以用加权融合。
4. **交叉编码器重排（Cross-encoder rerank）。** 取融合后的前 30 名，用交叉编码器（查询 + 文档一起，对每对打分）重排，保留前 5。交叉编码器每对速度比双编码器慢，但精确度远更高。只在前 30 上运行可以分摊成本。

三路检索（BM25 + 密集 + 学习式稀疏如 SPLADE）在 2026 年基准测试中优于两路，但需要支持学习式稀疏索引的基础设施。对大多数团队而言，两路加交叉编码器重排是最优选择。

## 动手实现

### 步骤一：从零实现 BM25

```python
import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        if not corpus:
            raise ValueError("corpus must not be empty")
        self.corpus = [tokenize(d) for d in corpus]
        self.k1 = k1
        self.b = b
        self.n_docs = len(self.corpus)
        self.avg_dl = sum(len(d) for d in self.corpus) / self.n_docs
        self.df = Counter()
        for doc in self.corpus:
            for term in set(doc):
                self.df[term] += 1

    def idf(self, term):
        n = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))

    def score(self, query, doc_idx):
        q_tokens = tokenize(query)
        doc = self.corpus[doc_idx]
        dl = len(doc)
        freq = Counter(doc)
        score = 0.0
        for term in q_tokens:
            f = freq.get(term, 0)
            if f == 0:
                continue
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            score += self.idf(term) * numerator / denominator
        return score

    def rank(self, query, top_k=10):
        scored = [(self.score(query, i), i) for i in range(self.n_docs)]
        scored.sort(reverse=True)
        return scored[:top_k]
```

两个值得了解的参数：`k1=1.5` 控制词频饱和度，越高则词语重复的权重越大；`b=0.75` 控制长度归一化，0 忽略文档长度，1 完全归一化。这些默认值来自 Robertson 在原始论文中的建议，很少需要调整。

### 步骤二：用双编码器做密集检索

```python
from sentence_transformers import SentenceTransformer
import numpy as np


def build_dense_index(corpus, model_id="sentence-transformers/all-MiniLM-L6-v2"):
    encoder = SentenceTransformer(model_id)
    embeddings = encoder.encode(corpus, normalize_embeddings=True)
    return encoder, embeddings


def dense_search(encoder, embeddings, query, top_k=10):
    q_emb = encoder.encode([query], normalize_embeddings=True)
    sims = (embeddings @ q_emb.T).flatten()
    order = np.argsort(-sims)[:top_k]
    return [(float(sims[i]), int(i)) for i in order]
```

L2 归一化嵌入后，点积等于余弦相似度。`all-MiniLM-L6-v2` 是 384 维，速度快，对大多数英文检索已足够。多语言场景使用 `paraphrase-multilingual-MiniLM-L12-v2`。追求最高精度时用 `bge-large-en-v1.5` 或 `e5-large-v2`。

### 步骤三：倒数排名融合

```python
def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, (_, doc_idx) in enumerate(ranking):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, doc_idx) for doc_idx, score in fused]
```

`k=60` 常数来自原始 RRF 论文。`k` 越大，排名差异的贡献越平缓；越小，顶部排名越占主导。60 是已发布的默认值，很少需要调整。

### 步骤四：混合搜索 + 重排

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def hybrid_search(query, bm25, encoder, dense_embeddings, corpus, top_k=5, pool_size=30, reranker=reranker):
    sparse_ranking = bm25.rank(query, top_k=pool_size)
    dense_ranking = dense_search(encoder, dense_embeddings, query, top_k=pool_size)
    fused = reciprocal_rank_fusion([sparse_ranking, dense_ranking])[:pool_size]

    pairs = [(query, corpus[doc_idx]) for _, doc_idx in fused]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(scores, [doc_idx for _, doc_idx in fused]), reverse=True)
    return reranked[:top_k]
```

三阶段组合。BM25 找词汇匹配。密集检索找语义匹配。RRF 合并两个排名，无需分数校准。交叉编码器利用查询-文档对对前 30 名重新打分，捕获双编码器遗漏的细粒度相关性，保留前 5。

### 步骤五：评估

| 指标 | 含义 |
|------|------|
| Recall@k | 在正确文档存在的查询中，有多少比例出现在前 k 名中？ |
| MRR（平均倒数排名） | 第一个相关文档排名的倒数的平均值。 |
| nDCG@k | 考虑相关度分级，不仅仅是二元相关/不相关。 |

对于 RAG，**Recall@k** 是检索器最重要的指标。如果正确段落不在检索集中，阅读器就无法作答。

调试建议：对失败的查询，对比稀疏和密集的排名。如果一个找到了正确文档而另一个没有，说明存在词汇不匹配（修复：补充缺失的那一半）或语义歧义（修复：更好的嵌入或重排器）。

## 生产使用

2026 年技术栈：

| 规模 | 技术栈 |
|------|--------|
| 1k-100k 文档 | 内存中 BM25 + `all-MiniLM-L6-v2` 嵌入 + RRF，无需独立数据库 |
| 100k-10M 文档 | FAISS 或 pgvector 做密集 + Elasticsearch / OpenSearch 做 BM25，并行运行 |
| 10M+ 文档 | 支持混合的 Qdrant / Weaviate / Vespa / Milvus，顶部 30 加交叉编码器重排 |
| 最高质量前沿 | 三路（BM25 + 密集 + SPLADE）+ ColBERT 后期交互重排 |

无论选择什么，都要预算评估时间。先对检索召回率做基准，再对端到端 RAG 准确性做基准。阅读器无法修复检索器遗漏的内容。

### 2026 年生产 RAG 总结出的经验教训

- **80% 的 RAG 失败源于数据摄取和分块，而非模型。** 团队花几周时间更换 LLM 和调整提示，而检索器每三次查询就悄悄返回错误上下文。先修复分块。
- **分块策略比分块大小更重要。** 固定大小分割会破坏表格、代码和嵌套标题。句子感知分块是默认选择；语义分块或 LLM 分块对技术文档和产品手册有明显收益。
- **父文档模式（Parent-doc pattern）。** 检索小的"子"块以保证精确度。当来自同一父节的多个子块出现时，换入父块以保留上下文。这能持续提升答案质量，无需重新训练。
- **k_rerank=3 通常最优。** 超过这个数量的每个额外块都增加 token 成本和生成延迟，而不提升答案质量。如果 k=8 仍优于 k=3，说明重排器表现不佳。
- **HyDE / 查询扩展。** 从查询生成一个假设答案，对其进行嵌入，然后检索。弥合了短问题和长文档之间的措辞差距。无需训练即可免费提升精确度。
- **上下文预算控制在 8K token 以内。** 持续触及这个限制说明重排器阈值太宽松。
- **版本控制一切。** 提示、分块规则、嵌入模型、重排器。任何漂移都会悄悄破坏答案质量。对忠实度、上下文精确率和未答率设置 CI 门控，阻止回归影响用户。
- **三路检索（BM25 + 密集 + SPLADE 等学习式稀疏）在 2026 年基准中优于两路**，尤其是在混合专有名词和语义的查询中。当基础设施支持 SPLADE 索引时部署。

据 2026 年行业测量，合理的检索设计可将幻觉率降低 70-90%。大多数 RAG 性能提升来自更好的检索，而非模型微调。

## 上手实践

将以下内容保存为 `outputs/skill-retrieval-picker.md`：

```markdown
---
name: retrieval-picker
description: Pick a retrieval stack for a given corpus and query pattern.
version: 1.0.0
phase: 5
lesson: 14
tags: [nlp, retrieval, rag, search]
---

Given requirements (corpus size, query pattern, latency budget, quality bar, infra constraints), output:

1. Stack. BM25 only, dense only, hybrid (BM25 + dense + RRF), hybrid + cross-encoder rerank, or three-way (BM25 + dense + learned-sparse).
2. Dense encoder. Name the specific model. Match to language(s), domain, and context length.
3. Reranker. Name the specific cross-encoder model if used. Flag that rerank adds 30-100ms latency on top-30.
4. Evaluation plan. Recall@10 is the primary retriever metric. MRR for multi-answer. Baseline first, incremental improvements measured against it.

Refuse to recommend dense-only for corpora with named entities, error codes, or product SKUs unless the user has evidence dense handles exact matches. Refuse to skip reranking for high-stakes retrieval (legal, medical) where the final top-5 decides the user's answer.
```

## 练习

1. **简单。** 在 500 篇文档的语料库上实现上述 `hybrid_search`，测试 20 个查询，比较纯 BM25、纯密集和混合三种方案的 Recall@5。
2. **中等。** 添加 MRR 计算。对每个已知正确文档的测试查询，找出正确文档在 BM25、密集和混合排名中的位置，报告各自的 MRR。
3. **困难。** 用 MultipleNegativesRankingLoss（Sentence Transformers）在你的领域上微调密集编码器，构建 500 对查询-文档训练集，比较微调前后的召回率。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| BM25 | 关键词搜索 | Okapi BM25，按词频、IDF 和文档长度对文档打分。 |
| 密集检索（Dense retrieval） | 向量搜索 | 将查询和文档编码为向量，找最近邻。 |
| 双编码器（Bi-encoder） | 嵌入模型 | 独立编码查询和文档，查询时速度快。 |
| 交叉编码器（Cross-encoder） | 重排模型 | 联合编码查询和文档，慢但准确。 |
| RRF | 排名融合 | 通过对 `1/(k + rank)` 求和合并两个排名列表。 |
| Recall@k | 检索指标 | 相关文档出现在前 k 名的查询比例。 |

## 延伸阅读

- [Robertson and Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) — BM25 权威论述
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — DPR，标准双编码器
- [Formal et al. (2021). SPLADE: Sparse Lexical and Expansion Model](https://arxiv.org/abs/2107.05720) — 缩短与密集检索差距的学习式稀疏检索器
- [Cormack, Clarke, Büttcher (2009). Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — RRF 论文
- [Khattab and Zaharia (2020). ColBERT: Efficient and Effective Passage Search](https://arxiv.org/abs/2004.12832) — 后期交互检索
