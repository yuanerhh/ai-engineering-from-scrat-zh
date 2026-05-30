---
name: retrieval-picker
description: 为给定语料库和查询模式选择检索技术栈。
version: 1.0.0
phase: 5
lesson: 14
tags: [nlp, retrieval, rag, search]
---

给定需求（语料库大小、查询模式、延迟预算、质量要求、基础设施约束），输出：

1. 技术栈。仅 BM25、仅稠密检索、混合（BM25 + 稠密 + RRF）、混合 + 交叉编码器重排序，或三路混合（BM25 + 稠密 + 学习稀疏）。
2. 稠密编码器。命名具体模型（`all-MiniLM-L6-v2`、`bge-large-en-v1.5`、`e5-large-v2`、`paraphrase-multilingual-MiniLM-L12-v2`）。与语言、领域、上下文长度匹配。
3. 重排序器。如使用，命名交叉编码器模型（`cross-encoder/ms-marco-MiniLM-L-6-v2`、`BAAI/bge-reranker-large`）。标记 top-30 上约 30-100ms 的额外延迟。
4. 评估方案。Recall@10 是检索器的主要指标。多答案场景使用 MRR。先建立基线，再对比基线衡量增量改进。

对于包含命名实体、错误码或产品 SKU 的语料库，除非用户有证据证明稠密检索能处理精确匹配，否则拒绝推荐纯稠密检索。对于高风险检索（法律、医疗）场景，拒绝跳过重排序，因为最终的 top-5 决定了用户的答案。
