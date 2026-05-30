---
name: embedding-picker
description: 为给定语料库和部署场景选择嵌入模型、维度和检索模式。
version: 1.0.0
phase: 5
lesson: 22
tags: [nlp, embeddings, retrieval]
---

给定语料库（大小、语言、领域、平均长度）、部署目标（云端 / 边缘 / 本地）、延迟预算和存储预算，输出：

1. 模型。命名检查点或 API。一句话说明理由。
2. 维度。完整 / Matryoshka 截断 / int8 量化。理由与存储预算相关联。
3. 模式。密集 / 稀疏 / 多向量 / 混合。理由。
4. 若模型卡有要求，提供查询前缀 / 模板。
5. 评估计划。与领域相关的 MTEB 任务 + 以 nDCG@10 为指标的领域留出集评估。

拒绝在没有领域验证的情况下将 Matryoshka 截断到 <64 维。拒绝在不足 1 万个段落的语料库中使用 ColBERTv2（开销不合理）。标记超长文档语料库（>8k token）路由到 512-token 窗口模型的情况。
