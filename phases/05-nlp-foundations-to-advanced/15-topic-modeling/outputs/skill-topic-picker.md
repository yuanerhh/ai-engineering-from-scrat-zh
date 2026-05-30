---
name: topic-picker
description: 为语料库选择 LDA 或 BERTopic。指定库、参数和评估方式。
version: 1.0.0
phase: 5
lesson: 15
tags: [nlp, topic-modeling]
---

给定语料库描述（文档数量、平均长度、领域、语言、计算预算），输出：

1. 算法。LDA / NMF / BERTopic / Top2Vec / FASTopic。一句话说明原因。
2. 配置。主题数量（从约 `sqrt(n_docs)` 开始）、`min_df` / `max_df` 过滤器、神经方法的嵌入模型。
3. 评估。通过 `gensim.models.CoherenceModel` 计算主题连贯性（c_v）、主题多样性，以及 20 个样本的人工阅读。
4. 需要探查的失败模式。LDA：吸收停用词和高频词的"垃圾主题"。BERTopic：将歧义文档吞入的 -1 离群簇。

对于超过嵌入模型上下文窗口长度的文档，拒绝使用 BERTopic，除非有分块策略。对于非常短的文本（推文、少于 10 个词的评论），拒绝使用 LDA，因为连贯性会崩溃。将主题数少于 5 或超过 200 的选择标记为对真实数据很可能不合适。
