---
name: multilingual-picker
description: 为多语言 NLP 任务选择源语言、目标模型和评估方案。
version: 1.0.0
phase: 5
lesson: 18
tags: [nlp, multilingual, cross-lingual]
---

给定需求（目标语言、任务类型、每种语言的可用标注数据），输出：

1. 微调源语言。默认英语；若目标语言有类型学上接近的高资源语言，可参考 LANGRANK 或 qWALS。
2. 基础模型。XLM-R（分类）、mT5（生成）、NLLB（翻译）、Aya-23（生成式 LLM）。
3. 少样本预算。如有条件，从 100-500 个目标语言样本开始。仅在标注不可行时使用零样本。
4. 评估方案。按语言分别计算准确率（而非聚合指标），跨语言一致性，非拉丁文字的实体级 F1。

拒绝在没有按语言分别评估的情况下上线多语言模型——聚合指标会掩盖长尾失败。将分词覆盖率低的文字（阿姆哈拉语、提格里尼亚语、许多非洲语言）标记为需要支持字节回退的模型（带 `byte_fallback=True` 的 SentencePiece，或如 GPT-2 的字节级分词器）。
