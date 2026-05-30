---
name: skill-bpe-vs-wordpiece
description: 为给定语料库和部署目标选择分词算法、词表大小和库。
version: 1.0.0
phase: 5
lesson: 19
tags: [nlp, tokenization]
---

给定语料库（大小、语言、领域）和部署目标（从零训练 / 微调 / API 兼容推理），输出：

1. 算法。BPE、Unigram 或 WordPiece。一句话说明原因。
2. 库。SentencePiece、HF Tokenizers 或 tiktoken。说明原因。
3. 词表大小。取整到最近的 1k。原因与模型大小和语言覆盖范围挂钩。
4. 覆盖率设置。`character_coverage`、`byte_fallback`、特殊词元列表。
5. 验证方案。留出集上的平均每词 token 数、OOV 率、压缩比、往返解码一致性。

对于包含稀有文字内容的语料库，拒绝训练 `character_coverage` 低于 0.995 的分词器。拒绝在 CI 中没有冻结 `tokenizer.json` 哈希检查的情况下上线词表。将任何词表小于 16k 的单语分词器标记为可能规格不足。
