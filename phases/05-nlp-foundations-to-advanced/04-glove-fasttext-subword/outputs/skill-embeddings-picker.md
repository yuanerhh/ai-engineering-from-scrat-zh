---
name: skill-embeddings-picker
description: 为新的语言模型或文本处理流水线选择分词策略。
version: 1.0.0
phase: 5
lesson: 04
tags: [nlp, tokenization, embeddings]
---

根据任务和数据集描述，输出以下内容：

1. 分词策略（词级、BPE、WordPiece、SentencePiece、字节级 BPE）。一句话说明原因。
2. 词汇表大小目标。纯英文语言模型：32k；多语言：64k-100k；代码：50k-100k。
3. 带有精确训练命令的库调用。注明所用库（Hugging Face `tokenizers`、`sentencepiece`）并引用参数。
4. 一个可复现性陷阱。分词器与模型不匹配是最常见的隐性生产 bug。注明哪个分词器与哪个预训练检查点配对，并警告不要随意替换。

当用户在微调预训练 LLM 时，拒绝推荐训练自定义分词器（微调必须使用预训练分词器）。拒绝为任何生产推理路径推荐词级分词。将非英文或多文字语料库标记为需要带字节回退的 SentencePiece。
