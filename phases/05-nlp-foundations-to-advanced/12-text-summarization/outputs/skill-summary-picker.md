---
name: summary-picker
description: 选择抽取式或生成式摘要方法，指定库，并添加事实性检查。
version: 1.0.0
phase: 5
lesson: 12
tags: [nlp, summarization]
---

给定任务描述（文档类型、合规要求、长度、计算预算），输出：

1. 方法。抽取式或生成式。一句话说明原因。
2. 起始模型/库。明确命名：`sumy.TextRankSummarizer`、`facebook/bart-large-cnn`、`google/pegasus-pubmed` 或 LLM 提示词。
3. 评估方案。ROUGE-1、ROUGE-2、ROUGE-L（使用带词干提取的 `rouge-score`）。生成式摘要还需添加事实性检查。
4. 需要探查的一个失败模式。实体替换是生成式新闻摘要中最常见的问题；标记源文本实体未出现在摘要中的样本。

对于医疗、法律、金融或受监管内容，拒绝在没有事实性门控的情况下使用生成式摘要。将超过模型上下文窗口的输入标记为需要分块 map-reduce 摘要，而非直接截断。
