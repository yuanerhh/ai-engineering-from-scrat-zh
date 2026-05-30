---
name: ner-picker
description: 为给定的抽取任务选择合适的 NER 方法。
version: 1.0.0
phase: 5
lesson: 06
tags: [nlp, ner, extraction]
---

根据任务描述（领域、标签集、语言、延迟、数据量），输出以下内容：

1. 方法。基于规则 + 词典、CRF、BiLSTM-CRF 或 Transformer 微调。
2. 起始模型。明确命名（spaCy 模型 ID，如 `en_core_web_sm` / `en_core_web_trf`；Hugging Face 检查点 ID，如 `dslim/bert-base-NER`；或"自定义，从头训练"）。
3. 标注策略。BIO、BILOU 或基于区间（span-based）。一句话说明理由。
4. 评估。使用 `seqeval`。始终报告实体级别 F1，绝不报告 token 级别。

对于少于 500 个标注样本的情况，拒绝推荐微调 Transformer，除非用户已有预训练领域模型（如医疗领域的 BioBERT）。将嵌套实体标记为需要基于区间或多轮模型。如果用户提到"生产规模"同时使用开箱即用的 CoNLL-2003 标签，则要求进行词典审查。
