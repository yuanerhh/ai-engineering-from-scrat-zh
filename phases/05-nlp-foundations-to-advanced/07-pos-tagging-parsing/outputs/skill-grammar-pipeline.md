---
name: grammar-pipeline
description: 为下游 NLP 任务设计经典的词性标注 + 依存解析流水线。
version: 1.0.0
phase: 5
lesson: 07
tags: [nlp, pos, parsing]
---

给定一个下游任务（信息抽取、改写验证、查询分解、词形还原），你需要输出：

1. 标注集。纯英文遗留流水线使用 Penn Treebank，多语言或跨语言场景使用 Universal Dependencies。
2. 库。大多数生产环境使用 spaCy（`en_core_web_sm` / `_lg` / `_trf`），学术级多语言使用 stanza，最高 UD 精度使用 trankit。
3. 集成代码片段。调用库并消费 `.pos_`、`.dep_`、`.head` 的 3-5 行代码。
4. 需要测试的失败模式。名词-动词歧义（`saw`、`book`、`can`）和 PP 附着歧义是经典陷阱。抽取 20 个输出并人工检查。

拒绝推荐自行构建解析器。从零开始构建解析器是研究项目，而非应用任务。将任何消费词性标注却未处理大小写变体的流水线标记为脆弱。
