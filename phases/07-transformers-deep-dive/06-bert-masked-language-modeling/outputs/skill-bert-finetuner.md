---
name: bert-finetuner
description: 为新的分类、抽取或检索任务规划 BERT 微调方案。
version: 1.0.0
phase: 7
lesson: 6
tags: [bert, fine-tuning, nlp]
---

给定下游任务（分类 / NER / 检索 / 重排序 / NLI）、标注数据量和部署约束（延迟、设备），输出以下内容：

1. 骨干模型选择。模型名称（ModernBERT-base / large、DeBERTa-v3、multilingual-e5 等），一句话说明理由。英文任务需要 ≤8K 上下文时优先使用 ModernBERT。
2. 头部规格。分类：`[CLS]` → dropout → linear(num_classes)。NER：每 token 线性层 + 可选 CRF。检索：均值池化 + 对比损失。
3. 训练配置。优化器（AdamW，典型学习率 2e-5）、预热比例（6-10%）、训练轮数（3-5）、批次大小、fp16/bf16。
4. 评估方案。任务对应的指标（分类用准确率 + F1，NER 用实体级别 F1，检索用 MRR/NDCG）。保留集大小。
5. 失败模式检查。一个具体风险：标签泄露、类别不平衡、上下文截断、预训练与微调语料库的分词器不匹配。

拒绝在 BERT 上微调生成任务（文本生成）——推荐使用仅解码器架构。拒绝在少数类低于 10% 时不使用类别分层评估。标记在少于 1000 个标注样本时解冻完整骨干网络的微调方案——很可能过拟合。
