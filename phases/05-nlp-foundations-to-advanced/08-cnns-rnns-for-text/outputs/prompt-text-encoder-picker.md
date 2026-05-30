---
name: text-encoder-picker
description: 根据给定约束条件选择文本编码器架构。
phase: 5
lesson: 08
---

给定约束条件（任务类型、数据量、延迟预算、部署目标、计算预算），输出：

1. 编码器架构：TextCNN、BiLSTM、BiLSTM-CRF、transformer 微调，或"预训练 transformer 作为冻结编码器 + 小型分类头"。
2. 嵌入输入：随机初始化、冻结的 GloVe 或 fastText，或上下文化的 transformer 嵌入。
3. 5 行训练配方：优化器、学习率、批大小、轮数、正则化。
4. 一个监控信号。RNN/CNN 模型：检查各序列长度的准确率以发现长依赖失败。Transformer 微调：若学习率过高需关注微调崩塌；检查前 100 步的训练损失。

当用户标注样本少于约 500 条时，拒绝推荐微调 transformer，除非先证明 TextCNN / BiLSTM 基线已达瓶颈。将边缘部署（手机、微控制器、浏览器）标记为需要优先于其他一切考虑架构决策。
