---
name: classifier-designer
description: 为音频分类任务选择架构、数据增强方式、类别平衡策略和评估指标。
version: 1.0.0
phase: 6
lesson: 03
tags: [audio, classification, beats, ast]
---

给定音频分类任务（领域、标签数量、每段音频的标签密度、数据量、部署目标），输出以下内容：

1. 架构。k-NN-MFCC / 2D CNN / AST / BEATs / Whisper 编码器。一句话说明理由。
2. 数据增强。SpecAugment 参数（时间遮掩、频率遮掩次数）、mixup α、背景噪声混合程度。
3. 类别平衡。平衡采样器 vs 焦点损失 vs 类别权重。与尾部类别和头部类别的比例挂钩。
4. 损失 + 指标。CE / BCE / 焦点损失；主要指标（top-1 / mAP / macro-F1）及次要指标。
5. 数据划分 + 评估方案。分层 k 折，语音任务使用说话人独立划分，流数据使用时间划分。

拒绝对多标签任务仅使用 top-1 准确率评分；要求使用 mAP。拒绝在不使用说话人独立划分的情况下评估说话人条件任务。标记任何从头训练的架构（标注样本 < 1 万）——应从 SSL 预训练骨干网络开始。
