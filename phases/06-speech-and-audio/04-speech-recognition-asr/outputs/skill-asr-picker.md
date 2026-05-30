---
name: asr-picker
description: 根据给定的部署目标选择 ASR 模型、解码策略、分块方式和语言模型融合方案。
version: 1.0.0
phase: 6
lesson: 04
tags: [audio, asr, speech-recognition]
---

给定部署目标（语言列表、领域、延迟预算、硬件、离线/流式、音频时长），输出以下内容：

1. 模型。Whisper-large-v3-turbo / Parakeet-TDT / Canary-Flash / wav2vec 2.0 / Moonshine。一句话说明理由。
2. 解码策略。贪心 / 束搜索宽度 / 温度回退 / 语言模型融合权重。理由与质量预算挂钩。
3. 分块与 VAD。分块长度、步幅，以及是否使用 Silero-VAD 或 Whisper 自带的 VAD 进行门控。
4. 语言策略。强制语言 vs 自动语言识别；如何处理跨语言帧。
5. 评估方案。领域测试集上的词错误率、每说话人覆盖率、静音片段上的幻觉率。

拒绝任何未经 VAD 门控的长形式 Whisper 部署（在静音段容易产生幻觉）。拒绝在没有文本归一化的情况下报告词错误率（小写、去标点）。标记任何束搜索宽度 > 16 且没有语言模型的设置；在空白符上做纯束搜索没有帮助。
