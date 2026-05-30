---
name: asr-configurator
description: 为新的语音流水线选择 ASR 模型（Whisper 变体 / Moonshine / faster-whisper）和解码参数。
version: 1.0.0
phase: 7
lesson: 10
tags: [transformers, whisper, asr, speech]
---

给定语音任务（转录 / 翻译 / 流式 / 设备端）、语言、音频特征（噪声、口音、时长）以及延迟/质量目标，输出以下内容：

1. 模型选择。以下之一：faster-whisper large-v3-turbo（默认生产首选）、whisper large-v3（最高质量，多语言）、whisper medium（中等）、Moonshine base（边缘端）、distil-whisper（英语速度 2×）。一句话说明理由。
2. 量化。int8_float16（CPU 默认）、float16（GPU 默认）、fp32（研究）。标记显存影响。
3. 解码。束宽（通常 5，流式时为 1）、温度回退策略、对数概率阈值、无语音阈值、VAD 门控开/关。
4. 分块。30 秒固定窗口 vs 流式分块（通常 10 秒带 2 秒重叠）+ 基于 VAD 的分段。记录重叠部分的后合并策略。
5. 后处理。时间戳对齐（WhisperX 强制对齐）、标点恢复、说话人日志化（pyannote）。标记任务必需项。

拒绝在生产环境推荐原版 OpenAI Whisper（参考实现）——`faster-whisper` 速度快 4 倍且输出相同。拒绝在没有文档说明理由的情况下发布流式 ASR 而不使用 VAD。当输入可能是多说话人时，标记任何单说话人假设。
