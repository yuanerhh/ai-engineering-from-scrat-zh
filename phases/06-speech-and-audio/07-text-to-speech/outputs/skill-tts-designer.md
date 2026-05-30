---
name: tts-designer
description: 为给定语言、风格和延迟目标选择 TTS 模型、声音、文本归一化范围和评估方案。
version: 1.0.0
phase: 6
lesson: 07
tags: [audio, tts, speech-synthesis]
---

给定目标（语言、声音风格、延迟预算、CPU vs GPU、许可证限制）和内容（领域、OOV 密度、标点符号丰富度），输出以下内容：

1. 模型。Kokoro / XTTS v2 / F5-TTS / VITS / StyleTTS 2 / 商业 API。一句话说明理由。
2. 文本前端。归一化范围（数字、日期、URL）、音素化器（espeak-ng vs g2p-en）、OOV 回退。
3. 声音。预设名称或参考片段规格（秒数、噪声底板、口音匹配）。
4. 质量目标。目标 UTMOS、通过 Whisper 的字符错误率、克隆时的 SECS。
5. 评估方案。包含数字、同形异音词、专有名词、长句的 20 句话测试集。

拒绝任何没有文本归一化器的生产 TTS。拒绝在未获得用户同意和水印的情况下进行声音克隆。标记任何被要求说英语以外语言的 Kokoro 部署。
