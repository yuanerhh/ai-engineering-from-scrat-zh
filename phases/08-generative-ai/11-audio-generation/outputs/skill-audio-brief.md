---
name: audio-brief
description: 将音频创作简报转化为 TTS、音乐和音效的模型 + 提示词 + 评估方案。
version: 1.0.0
phase: 8
lesson: 11
tags: [audio, tts, music, sfx, codec]
---

给定音频简报（任务：TTS / 音乐 / 音效 / 声音克隆、时长、风格、声音或流派、许可证限制、实时或离线、质量要求），输出以下内容：

1. 模型 + 托管。ElevenLabs V3、OpenAI TTS、XTTS v2、Suno v4、Udio、Stable Audio 2.5、MusicGen 3.3B、AudioCraft 2 或 GPT-4o realtime。一句话说明理由。
2. 提示词格式。TTS：文本 + 声音提示（3-10 秒样本或声音 ID）+ 情感/节奏标签。音乐：流派 + 编曲 + 情绪 + BPM + 结构标记。音效：拟声词 + 来源 + 时长提示。
3. 编解码器 + 生成器 + 声码器链。指定具体编解码器（Encodec 32 kHz、DAC 44 kHz、自定义）和生成器选择（token 自回归 vs 流匹配）。
4. 种子 + 可复现性。种子锁定、版本锁定、提示词哈希。
5. 评估。TTS 使用 MOS（平均意见分）或 A/B 测试，音乐使用 CLAP 分数，TTS 转录使用字符错误率，音效使用用户听觉测试。
6. 保护措施。声音克隆授权 + 水印（PerTh / SynthID-audio）、音乐输出版权扫描、训练数据政策检查。

拒绝在没有声音所有人明确授权的情况下克隆任何声音（磁带时代的"3 秒提示"不算授权）。拒绝发布含有未授权参考素材的音乐。标记任何实时目标 < 200 ms 而不使用流式 token 自回归模型的情况——2026 年基于扩散的音频无法达到 300 ms 以下的 TTFB。
