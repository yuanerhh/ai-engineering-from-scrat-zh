---
name: whisper-tuner
description: 为给定语言、领域和延迟预算设计 Whisper 微调或推理流水线。
version: 1.0.0
phase: 6
lesson: 05
tags: [audio, whisper, asr, fine-tuning, lora]
---

给定目标（语言集、领域、音频时长分布、延迟预算、硬件）和数据（可用小时数、质量），输出以下内容：

1. 模型变体。Tiny / Base / Small / Medium / Large-v3 / Turbo。说明理由。
2. 运行时。vanilla / faster-whisper / whisperx / whisper-streaming。说明理由。
3. 微调方案。全参数微调 vs LoRA（r、target_modules）、冻结编码器策略、训练轮数。
4. 推理保护措施。VAD（Silero 或 Whisper 自带）、`temperature=0`、`condition_on_previous_text=False`、`no_speech_threshold`。
5. 评估。领域词错误率目标、文本归一化规则、静音片段的幻觉率检查。

拒绝在没有 VAD 的情况下将 Whisper 部署到任意音频上。拒绝在多分块任务中将 `condition_on_previous_text=True` 设置为 True 而不加防范。标记任何替换 Whisper 分词器或梅尔频谱流水线的微调方案。
