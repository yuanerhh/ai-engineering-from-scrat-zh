---
name: duplex-pipeline
description: 为语音代理工作负载选择全双工（Moshi）架构还是流水线（VAD + STT + LLM + TTS）架构。
version: 1.0.0
phase: 6
lesson: 15
tags: [moshi, hibiki, full-duplex, voice-agent, streaming]
---

给定工作负载（延迟目标、工具调用需求、语言覆盖范围、硬件预算、云端 vs 边缘端），输出以下内容：

1. 架构。全双工（Moshi / GPT-4o Realtime / Gemini Live）vs 流水线（LiveKit + STT + LLM + TTS，参见第 12 课）。一句话说明理由。
2. 模型。Moshi · Hibiki · Hibiki-Zero · Sesame CSM · GPT-4o Realtime · Gemini 2.5 Live · 传统流水线。说明理由。
3. 扩展性。每会话 GPU 成本（Moshi 占用一个 GPU 插槽）、最大并发会话数、冷启动影响。
4. 工具调用路径。如有需要——混合流水线（全双工 + 外部 LLM 处理工具调用）还是纯流水线。解释权衡。
5. 语言覆盖。全双工模型的语言支持范围有限；流水线继承 LLM 的多语言能力。

拒绝纯全双工架构用于需要工具调用 / 检索的企业代理——Moshi 是对话模型，不是代理框架。拒绝纯流水线架构用于延迟要求 < 250 ms 的对话代理——各阶段延迟叠加。拒绝在一张 GPU 上运行超过 4 个并发 Moshi 会话——会产生竞争。

示例输入："语言学习语音伙伴——对话流利度练习。英语 + 法语。响应速度 < 250 ms。每日活跃 1 万用户。"

示例输出：
- 架构：全双工（Moshi）。< 250 ms 延迟要求 + 对话流利度符合 Moshi 的优势。
- 模型：Moshi。英语 + 法语均原生支持。CC-BY 4.0 许可证。
- 扩展性：每 4-6 个并发会话需一张 L4 GPU → 按 10% 并发率、1 万 DAU 峰值约需 1500 张 GPU。为安静路径规划使用 Kyutai Pocket TTS + 本地 Whisper 的设备端轻量模式。
- 工具调用：最少——"显示语法提示"和"翻译这个短语"可通过小型 LLM 旁路模块路由；大多数交互是 Moshi 擅长的开放式对话。
- 语言覆盖：英语 + 法语（原生）；通过 Hibiki-Zero 适配支持西班牙语 / 德语 / 日语（每门新语言需要 1000 小时音频）。
