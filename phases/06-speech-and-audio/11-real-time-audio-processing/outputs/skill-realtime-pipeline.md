---
name: realtime-voice-pipeline
description: 为达到目标端到端延迟，选择传输协议、VAD、流式 STT、LLM、流式 TTS 和编排框架。
version: 1.0.0
phase: 6
lesson: 11
tags: [voice-agent, livekit, pipecat, silero, streaming, latency]
---

给定目标（延迟 P50/P95、语言、信道、离线 vs 云端、通话量），输出以下内容：

1. 传输协议。WebRTC（LiveKit / Daily）· WebSocket · SIP 中继（Twilio / Telnyx）。理由与抖动容忍度和使用场景挂钩。
2. VAD + 轮次检测。Silero VAD（开源，TPR 99.5%）· Cobra（商业）· LiveKit 轮次检测器。阈值、最小语音时长、静音挂起时间。
3. 流式 STT。Parakeet TDT（最快开源）· Kyutai STT（含刷新技巧）· Deepgram Nova-3（API，约 150 ms）· Whisper-streaming。说明理由。
4. LLM + 流式传输。在 TTS 启动前固定前 20 个 token。模型 + 流式配置 + 提示词注入防护。
5. 流式 TTS。Kokoro-82M（约 100 ms TTFA）· Orpheus · Cartesia Sonic · ElevenLabs Turbo。声音包或克隆保护（参见第 8 课）。
6. 编排框架。LiveKit Agents · Pipecat · Vapi · Retell · 自定义 Rust。理由与团队技能和扩展需求挂钩。
7. 可观测性。每阶段 P50/P95/P99 直方图；错误中断率；掉线率；抽样通话的词错误率。

拒绝在 STT 之前缓存完整话语的部署。拒绝不支持流式输出的 TTS。拒绝用平均延迟评估——要求 P95。拒绝每月超过 10 万分钟而不做自建方案成本对比的托管平台（Vapi / Retell）。

示例输入："汽车保险报价的语音代理。P95 < 500 ms。英语，美国。每周 5 万分钟。合规：类 HIPAA（日志中无 PII）。"

示例输出：
- 传输：LiveKit Agents + Twilio SIP。经过呼叫中心规模验证，支持 HIPAA 模式。
- VAD：Silero VAD @ 阈值 0.45，最小语音 220 ms，静音挂起 400 ms。LiveKit 轮次检测器叠加使用。
- STT：Deepgram Nova-3 英语（约 150 ms P95）；如需本地审计则回退到 Parakeet-TDT。
- LLM：通过 OpenAI realtime API 流式调用 GPT-4o；用后置过滤器防止提示词注入；前 20 个 token 固定给 TTS。
- TTS：Cartesia Sonic 2（约 150 ms TTFA，不使用声音克隆——使用预定义声音）。
- 编排：LiveKit Agents。通过 Hamming AI 进行生产可观测性。
- 日志：在持久化前用正则 + NER 过滤 CVV / SSN / 出生日期。保留 30 天。
