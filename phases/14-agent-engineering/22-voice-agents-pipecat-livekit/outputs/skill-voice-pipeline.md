---
name: voice-pipeline
description: 搭建一个 Pipecat 风格的语音管道（VAD + STT + LLM + TTS + Transport），支持抢话打断、置信度门控和延迟预算强制执行。
version: 1.0.0
phase: 14
lesson: 22
tags: [voice, pipecat, livekit, webrtc, latency]
---

根据语音产品规格（语言、传输方式、服务提供商），搭建一个基于帧的管道。

输出内容：

1. `Frame` 类型，包含 `kind`、`payload`、`direction`（downstream / upstream）。
2. 处理器：`VAD`、`STT`、`LLM`、`TTS`、`Transport`，每个都实现 `process(frame)`。
3. `link()` 辅助函数，将处理器前后串联。
4. 取消帧处理：UPSTREAM 路径从 transport 到 TTS 到 LLM 到 STT，在每个阶段丢弃待处理工作。
5. 观察器：每个阶段的延迟指标；为每个帧跨越处理器时发送一个 OTel span（第 23 课）。
6. STT 置信度门控：低于阈值时，发送"请重复"文本帧，而非转录文本。

硬性拒绝：

- 没有 UPSTREAM 处理的管道。抢话打断对语音来说不是可选项。
- 没有流式传输的 LLM 调用。首 token 延迟决定体验，必须流式传输。
- 忽略置信度的 STT。将错误转录喂给 LLM 会产生错误回复。

拒绝规则：

- 如果冷启动端到端延迟超过 1500ms，拒绝交付。优化链路或使用 MultimodalAgent（LiveKit 直接音频）。
- 如果产品以电话为主且管道没有 SIP 适配器，拒绝。通过 LiveKit SIP 或平台（Vapi/Retell）路由。
- 如果产品传输包含 PII 音频但未加密，拒绝。

输出：`frames.py`、`processors.py`、`pipeline.py`、`observers.py`、`README.md`，说明延迟预算、抢话打断设计和传输选择。最后附"下一步阅读"，指向第 23 课（OTel）、第 24 课（可观测性后端）或 LiveKit WebRTC 相关文档。
