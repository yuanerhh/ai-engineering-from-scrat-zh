---
name: voice-agent
description: 构建一个实时语音智能体，具备 800ms 以内首次出音、插话处理和对话中工具调用能力。
version: 1.0.0
phase: 19
lesson: 03
tags: [capstone, voice, webrtc, livekit, pipecat, asr, tts, streaming]
---

给定一个应用场景（客户支持、日程安排、零售助手），部署一个 WebRTC 语音智能体，将端到端首次出音时间控制在 800ms 以内，同时处理插话、工具调用和丢包问题。

构建计划：

1. 搭建一个 LiveKit Agents 1.0 房间，附带流式传输麦克风音频的 Web 客户端。添加 Twilio PSTN 网关以支持电话接入。
2. 运行流式 ASR（托管的 Deepgram Nova-3 或在 g5.xlarge 上运行的 faster-whisper Whisper-v3-turbo）。订阅部分和最终转录结果。
3. 在 20ms 帧上运行 Silero VAD v5。语音结束时，使用 LiveKit 轮次检测器对最新部分结果打分；仅当 VAD 静音 >= 500ms 且完成分数 >= 0.6 时，确认轮次结束。
4. 流式传输 LLM（GPT-4o-realtime、Gemini 2.5 Flash Live 或级联的 Claude Haiku 4.5）。在 200ms 内将第一个 Token 传递给 TTS。
5. 流式传输 TTS（Cartesia Sonic-2 或 ElevenLabs Flash v3）。第一个音频块必须在收到第一个 LLM Token 后 200ms 内离开服务器。
6. 插话处理：当 VAD 在 SPEAKING 或 THINKING 状态下检测到新的用户语音时，取消 TTS、丢弃剩余 LLM 输出、重新激活 ASR。发布 `tts_canceled` span。
7. 工具旁路通道：并发运行函数调用；如果延迟 > 300ms，发出确认填充语，确保音频流不中断。
8. 录制 100 通电话。衡量与保留转录的 WER、Hamming VAD 基准上的误切率、首次出音 p50、NISQA MOS，以及 3% 丢包下的表现。
9. 在单台 g5.xlarge 上使用合成呼叫方进行 50 路并发压力测试；报告持续的首次出音 p95。

评估标准：

| 权重 | 评估项 | 度量方式 |
|:-:|---|---|
| 25 | 端到端延迟 | 在 100 通录制电话中 p50 首次出音低于 800ms |
| 20 | 轮次切换质量 | Hamming VAD 基准上误切率低于 3% |
| 20 | 工具调用正确性 | 对话中工具调用返回正确数据且不造成音频停顿 |
| 20 | 丢包下的可靠性 | 注入 3% 丢包时的 WER 和轮次切换稳定性 |
| 15 | 评估框架完整性 | 带公开配置的可重现测量 |

硬性拒绝条件：

- 非流式管道（批量 ASR、批量 TTS）无法达到延迟目标。
- 任何不立即取消 TTS 缓冲区的插话策略。延迟取消会产生最严重的用户体验回退。
- 同步阻塞 LLM 流的工具调用。它们必须在旁路通道上运行。

拒绝规则：

- 拒绝在没有 VAD 或轮次检测器的情况下部署。固定超时的轮次切换会产生不可接受的误切率。
- 拒绝报告 MOS 时不注明是人工评分还是 NISQA 代理评分。
- 拒绝在没有至少 100 通录制电话且未公开通话追踪的情况下报告"p50 延迟低于 X"。

输出：一个包含 LiveKit 智能体 Worker、PSTN 网关配置、100 通电话评估框架、公开 Langfuse 语音仪表板的代码库，与一家托管竞争对手（Retell、Vapi 或直接使用 OpenAI Realtime API）的对比分析，以及一份说明你观察到的三大轮次切换失败及相应检测器调优措施的报告。
