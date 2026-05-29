# 顶点项目 03 — 实时语音助手（ASR 到 LLM 到 TTS）

> 一个体验流畅的语音智能体需要端到端延迟低于 800 毫秒，能判断你何时停止说话，处理打断（barge-in），并能在不停顿的情况下调用工具。Retell、Vapi、LiveKit Agents 和 Pipecat 在 2026 年都已达到这一标准。它们的共同形态是：流式 ASR、轮次检测器、流式 LLM、流式 TTS，全部通过 WebRTC 串联，并在每个环节都设置严格的延迟预算。构建一个，测量词错率（WER）、平均意见分（MOS）和错误截断率，然后在丢包环境下运行。

**类型：** 顶点项目  
**语言：** Python（智能体 + 流水线），TypeScript（Web 客户端）  
**前置条件：** 第 6 阶段（语音与音频），第 7 阶段（Transformer），第 11 阶段（LLM 工程），第 13 阶段（工具），第 14 阶段（智能体），第 17 阶段（基础设施）  
**涵盖阶段：** P6 · P7 · P11 · P13 · P14 · P17  
**预计时间：** 30 小时

## 问题背景

语音是 2025-2026 年移动最快的 AI 交互类别。技术天花板每季度都在下降。OpenAI Realtime API、Gemini 2.5 Live、Cartesia Sonic-2、ElevenLabs Flash v3、LiveKit Agents 1.0 和 Pipecat 0.0.70 都让 800 毫秒以内的首次音频输出触手可及。门槛不仅是延迟，还有交互感受：不打断用户，不被用户打断，从句中间的中断中恢复，对话中调用工具时不出现停顿，在颠簸的移动网络下依然稳定。

你无法通过拼接三个 REST 调用来达到这一标准。架构必须是端到端流式流水线。构建它，失败模式就会变得清晰可见：针对电话音频调优的 VAD（语音活动检测）被背景电视误触发，等待永远不会出现的标点符号的轮次检测器，在发出第一帧音频前缓冲 400 毫秒的 TTS。这个顶点项目就是在负载下逐一修复这些问题，并发布延迟与质量报告。

## 核心概念

流水线有五个流式阶段：**audio in（音频输入）**（来自浏览器或 PSTN 的 WebRTC）、**ASR**（来自 Deepgram Nova-3 或 faster-whisper 的流式部分转录）、**turn detection（轮次检测）**（VAD 加上一个读取部分转录以判断完成信号的小型轮次检测模型）、**LLM**（在判断轮次完成后立即流式输出 token）、**TTS**（在 LLM 第一个 token 后约 200 毫秒内流式输出音频）。

三个横切关注点。**Barge-in（打断）**：当用户在智能体说话时开始说话，TTS 立即取消，ASR 立即接管。**Tool use（工具调用）**：对话中途的函数调用（天气、日历）必须在侧信道运行，不能停止音频；如果延迟超过 300 毫秒，智能体会预先填充确认 token（"稍等……"）。**背压**：在丢包情况下，部分转录被暂存，VAD 提高语音门限阈值，智能体避免在未被确认的消息上继续说话。

测量标准是定量的。在 15 dB 信噪比下，Hamming VAD 基准测试的 WER 低于 8%。100 次测量通话的首次音频输出 p50 低于 800 毫秒。错误截断率低于 3%。TTS 的 MOS 高于 4.2。单台 g5.xlarge 上支持 50 路并发通话。这些数字就是交付物。

## 架构图

```
browser / Twilio PSTN
        |
        v
   WebRTC / SIP edge
        |
        v
  LiveKit Agents 1.0  (or Pipecat 0.0.70)
        |
   +----+--------------+--------------+-----------------+
   |                   |              |                 |
   v                   v              v                 v
  ASR              VAD v5         turn-detector     side-channel
(Deepgram         (Silero)          (LiveKit)        tools
 Nova-3 /         speech-gate    completion score    (weather,
 Whisper-v3)      per 20ms        on partials        calendar)
   |                   |              |
   +--------+----------+--------------+
            v
        LLM (streaming)
     GPT-4o-realtime / Gemini 2.5 Flash /
     cascaded Claude Haiku 4.5
            |
            v
        TTS streaming
     Cartesia Sonic-2 / ElevenLabs Flash v3
            |
            v
     audio back to caller
            |
            v
   OpenTelemetry voice traces -> Langfuse
```

## 技术栈

- 传输：LiveKit Agents 1.0（WebRTC）加 Twilio PSTN 网关；Pipecat 0.0.70 作为备选框架
- ASR：Deepgram Nova-3（流式，首次部分结果低于 300 毫秒）或自托管 faster-whisper Whisper-v3-turbo
- VAD：Silero VAD v5 加 LiveKit 轮次检测器（读取部分转录的小型 Transformer）
- LLM：OpenAI GPT-4o-realtime（高度集成）、Gemini 2.5 Flash Live，或级联 Claude Haiku 4.5（流式补全，独立音频通道）
- TTS：Cartesia Sonic-2（最低首字节延迟）、ElevenLabs Flash v3，或用于自托管的开源 Orpheus
- 工具：FastMCP 侧信道用于天气/日历/预订；工具超过 300 毫秒时智能体预先发出填充语
- 可观测性：OpenTelemetry 语音 span，Langfuse 语音追踪（带音频回放）
- 部署：单台 g5.xlarge（24GB VRAM）用于自托管 Whisper + Orpheus；托管 API 用于最低延迟

## 构建步骤

1. **WebRTC 会话。** 搭建 LiveKit 房间和一个流式传输麦克风音频的 Web 客户端。在服务器端，附加一个加入房间的智能体 worker。

2. **ASR 流式处理。** 将 20ms PCM 帧馈送给 Deepgram Nova-3（或 GPU 上的 faster-whisper）。订阅部分和最终转录结果。记录每次部分转录的延迟。

3. **VAD 和轮次检测器。** 在帧流上运行 Silero VAD v5。在语音结束事件时，对最新的部分转录触发 LiveKit 轮次检测器。只有当 VAD 显示静音 500 毫秒且轮次检测器得分超过 0.6 时，才确认"轮次完成"。

4. **LLM 流。** 在轮次完成时，用正在进行的对话加最终转录开始 LLM 调用。流式输出 token。在第一个 token 时，交接给 TTS。

5. **TTS 流。** Cartesia Sonic-2 流式返回音频块。第一个块必须在 LLM 第一个 token 后 200 毫秒内离开服务器。将块发送到 LiveKit 房间；客户端通过 WebRTC 抖动缓冲区播放。

6. **Barge-in（打断处理）。** 当 VAD 在 TTS 播放期间检测到新的用户语音时，立即取消 TTS 流，丢弃剩余 LLM 输出，重新激活 ASR。发布 `tts_canceled` span。

7. **工具侧信道。** 将天气和日历注册为函数调用工具。调用时并发触发；如果在 300 毫秒内未完成，让 LLM 发出"稍等，让我查一下"作为填充语；工具返回后恢复。

8. **评测框架。** 录制 100 次通话。计算 WER（对照保留转录）、错误截断率（用户句中 TTS 被取消）、首次音频输出 p50、TTS MOS（人工或 NISQA），以及抖动丢包测试（丢弃 3% 的数据包）。

9. **负载测试。** 用合成呼叫者在单台 g5.xlarge 上驱动 50 路并发通话。测量持续的首次音频输出 p95。

## 使用示例

```
caller: "what is the weather in tokyo tomorrow"
[asr  ] partial @280ms: "what is the"
[asr  ] partial @540ms: "what is the weather"
[turn ] completion score 0.82 at @820ms; commit
[llm  ] first token @960ms
[tool ] weather.tokyo tomorrow -> 68/52 partly cloudy @1140ms
[tts  ] first audio-out @1040ms: "Tokyo tomorrow will be partly cloudy..."
turn latency: 1040ms user-stop -> audio-out
```

## 交付物

`outputs/skill-voice-agent.md` 是交付物。给定一个领域（客服、日程安排或信息亭），它搭建一个经过测量指标调优的 LiveKit 智能体（包含 ASR/VAD/LLM/TTS 流水线）。评分标准：

| 权重 | 评分标准 | 衡量方式 |
|:-:|---|---|
| 25 | 端到端延迟 | 100 次录制通话中首次音频输出 p50 低于 800 毫秒 |
| 20 | 轮次接管质量 | Hamming VAD 基准上错误截断率低于 3% |
| 20 | 工具调用正确性 | 对话中途不停顿地返回正确数据的工具调用 |
| 20 | 丢包下的可靠性 | 注入 3% 丢包时的 WER 和轮次接管稳定性 |
| 15 | 评测框架完整性 | 带公开配置的可复现测量结果 |
| **100** | | |

## 练习题

1. 将 Deepgram Nova-3 替换为 g5.xlarge 上的 faster-whisper v3 turbo。测量延迟和 WER 差距。找出 CPU vs GPU 决策的关键点。

2. 添加打断仲裁策略：用户在工具调用期间打断时智能体如何处理？比较三种策略（硬取消、完成工具再停止、排队下一轮）。

3. 进行对抗性轮次检测器测试：让用户在句子中间停顿很长时间。调整 VAD 静音阈值和轮次检测器得分阈值，以最低错误截断率且不超过 900 毫秒。

4. 通过 Twilio 在 PSTN 上部署同一智能体。比较 PSTN 与 WebRTC 的首次音频输出时间。解释抖动缓冲区和编解码器的差异。

5. 为非英语（日语、西班牙语）添加语音活动检测。测量 Silero VAD v5 的误触发率与特定语言微调版本的差异。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Turn detection（轮次检测） | "发言结束" | 给定 VAD 静音和部分转录，判断用户是否说完话的分类器 |
| Barge-in（打断） | "中断处理" | 当 VAD 检测到新用户语音时，中途取消 TTS 播放 |
| First-audio-out（首次音频输出） | "延迟" | 从用户停止说话到第一个音频包离开服务器的时间 |
| VAD | "语音门控" | 将音频帧分类为语音或静音的模型；Silero VAD v5 是 2026 年默认选项 |
| Jitter buffer（抖动缓冲区） | "音频平滑" | 客户端缓冲区，短暂保存数据包以吸收网络抖动 |
| Filler（填充语） | "确认词元" | 智能体在工具较慢时发出的短语，用于避免静音 |
| MOS | "平均意见分" | 感知语音质量评分；NISQA 是自动化代理指标 |

## 延伸阅读

- [LiveKit Agents 1.0](https://github.com/livekit/agents) — 参考 WebRTC 智能体框架
- [Pipecat](https://github.com/pipecat-ai/pipecat) — 备选 Python 优先流式智能体框架
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — 集成语音模型参考
- [Deepgram Nova-3 文档](https://developers.deepgram.com/docs) — 流式 ASR 参考
- [Silero VAD v5](https://github.com/snakers4/silero-vad) — VAD 参考模型
- [Cartesia Sonic-2](https://docs.cartesia.ai) — 低延迟 TTS 参考
- [Retell AI 架构](https://docs.retellai.com) — 生产语音智能体架构
- [Vapi.ai 生产栈](https://docs.vapi.ai) — 备选生产参考
