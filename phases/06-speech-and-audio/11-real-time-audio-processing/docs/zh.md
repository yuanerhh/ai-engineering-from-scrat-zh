# 实时音频处理

> 批处理流水线处理一个文件。实时流水线在下一个 20 毫秒到来之前处理当前的 20 毫秒。每个对话式 AI、广播工作室和电话机器人都依赖于这个延迟预算生死存亡。

**类型：** 构建
**语言：** Python、Rust
**前置条件：** Phase 6 · 02（频谱图）、Phase 6 · 04（ASR）、Phase 6 · 07（文本转语音）
**时长：** 约 75 分钟

## 问题背景

你想要一个感觉有生命力的语音助手。人类对话轮换延迟约为 230 毫秒（沉默到响应）。超过 500 毫秒感觉机械；超过 1500 毫秒感觉坏掉了。2026 年完整**听 → 理解 → 响应 → 说话**循环的延迟预算为：

| 阶段 | 预算 |
|------|------|
| 麦克风 → 缓冲 | 20 ms |
| VAD | 10 ms |
| ASR（流式） | 150 ms |
| LLM（首个 token） | 100 ms |
| TTS（首个数据块） | 100 ms |
| 渲染 → 扬声器 | 20 ms |
| **总计** | **~400 ms** |

Moshi（Kyutai，2024）实现了 200 ms 全双工。GPT-4o-realtime（2024）约 320 ms。2022 年的级联流水线延迟为 2500 ms。10 倍改进来自三项技术：(1) 全链路流式处理，(2) 带部分结果的异步流水线，(3) 可中断生成。

## 核心概念

![带环形缓冲区、VAD 门控和中断处理的流式音频流水线](../assets/real-time.svg)

**帧/块/窗口（Frame / chunk / window）。** 实时音频以固定大小的块流动。常用选择：20 ms（16 kHz 下 320 个采样点）。下游所有环节都必须跟上这个节奏。

**环形缓冲区（Ring buffer）。** 固定大小的循环缓冲区。生产者线程写入新帧，消费者线程读取。防止热路径中的内存分配。大小 ≈ 最大延迟 × 采样率；2 秒 16 kHz 环形缓冲 = 32,000 个采样点。

**VAD（语音活动检测，Voice Activity Detection）。** 在无人说话时对下游工作进行门控。Silero VAD 4.0（2024）在 CPU 上每 30 ms 帧运行时间 < 1 ms。`webrtcvad` 是较旧的替代方案。

**流式 ASR（Streaming ASR）。** 在音频到来时输出部分转录的模型。Parakeet-CTC-0.6B 流式模式（NeMo，2024）以 320 ms 延迟实现 2-5% WER。Whisper-Streaming（Macháček et al.，2023）通过分块 Whisper 实现约 2 秒延迟的近流式处理。

**中断（Interruption）。** 当用户在助手说话时开口，你必须 (a) 检测到插话，(b) 停止 TTS，(c) 丢弃剩余的 LLM 输出。必须在 100 ms 内完成，否则用户会感觉助手"听不见"。

**WebRTC Opus 传输。** 20 ms 帧，48 kHz，自适应码率 8-128 kbps。浏览器和移动端标准。LiveKit、Daily.co、Pion 是 2026 年构建语音应用的主流技术栈。

**抖动缓冲区（Jitter buffer）。** 网络数据包乱序或延迟到达。抖动缓冲区重新排序并平滑；太小 → 可听见的间断，太大 → 延迟。典型值 60-80 ms。

### 常见坑

- **线程竞争。** Python 的 GIL + 重型模型会饿死音频线程。使用 C 回调音频库（sounddevice、PortAudio），让 Python 远离热路径。
- **采样率转换延迟。** 流水线内部的重采样增加 5-20 ms。要么提前重采样，要么使用零延迟重采样器（PolyPhase、`soxr_hq`）。
- **TTS 预热。** 即使是 Kokoro 这样的快速 TTS，首次请求也有 100-200 ms 的热身时间。在第一次真实轮次之前缓存模型 + 用虚拟运行预热。
- **回声消除（Echo cancellation）。** 没有 AEC，TTS 输出会重新进入麦克风并触发对机器人自身声音的 ASR。WebRTC AEC3 是开源默认方案。

## 动手实现

### 步骤一：环形缓冲区

```python
import collections

class RingBuffer:
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)
    def write(self, frame):
        self.buf.extend(frame)
    def read(self, n):
        return [self.buf.popleft() for _ in range(min(n, len(self.buf)))]
    def level(self):
        return len(self.buf)
```

容量决定最大缓冲延迟。16 kHz 下 32,000 个采样点 = 2 秒。

### 步骤二：VAD 门控

```python
def simple_energy_vad(frame, threshold=0.01):
    return sum(x * x for x in frame) / len(frame) > threshold ** 2
```

生产环境中用 Silero VAD 替换：

```python
import torch
vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
is_speech = vad(torch.tensor(frame), 16000).item() > 0.5
```

### 步骤三：流式 ASR

```python
# 通过 NeMo 使用 Parakeet-CTC-0.6B 流式模式
from nemo.collections.asr.models import EncDecCTCModelBPE
asr = EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")
# chunk_ms=320 ms，look_ahead_ms=80 ms
for chunk in audio_stream():
    partial_text = asr.transcribe_streaming(chunk)
    print(partial_text, end="\r")
```

### 步骤四：中断处理器

```python
class Dialog:
    def __init__(self):
        self.tts_task = None

    def on_user_speech(self, frame):
        if self.tts_task and not self.tts_task.done():
            self.tts_task.cancel()   # 插话中断
        # 然后送入流式 ASR

    def on_final_user_utterance(self, text):
        self.tts_task = asyncio.create_task(self.reply(text))

    async def reply(self, text):
        async for tts_chunk in llm_then_tts(text):
            speaker.write(tts_chunk)
```

依赖异步 I/O 和可取消的 TTS 流。WebRTC peerconnection.stop() 是操作音频轨道的标准方式。

## 生产使用

2026 年技术栈：

| 层 | 选择 |
|----|------|
| 传输 | LiveKit（WebRTC）或 Pion（Go） |
| VAD | Silero VAD 4.0 |
| 流式 ASR | Parakeet-CTC-0.6B 或 Whisper-Streaming |
| LLM 首个 token | Groq、Cerebras、vLLM-streaming |
| 流式 TTS | Kokoro 或 ElevenLabs Turbo v2.5 |
| 回声消除 | WebRTC AEC3 |
| 端到端原生 | OpenAI Realtime API 或 Moshi |

## 常见坑

- **为安全起见缓冲 500 ms。** 缓冲区本身就是延迟下限。缩小它。
- **不固定线程。** 音频回调运行在优先级低于 UI 的线程上 = 负载下卡顿。
- **TTS 块太小。** 小于 200 ms 的块使声码器伪影清晰可闻。320 ms 块是最佳点。
- **无抖动缓冲区。** 真实网络是抖动的；没有平滑会出现爆音。
- **单点错误处理。** 音频流水线必须防崩溃。一个异常就会终止会话。

## 上手实践

保存为 `outputs/skill-realtime-designer.md`。设计一个带有每阶段具体延迟预算的实时音频流水线。

## 练习

1. **简单。** 运行 `code/main.py`。模拟环形缓冲区 + 能量 VAD；打印模拟 10 秒流的每阶段延迟。
2. **中等。** 使用 `sounddevice`，构建一个以 20 ms 帧处理麦克风的直通循环，并在每帧打印 VAD 状态。
3. **困难。** 用 `aiortc` 构建全双工回声测试：浏览器 → WebRTC → Python → WebRTC → 浏览器。用 1 kHz 脉冲测量端到端延迟。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 环形缓冲区（Ring buffer） | 循环队列 | 固定大小、无锁（或 SPSC 锁）的音频帧 FIFO。 |
| VAD | 静音门控 | 标记语音与非语音的模型或启发式方法。 |
| 流式 ASR | 实时 STT | 随音频到来输出部分文本；有界前瞻。 |
| 抖动缓冲区（Jitter buffer） | 网络平滑器 | 重新排序乱序数据包的队列；典型值 60-80 ms。 |
| AEC | 回声消除 | 消除扬声器到麦克风的反馈路径。 |
| 插话（Barge-in） | 用户中断 | 系统在 TTS 播放过程中检测到用户说话；必须取消播放。 |
| 全双工（Full duplex） | 双向同时 | 用户和机器人可以同时说话；Moshi 支持全双工。 |

## 延伸阅读

- [Macháček et al. (2023). Whisper-Streaming](https://arxiv.org/abs/2307.14743) — 分块近流式 Whisper
- [Kyutai (2024). Moshi](https://kyutai.org/Moshi.pdf) — 200 ms 延迟全双工
- [LiveKit Agents framework (2024)](https://docs.livekit.io/agents/) — 生产音频智能体编排
- [Silero VAD repo](https://github.com/snakers4/silero-vad) — 亚毫秒 VAD，Apache 2.0
- [WebRTC AEC3 paper](https://webrtc.googlesource.com/src/+/main/modules/audio_processing/aec3/) — 开源回声消除
