# 构建语音助手流水线——Phase 6 综合实践

> 将第 01-11 课的所有内容拼接在一起。构建一个能倾听、推理并回应的语音助手。2026 年这是一个已解决的工程问题，而非研究问题——但集成细节决定能否成功交付。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 6 · 04、05、06、07、11；Phase 11 · 09（函数调用）；Phase 14 · 01（智能体循环）
**时长：** 约 120 分钟

## 问题背景

构建一个端到端助手：

1. 采集麦克风输入（16 kHz 单声道）。
2. 检测用户说话的开始/结束。
3. 流式转录。
4. 将转录传给能调用工具（计时器、天气、日历）的 LLM。
5. 将 LLM 文本流式传输到 TTS。
6. 将音频播放给用户。
7. 如果用户在回应过程中打断，则停止。

延迟目标：用户说完后 800 ms 内播出 TTS 首个音频字节（笔记本 CPU）。质量目标：无漏词，无静音幻觉字幕，无声音克隆泄漏，无提示注入成功。

## 核心概念

![语音助手流水线：麦克风 → VAD → STT → LLM+工具 → TTS → 扬声器](../assets/voice-assistant.svg)

### 七个组件

1. **音频采集。** 麦克风 → 16 kHz 单声道 → 20 ms 块。生产中通常用 Python 的 `sounddevice` 或原生 AudioUnit/ALSA/WASAPI。
2. **VAD（第 11 课）。** Silero VAD @ 阈值 0.5，最短语音 250 ms，静音挂起 500 ms。发出"开始"和"结束"信号。
3. **流式 STT（第 4-5 课）。** Whisper-streaming、Parakeet-TDT 或 Deepgram Nova-3（API）。部分 + 最终转录。
4. **带工具调用的 LLM。** GPT-4o / Claude 3.5 / Gemini 2.5 Flash。工具的 JSON Schema。流式 token。
5. **流式 TTS（第 7 课）。** Kokoro-82M（最快开源）或 Cartesia Sonic（商业）。在 20 个 LLM token 后启动 TTS。
6. **播放。** 扬声器输出；低带宽网络用 Opus 编码。
7. **中断处理器。** 如果 TTS 播放期间 VAD 触发，停止播放，取消 LLM，重启 STT。

### 你将遇到的三种失败模式

1. **首词截断。** VAD 启动稍晚。用户的"喂"被截掉。将启动阈值设为 0.3，而不是 0.5。
2. **中途中断混乱。** 用户打断后 LLM 仍继续生成；助手与用户同时说话。将 VAD → 取消 LLM 连接起来。
3. **静音幻觉。** Whisper 在静音预热帧上输出"Thanks for watching"。始终使用 VAD 门控。

### 2026 年生产参考技术栈

| 技术栈 | 延迟 | 许可证 | 备注 |
|--------|------|--------|------|
| LiveKit + Deepgram + GPT-4o + Cartesia | 350-500 ms | 商业 API | 2026 年行业默认 |
| Pipecat + Whisper-streaming + GPT-4o + Kokoro | 500-800 ms | 大多开源 | DIY 友好 |
| Moshi（全双工） | 200-300 ms | CC-BY 4.0 | 单模型；不同架构，见第 15 课 |
| Vapi / Retell（托管） | 300-500 ms | 商业 | 上线最快；定制有限 |
| Whisper.cpp + llama.cpp + Kokoro-ONNX | 离线 | 开源 | 隐私/边缘场景 |

## 动手实现

### 步骤一：分块麦克风采集（伪代码）

```python
import sounddevice as sd

def mic_stream(chunk_ms=20, sr=16000):
    q = queue.Queue()
    def cb(indata, frames, time, status):
        q.put(indata.copy().flatten())
    with sd.InputStream(channels=1, samplerate=sr, blocksize=int(sr * chunk_ms/1000), callback=cb):
        while True:
            yield q.get()
```

### 步骤二：VAD 门控的轮次采集

```python
def capture_turn(stream, vad, pre_roll_ms=300, silence_ms=500):
    buf, pre, triggered = [], collections.deque(maxlen=pre_roll_ms // 20), False
    silent = 0
    for chunk in stream:
        pre.append(chunk)
        if vad(chunk):
            if not triggered:
                buf = list(pre)
                triggered = True
            buf.append(chunk)
            silent = 0
        elif triggered:
            silent += 20
            buf.append(chunk)
            if silent >= silence_ms:
                return b"".join(buf)
```

### 步骤三：流式 STT → LLM → TTS

```python
async def turn(audio_bytes):
    transcript = await stt.transcribe(audio_bytes)
    async for token in llm.stream(transcript):
        async for audio in tts.stream(token):
            await speaker.play(audio)
```

### 步骤四：LLM 循环中的工具调用

```python
tools = [
    {"name": "get_weather", "parameters": {"location": "string"}},
    {"name": "set_timer", "parameters": {"seconds": "int"}},
]

async for chunk in llm.stream(user_text, tools=tools):
    if chunk.type == "tool_call":
        result = dispatch(chunk.name, chunk.args)
        continue_streaming(result)
    if chunk.type == "text":
        await tts.stream(chunk.text)
```

### 步骤五：中断处理

```python
tts_task = asyncio.create_task(tts_loop())
while True:
    chunk = await mic.get()
    if vad(chunk):
        tts_task.cancel()
        await speaker.stop()
        await new_turn()
        break
```

## 生产使用

参见 `code/main.py`，这是一个可运行的模拟程序，用桩模块连接所有七个组件，即使没有硬件也能看到流水线形状。对于真实实现，将桩替换为：

- `silero-vad`（`pip install silero-vad`）
- `deepgram-sdk` 或 `openai-whisper`
- `openai`（`gpt-4o`）或 `anthropic`
- `kokoro` 或 `cartesia`
- `sounddevice`（I/O）

## 常见坑

- **永久记录 PII。** 完整轮次音频在大多数司法管辖区属于 PII。保留 30 天，静态加密。
- **无插话支持。** 用户会打断。你的助手必须停止说话。
- **阻塞式 TTS。** 同步 TTS 会阻塞事件循环。使用异步或单独线程。
- **无工具调用错误处理。** 工具会失败。LLM 必须收到错误并重试一次，然后优雅降级。
- **过度激进的幻觉过滤器。** 过度过滤导致助手反复说"我无法帮助"。过滤不足导致助手什么都说。在保留集上校准。
- **无唤醒词选项。** 始终监听是隐私风险。添加唤醒词门控（Porcupine 或 openWakeWord）。

## 上手实践

保存为 `outputs/skill-voice-assistant-architect.md`。给定预算 + 规模 + 语言 + 合规约束，生成完整的技术栈规格。

## 练习

1. **简单。** 运行 `code/main.py`。它用桩模块模拟一次完整的端到端轮次，并打印每阶段延迟。
2. **中等。** 将 STT 桩替换为预录 `.wav` 上的真实 Whisper 模型。测量 WER 和端到端延迟。
3. **困难。** 添加工具调用：实现 `get_weather`（任意 API）和 `set_timer`。通过工具路由 LLM，并验证当用户说"设置 5 分钟计时器"时正确函数被触发，且语音回复确认了这一点。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 轮次（Turn） | 一次用户+助手来回 | 一次 VAD 界定的用户语音 + 一次 LLM-TTS 响应。 |
| 插话（Barge-in） | 中断 | 用户在助手说话时开口；助手停止。 |
| 唤醒词（Wake word） | "嘿，助手" | 短关键词检测器；Porcupine、Snowboy、openWakeWord。 |
| 端点检测（End-pointing） | 轮次结束 | VAD + 最小静音判断用户已说完。 |
| 预缓冲（Pre-roll） | 语音前缓冲 | 在 VAD 触发前保留 200-400 ms 的音频，避免首词截断。 |
| 工具调用（Tool call） | 函数调用 | LLM 生成 JSON；运行时分发；结果反馈到循环中。 |

## 延伸阅读

- [LiveKit — voice agent quickstart](https://docs.livekit.io/agents/) — 生产级参考
- [Pipecat — voice agent examples](https://github.com/pipecat-ai/pipecat) — DIY 友好框架
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — 托管语音原生路径
- [Kyutai Moshi](https://github.com/kyutai-labs/moshi) — 全双工参考（第 15 课）
- [Porcupine wake-word](https://picovoice.ai/products/porcupine/) — 唤醒词门控
- [Anthropic — tool use guide](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) — LLM 函数调用
