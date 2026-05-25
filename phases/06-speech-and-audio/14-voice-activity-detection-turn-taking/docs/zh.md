# 语音活动检测与轮次切换——Silero、Cobra 与冲刷技巧

> 每个语音智能体的成败取决于两个决策：用户现在在说话吗？他们说完了吗？VAD 回答前一个问题。轮次检测（VAD + 静音挂起 + 语义端点模型）回答后一个问题。任何一个错误，你的助手要么打断用户，要么永远不停嘴。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 6 · 11（实时音频处理）、Phase 6 · 12（语音助手流水线）
**时长：** 约 45 分钟

## 问题背景

语音智能体在每个 20 ms 块上做出三种不同的决策：

1. **这帧是语音吗？** — VAD。逐帧二值判断。
2. **用户是否开始了新的话语？** — 起始检测。
3. **用户说完了吗？** — 端点检测（轮次结束）。

简单的能量阈值方案在任何噪声下都会失效——交通噪声、键盘声、人群嘈杂声。2026 年的答案：Silero VAD（开源，深度学习）+ 轮次检测模型（语义端点）+ VAD 校准静音挂起。

## 核心概念

![VAD 级联：能量 → Silero → 轮次检测器 → 冲刷技巧](../assets/vad-turn-taking.svg)

### 三层 VAD 级联

**第一层：能量门控。** 最便宜。阈值 RMS 在 -40 dBFS。过滤明显的静音，但会在任何高于阈值的噪声上触发。

**第二层：Silero VAD**（2020-2026，MIT）。100 万参数。在 6000+ 种语言上训练。在单个 CPU 线程上每 30 ms 块运行约 1 ms。TPR 87.7% @ FPR 5%。开源默认选择。

**第三层：语义轮次检测器。** LiveKit 的轮次检测模型（2024-2026）或自定义小分类器。区分"句中停顿"与"说完了"。使用语言上下文（语调 + 最近单词），而非仅靠静音判断。

### 关键参数及默认值

- **阈值（Threshold）。** Silero 输出概率；在 > 0.5（默认）或 > 0.3（灵敏）时分类为语音。阈值越低 = 首词截断越少，误报越多。
- **最短语音时长（Minimum speech duration）。** 拒绝短于 250 ms 的语音——通常是咳嗽或椅子噪声。
- **静音挂起（Silence hangover）端点检测。** VAD 回零后，等待 500-800 ms 再宣告轮次结束。太短 → 打断用户。太长 → 感觉迟钝。
- **预缓冲（Pre-roll buffer）。** 在 VAD 触发前保留 300-500 ms 的音频。防止"喂"被截断。

### 冲刷技巧（Kyutai，2025）

流式 STT 模型有预见延迟（Kyutai STT-1B 为 500 ms，STT-2.6B 为 2.5 秒）。通常需要在语音结束后等待那么长时间才能获得转录。冲刷技巧：当 VAD 检测到语音结束时，**向 STT 发送冲刷信号**，强制立即输出。STT 以约 4× 实时速度处理，因此 500 ms 缓冲在约 125 ms 内完成。

端到端：125 ms VAD + 冲刷 STT = 会话级延迟。

### 2026 年 VAD 对比

| VAD | TPR @ FPR 5% | 延迟 | 许可证 |
|-----|--------------|------|--------|
| WebRTC VAD（Google，2013） | 50.0% | 30 ms | BSD |
| Silero VAD（2020-2026） | 87.7% | ~1 ms | MIT |
| Cobra VAD（Picovoice） | 98.9% | ~1 ms | 商业 |
| pyannote 分割 | 95% | ~10 ms | 类 MIT |

Silero 是正确的默认选择。Cobra 是合规/准确度升级方案。纯能量 VAD 在 2026 年生产环境中没有立足之地。

## 动手实现

### 步骤一：能量门控

```python
def energy_vad(chunk, threshold_dbfs=-40.0):
    rms = (sum(x * x for x in chunk) / len(chunk)) ** 0.5
    dbfs = 20.0 * math.log10(max(rms, 1e-10))
    return dbfs > threshold_dbfs
```

### 步骤二：Python 中的 Silero VAD

```python
from silero_vad import load_silero_vad, get_speech_timestamps

vad = load_silero_vad()
audio = torch.tensor(waveform_16k, dtype=torch.float32)
segments = get_speech_timestamps(
    audio, vad, sampling_rate=16000,
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=500,
    speech_pad_ms=300,
)
for s in segments:
    print(f"{s['start']/16000:.2f}s - {s['end']/16000:.2f}s")
```

### 步骤三：轮次结束状态机

```python
class TurnDetector:
    def __init__(self, silence_hangover_ms=500, min_speech_ms=250):
        self.state = "idle"
        self.speech_ms = 0
        self.silence_ms = 0
        self.silence_hangover_ms = silence_hangover_ms
        self.min_speech_ms = min_speech_ms

    def update(self, is_speech, chunk_ms=20):
        if is_speech:
            self.speech_ms += chunk_ms
            self.silence_ms = 0
            if self.state == "idle" and self.speech_ms >= self.min_speech_ms:
                self.state = "speaking"
                return "START"
        else:
            self.silence_ms += chunk_ms
            if self.state == "speaking" and self.silence_ms >= self.silence_hangover_ms:
                self.state = "idle"
                self.speech_ms = 0
                return "END"
        return None
```

### 步骤四：冲刷技巧骨架

```python
def flush_on_end(stt_client, audio_buffer):
    stt_client.send_audio(audio_buffer)
    stt_client.send_flush()
    return stt_client.recv_transcript(timeout_ms=150)
```

STT（Kyutai、Deepgram、AssemblyAI）必须支持冲刷才能工作。Whisper streaming 不支持——它是基于块的，始终等待数据块。

## 生产使用

| 场景 | VAD 选择 |
|------|---------|
| 开放、快速、通用 | Silero VAD |
| 商业呼叫中心 | Cobra VAD |
| 设备端（手机） | Silero VAD ONNX |
| 研究/说话人分离 | pyannote 分割 |
| 零依赖回退 | WebRTC VAD（传统方案） |
| 需要高质量轮次结束检测 | Silero + LiveKit 轮次检测器叠加 |

经验法则：除非真的别无选择，否则不要部署纯能量 VAD。

## 常见坑

- **固定阈值。** 在安静环境中有效，在嘈杂环境中失效。要么在设备上校准，要么切换到 Silero。
- **静音挂起太短。** 智能体在句中打断。500-800 ms 是对话语音的最佳点。
- **挂起太长。** 感觉迟钝。与目标用户进行 A/B 测试。
- **无预缓冲。** 用户音频的前 200-300 ms 丢失。始终保持滚动预缓冲。
- **忽略语义端点。** "嗯，让我想想……"包含长停顿。用户不喜欢在想说的话说到一半被打断。使用 LiveKit 的轮次检测器或类似方案。

## 上手实践

保存为 `outputs/skill-vad-tuner.md`。为特定工作负载选择 VAD 模型、阈值、挂起时间、预缓冲和轮次检测策略。

## 练习

1. **简单。** 运行 `code/main.py`。模拟语音 + 静音 + 语音 + 咳嗽序列，测试三层 VAD。
2. **中等。** 安装 `silero-vad`，处理 5 分钟录音，调整阈值以最小化首词截断和误触发。报告精确率/召回率。
3. **困难。** 构建迷你轮次检测器：Silero VAD + 一个基于最近 10 个单词嵌入的 3 层 MLP（使用 sentence-transformers）。在手动标注的轮次结束数据集上训练。比仅用 Silero 提升 10% F1。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| VAD | 语音检测器 | 逐帧二值判断：这是语音吗？ |
| 轮次检测（Turn detection） | 端点检测 | VAD + 静音挂起 + 语义端点。 |
| 静音挂起（Silence hangover） | 语音后等待 | 宣告轮次结束前的等待时间；500-800 ms。 |
| 预缓冲（Pre-roll） | 语音前缓冲 | VAD 触发前保留 300-500 ms 音频。 |
| 冲刷技巧（Flush trick） | Kyutai 方法 | VAD → 冲刷 STT → 125 ms 而非 500 ms 延迟。 |
| 语义端点（Semantic endpoint） | "他们是打算停止说话吗？" | 查看单词而非仅看静音的 ML 分类器。 |
| TPR @ FPR 5% | ROC 点 | 标准 VAD 基准；Silero 87.7%，WebRTC 50%。 |

## 延伸阅读

- [Silero VAD](https://github.com/snakers4/silero-vad) — 参考开源 VAD
- [Picovoice Cobra VAD](https://picovoice.ai/products/cobra/) — 商业精度领先方案
- [Kyutai — Unmute + flush trick](https://kyutai.org/stt) — 低于 200 ms 的工程技巧
- [LiveKit — turn detection](https://docs.livekit.io/agents/logic/turns/) — 生产中的语义端点
- [WebRTC VAD](https://webrtc.googlesource.com/src/) — 传统基准线
- [pyannote segmentation](https://github.com/pyannote/pyannote-audio) — 说话人分离级别分割
