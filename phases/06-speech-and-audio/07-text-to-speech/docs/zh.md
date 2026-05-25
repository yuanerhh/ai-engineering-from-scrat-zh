# 文本转语音（TTS）——从 Tacotron 到 F5 和 Kokoro

> ASR 将语音转为文本；TTS 将文本转为语音。2026 年的技术栈分三部分：文本 → token，token → 梅尔频谱图，梅尔频谱图 → 波形。每个部分都有一个可在笔记本电脑上运行的默认模型。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 6 · 02（频谱图与梅尔特征）、Phase 5 · 09（序列到序列）、Phase 7 · 05（完整 Transformer）
**时长：** 约 75 分钟

## 问题背景

给定字符串："Please remind me to water the plants at 6 pm."，你需要生成一段 3 秒的音频，听起来自然，韵律正确（停顿、重音），发音准确，并在语音助手场景下在 CPU 上 300 毫秒内完成。你还需要切换声音，处理代码混合输入（"remind me at 6 pm, daijoubu?"），并正确处理人名。

现代 TTS 流水线包含三个部分：

1. **文本前端（Text frontend）。** 归一化文本（日期、数字、邮箱），转换为音素或子词 token，预测韵律特征。
2. **声学模型（Acoustic model）。** 文本 → 梅尔频谱图。Tacotron 2（2017）、FastSpeech 2（2020）、VITS（2021）、F5-TTS（2024）、Kokoro（2024）。
3. **声码器（Vocoder）。** 梅尔频谱图 → 波形。WaveNet（2016）、WaveRNN、HiFi-GAN（2020）、BigVGAN（2022）、2024 年后的神经编解码声码器。

2026 年，随着端到端扩散和流匹配模型的出现，声学模型与声码器的边界趋于模糊。但三段式的心智模型在调试时仍然适用。

## 核心概念

![Tacotron、FastSpeech、VITS、F5/Kokoro 并排对比](../assets/tts.svg)

**Tacotron 2（2017）。** 序列到序列（seq2seq）：字符嵌入 → 双向 LSTM 编码器 → 位置敏感注意力 → 自回归 LSTM 解码器生成梅尔帧。速度慢（自回归），长文本下不稳定。仍作为基准线被引用。

**FastSpeech 2（2020）。** 非自回归。时长预测器（duration predictor）输出每个音素对应的梅尔帧数。单次前向，比 Tacotron 快 10 倍。自然度略低（单调对齐），但部署广泛。

**VITS（2021）。** 联合训练编码器 + 基于流的时长预测器 + HiFi-GAN 声码器，通过变分推断端到端优化。质量高，单一模型。2022-2024 年主流开源 TTS。变体：YourTTS（多说话人零样本）、XTTS v2（2024，Coqui）。

**F5-TTS（2024）。** 基于流匹配（flow matching）的扩散 Transformer。韵律自然，5 秒参考音频即可实现零样本声音克隆。2026 年开源 TTS 排行榜首位。3.35 亿参数。

**Kokoro（2024）。** 小型（8200 万参数），可在 CPU 运行，英语实时 TTS 最佳选择。仅支持英语封闭词汇，Apache 2.0 许可。

**OpenAI TTS-1-HD、ElevenLabs v2.5、Google Chirp-3。** 商业最前沿。ElevenLabs v2.5 的情感标签（"[whispered]"、"[laughing]"）和角色声音在 2026 年有声书制作中占据主导。

### 声码器演进

| 时代 | 声码器 | 延迟 | 质量 |
|------|--------|------|------|
| 2016 | WaveNet | 仅支持离线 | 发布时 SOTA |
| 2018 | WaveRNN | ~实时 | 良好 |
| 2020 | HiFi-GAN | 100× 实时 | 接近人类 |
| 2022 | BigVGAN | 50× 实时 | 跨说话人/语言泛化性好 |
| 2024 | SNAC、DAC（神经编解码器） | 与 AR 模型集成 | 离散 token，比特效率高 |

到 2026 年，大多数"TTS"模型都是从文本到波形的端到端模型；梅尔频谱图成为内部表示。

### 评估

- **MOS（平均意见分，Mean Opinion Score）。** 1-5 分，众包评估。仍是黄金标准，但速度极慢。
- **CMOS（对比 MOS，Comparative MOS）。** A vs B 偏好对比。每条标注的置信区间更窄。
- **UTMOS、DNSMOS。** 无参考神经 MOS 预测器。用于排行榜评测。
- **CER（字符错误率，Character Error Rate），通过 ASR 计算。** 将 TTS 输出送入 Whisper，计算与输入文本的 CER。作为可懂度代理指标。
- **SECS（说话人嵌入余弦相似度，Speaker Embedding Cosine Similarity）。** 声音克隆质量。

2026 年在 LibriTTS test-clean 上的数字：

| 模型 | UTMOS | CER（通过 Whisper） | 规模 |
|------|-------|-------------------|------|
| 真实音频 | 4.08 | 1.2% | — |
| F5-TTS | 3.95 | 2.1% | 3.35 亿 |
| XTTS v2 | 3.81 | 3.5% | 4.70 亿 |
| VITS | 3.62 | 3.1% | 2500 万 |
| Kokoro v0.19 | 3.87 | 1.8% | 8200 万 |
| Parler-TTS Large | 3.76 | 2.8% | 23 亿 |

## 动手实现

### 步骤一：音素化输入

```python
from phonemizer import phonemize
ph = phonemize("Hello world", language="en-us", backend="espeak")
# 'həloʊ wɜːld'
```

音素是通用桥梁。对于 VITS 级别以下的质量，避免直接输入原始文本。

### 步骤二：运行 Kokoro（2026 年 CPU 默认方案）

```python
from kokoro import KPipeline
tts = KPipeline(lang_code="a")  # "a" = 美式英语
audio, sr = tts("Please remind me to water the plants at 6 pm.", voice="af_bella")
# audio: float32 tensor，sr=24000
```

离线运行，单文件，8200 万参数。

### 步骤三：使用 F5-TTS 进行声音克隆

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="my_voice_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please remind me to water the plants.",
)
```

传入 5 秒参考音频片段及其文字稿；F5 克隆韵律和音色。

### 步骤四：从零构建 HiFi-GAN 声码器

内容过多，无法放入教程脚本，但结构如下：

```python
class HiFiGAN(nn.Module):
    def __init__(self, mel_channels=80, upsample_rates=[8, 8, 2, 2]):
        super().__init__()
        # 4 个上采样块，总计 256× 从梅尔采样率到音频采样率
        ...
    def forward(self, mel):
        return self.blocks(mel)  # -> 波形
```

训练：对抗损失（对短窗口的判别器）+ 梅尔频谱图重建损失 + 特征匹配损失。已商品化——从 `hifi-gan` 仓库或 nvidia-NeMo 使用预训练检查点。

### 步骤五：完整流水线（伪代码）

```python
text = "Please remind me at 6 pm."
phones = phonemize(text)
mel = acoustic_model(phones, speaker=alice)      # [T, 80]
wav = vocoder(mel)                                # [T * 256]
soundfile.write("out.wav", wav, 24000)
```

## 生产使用

2026 年技术栈：

| 场景 | 选择 |
|------|------|
| 实时英语语音助手 | Kokoro（CPU）或 XTTS v2（GPU） |
| 5 秒参考音频声音克隆 | F5-TTS |
| 商业角色声音 | ElevenLabs v2.5 |
| 有声书旁白 | ElevenLabs v2.5 或 XTTS v2 + 微调 |
| 低资源语言 | 在 5-20 小时目标语言数据上训练 VITS |
| 情感表达 / 情感标签 | ElevenLabs v2.5 或 StyleTTS 2 微调 |

2026 年开源领跑者：**F5-TTS 质量最佳，Kokoro 效率最高**。除非是历史研究，否则不要使用 Tacotron。

## 常见坑

- **缺少文本归一化器。** "Dr. Smith"读成"Doctor"还是"Drive"？"2026"读成"twenty twenty six"还是"two zero two six"？在音素化**之前**先归一化。
- **OOV 专有名词。** "Ghumare"→"ghyu-mair"？为未知 token 配置 grapheme-to-phoneme 回退模型。
- **削波（Clipping）。** 声码器输出很少削波，但推理时梅尔尺度不匹配可能超出 ±1.0。始终使用 `np.clip(wav, -1, 1)`。
- **采样率不匹配。** Kokoro 输出 24 kHz；下游流水线期望 16 kHz → 需重采样，否则产生混叠。

## 上手实践

保存为 `outputs/skill-tts-designer.md`。为给定的声音、延迟和语言目标设计 TTS 流水线。

## 练习

1. **简单。** 运行 `code/main.py`。从玩具词汇表构建音素字典，估计每个音素的时长，打印模拟的"梅尔"调度。
2. **中等。** 安装 Kokoro，用声音 `af_bella` 和 `am_adam` 合成同一句话。比较音频时长和主观质量。
3. **困难。** 录制 5 秒参考音频。使用 F5-TTS 克隆。报告参考和克隆输出之间的 SECS。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 音素（Phoneme） | 声音单位 | 抽象音类；英语有 39 个（ARPABet）。 |
| 时长预测器（Duration predictor） | 每个音素持续多长 | 非 AR 模型输出；每个音素对应的整数帧数。 |
| 声码器（Vocoder） | 梅尔 → 波形 | 将梅尔频谱图映射到原始样本的神经网络。 |
| HiFi-GAN | 标准声码器 | 基于 GAN；2020-2024 年主流。 |
| MOS | 主观质量 | 人类评分者的 1-5 分平均意见分。 |
| SECS | 声音克隆指标 | 目标和输出说话人嵌入的余弦相似度。 |
| F5-TTS | 2024 开源 SOTA | 流匹配扩散；零样本克隆。 |
| Kokoro | CPU 英语领先者 | 8200 万参数模型，Apache 2.0。 |

## 延伸阅读

- [Shen et al. (2017). Tacotron 2](https://arxiv.org/abs/1712.05884) — 序列到序列基准线
- [Kim, Kong, Son (2021). VITS](https://arxiv.org/abs/2106.06103) — 端到端基于流的模型
- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — 当前开源 SOTA
- [Kong, Kim, Bae (2020). HiFi-GAN](https://arxiv.org/abs/2010.05646) — 2026 年仍在部署的声码器
- [Kokoro-82M on HuggingFace](https://huggingface.co/hexgrad/Kokoro-82M) — 2024 年 CPU 友好英语 TTS
