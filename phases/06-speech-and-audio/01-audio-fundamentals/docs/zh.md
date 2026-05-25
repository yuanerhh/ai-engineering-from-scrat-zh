# 音频基础——波形、采样与傅里叶变换

> 波形是原始信号。频谱图是表示形式。梅尔特征是对机器学习友好的形式。每个现代 ASR 和 TTS 流水线都沿着这个阶梯攀登，而第一级就是理解采样和傅里叶变换。

**类型：** 学习
**语言：** Python
**前置条件：** Phase 1 · 06（向量与矩阵）、Phase 1 · 14（概率分布）
**时长：** 约 45 分钟

## 问题背景

麦克风产生一个压力随时间变化的信号。你的神经网络消耗张量。两者之间是一套约定俗成的规则，一旦违反，就会产生静默的 bug：模型训练得很好，但词错误率（WER）翻倍；或者 TTS 输出带有噪声；或者声音克隆系统记住了麦克风，而不是说话人。

语音系统中的每个 bug 都可以追溯到以下三个问题之一：

1. 数据是以什么采样率录制的，模型期望的是什么？
2. 信号是否发生了混叠？
3. 你在操作原始样本还是频率表示？

把这些搞清楚，Phase 6 的其余部分就都可以处理了。搞错了，即使是 Whisper-Large-v4 也会产生垃圾输出。

## 核心概念

![波形、采样、DFT 和频率分箱可视化](../assets/audio-fundamentals.svg)

**波形（Waveform）。** 一个在 `[-1.0, 1.0]` 范围内的一维浮点数数组。以样本编号为索引。转换为秒数：`t = n / sr`。在 16 kHz 下的 10 秒音频片段是一个有 160,000 个浮点数的数组。

**采样率（Sample Rate，sr）。** 每秒的采样数。2026 年常用采样率：

| 采样率 | 用途 |
|--------|------|
| 8 kHz | 电话、传统 VOIP。奈奎斯特频率 4 kHz 会消灭辅音，ASR 应避免使用。 |
| 16 kHz | ASR 标准。Whisper、Parakeet、SeamlessM4T v2 均使用 16 kHz。 |
| 22.05 kHz | 旧模型的 TTS 声码器训练。 |
| 24 kHz | 现代 TTS（Kokoro、F5-TTS、xTTS v2）。 |
| 44.1 kHz | CD 音频、音乐。 |
| 48 kHz | 电影、专业音频、高保真 TTS（VALL-E 2、NaturalSpeech 3）。 |

**奈奎斯特-香农定理（Nyquist-Shannon）。** 采样率为 `sr` 时，可以无歧义地表示频率最高为 `sr/2` 的信号。`sr/2` 边界称为*奈奎斯特频率*。超过奈奎斯特频率的能量会发生*混叠（aliasing）*——折叠到低频范围——从而损坏信号。降采样前始终要先做低通滤波。

**位深度（Bit depth）。** 16 位 PCM（有符号 int16，范围 ±32,767）是通用交换格式。音乐使用 24 位，内部 DSP 使用 32 位浮点。`soundfile` 等库读取 int16，但以 `[-1, 1]` 范围内的 float32 数组形式暴露。

**傅里叶变换（Fourier Transform）。** 任何有限信号都可以表示为不同频率正弦波的叠加。离散傅里叶变换（DFT）对 `N` 个样本计算 `N` 个复数系数——每个频率分箱（bin）一个。第 `k` 个分箱对应频率 `k · sr / N` Hz。幅度是该频率处的振幅，相角是相位。

**FFT（快速傅里叶变换）。** 当 `N` 为 2 的幂次时，以 `O(N log N)` 计算 DFT 的算法。每个音频库都在底层使用 FFT。在 16 kHz 下的 1024 样本 FFT 给出 512 个可用频率分箱，覆盖 0–8 kHz，频率分辨率为 15.6 Hz。

**分帧 + 加窗（Framing + Window）。** 我们不对整个音频片段做 FFT，而是将其切割成重叠的*帧*（通常 25ms，10ms 步幅），对每帧乘以窗函数（Hann、Hamming）以消除边缘不连续，然后对每帧做 FFT。这就是短时傅里叶变换（STFT，Short-Time Fourier Transform）。第 02 课从这里继续。

## 动手实现

### 步骤一：读取音频片段并绘制波形

`code/main.py` 仅使用标准库的 `wave` 模块以保持示例无外部依赖。生产环境中你会使用 `soundfile` 或 `torchaudio.load`（两者都返回 `(waveform, sr)` 元组）：

```python
import soundfile as sf
waveform, sr = sf.read("clip.wav", dtype="float32")  # shape (T,), sr=int
```

### 步骤二：从第一性原理合成正弦波

```python
import math

def sine(freq_hz, sr, seconds, amp=0.5):
    n = int(sr * seconds)
    return [amp * math.sin(2 * math.pi * freq_hz * i / sr) for i in range(n)]
```

在 16 kHz 下 440 Hz（音乐会 A 音）持续 1 秒是 16,000 个浮点数。使用 16 位 PCM 编码写入 `wave.open(..., "wb")`。

### 步骤三：手动计算 DFT

```python
def dft(x):
    N = len(x)
    out = []
    for k in range(N):
        re = sum(x[n] * math.cos(-2 * math.pi * k * n / N) for n in range(N))
        im = sum(x[n] * math.sin(-2 * math.pi * k * n / N) for n in range(N))
        out.append((re, im))
    return out
```

`O(N²)`——对 `N=256` 验证正确性是可以的，但对真实音频毫无用处。实际代码调用 `numpy.fft.rfft` 或 `torch.fft.rfft`。

### 步骤四：找到主频

幅度峰值索引 `k_star` 对应频率 `k_star * sr / N`。对 440 Hz 正弦波运行此代码，应该在分箱 `440 * N / sr` 处返回峰值。

### 步骤五：演示混叠

以 10 kHz（奈奎斯特 = 5 kHz）采样 7 kHz 的正弦波。7 kHz 超过奈奎斯特频率，折叠为 `10 − 7 = 3 kHz`。FFT 峰值出现在 3 kHz 处。这是经典的混叠演示，也是每个 DAC/ADC 都内置砖墙低通滤波器的原因。

## 生产使用

2026 年你实际会用到的技术栈：

| 任务 | 库 | 原因 |
|------|-----|------|
| 读写 WAV/FLAC/OGG | `soundfile`（libsndfile 封装） | 最快、稳定，返回 float32。 |
| 重采样 | `torchaudio.transforms.Resample` 或 `librosa.resample` | 内置正确的抗混叠。 |
| STFT / 梅尔特征 | `torchaudio` 或 `librosa` | GPU 友好；PyTorch 生态系统。 |
| 实时流处理 | `sounddevice` 或 `pyaudio` | 跨平台 PortAudio 绑定。 |
| 检查文件 | `ffprobe` 或 `soxi` | CLI，快速，报告采样率/声道/编解码器。 |

决策原则：**先匹配采样率，再匹配其他一切**。Whisper 期望 16 kHz 单声道 float32。传入 44.1 kHz 立体声，你会得到看起来像模型 bug 的垃圾输出。

## 上手实践

保存为 `outputs/skill-audio-loader.md`。该技能帮助你检查音频输入是否符合下游模型的期望，并在不符合时正确重采样。

## 练习

1. **简单。** 在 16 kHz 下合成 1 秒的 220 Hz + 440 Hz + 880 Hz 混合信号。运行 DFT。确认在预期分箱处出现三个峰值。
2. **中等。** 以 48 kHz 录制 3 秒的声音。使用 `torchaudio.transforms.Resample`（带抗混叠）降采样到 16 kHz，然后用朴素抽取（每隔三个样本取一个）降采样到 16 kHz。对两者做 FFT，混叠在哪里出现？
3. **困难。** 仅使用 `math` 和步骤三的 DFT，从零构建 STFT。帧大小 400，步幅 160，Hann 窗。用 `matplotlib.pyplot.imshow` 绘制幅度图。这就是第 02 课的频谱图。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 采样率（Sample rate） | 每秒多少样本 | ADC 测量信号的频率（Hz）。 |
| 奈奎斯特（Nyquist） | 可表示的最大频率 | `sr/2`；超过它的能量会向下混叠。 |
| 位深度（Bit depth） | 每个样本的分辨率 | `int16` = 65,536 个级别；`float32` = `[-1, 1]` 中 24 位精度。 |
| DFT | 序列的傅里叶变换 | `N` 个样本 → `N` 个复数频率系数。 |
| FFT | 快速 DFT | `O(N log N)` 算法，要求 `N` = 2 的幂次。 |
| 分箱（Bin） | 频率列 | `k · sr / N` Hz；分辨率 = `sr / N`。 |
| STFT | 频谱图的底层 | 逐帧加窗的 FFT。 |
| 混叠（Aliasing） | 奇怪的频率鬼影 | 超过奈奎斯特的能量镜像到更低分箱。 |

## 延伸阅读

- [Shannon (1949). Communication in the Presence of Noise](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) — 采样定理背后的论文
- [Smith — The Scientist and Engineer's Guide to Digital Signal Processing](https://www.dspguide.com/ch8.htm) — 免费、经典的 DSP 教材
- [librosa docs — audio primer](https://librosa.org/doc/latest/tutorial.html) — 带代码的实践指南
- [Heinrich Kuttruff — Room Acoustics (6th ed.)](https://www.routledge.com/Room-Acoustics/Kuttruff/p/book/9781482260434) — 解释为何真实世界的音频不是干净正弦波的参考书
- [Steve Eddins — FFT Interpretation notebook](https://blogs.mathworks.com/steve/2020/03/30/fft-spectrum-and-spectral-densities/) — 10 分钟理清频率分箱直觉
