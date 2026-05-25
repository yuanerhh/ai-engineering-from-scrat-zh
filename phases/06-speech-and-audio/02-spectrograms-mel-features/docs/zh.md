# 频谱图、梅尔尺度与音频特征

> 神经网络对原始波形的处理并不好，但对频谱图的处理要好得多，对梅尔频谱图更是如此。2026 年每一个 ASR、TTS 和音频分类器的成败，都取决于这个单一的预处理选择。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 6 · 01（音频基础）
**时长：** 约 45 分钟

## 问题背景

取一段 10 秒的 16 kHz 音频片段。这是 160,000 个浮点数，全在 `[-1, 1]` 内，与"狗吠声"或"单词 cat"这样的标签几乎没有相关性。原始波形包含信息，但模型难以从这种形式中提取。两个发音完全相同但相差 100ms 的音素，其原始样本完全不同。

频谱图解决了这个问题。它在人类感知所忽视的时间细节上（微秒级抖动）折叠信息，同时保留感知所关注的结构（在约 10-25ms 的时间窗口内，哪些频率有能量）。

梅尔频谱图（Mel spectrogram）更进一步。人类对音高的感知是对数的：100 Hz vs 200 Hz 听起来与 1000 Hz vs 2000 Hz 的"距离"相同。梅尔尺度扭曲频率轴以匹配人类感知。梅尔频谱图是 2010 年到 2026 年语音机器学习中最重要的特征。

## 核心概念

![波形到 STFT 到梅尔频谱图到 MFCC 的阶梯](../assets/mel-features.svg)

**短时傅里叶变换（STFT，Short-Time Fourier Transform）。** 将波形切割成重叠帧（典型：25ms 窗口，10ms 步幅 = 在 16 kHz 下 400 样本 / 160 样本）。对每帧乘以窗函数（默认 Hann；Hamming 的权衡略有不同）。对每帧做 FFT。将幅度谱堆叠成形状为 `(n_frames, n_freq_bins)` 的矩阵。这就是你的频谱图。

**对数幅度（Log-magnitude）。** 原始幅度跨越 5-6 个数量级。取 `log(|X| + 1e-6)` 或 `20 * log10(|X|)` 压缩动态范围。每个生产流水线都使用对数幅度，而不是原始幅度。

**梅尔尺度（Mel scale）。** 频率 `f`（Hz）到梅尔 `m` 的映射：`m = 2595 * log10(1 + f / 700)`。1 kHz 以下近似线性，1 kHz 以上近似对数。覆盖 0-8 kHz 的 80 个梅尔分箱是标准的 ASR 输入。

**梅尔滤波器组（Mel filterbank）。** 一组在梅尔尺度上等间距的三角形滤波器。每个滤波器是相邻 FFT 分箱的加权和。将 STFT 幅度与滤波器组矩阵相乘，一次矩阵乘法即可得到梅尔频谱图。

**对数梅尔频谱图（Log-mel spectrogram）。** `log(mel_spec + 1e-10)`。Whisper 的输入。Parakeet 的输入。SeamlessM4T 的输入。2026 年通用音频前端。

**梅尔频率倒谱系数（MFCCs，Mel-Frequency Cepstral Coefficients）。** 对对数梅尔频谱图应用 DCT（II 型），保留前 13 个系数。解相关特征并进一步压缩。在 CNN/Transformer 在原始对数梅尔上赶上之前（约 2015 年）是主流特征。仍用于说话人识别（x-vectors、ECAPA）。

**分辨率权衡。** 更大的 FFT = 更好的频率分辨率，但更差的时间分辨率。25ms / 10ms 是音频 ML 的默认值；音乐用 50ms / 12.5ms；瞬态检测（鼓点、爆破音）用 5ms / 2ms。

## 动手实现

### 步骤一：分帧

```python
def frame(signal, frame_len, hop):
    n = 1 + (len(signal) - frame_len) // hop
    return [signal[i * hop : i * hop + frame_len] for i in range(n)]
```

`frame_len=400, hop=160` 下，10 秒 16 kHz 的音频片段产生 998 帧。

### 步骤二：Hann 窗

```python
import math

def hann(N):
    return [0.5 * (1 - math.cos(2 * math.pi * n / (N - 1))) for n in range(N)]
```

FFT 之前逐元素相乘。消除在非零端点截断时产生的频谱泄漏。

### 步骤三：STFT 幅度

```python
def stft_magnitude(signal, frame_len=400, hop=160):
    win = hann(frame_len)
    frames = frame(signal, frame_len, hop)
    return [magnitudes(dft([w * s for w, s in zip(win, f)])) for f in frames]
```

生产环境使用 `torch.stft` 或 `librosa.stft`（基于 FFT，向量化）。这里的循环是教学用途；在 `code/main.py` 中对短片段运行。

### 步骤四：梅尔滤波器组

```python
def hz_to_mel(f):
    return 2595.0 * math.log10(1.0 + f / 700.0)

def mel_to_hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1)

def mel_filterbank(n_mels, n_fft, sr, fmin=0, fmax=None):
    fmax = fmax or sr / 2
    mels = [hz_to_mel(fmin) + (hz_to_mel(fmax) - hz_to_mel(fmin)) * i / (n_mels + 1)
            for i in range(n_mels + 2)]
    hzs = [mel_to_hz(m) for m in mels]
    bins = [int(h * n_fft / sr) for h in hzs]
    fb = [[0.0] * (n_fft // 2 + 1) for _ in range(n_mels)]
    for m in range(n_mels):
        for k in range(bins[m], bins[m + 1]):
            fb[m][k] = (k - bins[m]) / max(1, bins[m + 1] - bins[m])
        for k in range(bins[m + 1], bins[m + 2]):
            fb[m][k] = (bins[m + 2] - k) / max(1, bins[m + 2] - bins[m + 1])
    return fb
```

覆盖 0-8 kHz 的 80 个梅尔分箱，`n_fft=400` 给出一个 `(80, 201)` 矩阵。将 `(n_frames, 201)` 的 STFT 幅度乘以其转置，得到 `(n_frames, 80)` 的梅尔频谱图。

### 步骤五：对数梅尔

```python
def log_mel(mel_spec, eps=1e-10):
    return [[math.log(max(v, eps)) for v in frame] for frame in mel_spec]
```

常见替代方案：`librosa.power_to_db`（参考归一化 dB）、`10 * log10(power + eps)`。Whisper 使用更复杂的裁剪 + 归一化程序（见 Whisper 的 `log_mel_spectrogram`）。

### 步骤六：MFCC

```python
def dct_ii(x, n_coeffs):
    N = len(x)
    return [
        sum(x[n] * math.cos(math.pi * k * (2 * n + 1) / (2 * N)) for n in range(N))
        for k in range(n_coeffs)
    ]
```

对每个对数梅尔帧应用 DCT，保留前 13 个系数。这就是你的 MFCC 矩阵。第一个系数通常被丢弃（它编码整体能量）。

## 生产使用

2026 年技术栈：

| 任务 | 特征 |
|------|------|
| ASR（Whisper、Parakeet、SeamlessM4T） | 80 对数梅尔，10ms 步幅，25ms 窗口 |
| TTS 声学模型（VITS、F5-TTS、Kokoro） | 80 梅尔，5-12ms 步幅以实现精细时间控制 |
| 音频分类（AST、PANNs、BEATs） | 128 对数梅尔，10ms 步幅 |
| 说话人嵌入（ECAPA-TDNN、WavLM） | 80 对数梅尔或原始波形 SSL |
| 音乐（MusicGen、Stable Audio 2） | EnCodec 离散 token（不是梅尔） |
| 关键词检测 | 40 MFCCs 用于小型设备 |

经验法则：**如果你不是在做音乐，就从 80 对数梅尔开始。** 任何偏差都需要有充分理由。

## 2026 年仍会出现的坑

- **梅尔数量不匹配。** 训练时用 80 个梅尔，推理时用 128 个梅尔。静默失败，要在两端都记录特征形状。
- **上游采样率不匹配。** 在 22.05 kHz 计算的梅尔与 16 kHz 的不同。在特征提取*之前*先修正采样率。
- **dB vs 对数。** Whisper 期望对数梅尔，不是 dB 梅尔。部分 HuggingFace 流水线会自动检测，你的自定义代码不会。
- **归一化漂移。** 训练时做每句话归一化，推理时做全局归一化。这是一个让 WER 翻倍的生产 bug。
- **填充造成的泄漏。** 对音频片段末尾做零填充，会在末尾帧中产生平坦频谱。使用对称填充或复制边界。

## 上手实践

保存为 `outputs/skill-feature-extractor.md`。该技能为给定的模型目标选择特征类型、梅尔数量、帧/步幅和归一化方案。

## 练习

1. **简单。** 运行 `code/main.py`。它合成一个频率从 200 Hz 扫到 4000 Hz 的啁啾信号，并打印每帧的最大梅尔分箱索引。绘图（可选）并确认与扫频匹配。
2. **中等。** 使用 `n_mels` ∈ {40, 80, 128} 和 `frame_len` ∈ {200, 400, 800} 重新运行。测量时间轴上的尖峰带宽。哪种组合对啁啾信号的分辨率最好？
3. **困难。** 实现 `power_to_db`，比较一个小型 CNN 分类器在 AudioMNIST 上使用（a）原始对数梅尔，（b）带 `ref=max` 的 dB 梅尔，（c）MFCC-13 + delta + delta-delta 的 ASR 准确率。报告 top-1 准确率。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 帧（Frame） | 一个切片 | 送入一次 FFT 的 25ms 波形块。 |
| 步幅（Hop） | 跨越距离 | 连续帧之间的样本数；ASR 默认 10ms。 |
| 窗函数（Window） | Hann/Hamming | 将帧边缘逐渐衰减到零的逐点乘数。 |
| STFT | 频谱图生成器 | 分帧 + 加窗的 FFT；产生时间 × 频率矩阵。 |
| 梅尔（Mel） | 扭曲的频率 | 对数感知尺度；`m = 2595·log10(1 + f/700)`。 |
| 滤波器组（Filterbank） | 那个矩阵 | 将 STFT 投影到梅尔分箱的三角形滤波器。 |
| 对数梅尔（Log-mel） | Whisper 的输入 | `log(mel_spec + eps)`；2026 年标准化。 |
| MFCC | 传统特征 | 对数梅尔的 DCT；13 个系数，解相关。 |

## 延伸阅读

- [Davis, Mermelstein (1980). Comparison of parametric representations for monosyllabic word recognition](https://ieeexplore.ieee.org/document/1163420) — MFCC 论文
- [Stevens, Volkmann, Newman (1937). A Scale for the Measurement of the Psychological Magnitude Pitch](https://pubs.aip.org/asa/jasa/article-abstract/8/3/185/735757/) — 原始梅尔尺度论文
- [OpenAI — Whisper source, log_mel_spectrogram](https://github.com/openai/whisper/blob/main/whisper/audio.py) — 阅读参考实现
- [librosa feature extraction docs](https://librosa.org/doc/main/feature.html) — `mfcc`、`melspectrogram` 和 hop/window 的参考
- [NVIDIA NeMo — audio preprocessing](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/asr_all.html#featurizers) — Parakeet + Canary 模型的生产级流水线
