# 音频分类——从 MFCC 的 k-NN 到 AST 和 BEATs

> 从"狗叫声 vs 警笛声"到"这是哪种语言"，这些都是音频分类。特征是梅尔特征，架构每十年迭代一次，评估始终是 AUC、F1 和每类召回率。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 6 · 02（频谱图与梅尔特征）、Phase 3 · 06（CNN）、Phase 5 · 08（文本的 CNN 与 RNN）
**时长：** 约 75 分钟

## 问题背景

你拿到一段 10 秒的音频，想知道："这是什么？"城市声音（警笛、钻机、狗叫）、语音指令（是/否/停止）、语言识别（英语/西班牙语/阿拉伯语）、说话人情感（愤怒/中性），或环境声音（室内/室外、嘈杂）。这些都是*音频分类*，2026 年基准架构已经成熟：对数梅尔 → CNN 或 Transformer → softmax。

核心难点不在于网络本身，而在于数据。音频数据集有严重的类别不平衡、强烈的领域偏移（干净 vs 嘈杂），以及标签噪声（谁决定"城市杂音"vs"餐厅噪声"？）。80% 的问题来自数据整理、数据增强和评估，而不是将 CNN 换成 Transformer。

## 核心概念

![音频分类阶梯：从 MFCC 的 k-NN 到 AST 再到 BEATs](../assets/audio-classification.svg)

**MFCC 上的 k-NN（1990 年代基准线）。** 将每个音频片段的 MFCC 展平，计算与标注库的余弦相似度，返回前 K 个的多数投票。在干净的小型数据集（Speech Commands、ESC-50）上效果出奇地好。无需 GPU 即可运行。

**对数梅尔上的 2D CNN（2015-2019）。** 将 `(T, n_mels)` 对数梅尔当作图像处理。应用 ResNet-18 或 VGG 风格。对时间轴做全局均值池化。对类别做 softmax。2026 年大多数 Kaggle 竞赛的基准线仍然是这个。

**音频频谱图 Transformer，AST（2021-2024）。** 对对数梅尔做分块处理（如 16×16 块），加入位置嵌入，送入 ViT。有监督学习在 AudioSet 上的最高水平（mAP 0.485）。

**BEATs 和 WavLM-base（2024-2026）。** 在数百万小时数据上做自监督预训练。用有监督数据量的 1-10% 微调即可。2026 年非语音音频的默认起点。BEATs-iter3 在 AudioSet 上击败 AST 1-2 mAP，计算量只有四分之一。

**Whisper 编码器作为冻结主干（2024）。** 取 Whisper 的编码器，去掉解码器，附加一个线性分类器。在语言识别和简单事件分类上接近最高水平，且无需任何音频增强。这是"免费午餐"基准线。

### 类别不平衡才是真正的挑战

ESC-50：50 个类别，每类 40 个片段——平衡，容易。UrbanSound8K：10 个类别，10:1 不平衡。AudioSet：632 个类别，有 100,000:1 的长尾分布。有效技术：

- 训练时做均衡采样（评估时不做）。
- Mixup：线性插值两个音频片段（及其标签）作为增强。
- SpecAugment：随机遮盖时间和频率带。简单但关键。

### 评估

- 多类别互斥（Speech Commands）：top-1 准确率、top-5 准确率。
- 多类别多标签（AudioSet、UrbanSound 风格）：平均精度均值（mAP）。
- 严重不平衡：每类召回率 + 宏平均 F1。

2026 年你应该了解的数字：

| 基准测试 | 基准线 | 2026 SOTA | 来源 |
|---------|--------|---------|------|
| ESC-50 | 82%（AST） | 97.0%（BEATs-iter3） | BEATs 论文（2024） |
| AudioSet mAP | 0.485（AST） | 0.548（BEATs-iter3） | HEAR 排行榜 2026 |
| Speech Commands v2 | 98%（CNN） | 99.0%（Audio-MAE） | HEAR v2 结果 |

## 动手实现

### 步骤一：特征提取

```python
def featurize_mfcc(signal, sr, n_mfcc=13, n_mels=40, frame_len=400, hop=160):
    mag = stft_magnitude(signal, frame_len, hop)
    fb = mel_filterbank(n_mels, frame_len, sr)
    mels = apply_filterbank(mag, fb)
    log = log_transform(mels)
    return [dct_ii(frame, n_mfcc) for frame in log]
```

### 步骤二：固定长度汇总

```python
def summarize(mfcc_frames):
    n = len(mfcc_frames[0])
    mean = [sum(f[i] for f in mfcc_frames) / len(mfcc_frames) for i in range(n)]
    var = [
        sum((f[i] - mean[i]) ** 2 for f in mfcc_frames) / len(mfcc_frames) for i in range(n)
    ]
    return mean + var
```

简单但强大：时间轴上的均值 + 方差为 13 系数 MFCC 提供了 26 维固定嵌入。运行速度极快。早在 2017 年就在 ESC-50 上击败了最先进的神经网络基准线。

### 步骤三：k-NN

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(x * x for x in b)) or 1e-12
    return dot / (na * nb)

def knn_classify(q, bank, labels, k=5):
    sims = sorted(range(len(bank)), key=lambda i: -cosine(q, bank[i]))[:k]
    votes = Counter(labels[i] for i in sims)
    return votes.most_common(1)[0][0]
```

### 步骤四：升级到对数梅尔上的 CNN

在 PyTorch 中：

```python
import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(self, n_mels=80, n_classes=50):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):  # x: (B, 1, T, n_mels)
        return self.head(self.body(x).flatten(1))
```

300 万参数。在单张 RTX 4090 上对 ESC-50 训练约 10 分钟，准确率 80%+。

### 步骤五：2026 年默认方案——微调 BEATs

```python
from transformers import ASTFeatureExtractor, ASTForAudioClassification

ext = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=50,
    ignore_mismatched_sizes=True,
)

inputs = ext(audio, sampling_rate=16000, return_tensors="pt")
logits = model(**inputs).logits
```

对于 BEATs，通过 `beats` 库使用 `microsoft/BEATs-base`；transformers API 的形状相同。

## 生产使用

2026 年技术栈：

| 场景 | 从这里开始 |
|------|----------|
| 小型数据集（<1000 个片段） | MFCC 均值上的 k-NN（基准线）+ 音频增强 |
| 中型数据集（1K-100K） | BEATs 或 AST 微调 |
| 大型数据集（>100K） | 从头训练或微调 Whisper 编码器 |
| 实时、边缘 | 40-MFCC CNN，量化到 int8（关键词检测风格） |
| 多标签（AudioSet） | BEATs-iter3 + BCE 损失 + Mixup + SpecAugment |
| 语言识别 | MMS-LID，SpeechBrain VoxLingua107 基准线 |

决策原则：**从冻结主干开始，而不是从头训练**。微调 BEATs 头几个小时就能获得 95% 的 SOTA 效果，而不是几周。

## 上手实践

保存为 `outputs/skill-classifier-designer.md`。为给定的音频分类任务选择架构、数据增强、类别平衡策略和评估指标。

## 练习

1. **简单。** 运行 `code/main.py`。它在一个 4 类合成数据集（不同音高的纯音调）上训练 k-NN MFCC 基准线。报告混淆矩阵。
2. **中等。** 用 [均值, 方差, 偏度, 峰度] 替换 `summarize`。4 阶矩池化在同一合成数据集上是否优于均值+方差？
3. **困难。** 使用 `torchaudio`，在 ESC-50 第 1 折上训练 2D CNN。报告 5 折交叉验证准确率。添加 SpecAugment（时间遮盖=20，频率遮盖=10），报告提升幅度。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| AudioSet | 音频界的 ImageNet | Google 的 200 万片段、632 类弱标注 YouTube 数据集。 |
| ESC-50 | 小型分类基准 | 50 类 × 40 个环境声片段。 |
| AST | 音频频谱图 Transformer | 对数梅尔分块上的 ViT；2021 年 SOTA。 |
| BEATs | 自监督音频模型 | 微软模型，iter3 截至 2026 年领先 AudioSet。 |
| Mixup | 配对增强 | `x = λ·x1 + (1-λ)·x2; y = λ·y1 + (1-λ)·y2`。 |
| SpecAugment | 基于遮盖的增强 | 随机遮盖频谱图的时间和频率带。 |
| mAP | 主要多标签指标 | 跨类别和阈值的平均精度均值。 |

## 延伸阅读

- [Gong, Chung, Glass (2021). AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778) — 2021-2024 年的代表性架构
- [Chen et al. (2022, rev. 2024). BEATs: Audio Pre-Training with Acoustic Tokenizers](https://arxiv.org/abs/2212.09058) — 2024 年后的默认模型
- [Park et al. (2019). SpecAugment](https://arxiv.org/abs/1904.08779) — 主流音频增强方法
- [Piczak (2015). ESC-50 dataset](https://github.com/karolpiczak/ESC-50) — 仍在使用的 50 类基准
- [Gemmeke et al. (2017). AudioSet](https://research.google.com/audioset/) — 632 类 YouTube 分类体系；仍是黄金标准
