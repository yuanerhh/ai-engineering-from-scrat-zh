# 语音识别（ASR）——CTC、RNN-T 与注意力机制

> 语音识别就是在每个时间步做音频分类，由一个懂英语和沉默的序列模型将其串联起来。CTC、RNN-T 和注意力机制是三种实现方式，选一种并理解其原因。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 6 · 02（频谱图与梅尔特征）、Phase 5 · 08（文本的 CNN 与 RNN）、Phase 5 · 10（注意力机制）
**时长：** 约 45 分钟

## 问题背景

你有一段 10 秒的 16 kHz 音频，想得到一个字符串："turn on the kitchen lights"。挑战在于结构：音频帧与字符不是一一对应的。"okay"这个词可能持续 200ms 或 1200ms。沉默标点着话语。有些音素比其他音素更长。输出 token 的数量事先未知。

三种表述方式解决了这个问题：

1. **CTC（连接主义时序分类，Connectionist Temporal Classification）。** 输出每帧的 token 概率，包含一个特殊的*空白（blank）*。解码时折叠重复项并去除空白。非自回归，速度快。被 wav2vec 2.0、MMS 使用。
2. **RNN-T（循环神经网络转录器，Recurrent Neural Network Transducer）。** 联合网络根据编码器帧和先前 token 预测下一个 token。可流式处理。被 Google 设备端 ASR、NVIDIA Parakeet 使用。
3. **注意力编码器-解码器。** 编码器将音频压缩为隐藏状态，解码器交叉注意编码器输出，自回归地生成 token。被 Whisper、SeamlessM4T 使用。

2026 年，LibriSpeech test-clean 上的 SOTA WER 为 1.4%（Parakeet-TDT-1.1B，NVIDIA）和 1.58%（Whisper-Large-v3-turbo）。模型差异很小，但部署差异很大。

## 核心概念

![三种 ASR 表述方式：CTC、RNN-T、注意力编码器-解码器](../assets/asr-formulations.svg)

**CTC 直觉。** 让编码器输出 `T` 个帧级分布，每个分布跨越 `V+1` 个 token（V 个字符 + 空白）。对于长度为 `U < T` 的目标字符串 `y`，任何折叠后等于 `y` 的帧对齐都算数。CTC 损失对所有这些对齐求和。推理：逐帧 argmax，折叠重复项，去除空白。

优点：非自回归，可流式，零前瞻。缺点：*条件独立假设*——每帧的预测独立于其他帧，因此没有内部语言模型。通过束搜索或浅层融合加入外部语言模型来修复。

**RNN-T 直觉。** 添加一个*预测器（predictor）*网络嵌入 token 历史，以及一个*联合器（joiner）*将预测器状态与编码器帧结合成联合分布（`V+1`，+1 是 null/不发射）。显式建模了 CTC 所忽略的条件依赖。可流式处理，因为每一步只以过去的帧和过去的 token 为条件。

优点：可流式 + 内部语言模型。缺点：训练更复杂且消耗内存（3D 损失格）；RNN-T 损失核是整个库的一类。

**注意力编码器-解码器。** 对数梅尔帧上的编码器（6-32 个 Transformer 层）。解码器（6-32 个 Transformer 层）交叉注意编码器输出，自回归地生成 token。无对齐约束——注意力可以关注音频的任何位置。非流式，除非限制注意力（分块 Whisper-Streaming，2024）。

优点：离线 ASR 质量最高，易于使用标准 seq2seq 工具训练。缺点：自回归延迟与输出长度成正比；无工程支持就无法流式处理。

### WER：那个唯一的数字

**词错误率（Word Error Rate，WER）** = `(S + D + I) / N`，其中 S=替换，D=删除，I=插入，N=参考词数。在词级别匹配 Levenshtein 编辑距离。越低越好。WER 超过 20% 通常不可用；低于 5% 是朗读语音的人类水平。2026 年标准基准上的数字：

| 模型 | LibriSpeech test-clean | LibriSpeech test-other | 规模 |
|------|----------------------|----------------------|------|
| Parakeet-TDT-1.1B | 1.40% | 2.78% | 11亿参数 |
| Whisper-Large-v3-turbo | 1.58% | 3.03% | 8.09亿 |
| Canary-1B Flash | 1.48% | 2.87% | 10亿 |
| Seamless M4T v2 | 1.7% | 3.5% | 23亿 |

这些都是基于编码器-解码器或 RNN-T。纯 CTC 系统（wav2vec 2.0）在 test-clean 上约为 1.8-2.1%。

## 动手实现

### 步骤一：贪心 CTC 解码

```python
def ctc_greedy(frame_logits, blank=0, vocab=None):
    # frame_logits: 每帧概率向量的列表
    preds = [max(range(len(p)), key=lambda i: p[i]) for p in frame_logits]
    out = []
    prev = -1
    for p in preds:
        if p != prev and p != blank:
            out.append(p)
        prev = p
    return "".join(vocab[i] for i in out) if vocab else out
```

两条规则：折叠连续重复项，去除空白。示例：`a a _ _ a b b _ c` → `a a b c`。

### 步骤二：束搜索 CTC

```python
def ctc_beam(frame_logits, beam=8, blank=0):
    import math
    beams = [([], 0.0)]  # (tokens, log_prob)
    for p in frame_logits:
        log_p = [math.log(max(pi, 1e-10)) for pi in p]
        candidates = []
        for seq, lp in beams:
            for t, lpt in enumerate(log_p):
                new = seq[:] if t == blank else (seq + [t] if not seq or seq[-1] != t else seq)
                candidates.append((new, lp + lpt))
        candidates.sort(key=lambda x: -x[1])
        beams = candidates[:beam]
    return beams[0][0]
```

生产环境使用带语言模型融合的前缀树束搜索；这是概念性骨架。

### 步骤三：WER

```python
def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[len(r)][len(h)] / max(1, len(r))
```

### 步骤四：对 Whisper 进行推理

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("clip.wav")
print(result["text"])
```

2026 年最强通用 ASR 的单行代码。在 24 GB GPU 上以约 20 倍实时速度运行。

### 步骤五：使用 Parakeet 或 wav2vec 2.0 进行流式处理

```python
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="nvidia/parakeet-tdt-1.1b")
for chunk in streaming_audio():
    print(asr(chunk, return_timestamps=True))
```

流式 ASR 需要分块编码器注意力和携带状态；使用支持它的库（Parakeet 用 NeMo，带 `chunk_length_s` 的 `transformers` 流水线）。

## 生产使用

2026 年技术栈：

| 场景 | 选择 |
|------|------|
| 英语，离线，最高质量 | Whisper-large-v3-turbo |
| 多语言，鲁棒 | SeamlessM4T v2 |
| 流式，低延迟 | Parakeet-TDT-1.1B 或 Riva |
| 边缘，移动端，<500ms 延迟 | Whisper-Tiny 量化版或 Moonshine（2024） |
| 长音频 | 带 VAD 分块的 Whisper（WhisperX） |
| 领域专属（医疗、法律） | 微调 wav2vec 2.0 + 领域语言模型融合 |

## 2026 年仍会出现的坑

- **没有 VAD。** 对静音运行 Whisper 会产生幻觉（"Thanks for watching!"）。始终用 VAD 进行门控。
- **字符 vs 词 vs 子词 WER。** 报告归一化后（小写，去除标点）的词级 WER。
- **语言识别漂移。** Whisper 的自动语言识别会将嘈杂片段误路由到日语或威尔士语；当你知道语言时强制指定 `language="en"`。
- **不分块的长音频。** Whisper 有 30 秒的窗口。对任何更长的内容使用 `chunk_length_s=30, stride=5`。

## 上手实践

保存为 `outputs/skill-asr-picker.md`。为给定的部署目标选择模型、解码策略、分块和语言模型融合。

## 练习

1. **简单。** 运行 `code/main.py`。它贪心解码一个手工制作的 CTC 输出，并计算与参考文本的 WER。
2. **中等。** 正确实现步骤二中的前缀树束搜索（考虑空白合并规则）。与贪心解码在 10 个合成样本上比较。
3. **困难。** 使用 `whisper-large-v3-turbo` 在 LibriSpeech test-clean 上测试。计算前 100 个句子的 WER，与已发表的数字比较。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| CTC | 空白 token 损失 | 对所有帧到 token 对齐的边际化；非自回归。 |
| RNN-T | 流式损失 | CTC + 下一 token 预测器；处理词序。 |
| 注意力编码器-解码器 | Whisper 风格 | 编码器 + 交叉注意解码器；离线质量最佳。 |
| WER | 你报告的那个数字 | 词级的 `(S+D+I)/N`。 |
| 空白（Blank） | 空白 | CTC 中表示"本帧无发射"的特殊 token。 |
| 语言模型融合（LM fusion） | 外部语言模型 | 束搜索时加入加权语言模型 log 概率。 |
| VAD | 静音门控 | 语音活动检测器；裁剪非语音部分。 |

## 延伸阅读

- [Graves et al. (2006). Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf) — CTC 论文
- [Graves (2012). Sequence Transduction with RNNs](https://arxiv.org/abs/1211.3711) — RNN-T 论文
- [Radford et al. / OpenAI (2022). Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — 2022 年经典论文；2024 年 v3-turbo 扩展
- [NVIDIA NeMo — Parakeet-TDT card](https://huggingface.co/nvidia/parakeet-tdt-1.1b) — 2026 年开放 ASR 排行榜领先者
- [Hugging Face — Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — 25+ 模型的实时基准
