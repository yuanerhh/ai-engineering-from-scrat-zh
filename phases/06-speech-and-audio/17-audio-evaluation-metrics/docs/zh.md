# 音频评估——WER、MOS、UTMOS、MMAU、FAD 与开放排行榜

> 无法衡量就无法交付。本课列出 2026 年每项音频任务的指标：ASR（WER、CER、RTFx）、TTS（MOS、UTMOS、SECS、ASR 往返 WER）、音频语言（MMAU、LongAudioBench）、音乐（FAD、CLAP）和说话人（EER）。以及你进行对比的排行榜。

**类型：** 学习
**语言：** Python
**前置条件：** Phase 6 · 04、06、07、09、10；Phase 2 · 09（模型评估）
**时长：** 约 60 分钟

## 问题背景

每项音频任务都有多个指标，每个衡量不同维度。使用错误的指标，会让你交付一个在仪表盘上看起来很好但在生产中很糟糕的模型。2026 年的规范清单：

| 任务 | 主要指标 | 次要指标 |
|------|---------|---------|
| ASR | WER | CER · RTFx · 首词延迟 |
| TTS | MOS / UTMOS | SECS · ASR 往返 WER · CER · TTFA |
| 声音克隆 | SECS（ECAPA 余弦） | MOS · CER |
| 说话人验证 | EER | minDCF · 工作点 FAR/FRR |
| 说话人分离 | DER | JER · 说话人混淆 |
| 音频分类 | top-1 · mAP | 宏 F1 · 每类召回率 |
| 音乐生成 | FAD | CLAP · 听觉小组 MOS |
| 音频语言模型 | MMAU-Pro | LongAudioBench · AudioCaps FENSE |
| 流式 S2S | 延迟 P50/P95 | WER · MOS |

## 核心概念

![音频评估矩阵——指标 vs 任务 vs 2026 年排行榜](../assets/eval-landscape.svg)

### ASR 指标

**WER（词错误率，Word Error Rate）。** `(S + D + I) / N`。评分前转小写、去除标点、归一化数字。使用 `jiwer` 或 OpenAI 的 `whisper_normalizer`。< 5% = 朗读语音的人类水平。

**CER（字符错误率，Character Error Rate）。** 相同公式，字符级别。用于词分割有歧义的声调语言（普通话、粤语）。

**RTFx（实时倍率的倒数）。** 每秒挂钟时间处理的音频秒数。越高越好。Parakeet-TDT 达到 3380×。Whisper-large-v3 约 30×。

**首词延迟（First-token latency）。** 从音频输入到首个转录 token 的挂钟时间。对流式处理至关重要。Deepgram Nova-3：约 150 ms。

### TTS 指标

**MOS（平均意见分，Mean Opinion Score）。** 1-5 人类评分。黄金标准，但速度慢。每个样本收集 20+ 名听众，每个模型 100+ 个样本。

**UTMOS（2022-2026）。** 学习型 MOS 预测器。在标准基准上与人类 MOS 相关性约 0.9。F5-TTS：UTMOS 3.95；真实音频：4.08。

**SECS（说话人编码器余弦相似度，Speaker Encoder Cosine Similarity）。** 用于声音克隆。参考和克隆输出之间的 ECAPA 嵌入余弦值。> 0.75 = 可识别的克隆。

**ASR 往返 WER（WER-on-ASR-round-trip）。** 对 TTS 输出运行 Whisper，计算与输入文本的 WER。捕获可懂度退化。2026 年 SOTA：< 2% CER。

**TTFA（首个音频时间，Time-to-first-audio）。** 挂钟延迟。Kokoro-82M：约 100 ms；F5-TTS：约 1 秒。

### 声音克隆专项指标

**SECS + MOS + CER** 三元组。高 SECS 低 MOS 意味着音色正确但不自然；反之意味着声音自然但说话人错误。

### 说话人验证

**EER（等错误率，Equal Error Rate）。** 误接受率等于误拒绝率时的阈值。VoxCeleb1-O 上的 ECAPA：0.87%。

**minDCF（最小检测成本）。** 选定工作点（通常 FAR=0.01）的加权成本。比 EER 更贴近生产实际。

### 说话人分离

**DER（说话人分离错误率，Diarization Error Rate）。** `(FA + Miss + Confusion) / 总说话人时间`。漏检语音 + 误报语音 + 说话人混淆，各作为比例。AMI 会议：DER 约 10-20% 属正常。pyannote 3.1 + Precision-2 商业方案：良好录音条件下 < 10% DER。

**JER（Jaccard 错误率）。** DER 的替代方案，对短段偏差鲁棒。

### 音频分类

多标签：所有类别的 **mAP（平均精度均值）**。AudioSet：BEATs-iter3 达到 0.548 mAP。

多类别互斥：**top-1、top-5 准确率**。Speech Commands v2：top-1 99.0%（Audio-MAE）。

不平衡：**宏 F1** + **每类召回率**。报告每类结果——聚合准确率会掩盖哪些类别失败。

### 音乐生成

**FAD（弗雷歇音频距离，Fréchet Audio Distance）。** 真实与生成音频的 VGGish 嵌入分布之间的距离。MusicGen-small 在 MusicCaps 上：4.5。MusicLM：4.0。越低越好。

**CLAP 分数。** 使用 CLAP 嵌入的文本-音频对齐分数。> 0.3 = 合理对齐。

**听觉小组 MOS。** 消费级音乐的最终裁决。Suno v5 ELO 1293（来自 TTS Arena 的成对人类偏好）。

### 音频语言基准

**MMAU（大规模多音频理解，Massive Multi-Audio Understanding）。** 10000 个音频问答对。

**MMAU-Pro。** 1800 个难题，四个类别：语音/声音/音乐/多音频。4 选 1 随机概率 25%。Gemini 2.5 Pro 总体约 60%；所有模型的多音频约 22%。

**LongAudioBench。** 多分钟音频片段配语义查询。Audio Flamingo Next 超越 Gemini 2.5 Pro。

**AudioCaps / Clotho。** 字幕基准。SPICE、CIDEr、FENSE 指标。

### 流式语音到语音

**延迟 P50 / P95 / P99。** 从用户说话结束到首个可听响应的挂钟时间。Moshi：200 ms；GPT-4o Realtime：300 ms。

**WER / MOS** 对输出结果评估。

**插话响应性。** 从用户中断到助手静音的时间。目标 < 150 ms。

### 2026 年排行榜

| 排行榜 | 赛道 | 地址 |
|--------|------|------|
| Open ASR Leaderboard（HF） | 英语 + 多语言 + 长格式 | `huggingface.co/spaces/hf-audio/open_asr_leaderboard` |
| TTS Arena（HF） | 英语 TTS | `huggingface.co/spaces/TTS-AGI/TTS-Arena` |
| Artificial Analysis Speech | TTS + STT，成对投票 ELO | `artificialanalysis.ai/speech` |
| MMAU-Pro | LALM 推理 | `mmaubenchmark.github.io` |
| SpeakerBench / VoxSRC | 说话人识别 | `voxsrc.github.io` |
| MMAU 音乐子集 | 音乐 LALM | （MMAU 内） |
| HEAR 基准 | 自监督音频 | `hearbenchmark.com` |

## 动手实现

### 步骤一：带归一化的 WER

```python
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, Strip

transform = Compose([ToLowerCase(), RemovePunctuation(), Strip()])
score = wer(
    truth="Please turn on the lights.",
    hypothesis="please turn on the light",
    truth_transform=transform,
    hypothesis_transform=transform,
)
# ~0.17
```

### 步骤二：TTS ASR 往返 WER

```python
def ttr_wer(tts_model, asr_model, texts):
    errors = []
    for txt in texts:
        audio = tts_model.synthesize(txt)
        recog = asr_model.transcribe(audio)
        errors.append(wer(truth=txt, hypothesis=recog))
    return sum(errors) / len(errors)
```

### 步骤三：声音克隆的 SECS

```python
from speechbrain.inference.speaker import EncoderClassifier
sv = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb")

emb_ref = sv.encode_batch(load_wav("reference.wav"))
emb_clone = sv.encode_batch(load_wav("cloned.wav"))
secs = torch.nn.functional.cosine_similarity(emb_ref, emb_clone, dim=-1).item()
```

### 步骤四：音乐生成的 FAD

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()
score = fad.get_fad_score("generated_folder/", "reference_folder/")
```

### 步骤五：说话人验证的 EER（与第 6 课代码相同）

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        frr = sum(1 for s in same_scores if s < t) / len(same_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

## 生产使用

每次部署都配套固定的评估框架，在每次模型更新时运行。三条核心原则：

1. **评分前归一化。** 转小写、去除标点、展开数字。报告归一化规则。
2. **报告分布，而非平均值。** 延迟用 P50/P95/P99。分类用每类召回率。MMAU 用每类别结果。
3. **运行一个规范的公开基准。** 即使你的生产数据不同，在 Open ASR / TTS Arena / MMAU 上报告也能让审阅者进行苹果对苹果的比较。

## 常见坑

- **UTMOS 外推。** 在 VCTK 风格的干净语音上训练；对噪声/克隆/情感音频评分较差。
- **MOS 评判小组偏差。** 20 名 Amazon Mechanical Turk 工人 ≠ 20 名目标用户。赌注高时为领域专家小组付费。
- **FAD 依赖参考集。** 跨模型时与相同参考分布比较。
- **聚合 WER。** 总体 5% WER 可能掩盖口音语音上 30% WER。按人口学切片报告。
- **公开基准饱和。** 大多数前沿模型在标准基准上接近天花板。构建反映你流量的内部保留测试集。

## 上手实践

保存为 `outputs/skill-audio-evaluator.md`。为任何音频模型发布选择指标、基准测试和报告格式。

## 练习

1. **简单。** 运行 `code/main.py`。在玩具输入上计算 WER / CER / EER / SECS / FAD 类 / MMAU 类指标。
2. **中等。** 构建 TTS ASR 往返 WER 框架。将 Kokoro 或 F5-TTS 输出通过 Whisper 处理。对 50 个提示词计算 WER。标记 WER > 10% 的提示词。
3. **困难。** 在 MMAU-Pro 语音 + 多音频子集（各 50 项）上评测第 10 课选择的 LALM。报告每类别准确率，并与发布数字比较。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| WER | ASR 分数 | 归一化后词级别的 `(S+D+I)/N`。 |
| CER | 字符 WER | 用于声调语言或字符级系统。 |
| MOS | 人类意见 | 1-5 评分；20+ 名听众 × 100 个样本。 |
| UTMOS | ML MOS 预测器 | 学习型模型；与人类 MOS 相关性约 0.9。 |
| SECS | 声音克隆相似度 | 参考与克隆之间的 ECAPA 余弦。 |
| EER | 说话人验证分数 | FAR = FRR 时的阈值。 |
| DER | 说话人分离分数 | (FA + Miss + Confusion) / 总计。 |
| FAD | 音乐生成质量 | VGGish 嵌入上的弗雷歇距离。 |
| RTFx | 吞吐量 | 每秒挂钟时间处理的音频秒数。 |

## 延伸阅读

- [jiwer](https://github.com/jitsi/jiwer) — 带归一化工具的 WER/CER 库
- [UTMOS (Saeki et al. 2022)](https://arxiv.org/abs/2204.02152) — 学习型 MOS 预测器
- [Fréchet Audio Distance (Kilgour et al. 2019)](https://arxiv.org/abs/1812.08466) — 音乐生成标准
- [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — 2026 年实时排名
- [TTS Arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena) — 人类投票 TTS 排行榜
- [MMAU-Pro benchmark](https://mmaubenchmark.github.io/) — LALM 推理排行榜
- [HEAR benchmark](https://hearbenchmark.com/) — 音频 SSL 基准
