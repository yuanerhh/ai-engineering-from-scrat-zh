# 说话人识别与验证

> ASR 问的是"他们说了什么？"说话人识别问的是"谁说的？"数学上看起来一样——嵌入加余弦——但每个生产决策都取决于一个 EER 数字。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 6 · 02（频谱图与梅尔特征）、Phase 5 · 22（嵌入模型）
**时长：** 约 45 分钟

## 问题背景

用户说了一个口令词。你想知道：这个人是他们声称的那个人吗（*验证（verification）*，1:1），还是你的注册库中的某个人（*识别（identification）*，1:N）？或者两者都不是——这是一个未知说话人（*开放集（open-set）*）？

2018 年前：GMM-UBM + i-vector。EER 尚可，但对信道偏移（手机 vs 电脑）和情感脆弱。2018-2022 年：x-vector（用角度间隔训练的 TDNN 主干）。2022 年后：ECAPA-TDNN 和 WavLM-large 嵌入。到 2026 年，该领域由三个模型和一个指标主导。

该指标是 **EER**——等错误率（Equal Error Rate）。设置你的决策阈值，使错误接受率（False Accept Rate）= 错误拒绝率（False Reject Rate）。交叉点就是 EER。用于所有论文、所有排行榜、所有采购对话。

## 核心概念

![注册 + 验证流水线：嵌入 + 余弦 + EER](../assets/speaker-verification.svg)

**流水线。** 注册：录制目标说话人 5-30 秒的音频；计算固定维度嵌入（ECAPA-TDNN 为 192 维，WavLM-large 为 256 维）。验证：获取测试话语的嵌入；计算余弦相似度；与阈值比较。

**ECAPA-TDNN（2020 年，2026 年仍占主导）。** Emphasized Channel Attention, Propagation and Aggregation - Time-Delay Neural Network。带挤压激励的 1D 卷积块、多头注意力池化，随后是一个到 192 维的线性层。在 VoxCeleb 1+2（2700 个说话人，110 万个话语）上用加性角度间隔损失（AAM-softmax）训练。

**WavLM-SV（2022 年后）。** 用 AAM 损失微调预训练的 WavLM-large SSL 主干。质量更高但更慢——300+ MB vs 15 MB。

**x-vector（基准线）。** TDNN + 统计量池化。经典；仍适用于 CPU/边缘场景。

**AAM-softmax。** 在角度空间中加入间隔 `m` 的标准 softmax：对正确类别使用 `cos(θ + m)`。强制类间角度分离。典型 `m=0.2`，缩放 `s=30`。

### 评分

- **余弦相似度。** 注册嵌入与测试嵌入之间的余弦相似度，基于阈值做决策。
- **PLDA（概率 LDA，Probabilistic LDA）。** 将嵌入投影到潜在空间，同说话人 vs 不同说话人有闭合形式的似然比。叠加在余弦上可降低 EER 10-20%。2020 年前的标准；现在仅用于封闭集设置。
- **分数归一化（Score normalization）。** `S-norm` 或 `AS-norm`：针对一组冒充者的均值和标准差归一化每个分数。跨领域评估的必备操作。

### 2026 年应了解的数字

| 模型 | VoxCeleb1-O EER | 参数量 | 吞吐量（A100） |
|------|----------------|--------|-------------|
| x-vector（经典） | 3.10% | 500万 | 400× 实时 |
| ECAPA-TDNN | 0.87% | 1500万 | 200× 实时 |
| WavLM-SV large | 0.42% | 3.16亿 | 20× 实时 |
| Pyannote 3.1 分割 + 嵌入 | 0.65% | 600万 | 100× 实时 |
| ReDimNet（2024） | 0.39% | 2400万 | 100× 实时 |

### 说话人分离（Diarization）

多说话人音频中"谁说了什么"。流水线：VAD → 分段 → 对每段做嵌入 → 聚类（聚合层次或谱聚类）→ 平滑边界。现代技术栈：`pyannote.audio` 3.1，它将说话人分割 + 嵌入 + 聚类封装在一次调用中。2026 年在 AMI 上的 SOTA DER 约 15%（2022 年为 23%）。

## 动手实现

### 步骤一：MFCC 统计量的简单嵌入

```python
def embed_mfcc_stats(signal, sr):
    frames = featurize_mfcc(signal, sr, n_mfcc=13)
    mean = [sum(f[i] for f in frames) / len(frames) for i in range(13)]
    std = [
        math.sqrt(sum((f[i] - mean[i]) ** 2 for f in frames) / len(frames))
        for i in range(13)
    ]
    return mean + std  # 26维
```

远不是 SOTA——仅供教学。`code/main.py` 将其用作合成说话人数据上的概念验证。

### 步骤二：余弦相似度 + 阈值

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0

def verify(enroll, test, threshold=0.75):
    return cosine(enroll, test) >= threshold
```

### 步骤三：从相似度对计算 EER

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 1.0, 0.0)  # (fa, fr, threshold)
    for t in thresholds:
        fr = sum(1 for s in same_scores if s < t) / len(same_scores)
        fa = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        if abs(fa - fr) < abs(best[0] - best[1]):
            best = (fa, fr, t)
    return (best[0] + best[1]) / 2, best[2]
```

返回 (eer, 对应阈值)。两者都要报告。

### 步骤四：使用 SpeechBrain 的生产方案

```python
from speechbrain.pretrained import EncoderClassifier

clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# 注册：对 3-5 个干净样本的嵌入取平均
enroll = torch.stack([clf.encode_batch(load(x)) for x in enrollment_clips]).mean(0)
# 验证
score = clf.similarity(enroll, clf.encode_batch(load("test.wav"))).item()
verdict = score > 0.25   # ECAPA 典型阈值；在你的数据上调整
```

### 步骤五：使用 pyannote 进行说话人分离

```python
from pyannote.audio import Pipeline

pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
diarization = pipe("meeting.wav", num_speakers=None)
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f}–{turn.end:.1f}  {speaker}")
```

## 生产使用

2026 年技术栈：

| 场景 | 选择 |
|------|------|
| 封闭集 1:1 验证，边缘 | ECAPA-TDNN + 余弦阈值 |
| 开放集验证，云端 | WavLM-SV + AS-norm |
| 说话人分离（会议、播客） | `pyannote/speaker-diarization-3.1` |
| 反欺骗（重放/深度伪造检测） | AASIST 或 RawNet2 |
| 微型嵌入（关键词唤醒 + 注册） | Titanet-Small（NeMo） |

## 常见坑

- **信道不匹配。** 在 VoxCeleb（网络视频）上训练的模型 ≠ 电话音频。始终在目标信道上评估。
- **短话语。** 测试音频低于 3 秒时 EER 急剧下降。
- **有噪声的注册。** 一个嘈杂的注册样本会污染锚点。使用 ≥3 个干净样本并取平均。
- **跨条件使用固定阈值。** 始终在来自目标领域的保留开发集上调整阈值。
- **未归一化嵌入上的余弦。** 先做 L2 归一化；否则幅度会主导结果。

## 上手实践

保存为 `outputs/skill-speaker-verifier.md`。选择模型、注册协议、阈值调整计划和防欺诈保障。

## 练习

1. **简单。** 运行 `code/main.py`。构建合成"说话人"（不同音调特征），注册，在 100 对试验列表上计算 EER。
2. **中等。** 在 30 个 VoxCeleb1 话语上（5 个说话人 × 6 个话语）使用 SpeechBrain ECAPA。比较余弦 vs PLDA 的 EER。
3. **困难。** 用 `pyannote.audio` 构建完整的注册 → 说话人分离 → 验证流水线。在 AMI 开发集上评估 DER。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| EER | 核心指标 | 错误接受率 = 错误拒绝率时的阈值。 |
| 验证（Verification） | 1:1 | "这是 Alice 吗？" |
| 识别（Identification） | 1:N | "谁在说话？" |
| 开放集（Open-set） | 可能未知 | 测试集可能包含未注册的说话人。 |
| 注册（Enrollment） | 注册 | 计算说话人的参考嵌入。 |
| AAM-softmax | 损失函数 | 带加性角度间隔的 softmax；强制簇分离。 |
| PLDA | 经典评分 | 概率 LDA；在嵌入之上做似然比评分。 |
| DER | 说话人分离指标 | 说话人分离错误率——漏检 + 虚警 + 混淆。 |

## 延伸阅读

- [Snyder et al. (2018). X-Vectors: Robust DNN Embeddings for Speaker Recognition](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf) — 经典深度嵌入论文
- [Desplanques et al. (2020). ECAPA-TDNN](https://arxiv.org/abs/2005.07143) — 2020-2026 主导架构
- [Chen et al. (2022). WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900) — SV 和说话人分离的 SSL 主干
- [Bredin et al. (2023). pyannote.audio 3.1](https://github.com/pyannote/pyannote-audio) — 生产说话人分离 + 嵌入技术栈
- [VoxCeleb leaderboard (updated 2026)](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) — 跨模型的 EER 当前排名
