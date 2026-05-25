# 音频生成

> 音频是 16-48 kHz 的一维信号。五秒的片段有 80-240k 个采样点。没有 Transformer 能直接处理这么长的序列。2026 年每个生产级音频模型的解决方案都相同：神经编解码器（Encodec、SoundStream、DAC）以 50-75 Hz 将音频压缩为离散 token，Transformer 或扩散模型生成这些 token。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 6 · 02（音频特征）、Phase 6 · 04（ASR）、Phase 8 · 06（DDPM）
**时长：** 约 45 分钟

## 问题背景

三种音频生成任务：

1. **文本到语音（TTS）。** 给定文本，生成语音。干净的语音是窄带的，具有强烈的音素结构——Transformer 对 token 的处理很好地解决了这一问题。VALL-E（微软）、NaturalSpeech 3、ElevenLabs、OpenAI TTS。
2. **音乐生成。** 给定提示词（文本、旋律、和弦进行、流派），生成音乐。分布更广。MusicGen（Meta）、Stable Audio 2.5、Suno v4、Udio、Riffusion。
3. **音效/声音设计。** 给定提示词，生成环境音或拟音。AudioGen、AudioLDM 2、Stable Audio Open。

这三种都运行在相同的基础上：神经音频编解码器 + token 自回归或扩散生成器。

## 核心概念

![音频生成：编解码器 token + Transformer 或扩散](../assets/audio-generation.svg)

### 神经音频编解码器

Encodec（Meta，2022）、SoundStream（谷歌，2021）、Descript Audio Codec（DAC，2023）。卷积编码器将波形压缩为每时间步向量；残差向量量化（RVQ）将每个向量转换为 K 个码本索引的级联。解码器逆转这一过程。使用 8 个 RVQ 码本在 75 Hz 下以 2kbps 编码 24kHz 音频 = 每秒 600 个 token。

```
波形（16000 采样/秒）
    └─ 编码器卷积 ─┐
                   ├─ RVQ 层 1 → 75 Hz 的索引
                   ├─ RVQ 层 2 → 75 Hz 的索引
                   ├─ ...
                   └─ RVQ 层 8
```

### 其上的两种生成范式

**Token 自回归。** 将 RVQ token 展平为序列，运行仅解码器 Transformer。MusicGen 使用"延迟并行"以每流偏移量并行输出 K 个码本流。VALL-E 从文本提示词 + 3 秒语音样本生成语音 token。

**潜在扩散。** 将编解码器 token 打包为连续潜在表示，或用类别扩散建模。Stable Audio 2.5 在连续音频潜在表示上使用流匹配。AudioLDM 2 使用文本到梅尔频谱图到音频扩散。

2024-2026 年趋势：流匹配在音乐方面获胜（更快推理、更干净的样本），而 token 自回归在语音方面仍占主导，因为它自然是因果的且流式传输效果好。

## 生产格局

| 系统 | 任务 | 骨干 | 延迟 |
|------|------|------|------|
| ElevenLabs V3 | TTS | Token-AR + 神经声码器 | 首个 token ~300ms |
| OpenAI GPT-4o 音频 | 全双工语音 | 端到端多模态 AR | ~200ms |
| NaturalSpeech 3 | TTS | 潜在流匹配 | 非流式 |
| Stable Audio 2.5 | 音乐/音效 | DiT + 音频潜在流匹配 | 1 分钟片段约 10 秒 |
| Suno v4 | 完整歌曲 | 未披露；疑似 token-AR | 每首歌约 30 秒 |
| Udio v1.5 | 完整歌曲 | 未披露 | 每首歌约 30 秒 |
| MusicGen 3.3B | 音乐 | Encodec 32kHz 上的 Token-AR | 实时 |
| AudioCraft 2 | 音乐 + 音效 | 流匹配 | 5 秒片段约 5 秒 |
| Riffusion v2 | 音乐 | 频谱图扩散 | ~10 秒 |

## 动手实现

`code/main.py` 模拟了核心思想：在两种不同"风格"的合成"音频 token"序列上训练一个小型下一个 token Transformer（风格 A 是交替的低高 token，风格 B 是单调递增序列）。以风格为条件进行采样。

### 步骤一：合成音频 token

```python
def make_tokens(style, length, vocab_size, rng):
    if style == 0:  # "语音类"：交替
        return [i % vocab_size for i in range(length)]
    # "音乐类"：递增
    return [(i * 3) % vocab_size for i in range(length)]
```

### 步骤二：训练小型 token 预测器

以风格为条件的二元组风格预测器。重点是模式：编解码器 token → 交叉熵训练 → 自回归采样。

### 步骤三：条件采样

给定风格 token 和起始 token，从预测分布中采样下一个 token。继续 20-40 个 token。

## 常见陷阱

- **编解码器质量限制输出质量。** 如果编解码器无法忠实表示某种声音，再强的生成器也无济于事。DAC 是目前最好的开放选项。
- **RVQ 误差累积。** 每个 RVQ 层建模前一层的残差。第 1 层的误差会传播。在较高层使用温度为 0 的采样有帮助。
- **音乐结构。** 75 Hz 下 30 秒的 token 超过 20k 个。Transformer 很难处理。MusicGen 使用滑动窗口 + 提示词续写；Stable Audio 使用较短片段 + 交叉淡化。
- **边界处的伪影。** 生成片段之间的交叉淡化需要仔细的重叠相加。
- **对干净数据的渴望。** 音乐生成器需要数万小时的授权音乐。Suno/Udio 的 RIAA 诉讼（2024 年）将这一问题推到了台面上。
- **声音克隆伦理。** 3 秒样本加文本提示词足以让 VALL-E / XTTS / ElevenLabs 克隆一个声音。每个生产模型都需要滥用检测 + 退出名单。

## 生产使用

| 任务 | 2026 年技术栈 |
|------|-------------|
| 商业 TTS | ElevenLabs、OpenAI TTS 或 Azure Neural |
| 声音克隆（经同意验证） | XTTS v2（开放）或 ElevenLabs Pro |
| 背景音乐，快速 | Stable Audio 2.5 API、Suno 或 Udio |
| 带歌词的音乐 | Suno v4 或 Udio v1.5 |
| 音效/拟音 | AudioCraft 2、ElevenLabs SFX 或 Stable Audio Open |
| 实时语音代理 | GPT-4o realtime 或 Gemini Live |
| 开放权重音乐研究 | MusicGen 3.3B、Stable Audio Open 1.0、AudioLDM 2 |
| 配音/翻译 | HeyGen、ElevenLabs Dubbing |

## 上手实践

保存 `outputs/skill-audio-brief.md`。该技能接受音频简报（任务、时长、风格、声音、许可证），输出：模型 + 托管、提示词格式（流派标签、风格描述词、结构标记）、编解码器 + 生成器 + 声码器链、种子协议，以及评估计划（TTS 的 MOS / CLAP 分数 / CER / 用户 A/B 测试）。

## 练习

1. **简单。** 运行 `code/main.py` 并明确设置风格。验证生成的序列是否匹配风格的模式。
2. **中等。** 添加延迟并行解码：模拟 2 个 token 流，必须保持 1 步的偏移。训练一个联合预测器。
3. **困难。** 使用 HuggingFace transformers 在本地运行 MusicGen-small。用三个不同提示词生成 10 秒片段；对风格遵循度进行 A/B 测试。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 编解码器（Codec） | "神经压缩" | 音频的编码器/解码器；典型输出是 50-75 Hz token。 |
| RVQ | "残差向量量化" | K 个量化器的级联；每个建模前一个的残差。 |
| Token | "一个编解码器符号" | 码本中的离散索引；通常是 1024 或 2048。 |
| 延迟并行（Delayed parallel） | "偏移码本" | 以交错偏移量输出 K 个 token 流以减少序列长度。 |
| 流匹配（Flow matching） | "2024 年音频领域的胜者" | 更直线路径的扩散替代；采样更快。 |
| 声音提示（Voice prompt） | "3 秒样本" | 引导克隆声音的说话人嵌入或 token 前缀。 |
| 梅尔频谱图（Mel spectrogram） | "那个可视化" | 对数幅度感知频谱图；许多 TTS 系统使用。 |
| 声码器（Vocoder） | "梅尔到波形" | 将梅尔频谱图转换回音频的神经组件。 |

## 生产注意：音频是流式问题

音频是用户期望*在生成时*到达的输出模态，而非一次性全部到达。在生产术语中，这意味着 TPOT（每输出 token 时间）至关重要，因为用户的收听速度是目标吞吐量——而非阅读速度。对于在约 75 token/秒（Encodec）下进行 token 化的 16kHz 音频，服务器必须每个用户生成 ≥75 token/秒以保持播放流畅。

两个架构后果：

- **流匹配音频模型无法简单流式传输。** Stable Audio 2.5 和 AudioCraft 2 在一次传播中渲染固定长度的片段。要进行流式传输，需要分块并重叠边界——类似于滑动窗口扩散——与编解码器 AR 模型相比增加了 100-300ms 的延迟开销。

如果产品是"实时语音聊天"或"实时音乐续写"，选择编解码器 AR 路径。如果是"提交后渲染 30 秒片段"，流匹配在质量和总延迟上获胜。

## 延伸阅读

- [Défossez et al. (2022). Encodec: High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — 编解码器标准
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312) — 第一个广泛使用的神经音频编解码器
- [Kumar et al. (2023). High-Fidelity Audio Compression with Improved RVQGAN (DAC)](https://arxiv.org/abs/2306.06546) — DAC
- [Wang et al. (2023). Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (VALL-E)](https://arxiv.org/abs/2301.02111) — VALL-E
- [Copet et al. (2023). Simple and Controllable Music Generation (MusicGen)](https://arxiv.org/abs/2306.05284) — MusicGen
- [Liu et al. (2023). AudioLDM 2: Learning Holistic Audio Generation with Self-supervised Pretraining](https://arxiv.org/abs/2308.05734) — AudioLDM 2
- [Stability AI (2024). Stable Audio 2.5](https://stability.ai/news/introducing-stable-audio-2-5) — 带流匹配的 2025 文本到音乐
