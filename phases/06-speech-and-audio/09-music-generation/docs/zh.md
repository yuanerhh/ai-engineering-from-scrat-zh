# 音乐生成——MusicGen、Stable Audio、Suno 与版权地震

> 2026 年音乐生成：Suno v5 和 Udio v4 主导商业市场；MusicGen、Stable Audio Open 和 ACE-Step 领跑开源领域。技术问题基本已解决。法律问题（华纳音乐 5 亿美元和解、环球音乐和解）在 2025-2026 年重塑了整个领域。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 6 · 02（频谱图）、Phase 4 · 10（扩散模型）
**时长：** 约 75 分钟

## 问题背景

文本 → 30 秒到 4 分钟的音乐片段，包含歌词、人声和结构。三个子问题：

1. **纯器乐生成。** "lo-fi 嘻哈鼓点配温暖键盘" → 音频。MusicGen、Stable Audio、AudioLDM。
2. **歌曲生成（含人声 + 歌词）。** "关于德克萨斯雨夜的乡村歌曲" → 完整歌曲。Suno、Udio、YuE、ACE-Step。
3. **条件/可控生成。** 扩展现有片段、重新生成过渡段、切换流派、分轨提取或修复特定部分。Udio 的修复（inpainting）+ 分轨分离是 2026 年的标杆功能。

## 核心概念

![音乐生成：token 语言模型 vs 扩散，2026 年模型全景](../assets/music-generation.svg)

### 基于神经编解码器 token 的语言模型

Meta 的 **MusicGen**（2023 年，MIT 许可）及其众多衍生模型：以文本/旋律嵌入为条件，自回归预测 EnCodec token（32 kHz，4 个码本），用 EnCodec 解码。3 亿至 33 亿参数。强基准线，但超过 30 秒后表现下降。

**ACE-Step**（开源，4B XL 版于 2026 年 4 月发布）扩展了此方法，支持完整歌曲歌词条件生成。是开源社区最接近 Suno 的方案。

### 基于梅尔频谱图或隐变量的扩散

**Stable Audio（2023）** 和 **Stable Audio Open（2024）**：压缩音频上的潜在扩散（latent diffusion）。擅长循环音效、声音设计和环境音效纹理。不擅长结构完整的完整歌曲。

**AudioLDM / AudioLDM2**：通过类 T2I 风格的潜在扩散实现文本到音频，泛化至音乐、音效和语音。

### 混合方法（商业产品）——Suno、Udio、Lyria

权重闭源。可能是 AR 编解码器语言模型 + 基于扩散的声码器，配备专用的人声/鼓/旋律头。Suno v5（2026）是 ELO 1293 的质量领先者。Udio v4 新增修复 + 分轨分离（低音、鼓点、人声可分别下载）。

### 评估

- **FAD（弗雷歇音频距离，Fréchet Audio Distance）。** 使用 VGGish 或 PANNs 特征计算生成与真实音频分布之间的嵌入级距离。越低越好。MusicGen small 在 MusicCaps 上 FAD 为 4.5；SOTA 约 3.0。
- **音乐性（主观）。** 人类偏好。Suno v5 ELO 1293 领先。
- **文本-音频对齐。** 提示与输出之间的 CLAP 分数。
- **音乐性伪影。** 节拍跳跃、人声短语漂移、超过 30 秒后结构丢失。

## 2026 年模型全景

| 模型 | 参数量 | 时长 | 人声 | 许可证 |
|------|--------|------|------|--------|
| MusicGen-large | 33 亿 | 30 秒 | 无 | MIT |
| Stable Audio Open | 12 亿 | 47 秒 | 无 | Stability 非商业 |
| ACE-Step XL（2026 年 4 月） | 40 亿 | > 2 分钟 | 有 | Apache-2.0 |
| YuE | 70 亿 | > 2 分钟 | 有，多语言 | Apache-2.0 |
| Suno v5（闭源） | ? | 4 分钟 | 有，ELO 1293 | 商业 |
| Udio v4（闭源） | ? | 4 分钟 | 有 + 分轨 | 商业 |
| Google Lyria 3（闭源） | ? | 实时 | 有 | 商业 |
| MiniMax Music 2.5 | ? | 4 分钟 | 有 | 商业 API |

## 法律格局（2025-2026）

- **华纳音乐 vs Suno 和解。** 5 亿美元。华纳现在对 Suno 上的 AI 形象、音乐版权和用户生成曲目拥有监督权。环球音乐与 Udio 达成类似和解。
- **欧盟《人工智能法案》** + **加利福尼亚州 SB 942**：AI 生成音乐必须披露。
- MIT 许可的 **Riffusion / MusicGen** 无合规负担，但也没有商业人声。

安全可发布的模式：

1. 仅生成纯器乐（MusicGen、Stable Audio Open，MIT/CC0 输出）。
2. 使用商业 API（Suno、Udio、ElevenLabs Music），获取按次生成许可。
3. 在自有或已授权曲目上训练（大多数企业最终选择此方案）。
4. 为生成内容添加水印 + 元数据标签。

## 动手实现

### 步骤一：使用 MusicGen 生成

```python
from audiocraft.models import MusicGen
import torchaudio

model = MusicGen.get_pretrained("facebook/musicgen-small")
model.set_generation_params(duration=10)
wav = model.generate(["upbeat synthwave with driving drums, 128 BPM"])
torchaudio.save("out.wav", wav[0].cpu(), 32000)
```

三种规模：`small`（3 亿，速度快）、`medium`（15 亿）、`large`（33 亿）。Small 足以验证想法。

### 步骤二：旋律条件生成

```python
melody, sr = torchaudio.load("humming.wav")
wav = model.generate_with_chroma(
    ["jazz piano cover"],
    melody.squeeze(),
    sr,
)
```

MusicGen-melody 使用色度图（chromagram）并在切换音色的同时保留旋律。适用于"将这段旋律转化为弦乐四重奏"。

### 步骤三：FAD 评估

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()

fad.get_fad_score("generated_folder/", "reference_folder/")
```

计算 VGGish 嵌入距离。适用于流派级别的回归测试；不能替代人类听众。

### 步骤四：加入 LLM-音乐工作流

结合第 7-8 课的思路：

```python
prompt = "Write a 30-second jazz loop. Describe the drums, bass, and piano voicing."
description = llm.complete(prompt)
music = musicgen.generate([description], duration=30)
```

## 生产使用

| 目标 | 技术栈 |
|------|--------|
| 器乐声音设计 | Stable Audio Open |
| 游戏/自适应音乐 | Google Lyria RealTime（闭源） |
| 含人声完整歌曲（商业） | Suno v5 或 Udio v4，附明确许可 |
| 含人声完整歌曲（开源） | ACE-Step XL 或 YuE |
| 短广告歌曲 | MusicGen 以哼唱参考为旋律条件 |
| 音乐视频背景 | MusicGen + Stable Video Diffusion |

## 2026 年仍存在的坑

- **版权洗白提示词。** "泰勒·斯威夫特风格的歌曲"——商业 Suno/Udio 现已过滤，开源模型不会。添加自己的过滤词列表。
- **超过 30 秒后的重复/漂移。** AR 模型会循环。将多次生成交叉渐变，或使用 ACE-Step 保证结构连贯。
- **节拍漂移。** 模型会偏离 BPM。在提示词中使用 BPM 标签，并用 librosa 的 `beat_track` 进行后处理过滤。
- **人声可懂度。** Suno 表现优秀；开源模型歌词往往含糊不清。如果歌词重要，使用商业 API 或微调。
- **单声道输出。** 开源模型生成单声道或假立体声。用适当的立体声重建工具升级（ezst、Cartesia 立体声扩散）。

## 上手实践

保存为 `outputs/skill-music-designer.md`。为音乐生成部署选择模型、许可策略、时长/结构方案和披露元数据。

## 练习

1. **简单。** 运行 `code/main.py`。它以 ASCII 符号生成"生成式"和弦进行 + 鼓点模式——一个音乐生成卡通。如需要可通过任意 MIDI 渲染器回放。
2. **中等。** 安装 `audiocraft`，使用 MusicGen-small 对 4 种流派提示词各生成 10 秒片段，相对参考流派集测量 FAD。
3. **困难。** 使用 ACE-Step（或 MusicGen-melody），对同一旋律生成三种不同音色提示的变体。计算与提示词的 CLAP 相似度以验证对齐效果。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| FAD | 音频 FID | 真实与生成音频嵌入分布之间的弗雷歇距离。 |
| 色度图（Chromagram） | 旋律作为音高 | 每帧 12 维向量；旋律条件生成的输入。 |
| 分轨（Stems） | 乐器轨道 | 分离的低音/鼓点/人声/旋律 WAV 文件。 |
| 修复（Inpainting） | 重新生成某段 | 遮盖时间窗口；模型只重新生成该部分。 |
| CLAP | 文本-音频 CLIP | 对比音频-文本嵌入；评估文本-音频对齐。 |
| EnCodec | 音乐编解码器 | Meta 的神经编解码器，MusicGen 使用；32 kHz，4 个码本。 |

## 延伸阅读

- [Copet et al. (2023). MusicGen](https://arxiv.org/abs/2306.05284) — 开源自回归基准
- [Evans et al. (2024). Stable Audio Open](https://arxiv.org/abs/2407.14358) — 声音设计默认方案
- [ACE-Step](https://github.com/ace-step/ACE-Step) — 开源 40 亿参数完整歌曲生成器，2026 年 4 月
- [Suno v5 platform docs](https://suno.com) — 商业质量领导者
- [AudioLDM2](https://arxiv.org/abs/2308.05734) — 音乐 + 音效的潜在扩散
- [WMG-Suno settlement coverage](https://www.musicbusinessworldwide.com/suno-warner-music-settlement/) — 2025 年 11 月判例
