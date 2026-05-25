# 声音克隆与声音转换

> 声音克隆用他人的声音朗读你的文本。声音转换将你的声音重写成他人的声音，同时保留你说的内容。两者的核心原语相同：将说话人身份与内容分离。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 6 · 06（说话人识别与验证）、Phase 6 · 07（文本转语音）
**时长：** 约 75 分钟

## 问题背景

2026 年，5 秒音频就足以用消费级 GPU 高质量克隆任何人的声音。ElevenLabs、F5-TTS、OpenVoice v2、VoiceBox 都支持零样本或少样本克隆。这项技术既是福音（无障碍 TTS、配音、辅助声音），也是武器（诈骗电话、政治深度伪造、知识产权盗窃）。

两项紧密相关的任务：

- **声音克隆（TTS 侧）：** 文本 + 5 秒参考声音 → 该声音的音频。
- **声音转换（语音侧）：** 说话人 A 说 X 的源音频 + 说话人 B 的参考声音 → B 说 X 的音频。

两者都将波形分解为（内容、说话人、韵律），然后将一个来源的内容与另一个来源的说话人重新组合。

2026 年的关键法律约束：**欧盟《人工智能法案》（2026 年 8 月起强制执行）和加利福尼亚州 AB 2905 法案（2025 年生效）要求水印和同意验证**。你的流水线必须嵌入不可听水印，并拒绝未经同意的克隆。

## 核心概念

![声音克隆 vs 声音转换：分解、替换说话人、重组](../assets/voice-cloning.svg)

**零样本克隆（Zero-shot cloning）。** 将 5 秒音频片段传入在数千个说话人上训练的模型。说话人编码器将片段映射为说话人嵌入；TTS 解码器以该嵌入加文本为条件生成音频。

使用者：F5-TTS（2024）、YourTTS（2022）、XTTS v2（2024）、OpenVoice v2（2024）。

**少样本微调（Few-shot fine-tuning）。** 录制 5-30 分钟目标声音。LoRA 微调基础模型约一小时。质量从"还可以"跃升到"难以区分"。Coqui 和 ElevenLabs 都支持此模式；社区将其与 F5-TTS 结合使用。

**声音转换（Voice conversion，VC）。** 两类方法：

- **识别-合成（Recognition-synthesis）。** 运行类 ASR 模型提取内容表示（如软音素后验概率 PPG），然后以目标说话人嵌入重新合成。对语言和口音具有鲁棒性。KNN-VC（2023）、Diff-HierVC（2023）采用此方法。
- **解耦（Disentanglement）。** 训练一个自编码器，在瓶颈层的潜在空间中分离内容、说话人和韵律。推理时替换说话人嵌入。质量较低但速度更快。AutoVC（2019）、VITS-VC 变体采用此方法。

**基于神经编解码器的克隆（2024 年后）。** VALL-E、VALL-E 2、NaturalSpeech 3、VoiceBox——将音频视为来自 SoundStream / EnCodec 的离散 token，在 codec token 上训练大型自回归或流匹配模型。短提示下质量与 ElevenLabs 相当。

### 伦理，不是附加功能

**水印（Watermarking）。** PerTh 和 SilentCipher（2024）将约 16-32 位 ID 不可感知地嵌入音频。能在重新编码、流式传输和常见编辑后存活。生产就绪的开源方案。

**同意验证（Consent gates）。** 必须将每个克隆输出与可验证的同意记录配对。"我，Rohit，于 2026-04-22，授权将此声音用于 X 用途。"存储在防篡改日志中。

**检测（Detection）。** AASIST、RawNet2 和 Wav2Vec2-AASIST 作为检测器发布。ASVspoof 2025 挑战赛公布了最先进检测器针对 ElevenLabs、VALL-E 2 和 Bark 输出的 EER 为 0.8-2.3%。

### 2026 年数字

| 模型 | 零样本？ | SECS（目标相似度） | WER（可懂度） | 参数量 |
|------|---------|-----------------|------------|------|
| F5-TTS | 是 | 0.72 | 2.1% | 3.35 亿 |
| XTTS v2 | 是 | 0.65 | 3.5% | 4.70 亿 |
| OpenVoice v2 | 是 | 0.70 | 2.8% | 2.20 亿 |
| VALL-E 2 | 是 | 0.77 | 2.4% | 3.70 亿 |
| VoiceBox | 是 | 0.78 | 2.1% | 3.30 亿 |

SECS > 0.70 通常让大多数听众无法与目标声音区分。

## 动手实现

### 步骤一：识别-合成分解（main.py 中的代码示例）

```python
def clone_pipeline(ref_audio, text, target_embedder, tts_model):
    speaker_emb = target_embedder.encode(ref_audio)
    mel = tts_model(text, speaker=speaker_emb)
    return vocoder(mel)
```

概念上很简单；实现复杂度集中在 `tts_model` 和说话人编码器中。

### 步骤二：使用 F5-TTS 进行零样本克隆

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="rohit_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please add milk and bread to my list.",
)
```

参考文字稿必须与参考音频完全匹配，包括标点；不匹配会破坏对齐。

### 步骤三：使用 KNN-VC 进行声音转换

```python
import torch
from knnvc import KNNVC  # 2023 模型，https://github.com/bshall/knn-vc
vc = KNNVC.load("wavlm-base-plus")
out_wav = vc.convert(source="my_voice.wav", target_pool=["alice_1.wav", "alice_2.wav"])
```

KNN-VC 运行 WavLM 提取源和目标池的逐帧嵌入，然后用池中最近邻替换每个源帧。非参数化，一分钟目标语音即可使用。

### 步骤四：嵌入水印

```python
from silentcipher import SilentCipher
sc = SilentCipher(model="2024-06-01")
payload = b"consent_id:abc123;ts:1745353200"
watermarked = sc.embed(wav, sr=24000, message=payload)
detected = sc.detect(watermarked, sr=24000)   # 返回 payload 字节
```

约 32 位载荷，MP3 重新编码和轻微噪声后仍可检测。

### 步骤五：同意验证

```python
def cloned_inference(text, ref_audio, consent_record):
    assert verify_signature(consent_record), "需要签名同意"
    assert consent_record["speaker_id"] == hash_speaker(ref_audio)
    wav = tts.infer(ref_file=ref_audio, gen_text=text)
    wav = watermark(wav, payload=consent_record["id"])
    return wav
```

## 生产使用

2026 年技术栈：

| 场景 | 选择 |
|------|------|
| 5 秒零样本克隆，开源 | F5-TTS 或 OpenVoice v2 |
| 商业生产克隆 | ElevenLabs Instant Voice Clone v2.5 |
| 声音转换（重写） | KNN-VC 或 Diff-HierVC |
| 多说话人微调 | StyleTTS 2 + 说话人适配器 |
| 跨语言克隆 | XTTS v2 或 VALL-E X |
| 深度伪造检测 | Wav2Vec2-AASIST |

## 常见坑

- **参考文字稿不匹配。** F5-TTS 等模型要求参考文字稿与参考音频完全匹配，包括标点。
- **有混响的参考音频。** 回声会破坏克隆效果。录音要干燥、近距离麦克风。
- **情感不匹配。** 训练参考音频"欢快"会导致所有克隆都显得欢快。匹配参考情感与目标用途。
- **语言泄漏。** 克隆英语说话人再要求模型说法语，口音往往仍会保留；使用跨语言模型（XTTS、VALL-E X）。
- **无水印。** 从 2026 年 8 月起在欧盟不得合法部署。

## 上手实践

保存为 `outputs/skill-voice-cloner.md`。设计一个包含同意验证 + 水印 + 质量目标的克隆或转换流水线。

## 练习

1. **简单。** 运行 `code/main.py`。通过计算替换前后两个"说话人"的余弦相似度，演示说话人嵌入替换。
2. **中等。** 使用 OpenVoice v2 克隆自己的声音。测量参考和克隆之间的 SECS。通过 Whisper 测量 CER。
3. **困难。** 对 20 个克隆应用 SilentCipher 水印，通过 128 kbps MP3 编码+解码，检测载荷。报告比特准确率。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 零样本克隆（Zero-shot clone） | 5 秒足够 | 预训练模型 + 说话人嵌入；无需训练。 |
| PPG | 音素后验图 | 用作语言无关内容表示的逐帧 ASR 后验概率。 |
| KNN-VC | 最近邻转换 | 用最近的目标池帧替换每个源帧。 |
| 神经编解码器 TTS | VALL-E 风格 | 基于 EnCodec/SoundStream token 的 AR 模型。 |
| 水印（Watermark） | 不可听签名 | 嵌入音频中的比特，能在重新编码后存活。 |
| SECS | 克隆保真度 | 目标和克隆说话人嵌入的余弦相似度。 |
| AASIST | 深度伪造检测器 | 反欺骗模型；检测合成语音。 |

## 延伸阅读

- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — 开源 SOTA 零样本克隆
- [Baevski et al. / Microsoft (2023). VALL-E](https://arxiv.org/abs/2301.02111) 和 [VALL-E 2 (2024)](https://arxiv.org/abs/2406.05370) — 神经编解码器 TTS
- [Qian et al. (2019). AutoVC](https://arxiv.org/abs/1905.05879) — 基于解耦的声音转换
- [Baas, Waubert de Puiseau, Kamper (2023). KNN-VC](https://arxiv.org/abs/2305.18975) — 基于检索的声音转换
- [SilentCipher (2024) — 音频水印](https://github.com/sony/silentcipher) — 生产就绪的 32 位音频水印
- [ASVspoof 2025 results](https://www.asvspoof.org/) — 检测器与合成器的军备竞赛，2026 年更新
