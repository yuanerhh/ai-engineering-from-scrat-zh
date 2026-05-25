# Whisper——架构与微调

> Whisper 是一个 30 秒窗口的 Transformer 编码器-解码器，在 68 万小时多语言弱监督音频-文本对上训练。一种架构，多种任务，在 99 种语言上鲁棒。2026 年的参考 ASR。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 6 · 04（ASR）、Phase 5 · 10（注意力机制）、Phase 7 · 05（完整 Transformer）
**时长：** 约 75 分钟

## 问题背景

OpenAI 于 2022 年 9 月发布的 Whisper 是第一个以商品形式发布的 ASR 模型：粘贴音频，获取文本，99 种语言，鲁棒对抗噪声，在笔记本上运行。到 2024 年，OpenAI 发布了 Large-v3 和 Turbo 变体；到 2026 年，Whisper 已成为从播客转录到语音助手再到 YouTube 字幕的一切的默认基准。

但 Whisper 不是一个你可以永远当黑盒对待的流水线。领域偏移会摧毁它——技术术语、说话人口音、专有名词、短片段、静音。你需要了解：

1. 它内部究竟是什么。
2. 如何正确地给它提供分块、流式或长音频。
3. 什么时候需要微调，以及如何微调。

## 核心概念

![Whisper 编码器-解码器、任务、分块推理、微调](../assets/whisper.svg)

**架构。** 标准 Transformer 编码器-解码器。

- 输入：30 秒对数梅尔频谱图，80 个梅尔分箱，10ms 步幅 → 3000 帧。更短的音频零填充，更长的音频分块。
- 编码器：卷积下采样（步幅 2）+ `N` 个 Transformer 块。Large-v3：32 层，1280 维，20 个头。
- 解码器：`N` 个 Transformer 块，带因果自注意力 + 对编码器输出的交叉注意力。大小与编码器相同。
- 输出：51,865 个 token 词汇表上的 BPE token。

Large-v3 有 15.5 亿参数。Turbo 使用 4 层解码器（原为 32 层），WER 损失 <1% 的情况下延迟降低 8 倍。

**提示格式。** Whisper 是一个多任务模型，通过解码器提示中的特殊 token 引导：

```
<|startoftranscript|><|en|><|transcribe|><|notimestamps|> Hello world.<|endoftext|>
```

- `<|en|>` — 语言标签；强制翻译 vs 转录行为。
- `<|transcribe|>` 或 `<|translate|>` — 将任意语言输入翻译为英语，或逐字转录。
- `<|notimestamps|>` — 跳过词级时间戳（更快）。

提示就是让一个模型完成多种任务的机制。将 `<|en|>` 改为 `<|fr|>`，它就会转录法语。

**30 秒窗口。** 所有内容都固定在 30 秒。更长的音频需要分块；更短的音频需要填充。窗口本身不支持原生流式处理——这就是为什么 WhisperX、Whisper-Streaming 和 faster-whisper 存在。

**对数梅尔归一化。** `(log_mel - mean) / std`，其中统计数据来自 Whisper 自己的训练语料库。你*必须*使用 Whisper 的预处理（`whisper.audio.log_mel_spectrogram`），而不是 `librosa.feature.melspectrogram`。

### 2026 年的变体

| 变体 | 参数 | 延迟（A100） | WER（LibriSpeech-clean） |
|------|------|------------|------------------------|
| Tiny | 3900万 | 1× 实时 | 5.4% |
| Base | 7400万 | 1× | 4.1% |
| Small | 2.44亿 | 1× | 3.0% |
| Medium | 7.69亿 | 1× | 2.7% |
| Large-v3 | 15.5亿 | 2× | 1.8% |
| Large-v3-turbo | 8.09亿 | 8× | 1.58% |
| Whisper-Streaming（2024） | 15.5亿 | 流式 | 2.0% |

### 微调

2026 年规范工作流：

1. 收集 10-100 小时带对齐转录的目标领域音频。
2. 使用带 `generate_with_loss` 回调的 `transformers.Seq2SeqTrainer`。
3. 参数高效：在注意力层的 `q_proj`、`k_proj`、`v_proj` 上使用 LoRA，GPU 内存减少 4 倍，WER 代价 <0.3%。
4. 数据少于 10 小时时冻结编码器，只调整解码器。
5. 使用 Whisper 自己的分词器和提示格式；永远不要更换分词器。

社区结果：在 20 小时医疗听写上微调 Medium，将医疗词汇的 WER 从 12% 降至 4.5%。在 4 小时冰岛语上微调 Turbo，将 WER 从 18% 降至 6%。

## 动手实现

### 步骤一：开箱即用运行 Whisper

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe(
    "clip.wav",
    language="en",
    task="transcribe",
    temperature=0.0,
    condition_on_previous_text=False,  # 防止失控重复
)
print(result["text"])
for seg in result["segments"]:
    print(f"[{seg['start']:.2f}–{seg['end']:.2f}] {seg['text']}")
```

你应该始终覆盖的关键默认值：`temperature=0.0`（采样默认为 0.0 → 0.2 → 0.4 … 回退链），`condition_on_previous_text=False`（防止级联幻觉问题），和 `no_speech_threshold=0.6`（静音检测）。

### 步骤二：分块长音频

```python
# whisperx 是 2026 年带词级时间戳长音频的参考实现
import whisperx
model = whisperx.load_model("large-v3-turbo", device="cuda", compute_type="float16")
segments = model.transcribe("1hour.mp3", batch_size=16, chunk_size=30)
```

WhisperX 添加了：(1) Silero VAD 门控，(2) 通过 wav2vec 2.0 做词级对齐，(3) 通过 `pyannote.audio` 做说话人分离。2026 年生产转录的主力工具。

### 步骤三：使用 LoRA 微调

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo")
lora = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1, bias="none", task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(model, lora)
# model.print_trainable_parameters()  -> 约 300 万可训练参数 / 8.09 亿总参数
```

然后是标准 Trainer 循环。每 1000 步保存检查点。在保留集上用 WER 评估。

### 步骤四：检查每层学到了什么

```python
# 解码时抓取交叉注意力权重，观察解码器关注什么
with torch.inference_mode():
    out = model.generate(
        input_features=features,
        return_dict_in_generate=True,
        output_attentions=True,
    )
# out.cross_attentions: layer × head × step × src_len
```

用热力图可视化——你会看到对角线对齐，解码器步骤扫描编码器帧。这个对角线就是 Whisper 的词级时间戳概念。

## 生产使用

2026 年技术栈：

| 场景 | 选择 |
|------|------|
| 通用英语，离线 | 通过 `whisperx` 使用 Large-v3-turbo |
| 移动端/边缘 | Whisper-Tiny 量化（int8）或 Moonshine |
| 多语言长音频 | 通过 `whisperx` 使用 Large-v3 + 说话人分离 |
| 低资源语言 | 用 LoRA 微调 Medium 或 Turbo |
| 流式（2 秒延迟） | Whisper-Streaming 或 Parakeet-TDT |
| 词级时间戳 | WhisperX（通过 wav2vec 2.0 强制对齐） |

`faster-whisper`（CTranslate2 后端）是 2026 年最快的 CPU+GPU 推理运行时——比原版快 4 倍，输出完全相同。

## 2026 年仍会出现的坑

- **静音上的幻觉文本。** Whisper 在字幕上训练，包含"Thanks for watching!"、"Subscribe!"、歌词。调用前始终用 VAD 门控。
- **`condition_on_previous_text` 级联。** 一个幻觉会污染后续窗口。除非需要跨块流畅性，否则设为 `False`。
- **短片段填充。** 2 秒的音频填充到 30 秒，可能在尾部静音中产生幻觉。使用 `pad=False` 或 VAD 门控。
- **错误的梅尔统计。** 使用 librosa 的梅尔代替 Whisper 的，会产生接近随机的输出。使用 `whisper.audio.log_mel_spectrogram`。

## 上手实践

保存为 `outputs/skill-whisper-tuner.md`。为给定领域设计 Whisper 微调或推理流水线。

## 练习

1. **简单。** 运行 `code/main.py`。它对 Whisper 风格的提示进行分词，计算解码形状预算，并打印 10 分钟音频的分块计划。
2. **中等。** 安装 `faster-whisper`，转录一个 10 分钟的播客，与人工转录相比较 WER。尝试 `language="auto"` vs 强制 `language="en"`。
3. **困难。** 使用 HuggingFace `datasets`，选一种 Whisper 表现欠佳的语言（如乌尔都语），用 LoRA 在 2 小时数据上微调 Medium 2 个 epoch，报告 WER 改善。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 30 秒窗口 | Whisper 的限制 | 硬性输入上限；对更长音频分块。 |
| SOT | 转录开始标记 | `<|startoftranscript|>` 启动解码器提示。 |
| 时间戳 token | 时间对齐 | 51k 词汇表中的每 0.02 秒偏移都是一个特殊 token。 |
| Turbo | 快速变体 | 4 层解码器，速度 8 倍，WER 退化 <1%。 |
| WhisperX | 长音频封装器 | VAD + Whisper + wav2vec 对齐 + 说话人分离。 |
| LoRA 微调 | 高效微调 | 在注意力层添加低秩适配器；训练约 0.3% 的参数。 |
| 幻觉 | 静默失败 | Whisper 从噪声/静音中产生流畅的英文。 |

## 延伸阅读

- [Radford et al. (2022). Whisper paper](https://arxiv.org/abs/2212.04356) — 原始架构和训练方法
- [OpenAI (2024). Whisper Large-v3-turbo release](https://github.com/openai/whisper/discussions/2363) — 4 层解码器，8 倍加速
- [Bain et al. (2023). WhisperX](https://arxiv.org/abs/2303.00747) — 长音频、词级对齐、说话人分离
- [Systran — faster-whisper repo](https://github.com/SYSTRAN/faster-whisper) — CTranslate2 后端，速度 4 倍
- [HuggingFace — Whisper fine-tune tutorial](https://huggingface.co/blog/fine-tune-whisper) — 规范的 LoRA / 全量微调教程
