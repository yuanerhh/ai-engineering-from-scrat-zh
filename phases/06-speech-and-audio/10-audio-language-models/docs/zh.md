# 音频语言模型——Qwen2.5-Omni、Audio Flamingo、GPT-4o Audio

> 2026 年的音频语言模型可对语音、环境声音和音乐进行推理。Qwen2.5-Omni-7B 在 MMAU-Pro 上与 GPT-4o Audio 持平。Audio Flamingo Next 在 LongAudioBench 上超越 Gemini 2.5 Pro。开源与闭源的差距基本消弭——多音频任务除外，在那里所有人都几乎等同于随机猜测。

**类型：** 学习
**语言：** Python
**前置条件：** Phase 6 · 04（ASR）、Phase 12 · 03（视觉语言模型）、Phase 7 · 10（音频 Transformer）
**时长：** 约 45 分钟

## 问题背景

给定 5 秒音频：狗叫声、有人喊"stop!"，然后是沉默。有用的问题跨越多个维度：

- **转录。** "说了什么？"——ASR 领域。
- **语义推理。** "这个人处于危险中吗？"——需要联合理解狗叫 + 喊声 + 沉默。
- **音乐推理。** "哪些乐器演奏旋律？"
- **长音频检索。** "在这 90 分钟的讲座中，讲师在哪里解释了梯度下降？"

能用一个提示词回答上述所有问题的单一模型称为**音频语言模型**（LALM / ALM）。与纯 ASR 不同：LALM 生成自由形式的自然语言答案，而不仅仅是文字稿。

## 核心概念

![音频语言模型：音频编码器 + 投影器 + LLM 解码器](../assets/alm-architecture.svg)

### 三组件模板

2026 年每个 LALM 都有相同的骨架：

1. **音频编码器（Audio encoder）。** Whisper 编码器、BEATs、CLAP、WavLM，或各模型自定义编码器。
2. **投影器（Projector）。** 将音频编码器特征桥接到 LLM token 嵌入空间的线性层或 MLP。
3. **LLM。** 基于 Llama / Qwen / Gemma 的解码器。接受交错的文本 + 音频 token；生成文本。

训练：

- **阶段一。** 冻结编码器 + LLM；仅在 ASR / 字幕数据上训练投影器。
- **阶段二。** 在指令跟随音频任务（问答、推理、音乐理解）上进行全量 / LoRA 微调。
- **阶段三（可选）。** 添加语音输入/输出，增加语音解码器。Qwen2.5-Omni 和 AF3-Chat 采用此方式。

### 2026 年模型全景

| 模型 | 骨干 | 音频编码器 | 输出模态 | 访问方式 |
|------|------|----------|---------|---------|
| Qwen2.5-Omni-7B | Qwen2.5-7B | 自定义 + Whisper | 文本 + 语音 | Apache-2.0 |
| Qwen3-Omni | Qwen3 | 自定义 | 文本 + 语音 | Apache-2.0 |
| Audio Flamingo 3 | Qwen2 | AF-CLAP | 文本 | NVIDIA 非商业 |
| Audio Flamingo Next | Qwen2 | AF-CLAP v2 | 文本 | NVIDIA 非商业 |
| SALMONN | Vicuna | Whisper + BEATs | 文本 | Apache-2.0 |
| LTU / LTU-AS | Llama | CAV-MAE | 文本 | Apache-2.0 |
| GAMA | Llama | AST + Q-Former | 文本 | Apache-2.0 |
| Gemini 2.5 Flash/Pro（闭源） | Gemini | 专有 | 文本 + 语音 | API |
| GPT-4o Audio（闭源） | GPT-4o | 专有 | 文本 + 语音 | API |

### 基准测试实况（2026）

**MMAU-Pro。** 1800 个问答对，涵盖语音/声音/音乐/混合。包含多音频子集。

| 模型 | 总体 | 语音 | 声音 | 音乐 | 多音频 |
|------|------|------|------|------|--------|
| Gemini 2.5 Pro | ~60% | 73.4% | 51.9% | 64.9% | ~22% |
| Gemini 2.5 Flash | ~57% | 73.4% | 50.5% | 64.9% | 21.2% |
| GPT-4o Audio | 52.5% | — | — | — | 26.5% |
| Qwen2.5-Omni-7B | 52.2% | 57.4% | 47.6% | 61.5% | ~20% |
| Audio Flamingo 3 | ~54% | — | — | — | — |
| Audio Flamingo Next | LongAudioBench SOTA | — | — | — | — |

**多音频一列的数据触目惊心。** 4 选 1 随机概率 = 25%；大多数模型分数与此相当。LALM 仍难以比较两段音频。

### 2026 年 LALM 的适用场景

- **呼叫中心录音合规审计。** "客服是否提及了必要的披露信息？"
- **无障碍访问。** 向聋哑用户描述声音事件（不仅仅是转录）。
- **内容审核。** 检测暴力语言 + 威胁语气 + 背景上下文。
- **播客/会议章节划分。** 语义摘要，而非仅分说话人轮次。
- **音乐目录分析。** "找出所有在 B 段转调的曲目。"

### 尚不适用的场景

- 精细音乐理论（和弦级别以下）。
- 长对话中的说话人归因推理（超过 10 分钟后退化）。
- 多音频比较（22-26% 几乎等同于随机）。
- 实时流式推理（大多数模型为离线批量推理）。

## 动手实现

### 步骤一：查询 Qwen2.5-Omni

```python
from transformers import AutoModelForCausalLM, AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype="auto")

audio, sr = load_wav("clip.wav", sr=16000)
messages = [{
    "role": "user",
    "content": [
        {"type": "audio", "audio": audio},
        {"type": "text", "text": "What sounds do you hear, and what's happening?"},
    ],
}]
inputs = processor.apply_chat_template(messages, tokenize=True, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output[0], skip_special_tokens=True))
```

### 步骤二：投影器模式

```python
import torch.nn as nn

class AudioProjector(nn.Module):
    def __init__(self, audio_dim=1280, llm_dim=4096):
        super().__init__()
        self.down = nn.Linear(audio_dim, llm_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(llm_dim, llm_dim)

    def forward(self, audio_features):
        return self.up(self.act(self.down(audio_features)))
```

就这样。投影器通常是 1-3 个线性层。在 ASR 对上（音频 → 文字稿）训练它是阶段一的预训练任务。

### 步骤三：MMAU / LongAudioBench 基准测试

```python
from datasets import load_dataset
mmau = load_dataset("MMAU/MMAU-Pro")

correct = 0
for item in mmau["test"]:
    answer = call_model(item["audio"], item["question"], item["choices"])
    if answer == item["correct_choice"]:
        correct += 1
print(f"Accuracy: {correct / len(mmau['test']):.3f}")
```

分类别报告（语音/声音/音乐/多音频）。聚合数字会掩盖模型的失败点。

## 生产使用

| 任务 | 2026 年选择 |
|------|-----------|
| 自由形式音频问答（开源） | Qwen2.5-Omni-7B |
| 开源长音频最佳 | Audio Flamingo Next |
| 闭源最佳 | Gemini 2.5 Pro |
| 语音输入/输出智能体 | Qwen2.5-Omni 或 GPT-4o Audio |
| 音乐推理 | Audio Flamingo 3 或 2（音乐专用 AF-CLAP） |
| 呼叫中心审计 | Gemini 2.5 Pro via API，结合策略文档 RAG |

## 常见坑

- **过度信任多音频结果。** 如果任务需要"哪段音频包含 X"，随机概率级别的表现是真实存在的。
- **长音频退化。** 超过 10 分钟后，大多数模型的说话人归因会崩溃。先做分离（第 6 课），再做摘要。
- **静音上的幻觉。** 使用 Whisper 编码器的 LALM 继承了同款 Whisper 风格问题。使用 VAD 门控。
- **基准测试挑选。** 厂商博客会突出最佳类别结果。自行运行 MMAU-Pro 多音频子集。

## 上手实践

保存为 `outputs/skill-alm-picker.md`。为给定的音频理解任务选择 LALM + 基准子集 + 输出模态（文本 vs 语音）。

## 练习

1. **简单。** 运行 `code/main.py`，查看玩具投影器模式 + 模拟 LALM 路由（音频嵌入，文本 token）→ 输出 token 的演示。
2. **中等。** 在 100 个 MMAU-Pro 语音条目上评测 Qwen2.5-Omni-7B。与论文报告的数字比较。
3. **困难。** 构建最小音频字幕基准线：BEATs 编码器 + 2 层投影器 + 冻结 Llama-3.2-1B。仅在 AudioCaps 上微调投影器。在 Clotho-AQA 上与 SALMONN 比较。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| LALM | 音频版 ChatGPT | 音频编码器 + 投影器 + LLM 解码器。 |
| 投影器（Projector） | 适配器 | 将音频特征映射到 LLM 嵌入空间的小型 MLP。 |
| MMAU | 那个基准 | 10000 个跨语音、声音、音乐的音频问答对。 |
| MMAU-Pro | 更难的 MMAU | 1800 个多音频/推理密集型问题。 |
| LongAudioBench | 长格式评估 | 多分钟音频片段配语义查询。 |
| 语音输入/输出（Voice-in/voice-out） | 语音原生 | 模型直接接受并生成语音，无需文本中转。 |

## 延伸阅读

- [Chu et al. (2024). Qwen2-Audio](https://arxiv.org/abs/2407.10759) — 参考架构
- [Alibaba (2025). Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) — 语音输入语音输出
- [NVIDIA (2025). Audio Flamingo 3](https://arxiv.org/abs/2507.08128) — 开源长音频领先者
- [NVIDIA (2026). Audio Flamingo Next](https://arxiv.org/abs/2604.10905) — LongAudioBench SOTA
- [Tang et al. (2023). SALMONN](https://arxiv.org/abs/2310.13289) — 双编码器先驱
- [MMAU-Pro leaderboard](https://mmaubenchmark.github.io/) — 2026 年实时排名
