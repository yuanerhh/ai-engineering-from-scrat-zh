# 流式语音到语音——Moshi、Hibiki 与全双工对话

> 2024-2026 年重新定义了语音 AI。Moshi 提供了一个以 200 ms 延迟同时监听和说话的单一模型。Hibiki 逐块进行语音到语音翻译。两者都放弃了 ASR → LLM → TTS 流水线，转向基于 Mimi 编解码器 token 的统一全双工架构。这是新的参考设计。

**类型：** 学习
**语言：** Python
**前置条件：** Phase 6 · 13（神经音频编解码器）、Phase 6 · 11（实时音频处理）、Phase 7 · 05（完整 Transformer）
**时长：** 约 75 分钟

## 问题背景

基于第 11 + 12 课构建的每个语音智能体都有一个基本延迟下限，约为 300-500 ms：VAD 触发，STT 处理，LLM 推理，TTS 生成。每个阶段都有自己的最小延迟。你可以调优和并行化，但流水线结构限制了上限。

Moshi（Kyutai，2024-2026）提出了一个不同的问题：如果没有流水线会怎样？如果一个模型直接连续地接收音频并输出音频，文本作为中间"内心独白"而非必需阶段呢？

答案是**全双工语音到语音**。理论延迟 160 ms（80 ms Mimi 帧 + 80 ms 声学延迟）。在单张 L4 GPU 上实际延迟 200 ms。这是最先进级联式语音智能体延迟的一半。

## 核心概念

![Moshi 架构：两个并行 Mimi 流 + 内心独白文本](../assets/moshi-hibiki.svg)

### Moshi 架构

**输入。** 两个 Mimi 编解码器流，均为 12.5 Hz × 8 个码本：

- 流 1：用户音频（Mimi 编码，持续到达）
- 流 2：Moshi 自身音频（由 Moshi 生成）

**Transformer。** 70 亿参数的时序 Transformer 处理两个流和一个文本"内心独白"流。在每个 80 ms 步骤中，它：

1. 消费最新的用户 Mimi token（8 个码本）。
2. 消费最近的 Moshi Mimi token（8 个码本，由 Moshi 生成）。
3. 生成下一个 Moshi 文本 token（内心独白）。
4. 通过小型深度 Transformer 生成下一个 Moshi Mimi token（8 个码本）。

三个流——用户音频、Moshi 音频、Moshi 文本——并行运行。Moshi 在说话时可以听到用户；用户打断时可以中断自己；可以在不打断主要话语的情况下进行回应性表达（"嗯嗯"）。

**深度 Transformer（Depth transformer）。** 在一帧内，8 个码本不是并行预测的——它们之间存在码本间依赖关系。一个小型 2 层"深度 Transformer"在 80 ms 内顺序预测它们。这是 AR 编解码器 LM 的标准分解方式（VALL-E、VibeVoice 也采用）。

### 为什么内心独白文本有帮助

没有显式文本，模型必须在声学流中隐式建模语言。Moshi 的洞见：强制它在音频旁边输出文本 token。文本流本质上是 Moshi 所说内容的文字稿。这提高了语义连贯性，使替换语言模型头更容易，并免费提供文字稿。

### Hibiki：流式语音到语音翻译

相同架构，在翻译对上训练。源语言音频输入，目标语言音频输出，持续进行。Hibiki-Zero（2026 年 2 月）消除了对词级对齐训练数据的需求——使用句级数据 + GRPO 强化学习进行延迟优化。

最初支持四种语言对；可以用约 1000 小时数据适配新语言。

### 更广泛的 Kyutai 技术栈（2026）

- **Moshi** — 全双工对话（法语优先，英语支持良好）
- **Hibiki / Hibiki-Zero** — 同声传译
- **Kyutai STT** — 流式 ASR（500 ms 或 2.5 秒前瞻）
- **Kyutai Pocket TTS** — 1 亿参数 TTS，可在 CPU 运行（2026 年 1 月）
- **Unmute** — 在公共服务器上组合这些组件的完整流水线

L40S GPU 上的吞吐量：64 个并发会话，3× 实时速度。

### Sesame CSM——近亲

Sesame CSM（2025）使用类似思路——带 Mimi 编解码器头的 Llama-3 骨干。但 CSM 是单向的（接受上下文 + 文本，生成语音），而非全双工。它是市场上"声音存在感"最好的 TTS；与 Moshi 的全双工能力并不完全相同。

### 2026 年性能数字

| 模型 | 延迟 | 使用场景 | 许可证 |
|------|------|---------|--------|
| Moshi | 200 ms（L4） | 全双工英语/法语对话 | CC-BY 4.0 |
| Hibiki | 12.5 Hz 帧率 | 法语 ↔ 英语流式翻译 | CC-BY 4.0 |
| Hibiki-Zero | 同上 | 5 种语言对，无对齐数据 | CC-BY 4.0 |
| Sesame CSM-1B | 200 ms TTFA | 上下文条件 TTS | Apache-2.0 |
| GPT-4o Realtime | ~300 ms | 闭源，OpenAI API | 商业 |
| Gemini 2.5 Live | ~350 ms | 闭源，Google API | 商业 |

## 动手实现

### 步骤一：接口

Moshi 暴露一个 WebSocket 服务器，接收 80 ms 的 Mimi 编码音频块并返回 80 ms 的 Mimi 编码音频块。双向。持续不断。

```python
import asyncio
import websockets
from moshi.client_utils import encode_audio_mimi, decode_audio_mimi

async def moshi_chat():
    async with websockets.connect("ws://localhost:8998/api/chat") as ws:
        mic_task = asyncio.create_task(stream_mic_to(ws))
        spk_task = asyncio.create_task(stream_from_to_speaker(ws))
        await asyncio.gather(mic_task, spk_task)
```

### 步骤二：全双工循环

```python
async def stream_mic_to(ws):
    async for chunk_80ms in mic_stream_at_12_5_hz():
        mimi_tokens = encode_audio_mimi(chunk_80ms)
        await ws.send(serialize(mimi_tokens))

async def stream_from_to_speaker(ws):
    async for msg in ws:
        mimi_tokens, text_token = deserialize(msg)
        audio = decode_audio_mimi(mimi_tokens)
        await play(audio)
```

两个方向同时运行。Python asyncio 或 Rust futures 是标准传输方式。

### 步骤三：训练目标（概念性）

对于每个 80 ms 帧 `t`：

- 输入：`user_mimi[0..t]`、`moshi_mimi[0..t-1]`、`moshi_text[0..t-1]`
- 预测：`moshi_text[t]`，然后预测 `moshi_mimi[t, codebook_0..7]`

文本在音频之前预测（内心独白）；音频在深度 Transformer 内按码本顺序预测。

### 步骤四：Moshi 的优势与劣势

Moshi 的优势：

- 在廉价硬件上实现亚 250 ms 端到端延迟。
- 自然的回应性表达和中断处理。
- 无需流水线胶水代码。

Moshi 的劣势：

- 工具调用（未为此训练；需要独立的 LLM 路径）。
- 长推理（Moshi 是约 80 亿参数的对话模型，不是 Claude/GPT-4）。
- 小众话题的事实准确性。
- 大多数生产企业用例（2026 年仍使用流水线）。

## 生产使用

| 场景 | 选择 |
|------|------|
| 最低延迟语音伴侣 | Moshi |
| 实时翻译通话 | Hibiki |
| 语音演示/研究 | Moshi、CSM |
| 带工具的企业智能体 | 流水线（第 12 课），而非 Moshi |
| 上下文中的自定义声音 TTS | Sesame CSM |
| 任意语言的语音到语音 | GPT-4o Realtime 或 Gemini 2.5 Live（商业） |

## 常见坑

- **工具调用有限。** Moshi 是对话模型，不是智能体框架。结合流水线使用工具。
- **特定声音条件。** Moshi 使用单一训练角色；克隆需要独立的训练运行。
- **语言覆盖。** 法语 + 英语表现优秀；其他语言有限。Hibiki-Zero 有所帮助，但仍需训练数据。
- **资源成本。** 完整的 Moshi 会话占用一个 GPU 槽；不是廉价的共享租户部署模式。

## 上手实践

保存为 `outputs/skill-duplex-pipeline.md`。为语音智能体工作负载选择流水线 vs 全双工架构，并说明原因。

## 练习

1. **简单。** 运行 `code/main.py`。以符号方式模拟双流 + 内心独白架构。
2. **中等。** 从 HuggingFace 拉取 Moshi，运行服务器，测试一次对话。从用户语音结束到 Moshi 开始响应，测量挂钟延迟。
3. **困难。** 取第 12 课的流水线智能体，在 20 个匹配测试话语上与 Moshi 比较 P50 延迟。写出流水线架构在哪些情况下仍然更优。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 全双工（Full-duplex） | 同时听和说 | 同一模型上同时激活两个音频流。 |
| 内心独白（Inner monologue） | 模型的文本流 | Moshi 在音频输出旁边输出文本 token。 |
| 深度 Transformer（Depth transformer） | 码本间预测器 | 在一个 80 ms 帧内预测 8 个码本的小型 Transformer。 |
| Mimi | Kyutai 的编解码器 | 12.5 Hz × 8 个码本；语义+声学；驱动 Moshi。 |
| 流式 S2S | 实时音频 → 音频 | 逐块翻译/对话，无流水线阶段。 |
| 回应性表达（Back-channeling） | "嗯嗯"回应 | Moshi 可以发出小型确认而不打断自己的轮次。 |

## 延伸阅读

- [Défossez et al. (2024). Moshi — speech-text foundation model](https://arxiv.org/html/2410.00037v2) — 论文
- [Kyutai Labs (2026). Hibiki-Zero](https://arxiv.org/abs/2602.12345) — 无对齐数据的流式翻译
- [Sesame (2025). Crossing the uncanny valley of voice](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice) — CSM 规格
- [Kyutai — Moshi repo](https://github.com/kyutai-labs/moshi) — 安装 + 服务器
- [OpenAI — Realtime API](https://platform.openai.com/docs/guides/realtime) — 闭源商业同类产品
- [Kyutai — Delayed Streams Modeling](https://github.com/kyutai-labs/delayed-streams-modeling) — 底层 STT/TTS 框架
