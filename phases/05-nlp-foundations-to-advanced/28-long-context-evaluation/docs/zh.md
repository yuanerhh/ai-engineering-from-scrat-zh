# 长上下文评估——NIAH、RULER、LongBench、MRCR

> Gemini 3 Pro 宣传 1000 万 token 的上下文。在 100 万 token 时，8 针 MRCR 得分降至 26.3%。宣传的不等于可用的。长上下文评估告诉你实际部署模型的真实容量。

**类型：** 学习
**语言：** Python
**前置条件：** Phase 5 · 13（问答系统）、Phase 5 · 23（RAG 分块策略）
**时长：** 约 60 分钟

## 问题背景

你有一份 200 页的合同。模型声称有 100 万 token 的上下文。你把合同粘贴进去，问："终止条款是什么？"模型回答了——但答的是封面页的内容，因为终止条款在 12 万 token 深处，超出了模型实际关注的范围。

这就是 2026 年的上下文容量差距。规格说明书说 100 万或 1000 万，现实是其中 60-70% 可用，而"可用"取决于任务类型：

- **检索（干草堆中的单根针）：** 在前沿模型上几乎完美，直到宣传的最大值。
- **多跳推理 / 聚合：** 在大多数模型上超过约 12.8 万 token 后急剧退化。
- **分散事实推理：** 最先失败的任务。

长上下文评估就是测量这些维度。本课介绍各基准测试的名称、实际衡量的内容，以及如何为你的领域构建自定义针测试。

## 核心概念

![NIAH 基准、RULER 多任务、LongBench 整体评估](../assets/long-context-eval.svg)

**干草堆中的针（Needle-in-a-Haystack，NIAH，2023）。** 将一个事实（"魔法词是菠萝"）放置在长上下文中的受控深度。要求模型检索它。横扫深度 × 长度。这是原始的长上下文基准测试。前沿模型现在已经把它做满了；它是必要但不充分的基准线。

**RULER（英伟达，2024）。** 4 个类别的 13 种任务类型：检索（单键/多键/多值）、多跳追踪（变量追踪）、聚合（高频词统计）、问答。可配置上下文长度（4k 到 128k+）。揭示了那些通过 NIAH 但在多跳推理上失败的模型。在 2024 年发布时，17 个声称 32k+ 上下文的模型中，只有一半在 32k 时保持了质量。

**LongBench v2（2024）。** 503 道多项选择题，上下文 8k-200 万词，六个任务类别：单文档问答、多文档问答、长上下文学习、长对话、代码仓库、长结构化数据。是评估真实世界长上下文行为的生产基准。

**MRCR（多轮指代消解，Multi-Round Coreference Resolution）。** 大规模多轮指代消解，有 8 针、24 针、100 针变体。揭示模型在注意力退化前能同时处理多少个事实。

**NoLiMa。** "非词汇针"。针与查询没有字面重叠；检索需要一步语义推理。比 NIAH 更难。

**HELMET。** 拼接多个文档，从其中任一文档提问。测试选择性注意力。

**BABILong。** 将 bAbI 推理链嵌入不相关的干草堆中。测试干草堆中的推理能力，而不仅仅是检索。

### 实际需要报告什么

- **宣传的上下文窗口。** 规格说明书上的数字。
- **有效检索长度（Effective retrieval length）。** 在某个阈值（如 90%）下通过 NIAH。
- **有效推理长度（Effective reasoning length）。** 在该阈值下通过多跳或聚合。
- **退化曲线。** 按任务类型绘制的准确率 vs 上下文长度图。

你的规格说明书上需要两个数字：有效检索长度和有效推理长度。通常有效推理长度是宣传窗口的 25-50%。

## 动手实现

### 步骤一：为你的领域构建自定义 NIAH

见 `code/main.py`。骨架代码：

```python
def build_haystack(filler_text, needle, depth_ratio, total_tokens):
    if not (0.0 <= depth_ratio <= 1.0):
        raise ValueError(f"depth_ratio must be in [0, 1], got {depth_ratio}")
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be positive, got {total_tokens}")

    filler_tokens = tokenize(filler_text)
    needle_tokens = tokenize(needle)
    if not filler_tokens:
        raise ValueError("filler_text produced no tokens")

    # 重复填充文本直到足够长以填满干草堆主体
    body_len = max(total_tokens - len(needle_tokens), 0)
    while len(filler_tokens) < body_len:
        filler_tokens = filler_tokens + filler_tokens
    filler_tokens = filler_tokens[:body_len]

    insert_at = min(int(body_len * depth_ratio), body_len)
    haystack = filler_tokens[:insert_at] + needle_tokens + filler_tokens[insert_at:]
    return " ".join(haystack)


def score_niah(model, haystack, question, expected):
    answer = model.complete(f"Context: {haystack}\nQ: {question}\nA:", max_tokens=50)
    return 1 if expected.lower() in answer.lower() else 0
```

横扫 `depth_ratio` ∈ {0, 0.25, 0.5, 0.75, 1.0} × `total_tokens` ∈ {1k, 4k, 16k, 64k}。绘制热力图，这就是目标模型的 NIAH 卡片。

### 步骤二：多针变体

```python
def build_multi_needle(filler, needles, total_tokens):
    depths = [0.1, 0.4, 0.7]
    chunks = [filler[:int(total_tokens * 0.1)]]
    for depth, needle in zip(depths, needles):
        chunks.append(needle)
        next_chunk = filler[int(total_tokens * depth): int(total_tokens * (depth + 0.3))]
        chunks.append(next_chunk)
    return " ".join(chunks)
```

"三个魔法词是什么？"这样的问题需要检索全部三个。单针成功并不能预测多针成功。

### 步骤三：多跳变量追踪（RULER 风格）

```python
haystack = """X1 = 42. ... (filler) ... X2 = X1 + 10. ... (filler) ... X3 = X2 * 2."""
question = "What is X3?"
```

这道题需要串联三次赋值运算。前沿模型在 128k token 时，这类任务的准确率通常降至 50-70%。

### 步骤四：在你的技术栈上评估 LongBench v2

```python
from datasets import load_dataset
longbench = load_dataset("THUDM/LongBench-v2")

def eval_model_on_longbench(model, subset="single-doc-qa"):
    tasks = [x for x in longbench["test"] if x["task"] == subset]
    correct = 0
    for x in tasks:
        answer = model.complete(x["context"] + "\n\nQ: " + x["question"], max_tokens=20)
        if normalize(answer) == normalize(x["answer"]):
            correct += 1
    return correct / len(tasks)
```

按类别报告准确率。聚合分数会掩盖任务级别的巨大差异。

## 常见坑

- **只做 NIAH 评估。** 在 100 万 token 上通过 NIAH 不说明多跳能力。始终运行 RULER 或自定义多跳测试。
- **均匀深度采样。** 许多实现只测试 depth=0.5。要测试 depth=0, 0.25, 0.5, 0.75, 1.0——"迷失在中间（lost in the middle）"效应是真实存在的。
- **填充文本与针有词汇重叠。** 如果针与填充文本共享关键词，检索就会变得微不足道。使用 NoLiMa 风格的无重叠针。
- **忽略延迟。** 100 万 token 的提示预填充（prefill）需要 30-120 秒。在测量准确率的同时测量首 token 时间（time-to-first-token）。
- **厂商自报数字。** OpenAI、Google、Anthropic 都发布自己的评估分数。始终针对你的使用场景独立重新运行。

## 生产使用

2026 年技术栈：

| 场景 | 基准测试 |
|------|---------|
| 快速健全性检查 | 3 个深度 × 3 个长度的自定义 NIAH |
| 生产模型选型 | 在目标长度下运行 RULER（13 个任务） |
| 真实世界问答质量 | LongBench v2 单文档问答子集 |
| 多跳推理 | BABILong 或自定义变量追踪 |
| 对话/会话 | 在目标长度下运行 MRCR 8 针 |
| 模型升级回归测试 | 固定内部 NIAH + RULER 测试套件，对每个新模型运行 |

生产经验法则：在你的目标长度上同时通过 NIAH + 1 个推理任务之前，永远不要信任某个上下文窗口。

## 上手实践

将以下内容保存为 `outputs/skill-long-context-eval.md`：

```markdown
---
name: long-context-eval
description: Design a long-context evaluation battery for a given model and use case.
version: 1.0.0
phase: 5
lesson: 28
tags: [nlp, long-context, evaluation]
---

Given a target model, target context length, and use case, output:

1. Tests. NIAH depth × length grid; RULER multi-hop; custom domain task.
2. Sampling. Depths 0, 0.25, 0.5, 0.75, 1.0 at each length.
3. Metrics. Retrieval pass rate; reasoning pass rate; time-to-first-token; cost-per-query.
4. Cutoff. Effective retrieval length (90% pass) and effective reasoning length (70% pass). Report both.
5. Regression. Fixed harness, rerun on every model upgrade, surface deltas.

Refuse to trust a context window from the model card alone. Refuse NIAH-only evaluation for any multi-hop workload. Refuse vendor self-reported long-context scores as independent evidence.
```

## 练习

1. **简单。** 构建 3 个深度（0.25, 0.5, 0.75）× 3 个长度（1k, 4k, 16k）的 NIAH。在任意模型上运行。将通过率绘制为 3×3 热力图。
2. **中等。** 添加 3 针变体。在每个长度下测量全部 3 根针的检索率。与相同长度下的单针通过率比较。
3. **困难。** 构建一个变量追踪任务（X1 → X2 → X3，共 3 跳），嵌入 64k 的填充文本中。在 3 个前沿模型上测量准确率。报告每个模型的有效推理长度。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| NIAH（干草堆中的针） | 干草堆测试 | 在填充文本中植入一个事实，让模型检索它。 |
| RULER | 增强版 NIAH | 涵盖检索/多跳/聚合/问答的 13 种任务类型。 |
| 有效上下文（Effective context） | 真实容量 | 准确率仍高于阈值的长度。 |
| 迷失在中间（Lost in the middle） | 深度偏差 | 模型对长输入中间部分的内容关注不足。 |
| 多针（Multi-needle） | 同时多个事实 | 多个植入点；测试注意力协调能力，而不仅仅是检索能力。 |
| MRCR | 多轮指代消解 | 8、24 或 100 针指代消解；揭示注意力饱和点。 |
| NoLiMa | 非词汇针 | 针与查询不共享任何字面 token；需要推理才能检索。 |

## 延伸阅读

- [Kamradt (2023). Needle in a Haystack analysis](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) — 原始 NIAH 仓库
- [Hsieh et al. (2024). RULER: What's the Real Context Size of Your Long-Context LMs?](https://arxiv.org/abs/2404.06654) — 多任务基准论文
- [Bai et al. (2024). LongBench v2](https://arxiv.org/abs/2412.15204) — 真实世界长上下文评估
- [Modarressi et al. (2024). NoLiMa: Non-lexical needles](https://arxiv.org/abs/2404.06666) — 更难的针
- [Kuratov et al. (2024). BABILong](https://arxiv.org/abs/2406.10149) — 干草堆中的推理
- [Liu et al. (2024). Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) — 深度偏差论文
