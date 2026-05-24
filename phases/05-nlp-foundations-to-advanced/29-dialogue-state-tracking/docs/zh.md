# 对话状态追踪

> "我想要一家北部便宜的餐厅……实际上改成中等价位……再加上意大利菜。"三轮对话，三次状态更新。对话状态追踪（DST）保持槽位-值字典同步，让预订得以成功。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 5 · 17（聊天机器人）、Phase 5 · 20（结构化输出）
**时长：** 约 75 分钟

## 问题背景

在面向任务的对话系统中，用户的目标被编码为一组槽位-值对：`{cuisine: italian, area: north, price: moderate}`。每轮用户输入都可以添加、更改或删除一个槽位。系统必须读取整个对话，并正确输出当前状态。

搞错一个槽位，系统就会预订错误的餐厅、安排错误的航班，或者扣错账单。DST 是用户所说内容与后端执行之间的枢纽。

2026 年，尽管有了 LLM，DST 仍然重要的原因：

- 合规敏感领域（银行、医疗、航空预订）需要确定性的槽位值，而不是自由形式的生成。
- 工具调用型 Agent 在调用 API 之前仍然需要槽位解析。
- 多轮纠正比看起来难得多："实际上不，改成周四。"

现代流水线：经典 DST 概念 + LLM 提取器 + 结构化输出护栏。

## 核心概念

![DST：对话历史 → 槽位-值状态](../assets/dst.svg)

**任务结构。** 模式（Schema）定义了领域（餐厅、酒店、出租车）及其槽位（菜系、区域、价格、人数）。每个槽位可以是空的、从封闭集合中填充的值（价格：{便宜, 中等, 昂贵}），或自由形式的值（名称："铜壶餐厅"）。

**两种 DST 表述方式。**

- **分类（Classification）。** 对每个（槽位, 候选值）对，预测是/否。适用于封闭词汇槽位。2020 年前的标准方案。
- **生成（Generation）。** 给定对话，以自由文本生成槽位值。适用于开放词汇槽位。现代默认方案。

**评估指标。** 联合目标准确率（Joint Goal Accuracy，JGA）——所有槽位都正确的轮次占总轮次的比例。全有或全无，非常严格。2026 年 MultiWOZ 2.4 排行榜最高约 83%。

**架构。**

1. **基于规则（槽位正则 + 关键词）。** 对窄领域而言是强基准线，可调试。
2. **TripPy / BERT-DST。** 基于复制的生成，使用 BERT 编码。LLM 时代前的标准。
3. **LDST（LLaMA + LoRA）。** 带领域-槽位提示的指令微调 LLM。在 MultiWOZ 2.4 上达到 ChatGPT 级别的质量。
4. **无本体（Ontology-free，2024-26）。** 跳过模式；直接生成槽位名称和值。可处理开放领域。
5. **提示 + 结构化输出（2024-26）。** LLM + Pydantic 模式 + 约束解码。5 行代码，可直接投入生产。

### 经典失败模式

- **跨轮次指代消解。** "就选第一个吧。"需要解析"第一个"是哪个选项。
- **覆盖 vs 追加。** 用户说"加上意大利菜"，是替换菜系还是追加？
- **隐式确认。** "好的没问题"——这算是接受了所提供的预订吗？
- **纠正。** "实际上改成晚上 7 点。"必须更新时间而不清除其他槽位。
- **对上一轮系统话语的指代。** "是的，就那个。""那个"是哪个？

## 动手实现

### 步骤一：基于规则的槽位提取器

见 `code/main.py`。正则 + 同义词词典覆盖窄领域中 70% 的标准话语：

```python
CUISINE_SYNONYMS = {
    "italian": ["italian", "pasta", "pizza", "italy"],
    "chinese": ["chinese", "chow mein", "noodles"],
}


def extract_cuisine(utterance):
    for canonical, synonyms in CUISINE_SYNONYMS.items():
        if any(syn in utterance.lower() for syn in synonyms):
            return canonical
    return None
```

在标准词汇之外很脆。适用于确定性的槽位确认。

### 步骤二：状态更新循环

```python
def update_state(state, utterance):
    new_state = dict(state)
    for slot, extractor in SLOT_EXTRACTORS.items():
        value = extractor(utterance)
        if value is not None:
            new_state[slot] = value
    for slot in NEGATION_CLEARS:
        if is_negated(utterance, slot):
            new_state[slot] = None
    return new_state
```

三个不变量：

- 永远不重置用户没有触及的槽位。
- 显式否定（"不要那个菜系了"）必须清除。
- 用户纠正（"实际上……"）必须覆盖而不是追加。

### 步骤三：使用结构化输出的 LLM 驱动 DST

```python
from pydantic import BaseModel
from typing import Literal, Optional
import instructor

class RestaurantState(BaseModel):
    cuisine: Optional[Literal["italian", "chinese", "indian", "thai", "any"]] = None
    area: Optional[Literal["north", "south", "east", "west", "center"]] = None
    price: Optional[Literal["cheap", "moderate", "expensive"]] = None
    people: Optional[int] = None
    day: Optional[str] = None


def llm_dst(history, llm):
    prompt = f"""You track the slot values of a restaurant booking across turns.
Dialogue so far:
{render(history)}

Update the state based on the latest user turn. Output only the JSON state."""
    return llm(prompt, response_model=RestaurantState)
```

Instructor + Pydantic 保证返回有效的状态对象。无需正则，无模式不匹配，无幻化槽位。

### 步骤四：JGA 评估

```python
def joint_goal_accuracy(predicted_states, gold_states):
    correct = sum(1 for p, g in zip(predicted_states, gold_states) if p == g)
    return correct / len(predicted_states)
```

校准：系统在多少比例的轮次上把所有槽位都答对了？对于 MultiWOZ 2.4，2026 年的顶级系统：80-83%。你的领域内系统应该在你的窄词汇上超过这个数字，否则 LLM 基准线就已经打败你了。

### 步骤五：处理纠正

```python
CORRECTION_CUES = {"actually", "no wait", "on second thought", "change that to"}


def is_correction(utterance):
    return any(cue in utterance.lower() for cue in CORRECTION_CUES)
```

检测到纠正时，覆盖最后更新的槽位，而不是追加。没有 LLM 的帮助很难做好。现代模式：始终让 LLM 从历史记录重新生成完整状态，而不是增量更新——这自然地处理了纠正。

## 常见坑

- **全历史重新生成的成本。** 让 LLM 每轮重新生成状态会消耗 O(n²) 的总 token 数。限制历史长度或对较早的轮次做摘要。
- **模式漂移（Schema drift）。** 事后添加新槽位会破坏旧训练数据，要对模式做版本管理。
- **大小写敏感性。** "Italian"、"italian"、"ITALIAN"——在所有地方都做归一化。
- **隐式继承。** 如果用户之前指定了"4人"，新的时间请求不应该清除人数槽位。始终传递完整历史。
- **自由形式 vs 封闭集合。** 名称、时间和地址需要自由形式槽位；菜系和区域是封闭的。在模式中两者都要有。

## 生产使用

2026 年技术栈：

| 场景 | 方法 |
|------|------|
| 窄领域（一两个意图） | 基于规则 + 正则 |
| 宽领域，有标注数据 | LDST（LLaMA + LoRA 在 MultiWOZ 风格数据上微调） |
| 宽领域，无标注，可生产 | LLM + Instructor + Pydantic 模式 |
| 语音/口语 | ASR + 归一化器 + LLM-DST |
| 多领域预订流程 | 带每领域 Pydantic 模型的模式引导 LLM |
| 合规敏感 | 基于规则作主要，LLM 作回退并加确认流程 |

## 上手实践

将以下内容保存为 `outputs/skill-dst-designer.md`：

```markdown
---
name: dst-designer
description: Design a dialogue state tracker — schema, extractor, update policy, evaluation.
version: 1.0.0
phase: 5
lesson: 29
tags: [nlp, dialogue, task-oriented]
---

Given a use case (domain, languages, vocab openness, compliance needs), output:

1. Schema. Domain list, slots per domain, open vs closed vocabulary per slot.
2. Extractor. Rule-based / seq2seq / LLM-with-Pydantic. Reason.
3. Update policy. Regenerate-whole-state / incremental; correction handling; negation handling.
4. Evaluation. Joint Goal Accuracy on a held-out dialogue set, slot-level precision/recall, confusion on the hardest slot.
5. Confirmation flow. When to explicitly ask the user to confirm (destructive actions, low-confidence extractions).

Refuse LLM-only DST for compliance-sensitive slots without a rule-based secondary check. Refuse any DST that cannot roll back a slot on user correction. Flag schemas without version tags.
```

## 练习

1. **简单。** 在 `code/main.py` 中为 3 个槽位（菜系、区域、价格）构建基于规则的状态追踪器。在 10 个手工制作的对话上测试，测量 JGA。
2. **中等。** 在同一数据集上使用 Instructor + Pydantic + 小型 LLM。比较 JGA，检查最难的轮次。
3. **困难。** 同时实现两者并进行路由：基于规则作主要，当基于规则只发出少于 2 个高置信度槽位时用 LLM 作回退。测量组合 JGA 和每轮推理成本。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| DST（对话状态追踪） | 对话状态追踪 | 在对话轮次中维护槽位-值字典。 |
| 槽位（Slot） | 用户意图单元 | 后端需要的命名参数（菜系、日期）。 |
| 领域（Domain） | 任务区域 | 餐厅、酒店、出租车——槽位集合。 |
| JGA（联合目标准确率） | 联合目标准确率 | 所有槽位都正确的轮次比例。全有或全无。 |
| MultiWOZ | 基准数据集 | 多领域 WOZ 数据集；标准 DST 评估基准。 |
| 无本体 DST（Ontology-free DST） | 无模式 | 直接生成槽位名称和值，无固定列表。 |
| 纠正（Correction） | "实际上……" | 覆盖之前已填充槽位的轮次。 |

## 延伸阅读

- [Budzianowski et al. (2018). MultiWOZ — A Large-Scale Multi-Domain Wizard-of-Oz](https://arxiv.org/abs/1810.00278) — 经典基准数据集
- [Feng et al. (2023). Towards LLM-driven Dialogue State Tracking (LDST)](https://arxiv.org/abs/2310.14970) — LLaMA + LoRA 指令微调用于 DST
- [Heck et al. (2020). TripPy — A Triple Copy Strategy for Value Independent Neural Dialog State Tracking](https://arxiv.org/abs/2005.02877) — 基于复制的 DST 主力方案
- [King, Flanigan (2024). Unsupervised End-to-End Task-Oriented Dialogue with LLMs](https://arxiv.org/abs/2404.10753) — 基于 EM 的无监督面向任务对话
- [MultiWOZ leaderboard](https://github.com/budzianowski/multiwoz) — 规范 DST 结果
