# 结构化输出与约束解码

> 要求 LLM 返回 JSON，大多数时候能得到 JSON。在生产中，"大多数"才是问题所在。约束解码通过在采样前编辑 logits，将"大多数"变为"始终"。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 5 · 17（聊天机器人）、Phase 5 · 19（子词分词）
**时长：** 约 60 分钟

## 问题背景

分类器提示 LLM："返回 {positive, negative, neutral} 中的一个。"模型返回："情感是正面的——这条评论非常有利，因为顾客明确表示他们……"。你的解析器崩溃了，分类器的 F1 为 0.0。

自由形式生成不是契约，而是建议。生产系统需要契约。

2026 年存在三个层次：

1. **提示。** 礼貌地询问，"只返回 JSON 对象。"在前沿模型上有效约 80%，在小模型上更低。
2. **原生结构化输出 API。** OpenAI `response_format`、Anthropic tool use、Gemini JSON mode。对支持的模式可靠，但供应商锁定。
3. **约束解码（Constrained decoding）。** 在每个生成步骤修改 logits，使模型**无法**输出无效 token。从构造上保证 100% 有效，适用于任何本地模型。

本课为三者建立直觉，并指出何时该用哪种。

## 核心概念

![约束解码在每步屏蔽无效 token](../assets/constrained-decoding.svg)

**约束解码的工作原理。** 在每个生成步骤，LLM 为完整词汇表（约 10 万 token）生成 logit 向量。**Logit 处理器**位于模型和采样器之间。它根据当前在目标文法（JSON Schema、正则表达式、上下文无关文法）中的位置计算哪些 token 有效，并将所有无效 token 的 logit 设为负无穷。对剩余 logit 做 softmax 后，概率质量只分布在有效续写上。

2026 年的实现：

- **Outlines。** 将 JSON Schema 或正则表达式编译为有限状态机（FSM）。每个 token 的有效下一 token 查询为 O(1)。基于 FSM，递归模式需要展平。
- **XGrammar / llguidance。** 上下文无关文法引擎，处理递归 JSON Schema，解码开销接近零。OpenAI 在其 2025 年结构化输出实现中致谢了 llguidance。
- **vLLM 引导解码。** 通过 Outlines、XGrammar 或 lm-format-enforcer 后端内置 `guided_json`、`guided_regex`、`guided_choice`、`guided_grammar`。
- **Instructor。** 基于 Pydantic 的任意 LLM 包装器，验证失败时自动重试。跨供应商，但不修改 logits——依赖重试和结构化输出感知提示。

### 反直觉的结果

约束解码通常比无约束生成**更快**。两个原因：其一，它缩小了下一个 token 的搜索空间；其二，对于强制 token（如脚手架 `{"name": "` ——每个字节都是确定的），聪明的实现完全跳过 token 生成。

### 会让你付出代价的坑

字段顺序很重要。把 `answer` 放在 `reasoning` 前面，模型在思考之前就提交了答案。JSON 有效，但答案是错的，没有任何验证能发现。

```json
// 不好的做法
{"answer": "yes", "reasoning": "because ..."}

// 好的做法
{"reasoning": "... therefore ...", "answer": "yes"}
```

Schema 字段顺序是逻辑，不是格式。

## 动手实现

### 步骤一：从零实现正则约束生成

见 `code/main.py` 的独立 FSM 实现。30 行的核心思想：

```python
def mask_logits(logits, valid_token_ids):
    mask = [float("-inf")] * len(logits)
    for tid in valid_token_ids:
        mask[tid] = logits[tid]
    return mask


def generate_constrained(model, tokenizer, prompt, fsm):
    ids = tokenizer.encode(prompt)
    state = fsm.initial_state
    while not fsm.is_accept(state):
        logits = model.next_token_logits(ids)
        valid = fsm.valid_tokens(state, tokenizer)
        logits = mask_logits(logits, valid)
        tok = sample(logits)
        ids.append(tok)
        state = fsm.transition(state, tok)
    return tokenizer.decode(ids)
```

FSM 追踪已满足文法的哪些部分。`valid_tokens(state, tokenizer)` 计算哪些词汇 token 能在不离开接受路径的情况下推进 FSM。

### 步骤二：用 Outlines 处理 JSON Schema

```python
from pydantic import BaseModel
from typing import Literal
import outlines


class Review(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    evidence_span: str


model = outlines.models.transformers("meta-llama/Llama-3.2-3B-Instruct")
generator = outlines.generate.json(model, Review)

result = generator("Classify: 'The wait staff was attentive and the food arrived hot.'")
print(result)
# Review(sentiment='positive', confidence=0.93, evidence_span='attentive ... hot')
```

零验证错误，永远。FSM 使无效输出不可达。

### 步骤三：用 Instructor 实现跨供应商 Pydantic

```python
import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field


class Invoice(BaseModel):
    vendor: str
    total_usd: float = Field(ge=0)
    line_items: list[str]


client = instructor.from_anthropic(Anthropic())
invoice = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=1024,
    response_model=Invoice,
    messages=[{"role": "user", "content": "Extract from: 'Acme Corp $420. Widget, Gizmo.'"}],
)
```

机制不同。Instructor 不修改 logits，而是将 schema 格式化进提示，解析输出，验证失败时重试（默认 3 次）。适用于任何供应商，重试会增加延迟和成本。跨供应商可移植性是卖点。

### 步骤四：原生供应商 API

```python
from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="gpt-5",
    input=[{"role": "user", "content": "Classify: 'The food was cold.'"}],
    text={"format": {"type": "json_schema", "name": "sentiment",
          "schema": {"type": "object", "required": ["sentiment"],
                     "properties": {"sentiment": {"type": "string",
                                                  "enum": ["positive", "negative", "neutral"]}}}}},
)
print(response.output_parsed)
```

服务端约束解码。对支持的 schema 与 Outlines 可靠性相当。无需本地模型管理，但锁定到供应商。

## 常见坑

- **递归 schema。** Outlines 将递归展平到固定深度。树形结构输出（嵌套评论、AST）需要 XGrammar 或 llguidance（基于 CFG）。
- **超大枚举。** 1 万个选项的枚举编译缓慢或超时。改用检索器：先预测 top-k 候选，然后约束到这些候选。
- **文法过于严格。** 强制 `date: "YYYY-MM-DD"` 正则后，模型无法对缺失的日期输出 `"unknown"`，它会通过编造日期来补偿。允许 `null` 或哨兵值。
- **过早提交。** 见上面的字段顺序坑，始终将推理放在前面。
- **不带 schema 的供应商 JSON 模式。** 纯 JSON 模式只保证 JSON 语法有效，不保证对你的用例有效。始终提供完整 schema。

## 生产使用

2026 年技术栈：

| 场景 | 选择 |
|------|------|
| OpenAI/Anthropic/Google 模型，简单 schema | 原生供应商结构化输出 |
| 任意供应商，Pydantic 工作流，可接受重试 | Instructor |
| 本地模型，需要 100% 有效性，平坦 schema | Outlines（FSM） |
| 本地模型，递归 schema | XGrammar 或 llguidance |
| 自托管推断服务器 | vLLM 引导解码 |
| 批处理，可接受重试 | Instructor + 最便宜的模型 |

## 上手实践

将以下内容保存为 `outputs/skill-structured-output-picker.md`：

```markdown
---
name: structured-output-picker
description: Choose a structured output approach, schema design, and validation plan.
version: 1.0.0
phase: 5
lesson: 20
tags: [nlp, llm, structured-output]
---

Given a use case (provider, latency budget, schema complexity, failure tolerance), output:

1. Mechanism. Native vendor structured output, Instructor retries, Outlines FSM, or XGrammar CFG. One-sentence reason.
2. Schema design. Field order (reasoning first, answer last), nullable fields for "unknown", enum vs regex, required fields.
3. Failure strategy. Max retries, fallback model, graceful `null` handling, out-of-distribution refusal.
4. Validation plan. Schema compliance rate (target 100%), semantic validity (LLM-judge), field-coverage rate, latency p50/p99.

Refuse any design that puts `answer` or `decision` before reasoning fields. Refuse to use bare JSON mode without a schema. Flag recursive schemas behind an FSM-only library.
```

## 练习

1. **简单。** 在不使用约束解码的情况下，提示小型开源模型（如 Llama-3.2-3B）返回 `Review(sentiment, confidence, evidence_span)`，在 100 条评论上测量能解析为有效 JSON 的比例。
2. **中等。** 在同一语料库上使用 Outlines JSON 模式，比较合规率、延迟和语义准确率。
3. **困难。** 从零实现电话号码（`\d{3}-\d{3}-\d{4}`）的正则约束解码器，在 1000 个样本上验证 0 个无效输出。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 约束解码（Constrained decoding） | 强制有效输出 | 在每个生成步骤屏蔽无效 token 的 logit。 |
| Logit 处理器（Logit processor） | 约束的东西 | 函数：`(logits, state) -> masked_logits`。 |
| FSM | 有限状态机 | 编译后的文法表示；O(1) 有效下一 token 查询。 |
| CFG | 上下文无关文法 | 处理递归的文法；比 FSM 慢但表达力更强。 |
| Schema 字段顺序 | 这重要吗？ | 重要——第一个字段先提交；始终将推理放在答案前面。 |
| 引导解码（Guided decoding） | vLLM 的叫法 | 相同概念，集成到推断服务器中。 |
| JSON 模式（JSON mode） | OpenAI 的早期版本 | 保证 JSON 语法；不保证与 schema 匹配。 |

## 延伸阅读

- [Willard, Louf (2023). Efficient Guided Generation for LLMs](https://arxiv.org/abs/2307.09702) — Outlines 论文
- [XGrammar paper (2024)](https://arxiv.org/abs/2411.15100) — 快速 CFG 约束解码
- [vLLM — Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs.html) — 推断服务器集成
- [OpenAI — Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs) — API 参考 + 注意事项
- [Instructor library](https://python.useinstructor.com/) — 跨供应商的 Pydantic + 重试
- [JSONSchemaBench (2025)](https://arxiv.org/abs/2501.10868) — 6 种约束解码框架的基准测试
