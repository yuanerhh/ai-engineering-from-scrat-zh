# 提示词缓存与上下文缓存

> 你的系统提示词有 4,000 个 token。你的 RAG 上下文有 20,000 个 token。你在每次请求时都发送这两者，而且每次都要付费。提示词缓存让提供商在他们那边保留这个前缀热缓存，复用时只收取正常价格的 10%。正确使用，它可以将推理成本削减 50-90%，将首 token 延迟降低 40-85%。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 11 · 01（提示词工程）、Phase 11 · 05（上下文工程）、Phase 11 · 11（缓存与成本）
**预计时间：** ~60 分钟

## 问题所在

一个编码智能体在对话的每个轮次都向 Claude 发送相同的 15,000 个 token 的系统提示词。20 个轮次在 $3/M 输入 token 的价格下，光输入成本就要 $0.90——还没计算用户的实际消息。乘以每天 10,000 个对话，账单就达到了每天 $9,000，全是从不改变的文本。

你无法缩短提示词而不损害质量。你无法避免发送它——模型在每个轮次都需要它。唯一的办法是停止为提供商已经见过的前缀支付全价。

这就是提示词缓存。Anthropic 在 2024 年 8 月发布了它（2025 年增加了 1 小时扩展 TTL 变体），OpenAI 在同年晚些时候将其自动化，Google 在 Gemini 1.5 推出时发布了显式上下文缓存，现在三者都在其前沿模型上将其作为一等功能提供。

## 核心概念

**机制。** 当请求的前缀与最近请求的前缀匹配时，提供商从上一次运行中提供 KV 缓存，而不是重新编码 token。第一次你支付少量写入溢价，之后每次都享受大幅读取折扣。

**2026 年三个提供商的方式。**

| 提供商 | API 风格 | 命中折扣 | 写入溢价 | 默认 TTL | 最小可缓存大小 |
|--------|---------|---------|---------|---------|------------|
| Anthropic | 内容块上的显式 `cache_control` 标记 | 输入减 90% | 25% 附加费 | 5 分钟（可扩展至 1 小时） | 1,024 token（Sonnet/Opus），2,048（Haiku） |
| OpenAI | 自动前缀检测 | 输入减 50% | 无 | 最多 1 小时（尽力而为） | 1,024 token |
| Google（Gemini） | 显式 `CachedContent` API | 按存储计费；读取约为正常价的 25% | 每 token·小时存储费 | 用户设置（默认 1 小时） | 4,096 token（Flash），32,768（Pro） |

**不变的规则。** 三者都只缓存前缀。如果任何 token 在请求之间有所不同，第一个不同 token 之后的所有内容都是未命中。将**稳定**部分放在顶部，**可变**部分放在底部。

### 缓存友好的布局

```
[系统提示词]          <-- 缓存这个
[工具定义]            <-- 缓存这个
[少样本示例]          <-- 缓存这个
[检索到的文档]        <-- 如果复用则缓存，否则不缓存
[对话历史]            <-- 缓存到上一轮次
[当前用户消息]        <-- 永不缓存（每次都不同）
```

违反顺序——将用户消息放在系统提示词上方，在少样本示例之间穿插动态检索——缓存永远不会命中。

### 盈亏平衡计算

Anthropic 25% 的写入溢价意味着缓存块必须被读取至少两次才能净节省资金。1 次写入 + 1 次读取平均每次请求成本为 0.675x（节省 32%）；1 次写入 + 10 次读取平均为 0.205x（节省 80%）。经验法则：缓存你预期在 TTL 内至少复用 3 次的任何内容。

## 构建它

### 步骤 1：带显式标记的 Anthropic 提示词缓存

```python
import anthropic

client = anthropic.Anthropic()

SYSTEM = [
    {
        "type": "text",
        "text": "You are a senior Python reviewer. Follow the rubric exactly.\n\n" + RUBRIC_15K_TOKENS,
        "cache_control": {"type": "ephemeral"},
    }
]

def review(code: str):
    return client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=SYSTEM,
        messages=[{"role": "user", "content": code}],
    )
```

`cache_control` 标记告诉 Anthropic 将该块存储 5 分钟。在该窗口内复用会命中；超时后重新写入。

**响应用量字段：**

```python
response = review(code_a)
response.usage
# InputTokensUsage(
#     input_tokens=120,
#     cache_creation_input_tokens=15023,   # 按 1.25x 收费
#     cache_read_input_tokens=0,
#     output_tokens=340,
# )

response_b = review(code_b)
response_b.usage
# cache_creation_input_tokens=0
# cache_read_input_tokens=15023           # 按 0.1x 收费
```

在 CI 中检查两个字段——如果 `cache_read_input_tokens` 在请求间始终为零，说明你的缓存键在漂移。

### 步骤 2：1 小时扩展 TTL

对于长时间运行的批处理作业，5 分钟的默认 TTL 会在作业之间过期。设置 `ttl`：

```python
{"type": "text", "text": RUBRIC, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
```

1 小时 TTL 花费 2 倍写入溢价（基准价的 50% 而不是 25%），但对于任何复用前缀超过 5 次的批处理，回报很快。

### 步骤 3：OpenAI 自动缓存

OpenAI 没有给你任何可配置的东西。任何超过 1,024 个 token 且与最近请求匹配的前缀都会自动获得 50% 的折扣。

```python
from openai import OpenAI
client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},   # 长且稳定
        {"role": "user", "content": user_msg},
    ],
)
resp.usage.prompt_tokens_details.cached_tokens  # 享受折扣的部分
```

同样的缓存友好布局规则适用。有两件事会破坏 OpenAI 的缓存但不会破坏 Anthropic 的：更改 `user` 字段（用作缓存键组件）和重新排序工具。

### 步骤 4：Gemini 显式上下文缓存

Gemini 将缓存视为你创建和命名的一等对象：

```python
from google import genai
from google.genai import types

client = genai.Client()

cache = client.caches.create(
    model="gemini-3-pro",
    config=types.CreateCachedContentConfig(
        display_name="rubric-v3",
        system_instruction=RUBRIC,
        contents=[FEW_SHOT_EXAMPLES],
        ttl="3600s",
    ),
)

resp = client.models.generate_content(
    model="gemini-3-pro",
    contents=["Review this code:\n" + code],
    config=types.GenerateContentConfig(cached_content=cache.name),
)
```

Gemini 在缓存存活期间按 token·小时收取存储费，读取时约为正常输入价格的 25%。当你在数天内跨多个会话复用相同的大型提示词时，这是合适的选择。

### 步骤 5：在生产中测量命中率

查看 `code/main.py` 了解模拟的三提供商记账器，它追踪写入/读取/未命中次数并计算每 1K 请求的混合成本。在目标命中率上设置部署门槛——大多数生产 Anthropic 设置在预热后应看到超过 80% 的读取比例。

## 2026 年仍在出现的陷阱

- **顶部的动态时间戳。** 系统提示词顶部的 `"当前时间：2026-04-22 15:30:02"`。每次请求都未命中。将时间戳移到缓存断点以下。
- **工具重新排序。** 以稳定顺序序列化工具——两次部署之间的字典重组会破坏每次命中。
- **自由文本近似重复。** "You are helpful." vs "You are a helpful assistant."——一个字节的差异 = 完全未命中。
- **太小的块。** Anthropic 强制执行 1,024 个 token 的下限（Haiku 为 2,048）。较小的块会悄悄地不缓存。
- **盲目的成本仪表板。** 将"输入 token"分为已缓存和未缓存。否则流量下降看起来像缓存胜利。

## 使用它

2026 年缓存技术栈：

| 情况 | 选择 |
|------|------|
| 带稳定 10k+ 系统提示词、多轮次的智能体 | Anthropic `cache_control`，5 分钟 TTL |
| 复用前缀 30+ 分钟的批处理作业 | Anthropic，`ttl: "1h"` |
| GPT-5 上的无服务器端点，无自定义基础设施 | OpenAI 自动（只需让你的前缀稳定且长） |
| 多天复用巨大的代码/文档语料库 | Gemini 显式 `CachedContent` |
| 跨提供商回退 | 保持可缓存的前缀布局在提供商之间完全相同，这样任何命中都有效 |

与语义缓存（Phase 11 · 11）结合用于用户消息层：提示词缓存处理**token 完全相同**的复用，语义缓存处理**含义相同**的复用。

## 练习

1. **简单。** 用 5,000 个 token 的系统提示词与 Claude 进行 10 轮对话。不使用 `cache_control` 运行，然后使用。报告每种情况下的输入 token 账单。
2. **中等。** 编写一个测试工具，给定提示词模板和请求日志，计算每个提供商（Anthropic 5 分钟、Anthropic 1 小时、OpenAI 自动、Gemini 显式）的预期命中率和美元节省。
3. **困难。** 构建一个布局优化器：给定提示词和标记了 `stable=True/False` 的字段列表，重写提示词以在最大缓存友好位置放置单个缓存断点而不丢失信息。在真实的 Anthropic 端点上验证。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 提示词缓存（Prompt caching） | "让长提示词变便宜" | 为匹配的前缀复用提供商侧 KV 缓存；重复输入 token 享受 50-90% 折扣 |
| `cache_control` | "Anthropic 标记" | 声明"到此为止的所有内容都可缓存"的内容块属性；`{"type": "ephemeral"}` |
| 缓存写入（Cache write） | "支付溢价" | 填充缓存的第一次请求；在 Anthropic 以约 1.25x 输入价格计费，OpenAI 免费 |
| 缓存读取（Cache read） | "折扣" | 匹配前缀的后续请求；以 10%（Anthropic）、50%（OpenAI）、约 25%（Gemini）计费 |
| TTL | "它能存活多久" | 缓存保持热状态的秒数；Anthropic 默认 5 分钟（可扩展至 1 小时），OpenAI 尽力而为最多 1 小时，Gemini 用户设置 |
| 扩展 TTL（Extended TTL） | "1 小时 Anthropic 缓存" | `{"type": "ephemeral", "ttl": "1h"}`；2 倍写入溢价，但对于批量复用超过 5 次的情况物超所值 |
| 前缀匹配（Prefix match） | "为什么我的缓存未命中" | 只有当从开始到断点的每个 token 都字节完全相同时，缓存才会命中 |
| 上下文缓存（Context caching，Gemini） | "显式的那个" | Google 的命名、按存储计费的缓存对象；最适合大型语料库的多天复用 |

## 延伸阅读

- [Anthropic — 提示词缓存](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — `cache_control`、1 小时 TTL、盈亏平衡表
- [OpenAI — 提示词缓存](https://platform.openai.com/docs/guides/prompt-caching) — 自动前缀匹配
- [Google — 上下文缓存](https://ai.google.dev/gemini-api/docs/caching) — `CachedContent` API 和存储定价
- [Anthropic 工程 — 长上下文工作负载的提示词缓存](https://www.anthropic.com/news/prompt-caching) — 带延迟数字的原始发布文章
- Phase 11 · 05（上下文工程）— 如何切分提示词以便缓存生效
- Phase 11 · 11（缓存与成本）— 将提示词缓存与用户消息上的语义缓存配对
- [Pope 等，"高效扩展 Transformer 推理"（2022）](https://arxiv.org/abs/2211.05102) — 提示词缓存向用户暴露的 KV 缓存内存模型；解释为什么缓存前缀重新读取比重新计算便宜约 10 倍
- [Agrawal 等，"SARATHI：通过分块预填充搭载解码的高效 LLM 推理"（2023）](https://arxiv.org/abs/2308.16369) — 预填充是提示词缓存跳过的阶段；这篇论文解释为什么缓存命中时 TTFT 大幅下降而 TPOT 不受影响
- [Leviathan 等，"通过推测解码实现 Transformer 快速推理"（2023）](https://arxiv.org/abs/2211.17192) — 提示词缓存与推测解码、Flash Attention 和 MQA/GQA 并列，作为弯曲推理成本曲线的杠杆；阅读此文了解其他三个
