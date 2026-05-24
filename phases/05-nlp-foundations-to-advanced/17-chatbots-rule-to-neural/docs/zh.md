# 聊天机器人——从规则到神经到 LLM 智能体

> ELIZA 用模式匹配回复。DialogFlow 映射意图。GPT 从权重中作答。Claude 运行工具并验证结果。每个时代都解决了上一个时代最严重的失败。

**类型：** 学习
**语言：** Python
**前置条件：** Phase 5 · 13（问答系统）、Phase 5 · 14（信息检索）
**时长：** 约 75 分钟

## 问题背景

用户说"我想改签我的航班"。系统必须弄清楚用户想要什么，缺少哪些信息，如何获取，以及如何完成操作。然后用户说"等等，如果我取消改怎样？"，系统必须记住上下文，切换任务，并保持状态。

对于机器学习系统而言，对话是困难的。输入是开放式的，输出必须在多个回合中保持连贯，系统可能需要对外部世界采取行动（改签航班、扣款），每一个错误步骤对用户都是可见的。

聊天机器人架构经历了四种范式的演变，每一种都是因为上一种的失败过于明显而被引入。本课按顺序介绍它们。2026 年的生产场景是后两种的混合。

## 核心概念

![聊天机器人演进：规则 → 检索 → 神经 → 智能体](../assets/chatbot.svg)

**规则式（ELIZA、AIML、DialogFlow）。** 手工编写的模式匹配用户输入并生成回复。意图分类器路由到预定义的流程，槽填充（Slot-filling）状态机收集所需信息。在其设计的窄范围内效果出色，但超出范围立刻失效。在不容忍幻觉的安全关键领域（银行认证、机票预订）仍在部署。

**检索式。** FAQ 风格系统。编码每对（用户话语、回复），运行时编码用户消息并检索最近似的存储回复。类似 Zendesk 经典的"相似文章"功能。比规则更好地处理改写。不生成内容，因此不会产生幻觉。

**神经式（seq2seq）。** 在对话日志上训练的编码器-解码器，从零生成回复。流畅，但容易产生泛化输出（"我不知道"）和事实漂移，从不可靠地保持话题。这是 2016-2019 年谷歌、Facebook 和微软聊天机器人都令人失望的原因。

**LLM 智能体（Agents）。** 语言模型包裹在一个循环中，进行规划、调用工具并验证结果。不是一个有着长提示的聊天机器人，而是智能体循环：规划 → 调用工具 → 观察结果 → 决定下一步。检索优先的锚定（RAG）防止幻觉，工具调用让它真正能够执行操作。这是 2026 年的架构。

四种范式并非顺序替换。2026 年的生产聊天机器人路由经过所有四种：规则式用于认证和破坏性操作，检索式用于 FAQ，神经生成用于自然措辞，LLM 智能体用于模糊的开放式查询。

## 动手实现

### 步骤一：规则式模式匹配

```python
import re


class RulePattern:
    def __init__(self, pattern, response_template):
        self.regex = re.compile(pattern, re.IGNORECASE)
        self.template = response_template


PATTERNS = [
    RulePattern(r"my name is (\w+)", "Nice to meet you, {0}."),
    RulePattern(r"i (need|want) (.+)", "Why do you {0} {1}?"),
    RulePattern(r"i feel (.+)", "Why do you feel {0}?"),
    RulePattern(r"(.*)", "Tell me more about that."),
]


def rule_based_respond(user_input):
    for pattern in PATTERNS:
        m = pattern.regex.match(user_input.strip())
        if m:
            return pattern.template.format(*m.groups())
    return "I don't understand."
```

20 行实现的 ELIZA。反射技巧（"I feel sad" → "Why do you feel sad"）是 Weizenbaum 1966 年经典心理治疗师演示。至今仍有启发意义。

### 步骤二：检索式（FAQ）

此示例代码段需要 `pip install sentence-transformers`（会引入 torch）。本课的可运行 `code/main.py` 使用标准库的 Jaccard 相似度代替，因此无需外部依赖即可运行。

```python
from sentence_transformers import SentenceTransformer
import numpy as np


FAQ = [
    ("how do i reset my password", "Go to Settings > Security > Reset Password."),
    ("how do i cancel my order", "Go to Orders, find the order, click Cancel."),
    ("what is your return policy", "30-day returns on unused items, original packaging."),
]


encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
faq_questions = [q for q, _ in FAQ]
faq_embeddings = encoder.encode(faq_questions, normalize_embeddings=True)


def faq_respond(user_input, threshold=0.5):
    q_emb = encoder.encode([user_input], normalize_embeddings=True)[0]
    sims = faq_embeddings @ q_emb
    best = int(np.argmax(sims))
    if sims[best] < threshold:
        return None
    return FAQ[best][1]
```

基于阈值的拒绝是关键设计选择。如果最佳匹配不够接近，返回 `None` 并让系统升级处理。

### 步骤三：神经生成（基准线）

使用小型指令微调的编码器-解码器（FLAN-T5）或微调的对话模型。2026 年单独使用不适合生产（存在矛盾、话题漂移、事实胡说），但在混合系统中用于自然措辞。DialoGPT 风格的仅解码器模型需要明确的对话轮次分隔符和 EOS 处理才能产生连贯回复；FLAN-T5 的 text2text pipeline 开箱即用。

```python
from transformers import pipeline

chatbot = pipeline("text2text-generation", model="google/flan-t5-small")

response = chatbot("Respond politely to: Hi there!", max_new_tokens=40)
print(response[0]["generated_text"])
```

### 步骤四：LLM 智能体循环

2026 年生产形态：

```python
def agent_loop(user_message, tools, llm, max_steps=5):
    history = [{"role": "user", "content": user_message}]
    for _ in range(max_steps):
        response = llm(history, tools=tools)
        tool_call = response.get("tool_call")
        if tool_call:
            tool_name = tool_call.get("name")
            args = tool_call.get("arguments")
            if not isinstance(tool_name, str) or tool_name not in tools:
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": str(tool_name), "content": f"error: unknown tool {tool_name!r}"})
                continue
            if not isinstance(args, dict):
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": tool_name, "content": f"error: arguments must be a dict, got {type(args).__name__}"})
                continue
            fn = tools[tool_name]
            result = fn(**args)
            history.append({"role": "assistant", "tool_call": tool_call})
            history.append({"role": "tool", "name": tool_name, "content": result})
        else:
            return response["content"]
    return "I could not complete the task in the step budget."
```

三个要点：工具是 LLM 可以调用的可执行函数；当 LLM 返回最终答案而非工具调用时循环终止；步骤预算防止在模糊任务上无限循环。

实际生产还需添加：检索优先锚定（每次 LLM 调用前注入相关文档）、护栏（拒绝不经确认的破坏性操作）、可观测性（记录每个步骤）以及评估（自动检查智能体行为是否符合规范）。

### 步骤五：混合路由

```python
def hybrid_chat(user_input):
    if is_destructive_action(user_input):
        return structured_flow(user_input)

    faq_answer = faq_respond(user_input, threshold=0.6)
    if faq_answer:
        return faq_answer

    return agent_loop(user_input, tools, llm)


def is_destructive_action(text):
    danger_words = ["delete", "cancel", "charge", "refund", "transfer"]
    return any(w in text.lower() for w in danger_words)
```

模式：对任何破坏性操作用确定性规则，对固定 FAQ 用检索，其他一切用 LLM 智能体。这是 2026 年客服系统的标准部署形态。

## 生产使用

2026 年技术栈：

| 使用场景 | 架构 |
|---------|------|
| 预订、支付、认证 | 规则式状态机 + 槽填充 |
| 客服 FAQ | 对精心策划答案的检索 |
| 开放式帮助聊天 | LLM 智能体 + RAG + 工具调用 |
| 内部工具 / IDE 助手 | LLM 智能体 + 工具调用（搜索、读取、写入） |
| 伴侣 / 角色聊天机器人 | 带人设系统提示和知识检索的微调 LLM |

在生产中始终使用混合路由。没有任何单一架构能很好地处理所有请求。路由层本身通常是一个小型意图分类器。

## 仍在困扰生产的失败模式

- **自信捏造。** LLM 智能体声称完成了实际上未完成的操作。缓解措施：验证结果、记录工具调用，绝不让 LLM 在没有成功工具返回的情况下声称已完成某事。
- **提示注入（Prompt injection）。** 用户插入覆盖系统提示的文本，在 OWASP LLM 应用 Top 10（2025）中排名 LLM01。两种形式：直接注入（粘贴到聊天中）和间接注入（隐藏在智能体读取的文档、邮件或工具输出中）。

  攻击成功率因场景而异，在通用工具使用和编码基准中的前沿模型上约为 0.5-8.5%。针对 AI 编码智能体的自适应攻击和易受攻击的编排环境已达到约 84%。生产 CVE 包括 EchoLeak（CVE-2025-32711，CVSS 9.3）——Microsoft 365 Copilot 中由攻击者控制的邮件触发的零点击数据泄露漏洞。

  缓解措施：在整个循环中将用户输入视为不可信；工具调用前进行净化；将工具输出与主提示隔离；使用计划-验证-执行（PVE，Plan-Verify-Execute）模式（智能体先规划，然后在执行前对照计划验证每个操作——这可以阻止工具结果注入新的未计划操作）；对破坏性操作要求用户确认；对工具权限范围应用最小权限原则。

  任何提示工程都无法完全消除这种风险，需要外部运行时防御层（LLM Guard、白名单验证、语义异常检测）。
- **范围蔓延（Scope creep）。** 智能体因工具调用返回了切线相关信息而偏离任务。缓解措施：缩小工具契约；保持系统提示专注；添加对偏离任务率的评估。
- **无限循环。** 智能体持续调用同一工具。缓解措施：步骤预算、工具调用去重、用 LLM 评审"是否正在取得进展"。
- **上下文窗口耗尽。** 长对话将最早的回合推出上下文。缓解措施：对旧回合摘要、按相似度检索相关过去回合，或使用长上下文模型。

## 上手实践

将以下内容保存为 `outputs/skill-chatbot-architect.md`：

```markdown
---
name: chatbot-architect
description: Design a chatbot stack for a given use case.
version: 1.0.0
phase: 5
lesson: 17
tags: [nlp, agents, chatbot]
---

Given a product context (user need, compliance constraints, available tools, data volume), output:

1. Architecture. Rule-based, retrieval, neural, LLM agent, or hybrid (specify which paths go where).
2. LLM choice if applicable. Name the model family (Claude, GPT-4, Llama-3.1, Mixtral). Match to tool-use quality and cost.
3. Grounding strategy. RAG sources, retrieval method (see lesson 14), tool contracts.
4. Evaluation plan. Task success rate, tool-call correctness, off-task rate, hallucination rate on held-out dialogs.

Refuse to recommend a pure-LLM agent for any destructive action (payments, account deletion, data modification) without a structured confirmation flow. Refuse to skip the prompt-injection audit if the agent has write access to anything.
```

## 练习

1. **简单。** 用 10 个模式实现上述规则式回复，构建一个咖啡店订购机器人。测试边缘情况：重复订购、修改、取消、意图不清。
2. **中等。** 构建混合 FAQ + LLM 回退系统。为某 SaaS 产品准备 50 条固定 FAQ，使用文档站点上的检索作为 LLM 回退。在 100 个真实支持问题上衡量拒绝率和准确性。
3. **困难。** 实现上述智能体循环，配备三个工具（搜索、读取用户数据、发送邮件）。用 50 个测试场景（包括提示注入尝试）进行评估，报告偏离任务率、任务失败率以及任何注入成功情况。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 意图（Intent） | 用户想要什么 | 分类标签（book_flight、reset_password），路由到处理器。 |
| 槽（Slot） | 一条信息 | 机器人需要的参数（日期、目的地），槽填充是一系列提问的过程。 |
| RAG | 检索后生成 | 检索相关文档，然后基于这些文档锚定 LLM 的回复。 |
| 工具调用（Tool call） | 函数调用 | LLM 输出带有名称和参数的结构化调用，运行时执行并返回结果。 |
| 智能体循环（Agent loop） | 规划-行动-验证 | 控制器交替运行 LLM 调用和工具调用，直到任务完成。 |
| 提示注入（Prompt injection） | 用户攻击提示 | 试图覆盖系统提示的恶意输入。 |

## 延伸阅读

- [Weizenbaum (1966). ELIZA — A Computer Program For the Study of Natural Language Communication](https://web.stanford.edu/class/cs124/p36-weizenabaum.pdf) — 原始规则式聊天机器人论文
- [Thoppilan et al. (2022). LaMDA: Language Models for Dialog Applications](https://arxiv.org/abs/2201.08239) — 谷歌的晚期神经聊天机器人论文，就在 LLM 智能体接管之前
- [Yao et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — 命名智能体循环模式的论文
- [Anthropic's guide on building effective agents](https://www.anthropic.com/research/building-effective-agents) — 2024 年生产指导，2026 年仍适用
- [Greshake et al. (2023). Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection](https://arxiv.org/abs/2302.12173) — 提示注入论文
- [OWASP Top 10 for LLM Applications 2025 — LLM01 Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) — 将提示注入列为首要安全关切的排名
- [AWS — Securing Amazon Bedrock Agents against Indirect Prompt Injections](https://aws.amazon.com/blogs/machine-learning/securing-amazon-bedrock-agents-a-guide-to-safeguarding-against-indirect-prompt-injections/) — 实用的编排层防御，包括计划-验证-执行和用户确认流程
- [EchoLeak (CVE-2025-32711)](https://www.vectra.ai/topics/prompt-injection) — 间接提示注入的标准零点击数据泄露 CVE，说明为何具有写权限的智能体需要运行时防御
