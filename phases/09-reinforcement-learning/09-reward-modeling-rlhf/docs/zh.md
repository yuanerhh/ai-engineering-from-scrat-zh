# 奖励建模与 RLHF

> 人类无法为"好的助手回复"编写奖励函数，但可以比较两个回复并选出更好的那个。用这些比较拟合一个奖励模型，然后对语言模型进行强化学习。Christiano 2017，InstructGPT 2022。将 GPT-3 变成 ChatGPT 的配方。2026 年它大部分正被 DPO 取代——但思维模型依然适用。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 5 · 05（情感分析）、Phase 9 · 08（PPO）
**时长：** 约 45 分钟

## 问题背景

你在下一个 token 预测目标上训练了一个语言模型。它能写语法正确的英语，但也会撒谎、啰嗦，还不会拒绝不当请求。你无法通过更多预训练来修复这个问题——互联网文本是问题所在，不是解药。

你想要一个*标量奖励*，能够说"对于指令 X，回复 A 比回复 B 更好"。手工编写这个奖励函数是不可能的。"有用性"不是 token 上的封闭表达式。但人类可以比较两个输出并标记偏好，这种比较可以大规模廉价地收集。

RLHF（Christiano 等 2017；Ouyang 等 2022）将偏好转化为奖励模型，然后通过 PPO 对语言模型进行优化。三个步骤：SFT → RM → PPO。这是推出 ChatGPT、Claude、Gemini 以及 2023–2025 年每个对齐语言模型的配方。

2026 年，PPO 步骤大部分被 DPO（Phase 10 · 08）取代，因为对对齐调优来说它更便宜且效果几乎相同。但*奖励模型*部分仍然是每个 Best-of-N 采样器、每个来自可验证奖励的强化学习流水线以及每个使用过程奖励模型的推理模型的基础。理解 RLHF，就理解了整个对齐技术栈。

## 核心概念

![三阶段 RLHF：SFT、基于配对偏好的 RM 训练、带 KL 惩罚的 PPO](../assets/rlhf.svg)

**第一阶段：监督微调（SFT，Supervised Fine-Tuning）。** 从预训练基础模型开始。在人类编写的目标行为示范（指令跟随回复、有用的答复等）上微调。结果：一个*倾向于好行为*但动作空间仍然无界的模型 `π_SFT`。

**第二阶段：奖励模型训练。**

- 收集对提示词 `x` 的回复对 `(y_+, y_-)`，由人类标注为"y_+ 优于 y_-"。
- 训练奖励模型 `R_φ(x, y)` 为 `y_+` 分配更高分数。
- 损失函数：**Bradley-Terry 配对逻辑斯谛（pairwise logistic）**：

  `L(φ) = -E[ log σ(R_φ(x, y_+) - R_φ(x, y_-)) ]`

  σ 是 sigmoid。奖励差值意味着偏好的对数几率。BT 自 1952 年（Bradley-Terry）以来一直是标准，也是现代 RLHF 的主流选择。

- `R_φ` 通常从 SFT 模型初始化，顶部加一个标量头。相同的 Transformer 主干；一个单线性层输出奖励。

**第三阶段：带 KL 惩罚的 PPO 对 RM 进行优化。**

- 从 `π_SFT` 初始化可训练策略 `π_θ`。保留冻结的*参考模型* `π_ref = π_SFT`。
- 回复 `y` 末尾的奖励：

  `r_total(x, y) = R_φ(x, y) - β · KL(π_θ(·|x) || π_ref(·|x))`

  KL 惩罚防止 `π_θ` 任意偏离 `π_SFT`——它是一个*正则化器*，而非硬信任域。`β` 通常为 `0.01`-`0.05`。
- 使用这个奖励运行 PPO（第 08 课）。优势在 token 级轨迹上计算，但 RM 只对完整回复打分。

**为什么要 KL？** 没有它，PPO 会愉快地找到奖励黑客（reward hacking）策略——RM 只在分布内的补全上训练过。一个分布外的回复可能得分高于任何人类编写的内容。KL 将 `π_θ` 保持在 RM 训练时所在的流形附近。这是 RLHF 中最重要的旋钮。

**2026 年现状：**

- **DPO**（Rafailov 2023）：封闭形式代数将第 2+3 阶段折叠为偏好数据上的单一监督损失。无需 RM，无需 PPO。对对齐基准的质量相同，计算量只需一小部分。在 Phase 10 · 08 中介绍。
- **GRPO**（DeepSeek 2024–2025）：带组相对基线的 PPO，奖励来自*验证器*（代码运行 / 数学答案匹配）而非人类训练的 RM。推理模型的主流。在 Phase 9 · 12 中介绍。
- **过程奖励模型（PRMs，Process reward models）：** 对部分解（每个推理步骤）打分，用于推理的 RLHF 和 GRPO 变体。
- **宪法 AI（Constitutional AI）/ RLAIF：** 用对齐的语言模型生成偏好，而非人类。扩展了偏好预算。

## 动手实现

本课使用表示为字符串的微型合成"提示词"和"回复"。RM 是词袋表示上的线性评分器。没有真实的语言模型——流水线的*形状*才是重点，而非规模。见 `code/main.py`。

### 步骤一：合成偏好数据

```python
PROMPTS = ["help me", "answer me", "explain this"]
GOOD_WORDS = {"clear", "specific", "kind", "thorough"}
BAD_WORDS = {"vague", "rude", "wrong", "short"}

def make_pair(rng):
    x = rng.choice(PROMPTS)
    y_good = rng.choice(list(GOOD_WORDS)) + " " + rng.choice(list(GOOD_WORDS))
    y_bad = rng.choice(list(BAD_WORDS)) + " " + rng.choice(list(BAD_WORDS))
    return (x, y_good, y_bad)
```

在真实 RLHF 中，这由人类标注者替代。形状——`(提示词, 偏好回复, 拒绝回复)`——是相同的。

### 步骤二：Bradley-Terry 奖励模型

线性分数：`R(x, y) = w · bag(y)`。训练最小化 BT 配对对数损失：

```python
def rm_train_step(w, x, y_pos, y_neg, lr):
    r_pos = dot(w, bag(y_pos))
    r_neg = dot(w, bag(y_neg))
    p = sigmoid(r_pos - r_neg)
    for tok, cnt in bag(y_pos).items():
        w[tok] += lr * (1 - p) * cnt
    for tok, cnt in bag(y_neg).items():
        w[tok] -= lr * (1 - p) * cnt
```

几百次更新后，`w` 为好词 token 分配正权重，为坏词分配负权重。

### 步骤三：基于 RM 的类 PPO 策略

我们的玩具策略从词汇表中产生单个 token。我们用 RM 给 token 打分，计算 `log π_θ(token | prompt)`，添加对参考模型的 KL 惩罚，并应用截断的 PPO 代理目标。

```python
def rlhf_step(theta, ref, w, prompt, rng, eps=0.2, beta=0.1, lr=0.05):
    logits_theta = policy_logits(theta, prompt)
    probs = softmax(logits_theta)
    token = sample(probs, rng)
    logits_ref = policy_logits(ref, prompt)
    probs_ref = softmax(logits_ref)
    reward = dot(w, bag([token])) - beta * kl(probs, probs_ref)
    # 对 theta 进行 PPO 风格更新，将 reward 视为回报
    ...
```

### 步骤四：监控 KL

每次更新追踪平均 `KL(π_θ || π_ref)`。如果它悄悄超过 `~5-10`，策略已经大幅偏离 `π_SFT`——`β` 上升太慢或奖励黑客正在开始。这是真实 RLHF 中的首要诊断指标。

### 步骤五：使用 TRL 的生产配方

一旦你理解了玩具流水线，这就是真实库用户编写的同一循环。Hugging Face 的 [TRL](https://huggingface.co/docs/trl) 是参考实现——`RewardTrainer` 用于第 2 阶段，`PPOTrainer`（内置 KL-对参考模型）用于第 3 阶段。

```python
# 第 2 阶段：从配对偏好训练奖励模型
from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
rm = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", num_labels=1
)

# 数据集行：{"prompt", "chosen", "rejected"} — Bradley-Terry 格式
trainer = RewardTrainer(
    model=rm,
    tokenizer=tok,
    train_dataset=preference_data,
    args=RewardConfig(output_dir="./rm", num_train_epochs=1, learning_rate=1e-5),
)
trainer.train()
```

```python
# 第 3 阶段：带对 SFT 参考模型 KL 惩罚的 PPO 对 RM 优化
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

policy = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")
ref    = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")  # 冻结

ppo = PPOTrainer(
    config=PPOConfig(learning_rate=1.41e-5, batch_size=64, init_kl_coef=0.05,
                     target_kl=6.0, adap_kl_ctrl=True),
    model=policy, ref_model=ref, tokenizer=tok,
)

for batch in dataloader:
    responses = ppo.generate(batch["query_ids"], max_new_tokens=128)
    rewards   = rm(torch.cat([batch["query_ids"], responses], dim=-1)).logits[:, 0]
    stats     = ppo.step(batch["query_ids"], responses, rewards)
    # stats 包含：mean_kl、clip_frac、value_loss——三个 PPO 诊断指标
```

库为你做三件事。`adap_kl_ctrl=True` 实现自适应 β 调度：如果观察到的 KL 超过 `target_kl`，β 加倍；如果低于一半，β 减半。参考模型按惯例冻结——你不能意外地与 `policy` 共享参数。而价值头与策略共享相同的主干（`AutoModelForCausalLMWithValueHead` 附加了一个标量 MLP 头），这就是为什么 TRL 分别报告 `policy/kl` 和 `value/loss`。

## 常见陷阱

- **过度优化 / 奖励黑客（Reward hacking）。** RM 是不完美的；`π_θ` 找到得分高但实际很差的对抗性补全。症状：奖励无限上升，而人工评估分数停滞或下降。修复：提前停止，提高 `β`，拓宽 RM 训练数据。
- **长度黑客（Length hacking）。** 在有用回复上训练的 RM 通常隐式地奖励长度。策略学会填充回复。补救：长度归一化奖励，或带长度感知 RM 的 RLAIF。
- **RM 太小。** RM 需要至少与策略一样大。小型 RM 无法忠实地给策略的输出打分。
- **KL 调整。** β 太低 → 漂移和奖励黑客。β 太高 → 策略几乎不变。标准技巧是自适应 β，以每步固定 KL 为目标。
- **偏好数据噪声。** 约 30% 的人类标注有噪声或歧义。通过在一致性过滤的数据上训练 RM 来校准，或在 BT 上使用温度。
- **离策略问题。** 第一轮之后，PPO 数据略微离策略。如第 08 课所示，监控截断比例。

## 生产使用

2026 年的 RLHF 是分层的：

| 层次 | 目标 | 方法 |
|------|------|------|
| 指令跟随、有用性、无害性 | 对齐 | DPO（Phase 10 · 08）优先于 RLHF-PPO。 |
| 推理正确性（数学、代码） | 能力 | 带验证器奖励的 GRPO（Phase 9 · 12）。 |
| 长视野多步任务 | 智能体 | 带逐步过程奖励模型的 PPO / GRPO。 |
| 安全 / 拒绝行为 | 安全 | 带独立安全 RM 的 RLHF-PPO，或宪法 AI。 |
| 推理时 Best-of-N | 快速对齐 | 解码时使用 RM；无需策略训练。 |
| 奖励蒸馏 | 推理计算 | 在冻结语言模型上训练小型"奖励头"。 |

RLHF 是 2022–2024 年的*主流*方法。2026 年，生产对齐流水线以 DPO 为主，仅对需要 RM 或安全关键步骤才用 PPO。

## 上手实践

保存为 `outputs/skill-rlhf-architect.md`：

```markdown
---
name: rlhf-architect
description: Design an RLHF / DPO / GRPO alignment pipeline for a language model, including RM, KL, and data strategy.
version: 1.0.0
phase: 9
lesson: 9
tags: [rl, rlhf, alignment, llm]
---

给定基础语言模型、目标行为（对齐 / 推理 / 拒绝 / 智能体）以及偏好或验证器预算，输出：

1. 阶段。SFT？RM？DPO？GRPO？附理由。
2. 偏好或验证器来源。人类、AI 反馈、基于规则、单元测试通过或奖励蒸馏。
3. KL 策略。固定 β、自适应 β 或 DPO（隐式 KL）。
4. 诊断。平均 KL、奖励稳定性、过度优化保护（保留的人工评估集）。
5. 安全门。红队测试集、拒绝率、与有用性 RM 分开的安全 RM。

拒绝在没有 KL 监控的情况下发布 RLHF-PPO。拒绝使用小于目标策略的 RM。拒绝纯长度奖励。将任何不保留盲测人工评估集的流水线标记为缺乏过度优化保护。
```

## 练习

1. **简单。** 在 `code/main.py` 中用 500 个合成偏好对训练 Bradley-Terry 奖励模型。在保留的 100 对上测量配对准确率。应超过 90%。
2. **中等。** 用 `β ∈ {0.0, 0.1, 1.0}` 运行玩具 PPO-RLHF 循环。对每个 β，绘制 RM 分数 vs 对参考模型的 KL 随更新次数的曲线。哪次运行出现了奖励黑客？
3. **困难。** 在同一偏好数据上实现 DPO（封闭形式偏好似然损失），并与 RLHF-PPO 流水线在计算量使用和最终 RM 分数上进行比较。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| RLHF | "对齐强化学习" | 三阶段 SFT + RM + PPO 流水线（Christiano 2017，Ouyang 2022）。 |
| 奖励模型（RM，Reward Model） | "评分网络" | 通过 Bradley-Terry 拟合到配对偏好的学习标量函数。 |
| Bradley-Terry | "配对逻辑斯谛损失" | `P(y_+ ≻ y_-) = σ(R(y_+) - R(y_-))`；标准 RM 目标。 |
| KL 惩罚（KL penalty） | "保持接近参考" | 奖励中的 `β · KL(π_θ || π_ref)`；防奖励黑客正则化器。 |
| 奖励黑客（Reward hacking） | "古德哈特定律" | 策略利用 RM 缺陷；症状：奖励上升，人工评估平稳。 |
| RLAIF | "AI 标注的偏好" | RLHF，标注来自另一个语言模型而非人类。 |
| PRM | "过程奖励模型" | 给部分推理步骤打分；用于推理流水线。 |
| 宪法 AI（Constitutional AI） | "Anthropic 的方法" | 由明确规则引导的 AI 生成偏好。 |

## 延伸阅读

- [Christiano et al. (2017). Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) — 开创 RLHF 的论文
- [Ouyang et al. (2022). InstructGPT — Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — ChatGPT 背后的配方
- [Stiennon et al. (2020). Learning to summarize with human feedback](https://arxiv.org/abs/2009.01325) — 用于摘要的早期 RLHF
- [Rafailov et al. (2023). Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — DPO；2026 年的 RLHF 后继默认方法
- [Bai et al. (2022). Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — RLAIF 和自我批评循环
- [Anthropic RLHF paper (Bai et al. 2022). Training a Helpful and Harmless Assistant](https://arxiv.org/abs/2204.05862) — HH 论文
- [Hugging Face TRL library](https://huggingface.co/docs/trl) — 生产 `RewardTrainer` 和 `PPOTrainer`
- [Hugging Face — Illustrating Reinforcement Learning from Human Feedback](https://huggingface.co/blog/rlhf) — 带图示的三阶段流水线规范解读
- [von Werra et al. (2020). TRL: Transformer Reinforcement Learning](https://github.com/huggingface/trl) — 该库；`examples/` 有 Llama、Mistral 和 Qwen 的端到端 RLHF 脚本
- [Sutton & Barto (2018). Ch. 17.4 — Designing Reward Signals](http://incompleteideas.net/book/RLbook2020.pdf) — 奖励假设视角；思考奖励黑客的基本前提
