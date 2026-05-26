# 近端策略优化（PPO）

> A2C 每次展开只更新一次就丢弃数据。PPO 用截断的重要性比率包裹策略梯度，让你可以在同一数据上做 10+ 轮训练而不让策略爆炸。Schulman 等（2017）。2026 年仍是默认的策略梯度算法。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 9 · 06（REINFORCE）、Phase 9 · 07（演员-评论家）
**时长：** 约 75 分钟

## 问题背景

A2C（第 07 课）是在策略的：梯度 `E_{π_θ}[A · ∇ log π_θ]` 需要从*当前* `π_θ` 采样的数据。执行一次更新后，`π_θ` 改变了；你使用的数据现在是离策略的。重用它，梯度就有偏差。

展开数据很昂贵。在 Atari 上，8 个环境 × 128 步的一次展开 = 1024 个转移，花费十几秒的环境时间。用一步梯度就丢弃这些数据太浪费了。

信任域策略优化（TRPO，Schulman 2015）是第一个修复方案：约束每次更新，使旧策略和新策略之间的 KL 散度保持在 `δ` 以下。理论上干净，但每次更新需要共轭梯度求解。2026 年没人运行 TRPO 了。

PPO（Schulman 等 2017）用简单的截断目标替换了硬信任域约束。多一行代码，每次展开十轮训练，无需共轭梯度，有足够好的理论保证。九年后，它仍然是从 MuJoCo 到 RLHF 一切场景的默认策略梯度算法。

## 核心概念

![PPO 截断代理目标：在 1 ± ε 处截断比率](../assets/ppo.svg)

**重要性比率（Importance ratio）。**

`r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t)`

这是新策略相对于收集数据的策略的似然比。`r_t = 1` 表示无变化。`r_t = 2` 表示新策略采取 `a_t` 的可能性是旧策略的两倍。

**截断代理目标（Clipped surrogate）。**

`L^{CLIP}(θ) = E_t [ min( r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t ) ]`

两项：

- 如果优势 `A_t > 0` 且比率试图超过 `1 + ε`，截断使梯度变平——不要把好的动作推到旧概率的 `+ε` 之上。
- 如果优势 `A_t < 0` 且比率试图超过 `1 - ε`（意味着我们会使坏动作相比截断后更可能发生），截断限制梯度——不要把坏动作推到 `-ε` 之下。

`min` 处理另一个方向：如果比率已经朝着*有益*方向移动，你仍然得到梯度（在不会伤害你的一侧没有截断）。

典型值 `ε = 0.2`。将目标绘制为 `r_t` 的函数：一个分段线性函数，在"好的一侧"有平顶，在"坏的一侧"有平底。

**完整的 PPO 损失：**

`L(θ, φ) = L^{CLIP}(θ) - c_v · (V_φ(s_t) - V_t^{target})² + c_e · H(π_θ(·|s_t))`

与 A2C 相同的演员-评论家结构。三个系数，通常 `c_v = 0.5`，`c_e = 0.01`，`ε = 0.2`。

**训练循环：**

1. 跨 `N` 个并行环境，每个 `T` 步，收集 `N × T` 个转移。
2. 计算优势（GAE），将其作为常数冻结。
3. 将 `π_{θ_old}` 作为当前 `π_θ` 的快照冻结。
4. 对 `K` 轮，对每个包含 `(s, a, A, V_target, log π_old(a|s))` 的小批量：
   - 计算 `r_t(θ) = exp(log π_θ(a|s) - log π_old(a|s))`。
   - 应用 `L^{CLIP}` + 价值损失 + 熵。
   - 梯度步。
5. 丢弃展开数据。返回步骤 1。

`K = 10` 和 64 的小批量大小是标准超参数组合。PPO 鲁棒：确切数字在 ±50% 内通常无关紧要。

**KL 惩罚变体。** 原始论文提出了一种使用自适应 KL 惩罚的替代方案：`L = L^{PG} - β · KL(π_θ || π_old)`，`β` 根据观察到的 KL 调整。截断版本占据主导；KL 变体在 RLHF 中延续（在那里，对参考策略的 KL 本来就是一个你始终想要的单独约束）。

## 动手实现

### 步骤一：在展开时捕获 `log π_old(a | s)`

```python
for step in range(T):
    probs = softmax(logits(theta, state_features(s)))
    a = sample(probs, rng)
    s_next, r, done = env.step(s, a)
    buffer.append({
        "s": s, "a": a, "r": r, "done": done,
        "v_old": value(w, state_features(s)),
        "log_pi_old": log(probs[a] + 1e-12),
    })
    s = s_next
```

快照在展开时一次性获取。在更新轮次期间不改变。

### 步骤二：计算 GAE 优势（第 07 课）

与 A2C 相同。在批量内归一化。

### 步骤三：截断代理更新

```python
for _ in range(K_EPOCHS):
    for mb in minibatches(buffer, size=64):
        for rec in mb:
            x = state_features(rec["s"])
            probs = softmax(logits(theta, x))
            logp = log(probs[rec["a"]] + 1e-12)
            ratio = exp(logp - rec["log_pi_old"])
            adv = rec["advantage"]
            surrogate = min(
                ratio * adv,
                clamp(ratio, 1 - EPS, 1 + EPS) * adv,
            )
            # 反向传播 -surrogate，添加价值损失，减去熵
            grad_logpi = onehot(rec["a"]) - probs
            if (adv > 0 and ratio >= 1 + EPS) or (adv < 0 and ratio <= 1 - EPS):
                pg_grad = 0.0  # 被截断
            else:
                pg_grad = ratio * adv
            for i in range(N_ACTIONS):
                for j in range(N_FEAT):
                    theta[i][j] += LR * pg_grad * grad_logpi[i] * x[j]
```

"截断 → 零梯度"模式是 PPO 的核心。如果新策略在有益方向上已经偏移太远，更新就停止。

### 步骤四：价值和熵

为评论家目标添加标准 MSE，并为演员添加熵奖励，与 A2C 相同。

### 步骤五：诊断

每次更新要观察三件事：

- **平均 KL** `E[log π_old - log π_θ]`。应该保持在 `[0, 0.02]`。如果超过 `0.1`，减少 `K_EPOCHS` 或 `LR`。
- **截断比例（Clip fraction）**——比率超出 `[1-ε, 1+ε]` 的样本比例。应该约为 `~0.1-0.3`。如果约为 `~0`，截断从未触发 → 提高 `LR` 或 `K_EPOCHS`。如果约为 `~0.5+`，你对展开数据过拟合了 → 降低它们。
- **解释方差（Explained variance）** `1 - Var(V_target - V_pred) / Var(V_target)`。评论家质量指标。应该随着评论家学习而趋向 1。

## 常见陷阱

- **截断系数调错。** `ε = 0.2` 是实际标准。降到 `0.1` 使更新过于谨慎；`0.3+` 带来不稳定。
- **训练轮数太多。** `K > 20` 经常导致不稳定，因为策略从 `π_old` 漂移太远。限制轮数，尤其对大型网络。
- **没有奖励归一化。** 大奖励尺度会吃掉截断范围。在计算优势之前归一化奖励（运行标准差）。
- **忘记优势归一化。** 每批零均值/单位标准差归一化是标准做法。跳过它会毁掉 PPO 在大多数基准上的表现。
- **学习率未衰减。** PPO 受益于线性 LR 衰减到零。常数 LR 通常更差。
- **重要性比率数学错误。** 始终用 `exp(log_new - log_old)` 以保证数值稳定性，而非 `new / old`。
- **梯度符号错误。** 最大化代理目标 = *最小化* `-L^{CLIP}`。符号翻转是最常见的 PPO bug。

## 生产使用

PPO 是 2026 年令人惊讶的广泛领域的默认强化学习算法：

| 使用场景 | PPO 变体 |
|---------|---------|
| MuJoCo / 机器人控制 | 带高斯策略的 PPO，GAE(0.95) |
| Atari / 离散游戏 | 带分类策略的 PPO，滚动 128 步展开 |
| LLM 的 RLHF | 带对参考模型 KL 惩罚的 PPO，奖励来自回复末尾的 RM |
| 大规模游戏智能体 | IMPALA + PPO（AlphaStar，OpenAI Five） |
| 推理 LLM | GRPO（第 12 课）——无评论家的 PPO 变体 |
| 仅偏好数据 | DPO——PPO+KL 的封闭形式折叠，无在线采样 |

PPO 的*损失形状*——截断代理 + 价值 + 熵——是 DPO、GRPO 以及几乎每个 RLHF 流水线的脚手架。

## 上手实践

保存为 `outputs/skill-ppo-trainer.md`：

```markdown
---
name: ppo-trainer
description: Produce a PPO training config and a diagnostic plan for a given environment.
version: 1.0.0
phase: 9
lesson: 8
tags: [rl, ppo, policy-gradient]
---

给定环境和训练预算，输出：

1. 展开大小。`N` 个环境 × `T` 步。
2. 更新调度。`K` 轮，小批量大小，LR 调度。
3. 代理参数。`ε`（截断）、`c_v`、`c_e`、开启优势归一化。
4. 优势。GAE(`λ`)，明确指定 `γ` 和 `λ`。
5. 诊断计划。KL、截断比例、解释方差阈值及警报。

拒绝 `K > 30` 或 `ε > 0.3`（不安全的信任域）。拒绝任何没有优势归一化或 KL/截断监控的 PPO 运行。将截断比例持续高于 0.4 标记为漂移。
```

## 练习

1. **简单。** 在 4×4 GridWorld 上运行 PPO（`ε=0.2, K=4`）。在匹配的环境步数下，比较与 A2C（每次展开一轮）的样本效率。
2. **中等。** 扫描 `K ∈ {1, 4, 10, 30}`。绘制回报 vs 环境步数，并追踪每次更新的平均 KL。在这个任务上，KL 在哪个 `K` 时爆炸？
3. **困难。** 用自适应 KL 惩罚替换截断代理（`KL > 2·target` 时 `β` 加倍，`KL < target/2` 时减半）。比较最终回报、稳定性和无截断效果。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 重要性比率（Importance ratio） | "r_t(θ)" | `π_θ(a|s) / π_old(a|s)`；与收集数据的策略的偏差。 |
| 截断代理（Clipped surrogate） | "PPO 的主要技巧" | `min(r·A, clip(r, 1-ε, 1+ε)·A)`；在有益侧超过截断后梯度变平。 |
| 信任域（Trust region） | "TRPO / PPO 意图" | 限制每次更新的 KL 以保证单调改进。 |
| KL 惩罚（KL penalty） | "软信任域" | 替代 PPO：`L - β · KL(π_θ || π_old)`。自适应 `β`。 |
| 截断比例（Clip fraction） | "截断触发频率" | 诊断指标——应为 0.1-0.3；超出意味着调错了。 |
| 多轮训练（Multi-epoch training） | "数据复用" | 对每次展开做 K 轮；以方差成本换取样本效率。 |
| 准在策略（On-policy-ish） | "大致在策略" | PPO 名义上是在策略的，但 K>1 轮使用略微离策略的数据，但安全。 |
| PPO-KL | "另一个 PPO" | KL 惩罚变体；在 RLHF 中使用，此时对参考模型的 KL 已经是约束。 |

## 延伸阅读

- [Schulman et al. (2017). Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — 原始论文
- [Schulman et al. (2015). Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) — TRPO，PPO 的前身
- [Andrychowicz et al. (2021). What Matters In On-Policy RL? A Large-Scale Empirical Study](https://arxiv.org/abs/2006.05990) — 每个 PPO 超参数的消融研究
- [Ouyang et al. (2022). Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — InstructGPT；PPO 在 RLHF 中的配方
- [OpenAI Spinning Up — PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) — 带 PyTorch 的清晰现代解说
- [CleanRL PPO implementation](https://github.com/vwxyzjn/cleanrl) — 许多论文使用的参考单文件 PPO
- [Hugging Face TRL — PPOTrainer](https://huggingface.co/docs/trl/main/en/ppo_trainer) — 语言模型上 PPO 的生产配方；与第 09 课（RLHF）配套阅读
- [Engstrom et al. (2020). Implementation Matters in Deep Policy Gradients](https://arxiv.org/abs/2005.12729) — "37 个代码级优化"论文；哪些 PPO 技巧是关键的，哪些只是传说
