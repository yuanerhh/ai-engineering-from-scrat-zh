# 时序差分学习——Q 学习与 SARSA

> 蒙特卡洛等待回合结束。TD 通过自举下一状态的价值估计，在每一步后更新。Q 学习是离策略且乐观的；SARSA 是在策略且保守的。两者都是一行代码。两者都是本阶段每个深度强化学习方法的基础。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 9 · 01（MDP）、Phase 9 · 02（动态规划）、Phase 9 · 03（蒙特卡洛）
**时长：** 约 75 分钟

## 问题背景

蒙特卡洛有效，但它有两个昂贵的要求：它需要能够终止的回合，而且只在最终回报到来后才更新。如果你的回合有 1,000 步，MC 要等 1,000 步才更新任何内容。它高方差、低偏差，实践中很慢。

动态规划则完全相反——零方差的自举备份——但需要已知模型。

时序差分（TD）学习折中两者。从一个单步转移 `(s, a, r, s')` 出发，形成单步目标 `r + γ V(s')`，将 `V(s)` 向它靠近。无需模型，无需完整回合。由于在右侧使用了近似的 `V` 而引入偏差，但方差比 MC 大幅降低，且从第一步起就能在线更新。

这是现代强化学习——DQN、A2C、PPO、SAC——全部立足的转折点。Phase 9 的其余部分都是在你将在本课中编写的单步 TD 更新上叠加的函数近似和技巧。

## 核心概念

![Q 学习 vs SARSA：离策略 max vs 在策略 Q(s', a')](../assets/td.svg)

**V 的 TD(0) 更新：**

`V(s) ← V(s) + α [r + γ V(s') - V(s)]`

括号内的量是 TD 误差（TD error）`δ = r + γ V(s') - V(s)`。它是 MC 中 `G_t - V(s_t)` 的在线类比。收敛需要 `α` 满足 Robbins-Monro 条件（`Σ α = ∞`，`Σ α² < ∞`）且所有状态被无限次访问。

**Q 学习（Q-learning）。** 一种用于控制的离策略 TD 方法：

`Q(s, a) ← Q(s, a) + α [r + γ max_{a'} Q(s', a') - Q(s, a)]`

`max` 假设从 `s'` 开始将遵循*贪婪*策略，无论智能体实际采取什么动作。这种解耦使 Q 学习在智能体通过 ε-贪婪探索时学习 `Q*`。Mnih 等（2015）将其转换为 Atari 上的深度 Q 学习（第 05 课）。

**SARSA。** 一种在策略 TD 方法：

`Q(s, a) ← Q(s, a) + α [r + γ Q(s', a') - Q(s, a)]`

名称来自元组 `(s, a, r, s', a')`。SARSA 使用智能体*实际*采取的下一个动作 `a'`，而非贪婪的 `argmax`。收敛到正在运行的任意 ε-贪婪策略 `π` 对应的 `Q^π`，极限 `ε → 0` 时变为 `Q*`。

**悬崖行走的差异。** 在经典悬崖行走任务中（掉落悬崖 = 奖励 -100），Q 学习学到沿悬崖边缘的最优路径，但在探索过程中偶尔会承受惩罚。SARSA 学到离悬崖一步距离的更安全路径，因为它将探索噪声纳入 Q 值。随着训练推进，`ε → 0` 时两者都达到最优。实践中这很重要：当部署时确实在进行探索时，SARSA 的行为更保守。

**期望 SARSA（Expected SARSA）。** 用 `Q(s', a')` 在策略 `π` 下的期望值替换它：

`Q(s, a) ← Q(s, a) + α [r + γ Σ_{a'} π(a'|s') Q(s', a') - Q(s, a)]`

方差比 SARSA 更低（无需对 `a'` 采样），相同的在策略目标。通常是现代教科书的默认选择。

**n 步 TD 与 TD(λ)。** 通过在自举前等待 `n` 步，在 TD(0) 和 MC 之间插值。`n=1` 是 TD，`n=∞` 是 MC。TD(λ) 用几何权重 `(1-λ)λ^{n-1}` 在所有 `n` 上取平均。大多数深度强化学习使用 `n` 在 3 到 20 之间。

## 动手实现

### 步骤一：ε-贪婪策略上的 SARSA

```python
def sarsa(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})

    def choose(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        s = env.reset()
        a = choose(s)
        while True:
            s_next, r, done = env.step(s, a)
            a_next = choose(s_next) if not done else None
            target = r + (gamma * Q[s_next][a_next] if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s, a = s_next, a_next
    return Q
```

八行代码。与 Q 学习*唯一*的区别在于目标行。

### 步骤二：Q 学习

```python
def q_learning(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    for _ in range(episodes):
        s = env.reset()
        while True:
            a = choose(s, Q, epsilon)
            s_next, r, done = env.step(s, a)
            target = r + (gamma * max(Q[s_next].values()) if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s = s_next
    return Q
```

`max` 将目标与行为解耦。就是这一个符号区分了在策略和离策略。

### 步骤三：学习曲线

每 100 个回合追踪平均回报。Q 学习在简单确定性 GridWorld 上收敛更快；SARSA 在悬崖行走上更保守。在 `code/main.py` 的 4×4 GridWorld 上，两者在约 2,000 个回合后接近最优（`α=0.1, ε=0.1`）。

### 步骤四：与 DP 真值比较

运行价值迭代（第 02 课）得到 `Q*`。检查 `max_{s,a} |Q_learned(s,a) - Q*(s,a)|`。健康的表格 TD 智能体在 10,000 个回合后，4×4 GridWorld 上误差约为 `~0.5`。

## 常见陷阱

- **初始 Q 值的重要性。** 乐观初始化（对负奖励任务 `Q = 0`）鼓励探索。悲观初始化可能永远困住贪婪策略。
- **α 调度。** 常数 `α` 适合非平稳问题。衰减 `α_n = 1/n` 理论上给出收敛保证，但实践中太慢——将 `α` 固定在 `[0.05, 0.3]` 并监控学习曲线。
- **ε 调度。** 从高开始（`ε=1.0`），衰减到 `ε=0.05`。"GLIE"（无限探索极限下贪婪，Greedy in the Limit with Infinite Exploration）是收敛条件。
- **Q 学习的最大偏差（max bias）。** `max` 算子在 `Q` 有噪声时向上偏。导致高估——Hasselt 的双 Q 学习（Double Q-learning，第 05 课 DDQN 使用）用两个 Q 表修复这个问题。
- **非终止回合。** TD 可以在没有终止状态的情况下学习，但需要限制步数或在限制处正确处理自举。标准做法：将上限视为非终止，继续自举。
- **状态哈希。** 如果状态是元组/张量，使用可哈希的键（tuple 而非 list；四舍五入的浮点数元组，而非原始值）。

## 生产使用

2026 年的 TD 应用格局：

| 任务 | 方法 | 原因 |
|------|------|------|
| 小型表格环境 | Q 学习 | 直接学习最优策略。 |
| 在策略安全关键场景 | SARSA / 期望 SARSA | 探索期间保守。 |
| 高维状态 | DQN（Phase 9 · 05） | 带回放和目标网络的神经网络 Q 函数。 |
| 连续动作 | SAC / TD3（Phase 9 · 07） | Q 网络上的 TD 更新；策略网络输出动作。 |
| LLM 强化学习（基于奖励模型） | PPO / GRPO（Phase 9 · 08, 12） | 带 GAE TD 风格优势的演员-评论家。 |
| 离线强化学习 | CQL / IQL（Phase 9 · 08） | 带保守正则化的 Q 学习。 |

2026 年论文中 90% 的"强化学习"都是 Q 学习或 SARSA 的某种演变。在深入阅读之前，先把表格更新烂熟于胸。

## 上手实践

保存为 `outputs/skill-td-agent.md`：

```markdown
---
name: td-agent
description: Pick between Q-learning, SARSA, Expected SARSA for a tabular or small-feature RL task.
version: 1.0.0
phase: 9
lesson: 4
tags: [rl, td-learning, q-learning, sarsa]
---

给定表格或小特征环境，输出：

1. 算法。Q 学习 / SARSA / 期望 SARSA / n 步变体。一句话理由，与在策略 vs 离策略和方差相关。
2. 超参数。α、γ、ε、衰减调度。
3. 初始化。Q_0 值（乐观 vs 零）及理由。
4. 收敛诊断。目标学习曲线，如果 DP 可行则进行 `|Q - Q*|` 检查。
5. 部署注意事项。推理时探索将如何表现？是否需要 SARSA 的保守性？

拒绝对状态空间 > 10⁶ 的问题应用表格 TD。拒绝在没有最大偏差警告的情况下发布 Q 学习智能体。标记任何训练中 ε 始终保持 1.0 的智能体（无利用阶段）。
```

## 练习

1. **简单。** 在 4×4 GridWorld 上实现 Q 学习和 SARSA。绘制 2,000 个回合内每 100 个回合的平均回报学习曲线。谁收敛更快？
2. **中等。** 构建一个悬崖行走环境（4×12，最后一行是悬崖，奖励 -100 并重置到起点）。比较 Q 学习和 SARSA 的最终策略。截图每个走过的路径。谁离悬崖更近？
3. **困难。** 实现双 Q 学习（Double Q-learning）。在带噪声奖励的 GridWorld 上（每步奖励加高斯噪声 σ=5），展示 Q 学习对 `V*(0,0)` 的高估达到显著程度，而双 Q 学习则没有。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| TD 误差（TD error） | "更新信号" | `δ = r + γ V(s') - V(s)`，自举残差。 |
| TD(0) | "单步 TD" | 每次转移后使用下一状态的估计更新。 |
| Q 学习（Q-learning） | "离策略 RL 入门" | 带 `max` 的 TD 更新；无论行为策略如何都学习 `Q*`。 |
| SARSA | "在策略 Q 学习" | 使用实际下一动作的 TD 更新；学习当前 ε-贪婪策略的 `Q^π`。 |
| 期望 SARSA（Expected SARSA） | "低方差 SARSA" | 用策略 π 下的期望值替换采样的 `a'`。 |
| GLIE | "正确的探索调度" | 无限探索极限下贪婪；Q 学习收敛的必要条件。 |
| 自举（Bootstrapping） | "在目标中使用当前估计" | 区分 TD 与 MC 的特征。偏差来源，但大幅降低方差。 |
| 最大偏差（Maximization bias） | "Q 学习高估" | `max` 作用于噪声估计时向上偏；双 Q 学习修复。 |

## 延伸阅读

- [Watkins & Dayan (1992). Q-learning](https://link.springer.com/article/10.1007/BF00992698) — 原始论文和收敛证明
- [Sutton & Barto (2018). Ch. 6 — Temporal-Difference Learning](http://incompleteideas.net/book/RLbook2020.pdf) — TD(0)、SARSA、Q 学习、期望 SARSA
- [Hasselt (2010). Double Q-learning](https://papers.nips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html) — 最大偏差的修复
- [Seijen, Hasselt, Whiteson, Wiering (2009). A Theoretical and Empirical Analysis of Expected SARSA](https://ieeexplore.ieee.org/document/4927542) — 期望 SARSA 的动机
- [Rummery & Niranjan (1994). On-line Q-learning using connectionist systems](https://www.researchgate.net/publication/2500611_On-Line_Q-Learning_Using_Connectionist_Systems) — 最初创造 SARSA 名称的论文
- [Sutton & Barto (2018). Ch. 7 — n-step Bootstrapping](http://incompleteideas.net/book/RLbook2020.pdf) — 将 TD(0) 推广到 TD(n)，从 Q 学习到资格迹再到 PPO 中 GAE 的路径
