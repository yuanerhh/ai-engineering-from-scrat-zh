# 马尔可夫决策过程——状态、动作与奖励

> 马尔可夫决策过程（MDP）是五件事：状态、动作、转移、奖励、折扣。强化学习中的一切——Q 学习、PPO、DPO、GRPO——都在这个框架上优化。学一次，强化学习的其余部分就能免费阅读。

**类型：** 学习
**语言：** Python
**前置条件：** Phase 1 · 06（概率与分布）、Phase 2 · 01（机器学习分类）
**时长：** 约 45 分钟

## 问题背景

你在写一个国际象棋机器人。或者库存规划器。或者交易代理。或者训练推理模型的 PPO 循环。四个不同的领域，一个令人惊讶的事实：所有四个都归结为相同的数学对象。

监督学习给你 `(x, y)` 对，要求你拟合一个函数。强化学习不给你标签——只有状态流、你采取的动作和一个标量奖励。那步棋赢了吗？补货决策省钱了吗？交易盈利了吗？LLM 刚生成的 token 是否让评判者给出了更高的奖励？

在将这个流形式化之前，你无法从中学习。"我看到了什么"、"我做了什么"、"接下来发生了什么"、"那有多好"——每一个都必须成为你可以推理的对象。这种形式化就是马尔可夫决策过程。本阶段的每个强化学习算法，包括最后的 RLHF 和 GRPO 循环，都在这个框架上优化。

## 核心概念

![马尔可夫决策过程：状态、动作、转移、奖励、折扣](../assets/mdp.svg)

**五个对象。**

- **状态** `S`。智能体做决定所需的一切。在 GridWorld 中是单元格，在国际象棋中是棋盘，在 LLM 中是上下文窗口加任何记忆。
- **动作** `A`。选择。上/下/左/右移动。走一步棋。生成一个 token。
- **转移** `P(s' | s, a)`。给定状态 `s` 和动作 `a`，下一个状态的分布。国际象棋中确定性，库存中随机性，LLM 解码中几乎确定性。
- **奖励** `R(s, a, s')`。标量信号。赢 = +1，输 = -1。收入减成本。GRPO 中的对数似然比项。
- **折扣** `γ ∈ [0, 1)`。未来奖励相对于当前的重要程度。`γ = 0.99` 购买约 100 步的视野；`γ = 0.9` 购买约 10 步。

**马尔可夫性质** `P(s_{t+1} | s_t, a_t) = P(s_{t+1} | s_0, a_0, …, s_t, a_t)`。未来只依赖于当前状态。如果不是这样，状态表示是不完整的——不是方法的失败，而是状态的失败。

**策略与回报。** 策略 `π(a | s)` 将状态映射到动作分布。回报 `G_t = r_t + γ r_{t+1} + γ² r_{t+2} + …` 是未来奖励的折扣和。价值 `V^π(s) = E[G_t | s_t = s]` 是在策略 `π` 下从 `s` 开始的期望回报。Q 值 `Q^π(s, a) = E[G_t | s_t = s, a_t = a]` 是从特定动作开始的期望回报。每个强化学习算法都估计这两者之一，然后相应地改进 `π`。

**贝尔曼方程。** 本阶段所有内容使用的不动点方程：

`V^π(s) = Σ_a π(a|s) Σ_{s', r} P(s', r | s, a) [r + γ V^π(s')]`
`Q^π(s, a) = Σ_{s', r} P(s', r | s, a) [r + γ Σ_{a'} π(a'|s') Q^π(s', a')]`

这将期望回报分解为"这一步的奖励"加上"降落位置的折扣价值"。递归。Phase 9 中的每个算法要么迭代这个方程到收敛（动态规划），要么从中采样（蒙特卡洛），要么单步自举它（时序差分）。

## 动手实现

### 步骤一：一个小型确定性 MDP

4×4 GridWorld。智能体从左上角开始，右下角为终止状态，每步奖励 -1，动作 `{up, down, left, right}`。见 `code/main.py`。

```python
GRID = 4
TERMINAL = (3, 3)
ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

def step(state, action):
    if state == TERMINAL:
        return state, 0.0, True
    dr, dc = ACTIONS[action]
    r, c = state
    nr = min(max(r + dr, 0), GRID - 1)
    nc = min(max(c + dc, 0), GRID - 1)
    return (nr, nc), -1.0, (nr, nc) == TERMINAL
```

五行代码。这就是整个环境。确定性转移，恒定步骤惩罚，吸收终止状态。

### 步骤二：展开一个策略

策略是从状态到动作分布的函数。最简单的：均匀随机。

```python
def uniform_policy(state):
    return {a: 0.25 for a in ACTIONS}

def rollout(policy, max_steps=200):
    s, total, steps = (0, 0), 0.0, 0
    for _ in range(max_steps):
        a = sample(policy(s))
        s, r, done = step(s, a)
        total += r
        steps += 1
        if done:
            break
    return total, steps
```

运行随机策略 1000 次。这个 4×4 棋盘的平均回报约为 -60 到 -80。最优回报是 -6（向右下方的直线路径）。缩小这个差距就是 Phase 9 的全部内容。

### 步骤三：通过贝尔曼方程精确计算 `V^π`

对于小型 MDP，贝尔曼方程是一个线性系统。枚举状态，应用期望，迭代直到值停止变化。

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in all_states()}
    while True:
        delta = 0.0
        for s in all_states():
            if s == TERMINAL:
                continue
            v = 0.0
            for a, pi_a in policy(s).items():
                s_next, r, _ = step(s, a)
                v += pi_a * (r + gamma * V[s_next])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

这是迭代策略评估。它是 Sutton & Barto 的第一个算法，也是后续每个强化学习方法的理论基础。

### 步骤四：`γ` 是有物理含义的超参数

有效视野约为 `1 / (1 - γ)`。`γ = 0.9` → 10 步。`γ = 0.99` → 100 步。`γ = 0.999` → 1000 步。

太低，智能体行为短视。太高，信用分配变得嘈杂，因为许多早期步骤共同承担远期奖励的责任。LLM RLHF 通常使用 `γ = 1`，因为回合很短且有界。控制任务使用 `0.95-0.99`。长视野策略游戏使用 `0.999`。

## 常见陷阱

- **非马尔可夫状态。** 如果你需要最近三次观察才能决策，"状态"不只是当前观察。修复：堆叠帧（Atari 上的 DQN 堆叠 4 帧）或使用循环状态（对观察进行 LSTM/GRU 处理）。
- **稀疏奖励。** 仅有胜利的奖励使得在大型状态空间中学习几乎不可能。塑造奖励（中间信号）或用模仿引导（Phase 9 · 09）。
- **奖励黑客。** 优化代理奖励往往产生病态行为。OpenAI 的赛艇智能体永远在圆圈中旋转收集道具，而不是完成比赛。始终从目标结果而非代理指标定义奖励。
- **折扣配置错误。** 在无限视野任务上 `γ = 1` 使每个值无穷大。始终通过有限视野或 `γ < 1` 进行限制。
- **奖励尺度。** {+100, -100} vs {+1, -1} 的奖励给出相同的最优策略，但梯度量级差异巨大。插入 PPO/DQN 之前归一化到约 `[-1, 1]`。

## 生产使用

2026 年的技术栈在接触代码之前将每个强化学习流水线归结为 MDP：

| 情况 | 状态 | 动作 | 奖励 | γ |
|------|------|------|------|---|
| 控制（运动、操控） | 关节角 + 速度 | 连续力矩 | 任务特定的塑造奖励 | 0.99 |
| 游戏（国际象棋、围棋、扑克） | 棋盘 + 历史 | 合法走法 | 赢=+1/输=-1 | 1.0（有限） |
| 库存/定价 | 库存 + 需求 | 订购数量 | 收入减成本 | 0.95 |
| LLM 的 RLHF | 上下文 token | 下一个 token | 结束时的奖励模型分数 | 1.0（回合约 200 token） |
| 推理的 GRPO | 提示词 + 部分回答 | 下一个 token | 结束时验证器 0/1 | 1.0 |

在写任何训练循环之前先写出五元组。大多数"强化学习不起作用"的错误报告都可追溯到纸面上就已损坏的 MDP 形式化。

## 上手实践

保存为 `outputs/skill-mdp-modeler.md`：

```markdown
---
name: mdp-modeler
description: Given a task description, produce a Markov Decision Process spec and flag formulation risks before training.
version: 1.0.0
phase: 9
lesson: 1
tags: [rl, mdp, modeling]
---

给定任务（控制/游戏/推荐/LLM 微调），输出：

1. 状态。精确的特征向量或张量规格。证明马尔可夫性质。
2. 动作。离散集合或连续范围。维度。
3. 转移。确定性、已知模型的随机性或仅采样。
4. 奖励。函数和来源。稀疏 vs 塑造。终止 vs 每步。
5. 折扣。值和视野理由。

拒绝交付任何状态非马尔可夫而未明确提及帧堆叠或循环状态的 MDP。拒绝任何未从目标结果定义的奖励。标记无限视野任务上任何 `γ ≥ 1.0` 的情况。标记任何奖励范围超过典型步骤奖励 100 倍的情况，这可能是梯度爆炸的来源。
```

## 练习

1. **简单。** 在 `code/main.py` 中实现 4×4 GridWorld 和随机策略展开。运行 10,000 个回合。报告回报的均值和标准差。与最优回报（-6）比较。
2. **中等。** 对均匀随机策略，用 `γ ∈ {0.5, 0.9, 0.99}` 运行 `policy_evaluation`。以 4×4 网格形式打印每个 `V`。解释为什么靠近终止状态的状态值随更大的 `γ` 增长更快。
3. **困难。** 将 GridWorld 变为随机：每个动作以概率 `p = 0.1` 滑向相邻方向。重新评估均匀策略。`V[start]` 变好了还是变差了？为什么？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| MDP | "强化学习设置" | 满足马尔可夫性质的元组 `(S, A, P, R, γ)`。 |
| 状态（State） | "智能体看到的" | 在所选策略类下对未来动态充分统计。 |
| 策略（Policy） | "智能体的行为" | 条件分布 `π(a | s)` 或确定性映射 `s → a`。 |
| 回报（Return） | "总奖励" | 从当前步开始的折扣和 `Σ γ^t r_t`。 |
| 价值（Value） | "状态有多好" | 从 `s` 开始在 `π` 下的期望回报。 |
| Q 值（Q-value） | "动作有多好" | 从 `s` 开始以第一个动作 `a` 在 `π` 下的期望回报。 |
| 贝尔曼方程（Bellman equation） | "动态规划递归" | 将价值/Q 分解为一步奖励加折扣后继价值的不动点分解。 |
| 折扣 `γ`（Discount） | "未来 vs 当前" | 远期奖励上的几何权重；有效视野约 `1/(1-γ)`。 |

## 延伸阅读

- [Sutton & Barto (2018). Reinforcement Learning: An Introduction, 2nd ed.](http://incompleteideas.net/book/RLbook2020.pdf) — 教科书。第 3 章介绍 MDP 和贝尔曼方程；第 1 章激励奖励假设
- [Bellman (1957). Dynamic Programming](https://press.princeton.edu/books/paperback/9780691146683/dynamic-programming) — 贝尔曼方程的起源
- [OpenAI Spinning Up — Part 1: Key Concepts](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) — 从深度强化学习角度的简洁 MDP 入门
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — MDP 和精确求解方法的运筹学参考
- [Littman (1996). Algorithms for Sequential Decision Making (PhD thesis)](https://www.cs.rutgers.edu/~mlittman/papers/thesis-main.pdf) — 将 MDP 作为动态规划特化的最清晰推导
