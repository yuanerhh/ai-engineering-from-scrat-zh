# 蒙特卡洛方法——从完整回合中学习

> 动态规划需要模型。蒙特卡洛什么都不需要，只需要回合。运行策略，观察回报，取平均值。强化学习中最简单的思想——也是解锁后续一切的那个。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 9 · 01（MDP）、Phase 9 · 02（动态规划）
**时长：** 约 75 分钟

## 问题背景

动态规划很优雅，但它假设你可以对每个状态和动作查询 `P(s' | s, a)`。现实世界中几乎没有什么是这样工作的。机器人无法解析地计算关节力矩后相机像素的分布。定价算法无法对每种可能的客户反应进行积分。LLM 无法枚举 token 之后所有可能的延续。

你需要一种只需要从环境中*采样*的方法。运行策略。得到轨迹 `s_0, a_0, r_1, s_1, a_1, r_2, …, s_T`。用它来估计价值。这就是蒙特卡洛。

从 DP 到 MC 的转变在哲学上很重要：我们从*已知模型 + 精确备份*转移到*采样展开 + 平均回报*。方差跳升，但适用性爆炸。本课之后的每个强化学习算法——TD、Q 学习、REINFORCE、PPO、GRPO——本质上都是蒙特卡洛估计器，有时在上面叠加了自举。

## 核心概念

![蒙特卡洛：展开、计算回报、平均；首次访问 vs 每次访问](../assets/monte-carlo.svg)

**核心思想，一行话：** `V^π(s) = E_π[G_t | s_t = s] ≈ (1/N) Σ_i G^{(i)}(s)`，其中 `G^{(i)}(s)` 是在策略 `π` 下访问 `s` 后观察到的回报。

**首次访问 vs 每次访问 MC。** 给定多次访问状态 `s` 的回合，首次访问 MC 只计算首次访问后的回报；每次访问 MC 计算所有访问。两者在极限下都是无偏的。首次访问分析更简单（iid 样本）。每次访问每个回合使用更多数据，实践中通常收敛更快。

**增量均值。** 不存储所有回报，而是更新运行平均值：

`V_n(s) = V_{n-1}(s) + (1/n) [G_n - V_{n-1}(s)]`

重组：`V_new = V_old + α · (target - V_old)`，其中 `α = 1/n`。将 `1/n` 换成常数步长 `α ∈ (0, 1)`，你得到跟踪 `π` 变化的非平稳 MC 估计器。这个变化就是从 MC 到 TD 再到每个现代强化学习算法的全部飞跃。

**探索现在是个问题。** DP 通过枚举访问每个状态。MC 只看策略访问的状态。如果 `π` 是确定性的，状态空间的整个区域永远不会被采样，它们的价值估计永远保持为零。三种修复方法，按历史顺序：

1. **探索性起始。** 从随机的 (s, a) 对开始每个回合。保证覆盖；实践中不现实（你无法将机器人"重置"到任意状态）。
2. **ε-贪婪。** 关于当前 Q 贪婪行动，但以概率 `ε` 选择随机动作。所有状态-动作对渐近地得到采样。
3. **离策略 MC。** 在行为策略 `μ` 下收集数据，通过重要性采样学习目标策略 `π`。高方差，但这是通向 DQN 等回放缓冲方法的桥梁。

**蒙特卡洛控制。** 评估 → 改进 → 评估，就像策略迭代，但评估是基于采样的：

1. 运行 `π`，获得一个回合。
2. 从观察到的回报更新 `Q(s, a)`。
3. 使 `π` 关于 `Q` ε-贪婪。
4. 重复。

在温和条件下（每对无限次访问，`α` 满足 Robbins-Monro）以概率 1 收敛到 `Q*` 和 `π*`。

## 动手实现

### 步骤一：展开 → (s, a, r) 列表

```python
def rollout(env, policy, max_steps=200):
    trajectory = []
    s = env.reset()
    for _ in range(max_steps):
        a = policy(s)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r))
        s = s_next
        if done:
            break
    return trajectory
```

没有模型，只有 `env.reset()` 和 `env.step(s, a)`。与 gym 环境相同的接口但精简了。

### 步骤二：计算回报（逆向扫描）

```python
def returns_from(trajectory, gamma):
    returns = []
    G = 0.0
    for _, _, r in reversed(trajectory):
        G = r + gamma * G
        returns.append(G)
    return list(reversed(returns))
```

一次传播，`O(T)`。逆向递推 `G_t = r_{t+1} + γ G_{t+1}` 避免重新求和。

### 步骤三：首次访问 MC 评估

```python
def mc_policy_evaluation(env, policy, episodes, gamma=0.99):
    V = defaultdict(float)
    counts = defaultdict(int)
    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for t, ((s, _, _), G) in enumerate(zip(trajectory, returns)):
            if s in seen:
                continue
            seen.add(s)
            counts[s] += 1
            V[s] += (G - V[s]) / counts[s]
    return V
```

三行完成工作：首次访问时标记状态为已见，增加计数，更新运行均值。

### 步骤四：ε-贪婪 MC 控制（在策略）

```python
def mc_control(env, episodes, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    counts = defaultdict(lambda: {a: 0 for a in ACTIONS})

    def policy(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for (s, a, _), G in zip(trajectory, returns):
            if (s, a) in seen:
                continue
            seen.add((s, a))
            counts[s][a] += 1
            Q[s][a] += (G - Q[s][a]) / counts[s][a]
    return Q, policy
```

### 步骤五：与 DP 黄金标准比较

你对 `V^π` 的 MC 估计应该随着回合 → ∞ 与第 02 课的 DP 结果一致。实践中：4×4 GridWorld 上 50,000 个回合能让你在 DP 答案的约 `0.1` 范围内。

## 常见陷阱

- **无限回合。** MC 要求回合*终止*。如果你的策略可能永远循环，设置 `max_steps` 上限并将达到上限视为隐式失败。随机策略的 GridWorld 经常超时——这是正常的，只要确保正确计数即可。
- **方差。** MC 使用完整回报。在长回合中，方差巨大——末尾一个不幸的奖励会同等程度地影响 `V(s_0)`。TD 方法（第 04 课）通过自举来削减这一问题。
- **状态覆盖。** 在新 Q 上有平局的贪婪 MC 只会尝试一个动作。你*必须*探索（ε-贪婪、探索性起始、UCB）。
- **非平稳策略。** 如果 `π` 改变了（如在 MC 控制中），旧回报来自不同的策略。常数 α MC 处理这个问题；样本平均 MC 不行。
- **离策略重要性采样。** 权重 `π(a|s)/μ(a|s)` 沿轨迹相乘。方差随视野爆炸。用每决策加权 IS 限制，或切换到 TD。

## 生产使用

2026 年蒙特卡洛方法的作用：

| 使用场景 | 为何用 MC |
|---------|---------|
| 短视野游戏（21点、扑克） | 回合自然终止；回报干净。 |
| 记录策略的离线评估 | 对存储轨迹的折扣回报取平均。 |
| 蒙特卡洛树搜索（AlphaZero） | 树叶节点的 MC 展开指导选择。 |
| LLM 强化学习评估 | 计算给定策略下采样补全的平均奖励。 |
| PPO 中的基线估计 | 优势目标 `A_t = G_t - V(s_t)` 使用 MC 的 `G_t`。 |
| 教学强化学习 | 实际有效的最简单算法——去掉自举以看到核心。 |

现代深度强化学习算法（PPO、SAC）通过 n 步回报或 GAE 在纯 MC（完整回报）和纯 TD（单步自举）之间插值。两个端点都是同一估计器的实例。

## 上手实践

保存为 `outputs/skill-mc-evaluator.md`：

```markdown
---
name: mc-evaluator
description: Evaluate a policy via Monte Carlo rollouts and produce a convergence report with DP-comparison if available.
version: 1.0.0
phase: 9
lesson: 3
tags: [rl, monte-carlo, evaluation]
---

给定环境（回合式，带有 reset+step API）和策略，输出：

1. 方法。首次访问 vs 每次访问 MC。理由。
2. 回合预算。目标数量，方差诊断，预期标准误差。
3. 探索计划。ε 调度（如需要）或探索性起始。
4. 黄金标准比较。表格任务的 DP 最优 V*；否则来自 Q 学习/PPO 基线的界。
5. 终止检查。最大步骤上限，超时，非终止轨迹的处理。

拒绝在无限时间上限的非回合式任务上运行 MC。拒绝报告表格任务每个状态少于 100 个回合的 V^π 估计。将零方差动作的任何策略标记为探索风险。
```

## 练习

1. **简单。** 在 4×4 GridWorld 上实现均匀随机策略的首次访问 MC 评估。运行 10,000 个回合。绘制 `V(0,0)` 作为回合数的函数与 DP 答案对比。
2. **中等。** 用 `ε ∈ {0.01, 0.1, 0.3}` 实现 ε-贪婪 MC 控制。比较 20,000 个回合后的平均回报。曲线是什么样的？偏差-方差权衡在哪里？
3. **困难。** 实现带重要性采样的*离策略* MC：在均匀随机策略 `μ` 下收集数据，为确定性最优策略 `π` 估计 `V^π`。比较朴素 IS、每决策 IS 和加权 IS。哪个方差最低？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 蒙特卡洛（Monte Carlo） | "随机采样" | 通过对分布的 iid 样本取平均来估计期望。 |
| 回报 `G_t`（Return） | "未来奖励" | 从步骤 `t` 到回合结束的折扣奖励和：`Σ_{k≥0} γ^k r_{t+k+1}`。 |
| 首次访问 MC（First-visit MC） | "每个状态只计一次" | 只有回合中的首次访问贡献价值估计。 |
| 每次访问 MC（Every-visit MC） | "使用所有访问" | 每次访问都贡献；略有偏差但样本效率更高。 |
| ε-贪婪（ε-greedy） | "探索噪声" | 以概率 `1-ε` 选贪婪动作；以概率 `ε` 随机动作。 |
| 重要性采样（Importance sampling） | "纠正从错误分布采样" | 用 `π(a|s)/μ(a|s)` 乘积重新加权回报，从 `μ` 数据估计 `V^π`。 |
| 在策略（On-policy） | "从自己的数据学习" | 目标策略 = 行为策略。普通 MC、PPO、SARSA。 |
| 离策略（Off-policy） | "从别人的数据学习" | 目标策略 ≠ 行为策略。重要性采样 MC、Q 学习、DQN。 |

## 延伸阅读

- [Sutton & Barto (2018). Ch. 5 — Monte Carlo Methods](http://incompleteideas.net/book/RLbook2020.pdf) — 规范处理
- [Singh & Sutton (1996). Reinforcement Learning with Replacing Eligibility Traces](https://link.springer.com/article/10.1007/BF00114726) — 首次访问 vs 每次访问分析
- [Precup, Sutton, Singh (2000). Eligibility Traces for Off-Policy Policy Evaluation](http://incompleteideas.net/papers/PSS-00.pdf) — 离策略 MC 和方差控制
- [Mahmood et al. (2014). Weighted Importance Sampling for Off-Policy Learning](https://arxiv.org/abs/1404.6362) — 现代低方差 IS 估计器
- [Tesauro (1995). TD-Gammon, A Self-Teaching Backgammon Program](https://dl.acm.org/doi/10.1145/203330.203343) — MC/TD 自博弈收敛到超人水平的第一个大规模实证示范
