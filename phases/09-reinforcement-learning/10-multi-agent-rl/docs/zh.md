# 多智能体强化学习

> 单智能体强化学习假设环境是平稳的。将两个学习中的智能体放入同一个世界，这个假设就会被打破：每个智能体都是另一个智能体环境的一部分，而两者都在不断变化。多智能体强化学习是在马尔可夫假设不再成立时让学习收敛的一套技巧。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 9 · 04（Q 学习）、Phase 9 · 06（REINFORCE）、Phase 9 · 07（演员-评论家）
**时长：** 约 45 分钟

## 问题背景

机器人学习在房间里导航是单智能体强化学习问题。足球队不是。AlphaStar vs 星际争霸对手不是。竞价智能体的市场不是。两辆车在十字路口协商不是。多对多的现实问题不是。

在每个多智能体设置中，从任何一个智能体的角度来看，其他智能体*就是*环境的一部分。随着它们学习并改变行为，环境变得非平稳。马尔可夫性质——"下一状态只依赖于当前状态和我的动作"——被违反了，因为下一状态还取决于*其他*智能体的选择，而它们的策略是移动的目标。

这打破了表格收敛证明（Q 学习的保证假设环境平稳）。也打破了朴素的深度强化学习：智能体相互追逐，永远无法收敛到稳定策略。你需要多智能体专用技术：集中训练/分布执行、反事实基线、联赛博弈、自博弈。

2026 年的应用：机器人群体、交通路由、自动驾驶车队、市场模拟器、多智能体语言模型系统（Phase 16），以及任何有多个智能玩家的游戏。

## 核心概念

![四种多智能体强化学习制度：独立、集中评论家、自博弈、联赛](../assets/marl.svg)

**形式化：马尔可夫博弈（Markov Game）。** MDP 的推广：状态 `S`，联合动作 `a = (a_1, …, a_n)`，转移 `P(s' | s, a)`，以及每个智能体的奖励 `R_i(s, a, s')`。每个智能体 `i` 在其自己的策略 `π_i` 下最大化自己的回报。如果奖励相同，则为**完全合作**。如果零和，则为**对抗**。如果混合，则为**一般和**。

**核心挑战：**

- **非平稳性。** 从智能体 `i` 的角度看，`P(s' | s, a_i)` 依赖于正在变化的 `π_{-i}`。
- **信用分配（Credit assignment）。** 共享奖励时，哪个智能体造成了它？
- **探索协调。** 智能体必须探索互补策略，而非冗余地探索相同状态。
- **可扩展性。** 联合动作空间随 `n` 指数增长。
- **部分可观测性。** 每个智能体只看到自己的观察；全局状态是隐藏的。

**四种主流制度：**

**1. 独立 Q 学习 / 独立 PPO（IQL, IPPO）。** 每个智能体学习自己的 Q 或策略，将其他智能体视为环境的一部分。简单，有时有效（尤其是当经验回放作为智能体建模平滑技巧时）。理论收敛：无。实践中：对松耦合任务没问题，对紧耦合任务很差。

**2. 集中训练，分布执行（CTDE，Centralized Training, Decentralized Execution）。** 最常见的现代范式。每个智能体有自己的*策略* `π_i`，以局部观察 `o_i` 为条件——部署时标准的分布执行。在*训练*期间，集中评论家 `Q(s, a_1, …, a_n)` 以完整全局状态和联合动作为条件。示例：
- **MADDPG**（Lowe 等 2017）：带每个智能体集中评论家的 DDPG。
- **COMA**（Foerster 等 2017）：反事实基线——"如果我采取了动作 `a'` 而不是实际动作，我的奖励会是多少？"——隔离我的贡献。
- **MAPPO** / 带共享评论家的 **IPPO**（Yu 等 2022）：带集中价值函数的 PPO。2026 年合作多智能体强化学习的主流。
- **QMIX**（Rashid 等 2018）：价值分解——`Q_tot(s, a) = f(Q_1(s, a_1), …, Q_n(s, a_n))`，带单调混合。

**3. 自博弈（Self-play）。** 同一智能体的两个副本互相对弈。对手的策略*就是*过去快照的我的策略。AlphaGo / AlphaZero / MuZero。OpenAI Five。最适合零和游戏；训练信号是对称的。

**4. 联赛博弈（League play）。** 自博弈对一般和/对抗环境的扩展：维护过去和当前策略的种群，从联赛中采样对手，对他们进行训练。添加利用者（专门击败当前最优策略）和主利用者（专门击败利用者）。AlphaStar（星际争霸 II）。当游戏存在"石头剪刀布"式策略循环时需要此方法。

**通信。** 允许智能体向彼此发送学习到的消息 `m_i`。在合作设置中有效。Foerster 等（2016）展示了可微的智能体间通信可以端到端训练。今天基于语言模型的多智能体系统（Phase 16）本质上以自然语言进行通信。

## 动手实现

本课使用一个 6×6 GridWorld，其中有两个合作智能体。它们从对角出发，必须到达共享目标。共享奖励：只要有任一智能体仍在移动，每步奖励 `-1`；当两者都到达时奖励 `+10`。见 `code/main.py`。

### 步骤一：多智能体环境

```python
class CoopGridWorld:
    def __init__(self):
        self.size = 6
        self.goal = (5, 5)

    def reset(self):
        return ((0, 0), (5, 0))  # 两个智能体

    def step(self, state, actions):
        a1, a2 = state
        new1 = move(a1, actions[0])
        new2 = move(a2, actions[1])
        done = (new1 == self.goal) and (new2 == self.goal)
        reward = 10.0 if done else -1.0
        return (new1, new2), reward, done
```

*联合*动作空间是 `|A|² = 16`。全局状态是两个位置。

### 步骤二：独立 Q 学习

每个智能体运行自己的 Q 表，以联合状态为键。每步：两者都选择 ε-贪婪动作，收集联合转移，每个用共享奖励更新自己的 Q。

```python
def independent_q(env, episodes, alpha, gamma, epsilon):
    Q1, Q2 = defaultdict(default_q), defaultdict(default_q)
    for _ in range(episodes):
        s = env.reset()
        while not done:
            a1 = epsilon_greedy(Q1, s, epsilon)
            a2 = epsilon_greedy(Q2, s, epsilon)
            s_next, r, done = env.step(s, (a1, a2))
            target1 = r + gamma * max(Q1[s_next].values())
            target2 = r + gamma * max(Q2[s_next].values())
            Q1[s][a1] += alpha * (target1 - Q1[s][a1])
            Q2[s][a2] += alpha * (target2 - Q2[s][a2])
            s = s_next
```

对这个任务有效，因为奖励密集且对齐。在紧耦合任务上失败（例如，一个智能体必须*等待*另一个的情况）。

### 步骤三：带分解价值更新的集中 Q

对联合动作使用一个 Q：`Q(s, a_1, a_2)`。从共享奖励更新。通过边际化在执行时去中心化：`π_i(s) = argmax_{a_i} max_{a_{-i}} Q(s, a_1, a_2)`。以指数增长的联合动作空间换取*正确的*全局视图。

### 步骤四：简单自博弈（对抗 2 智能体）

同一智能体，两个角色。训练智能体 A 对抗智能体 B；经过 `K` 个回合后，将 A 的权重复制到 B。对称训练，持续进步。AlphaZero 配方的微缩版。

## 常见陷阱

- **非平稳回放。** 带独立智能体的经验回放比单智能体更差，因为旧转移由现在已过时的对手生成。修复：按时效重新标注或加权。
- **信用分配歧义。** 长回合后的共享奖励；没有明确方式说明哪个智能体有贡献。修复：反事实基线（COMA），或按智能体进行奖励塑造。
- **策略漂移 / 相互追逐。** 每个智能体的最优响应随另一个的更新而改变。修复：集中评论家、慢学习率，或逐一冻结。
- **协调形式的奖励黑客。** 智能体找到设计者未预料的协调漏洞。拍卖智能体收敛到零出价。修复：仔细的奖励设计、行为约束。
- **探索冗余。** 两个智能体探索相同的状态-动作对。修复：每个智能体的熵奖励，或角色条件化。
- **联赛循环。** 纯自博弈可能陷入支配循环。修复：带多样化对手的联赛博弈。
- **样本爆炸。** `n` 个智能体 × 状态空间 × 联合动作。用函数近似来近似；因子化动作空间（每个智能体一个策略输出头）。

## 生产使用

2026 年多智能体强化学习应用地图：

| 领域 | 方法 | 说明 |
|------|------|------|
| 合作导航 / 操作 | MAPPO / QMIX | CTDE；共享评论家 + 分布执行演员。 |
| 双人游戏（国际象棋、围棋、扑克） | 带 MCTS 的自博弈（AlphaZero） | 零和；对称训练。 |
| 复杂多人（Dota、星际争霸） | 联赛博弈 + 模仿预训练 | OpenAI Five，AlphaStar。 |
| 自动驾驶车队 | CTDE MAPPO / 带注意力的 PPO | 部分观测；可变团队大小。 |
| 拍卖市场 | 博弈论均衡 + 强化学习 | `n` → ∞ 时的平均场强化学习。 |
| 语言模型多智能体系统（Phase 16） | 自然语言通信 + 角色条件化 | 强化学习循环在智能体规划层。 |

2026 年，多智能体强化学习最大的增长领域是基于语言模型的：语言模型智能体群体进行谈判、辩论、构建软件。强化学习表现为对*轨迹级*输出（而非 token 级）的偏好优化（Phase 16 · 03）。

## 上手实践

保存为 `outputs/skill-marl-architect.md`：

```markdown
---
name: marl-architect
description: Pick the right multi-agent RL regime (IPPO, CTDE, self-play, league) for a given task.
version: 1.0.0
phase: 9
lesson: 10
tags: [rl, multi-agent, marl, self-play]
---

给定有 `n` 个智能体的任务，输出：

1. 制度分类。合作 / 对抗 / 一般和。说明理由。
2. 算法。IPPO / MAPPO / QMIX / 自博弈 / 联赛。理由与耦合紧密度和奖励结构相关。
3. 信息访问。集中训练（哪些全局信息传入评论家）？分布执行？
4. 信用分配。反事实基线、价值分解或奖励塑造。
5. 探索计划。每个智能体的熵、基于种群的训练或联赛。

拒绝在紧耦合合作任务上使用独立 Q 学习。拒绝推荐有循环风险的一般和游戏使用自博弈。标记任何没有固定对手评估的多智能体强化学习流水线（精心挑选的自博弈数字很常见）。
```

## 练习

1. **简单。** 在 2 智能体合作 GridWorld 上训练独立 Q 学习。多少个回合后平均回报 > 0？绘制联合学习曲线。
2. **中等。** 添加"协调"任务：只有当两个智能体在同一回合步入目标时才算到达。独立 Q 学习还能收敛吗？什么地方失效了？
3. **困难。** 为 MAPPO 风格训练实现集中评论家，并在协调任务上与独立 PPO 比较收敛速度。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 马尔可夫博弈（Markov game） | "多智能体 MDP" | `(S, A_1, …, A_n, P, R_1, …, R_n)`；每个智能体有自己的奖励。 |
| CTDE | "集中训练，分布执行" | 训练时使用联合评论家；每个智能体的策略只使用局部观察。 |
| IPPO | "独立 PPO" | 每个智能体单独运行 PPO。简单基线；常被低估。 |
| MAPPO | "多智能体 PPO" | 带以全局状态为条件的集中价值函数的 PPO。 |
| QMIX | "单调价值分解" | `Q_tot = f_单调(Q_1, …, Q_n)` 允许分布 argmax。 |
| COMA | "反事实多智能体" | 优势 = 我的 Q 减去对我的动作边际化的期望 Q。 |
| 自博弈（Self-play） | "智能体对抗过去的自己" | 单个智能体，两个角色；零和游戏的标准方法。 |
| 联赛博弈（League play） | "种群训练" | 缓存过去策略，从池中采样对手；处理策略循环。 |

## 延伸阅读

- [Lowe et al. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG)](https://arxiv.org/abs/1706.02275) — 带集中评论家的 CTDE
- [Foerster et al. (2017). Counterfactual Multi-Agent Policy Gradients (COMA)](https://arxiv.org/abs/1705.08926) — 用于信用分配的反事实基线
- [Rashid et al. (2018). QMIX: Monotonic Value Function Factorisation](https://arxiv.org/abs/1803.11485) — 带单调性的价值分解
- [Yu et al. (2022). The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games (MAPPO)](https://arxiv.org/abs/2103.01955) — PPO 在多智能体强化学习中出人意料的强大表现
- [Vinyals et al. (2019). Grandmaster level in StarCraft II using multi-agent reinforcement learning (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z) — 规模化联赛博弈
- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270) — 零和游戏中的纯自博弈
- [Sutton & Barto (2018). Ch. 15 & Ch. 17](http://incompleteideas.net/book/RLbook2020.pdf) — 多智能体设置和 CTDE 旨在解决的非平稳性问题的教科书简短处理
- [Zhang, Yang & Başar (2021). Multi-Agent Reinforcement Learning: A Selective Overview](https://arxiv.org/abs/1911.10635) — 覆盖合作、竞争和混合多智能体强化学习及收敛结果的综述
