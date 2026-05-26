# 深度 Q 网络（DQN）

> 2013 年：Mnih 在原始像素上训练了一个 Q 学习网络，在七款 Atari 游戏上击败了每个经典强化学习智能体。2015 年：扩展到 49 款游戏，发表于 Nature，引爆了深度强化学习时代。DQN 是 Q 学习加三个让函数近似稳定的技巧。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 3 · 03（反向传播）、Phase 9 · 04（Q 学习、SARSA）
**时长：** 约 75 分钟

## 问题背景

表格 Q 学习需要为每个（状态, 动作）对单独存一个 Q 值。国际象棋棋盘有约 10⁴³ 个状态。一帧 Atari 图像是 210×160×3 = 100,800 个特征。表格强化学习在数千个状态下就已力不从心，更别说数十亿了。

修复思路事后看来显而易见：用神经网络 `Q(s, a; θ)` 替换 Q 表。但这个"事后显而易见"用了几十年才实现。朴素的函数近似与 Q 学习组合在"致命三角"下会发散——函数近似 + 自举 + 离策略学习。Mnih 等（2013, 2015）找到了三个让训练稳定的工程技巧：

1. **经验回放（Experience replay）** 去除转移间的相关性。
2. **目标网络（Target network）** 冻结自举目标。
3. **奖励截断（Reward clipping）** 归一化梯度量级。

Atari 上的 DQN 是第一次用单一架构、单套超参数从原始像素解决数十个控制问题。此后构建的一切"深度强化学习"——DDQN、Rainbow、Dueling、Distributional、R2D2、Agent57——都堆叠在这三个技巧的基础之上。

## 核心概念

![DQN 训练循环：环境、回放缓冲、在线网络、目标网络、贝尔曼 TD 损失](../assets/dqn.svg)

**目标函数。** DQN 最小化神经 Q 函数上的单步 TD 损失：

`L(θ) = E_{(s,a,r,s')~D} [ (r + γ max_{a'} Q(s', a'; θ^-) - Q(s, a; θ))² ]`

`θ` = 在线网络，每步通过梯度下降更新。`θ^-` = 目标网络，周期性从 `θ` 复制（每约 10,000 步）。`D` = 过去转移的回放缓冲区。

**三个技巧，按重要性排序：**

**经验回放。** 约 `10⁶` 个转移的环形缓冲区。每个训练步从中均匀随机采样一个小批量。这打破了时序相关性（连续帧几乎相同），让网络能多次从稀有奖励转移中学习，并去除连续梯度更新之间的相关性。没有它，带神经网络的在策略 TD 在 Atari 上会发散。

**目标网络。** 在贝尔曼方程两侧使用同一网络 `Q(·; θ)` 会使目标随每次更新移动——"追自己的尾巴"。修复方法：保留第二个网络 `Q(·; θ^-)`，其权重被冻结。每 `C` 步，将 `θ → θ^-` 复制一次。这让回归目标在数千次梯度步内保持稳定。软更新 `θ^- ← τ θ + (1-τ) θ^-`（DDPG、SAC 使用）是更平滑的变体。

**奖励截断。** Atari 奖励量级从 1 到 1000+ 不等。截断到 `{-1, 0, +1}` 防止任何单个游戏主导梯度。当奖励量级重要时不适用；对只关心符号的 Atari 没问题。

**双重 DQN（Double DQN）。** Hasselt（2016）修复最大偏差：用在线网络*选择*动作，用目标网络*评估*它。

`target = r + γ Q(s', argmax_{a'} Q(s', a'; θ); θ^-)`

可直接替换，效果一贯更好。默认使用。

**其他改进（Rainbow, 2017）：** 优先级回放（更多采样高 TD 误差的转移）、对决架构（分离 `V(s)` 和优势头）、噪声网络（学习探索）、n 步回报、分布式 Q（C51/QR-DQN）、多步自举。每个各带来几个百分点的提升，收益大致是可加的。

## 动手实现

此处的代码只用标准库，无 numpy——我们在微型连续 GridWorld 上使用手动实现的单隐藏层 MLP，每个训练步在微秒内完成。算法与 Atari 规模的 DQN 完全相同。

### 步骤一：回放缓冲区

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = []
        self.capacity = capacity
    def push(self, s, a, r, s_next, done):
        if len(self.buf) == self.capacity:
            self.buf.pop(0)
        self.buf.append((s, a, r, s_next, done))
    def sample(self, batch, rng):
        return rng.sample(self.buf, batch)
```

Atari 约 50,000 容量；玩具环境 5,000 足够。

### 步骤二：微型 Q 网络（手动 MLP）

```python
class QNet:
    def __init__(self, n_in, n_hidden, n_actions, rng):
        self.W1 = [[rng.gauss(0, 0.3) for _ in range(n_in)] for _ in range(n_hidden)]
        self.b1 = [0.0] * n_hidden
        self.W2 = [[rng.gauss(0, 0.3) for _ in range(n_hidden)] for _ in range(n_actions)]
        self.b2 = [0.0] * n_actions
    def forward(self, x):
        h = [max(0.0, sum(w * xi for w, xi in zip(row, x)) + b) for row, b in zip(self.W1, self.b1)]
        q = [sum(w * hi for w, hi in zip(row, h)) + b for row, b in zip(self.W2, self.b2)]
        return q, h
```

前向传播：线性 → ReLU → 线性。这就是整个网络。

### 步骤三：DQN 更新

```python
def train_step(online, target, batch, gamma, lr):
    grads = zeros_like(online)
    for s, a, r, s_next, done in batch:
        q, h = online.forward(s)
        if done:
            y = r
        else:
            q_next, _ = target.forward(s_next)
            y = r + gamma * max(q_next)
        td_error = q[a] - y
        accumulate_grads(grads, online, s, h, a, td_error)
    apply_sgd(online, grads, lr / len(batch))
```

结构与第 04 课的 Q 学习相同，只有两处区别：(a) 通过可微的 `Q(·; θ)` 反向传播，而非索引表格；(b) 目标使用 `Q(·; θ^-)`。

### 步骤四：外层循环

每个回合，对 `Q(·; θ)` 执行 ε-贪婪动作，将转移推入缓冲区，采样一个小批量，执行一步梯度，定期同步 `θ^- ← θ`。模式如下：

```python
for episode in range(N):
    s = env.reset()
    while not done:
        a = epsilon_greedy(online, s, epsilon)
        s_next, r, done = env.step(s, a)
        buffer.push(s, a, r, s_next, done)
        if len(buffer) >= batch:
            train_step(online, target, buffer.sample(batch), gamma, lr)
        if steps % sync_every == 0:
            target = copy(online)
        s = s_next
```

在带 16 维独热状态的微型 GridWorld 上，智能体约 500 个回合学到接近最优策略。在 Atari 上，将其扩展到 2 亿帧并添加 CNN 特征提取器。

## 常见陷阱

- **致命三角。** 函数近似 + 离策略 + 自举可能发散。DQN 用目标网络 + 回放来缓解；不要去掉任何一个。
- **探索。** ε 必须衰减，通常在训练前约 10% 内从 1.0 降到 0.01。前期探索不足，Q 网络会收敛到局部盆地。
- **高估。** 对噪声 Q 取 `max` 会向上偏。生产中始终使用双重 DQN。
- **奖励尺度。** 截断或归一化奖励；梯度量级与奖励量级成正比。
- **回放缓冲冷启动。** 缓冲区有几千个转移前不要开始训练。在约 20 个样本上早期梯度会过拟合。
- **目标同步频率。** 太频繁 ≈ 没有目标网络；太少 ≈ 目标过时。Atari DQN 使用 10,000 个环境步。经验法则：每约 1/100 训练周期同步一次。
- **观察预处理。** Atari DQN 堆叠 4 帧使状态满足马尔可夫性质。任何含速度信息的环境都需要帧堆叠或循环状态。

## 生产使用

2026 年，DQN 很少是最先进的方法，但仍是参考离策略算法：

| 任务 | 首选方法 | 为何不用 DQN？ |
|------|---------|--------------|
| 离散动作 Atari 类 | Rainbow DQN 或 Muesli | 相同框架，更多技巧。 |
| 连续控制 | SAC / TD3（Phase 9 · 07） | DQN 没有策略网络。 |
| 在策略 / 高吞吐量 | PPO（Phase 9 · 08） | 无回放缓冲区；更易扩展。 |
| 离线强化学习 | CQL / IQL / 决策 Transformer | 保守 Q 目标，无自举爆炸。 |
| 大型离散动作空间（推荐系统） | 带动作嵌入的 DQN 或 IMPALA | 可行；细节很重要。 |
| LLM 强化学习 | PPO / GRPO | 序列级而非步级；不同的损失。 |

这些经验教训依然适用。回放和目标网络出现在 SAC、TD3、DDPG、SAC-X、AlphaZero 的自博弈缓冲区以及每个离线强化学习方法中。奖励截断在 PPO 中以优势归一化的形式延续。这个架构是蓝图。

## 上手实践

保存为 `outputs/skill-dqn-trainer.md`：

```markdown
---
name: dqn-trainer
description: Produce a DQN training config (buffer, target sync, ε schedule, reward clipping) for a discrete-action RL task.
version: 1.0.0
phase: 9
lesson: 5
tags: [rl, dqn, deep-rl]
---

给定离散动作环境（观察形状、动作数量、视野、奖励尺度），输出：

1. 网络。架构（MLP / CNN / Transformer），特征维度，深度。
2. 回放缓冲区。容量、小批量大小、热身大小。
3. 目标网络。同步策略（每 C 步硬同步或软更新 τ）。
4. 探索。ε 起始 / 终止 / 调度长度。
5. 损失。Huber vs MSE，梯度裁剪值，奖励截断规则。
6. 双重 DQN。默认开启，除非有明确理由禁用。

拒绝发布没有目标网络、没有回放缓冲区或 ε 保持 1 的 DQN。拒绝连续动作任务（转向 SAC / TD3）。标记任何奖励范围超过每步均值 10 倍的情况，需要截断或尺度归一化。
```

## 练习

1. **简单。** 运行 `code/main.py`。绘制每回合回报曲线。多少个回合后滚动均值超过 -10？
2. **中等。** 禁用目标网络（贝尔曼目标两侧都用在线网络）。测量训练不稳定性——回报是否振荡或发散？
3. **困难。** 添加双重 DQN：用在线网络选择 `argmax a'`，目标网络评估。在带噪声奖励的 GridWorld 上，对比 1,000 个回合后有 / 无双重 DQN 时 `Q(s_0, best_a)` 与真实 `V*(s_0)` 的偏差。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| DQN | "深度 Q 学习" | 带神经网络 Q 函数、回放缓冲区和目标网络的 Q 学习。 |
| 经验回放（Experience replay） | "打乱的转移" | 每步梯度均匀采样的环形缓冲区；去除数据相关性。 |
| 目标网络（Target network） | "冻结的自举" | 贝尔曼目标中使用的周期复制 Q；稳定训练。 |
| 致命三角（Deadly triad） | "强化学习为何发散" | 函数近似 + 自举 + 离策略 = 无收敛保证。 |
| 双重 DQN（Double DQN） | "最大偏差修复" | 在线网络选择动作，目标网络评估它。 |
| 对决 DQN（Dueling DQN） | "V 和 A 头" | 分解 Q = V + A - mean(A)；相同输出，更好的梯度流。 |
| Rainbow | "所有技巧的集合" | DDQN + PER + 对决 + n 步 + 噪声 + 分布式于一体。 |
| PER | "优先级回放" | 按 TD 误差量级比例采样转移。 |

## 延伸阅读

- [Mnih et al. (2013). Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) — 引爆深度强化学习的 2013 年 NeurIPS 研讨会论文
- [Mnih et al. (2015). Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) — Nature 论文，49 款游戏 DQN
- [Hasselt, Guez, Silver (2016). Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) — DDQN
- [Wang et al. (2016). Dueling Network Architectures](https://arxiv.org/abs/1511.06581) — 对决 DQN
- [Hessel et al. (2018). Rainbow: Combining Improvements in Deep RL](https://arxiv.org/abs/1710.02298) — 技巧叠加论文
- [OpenAI Spinning Up — DQN](https://spinningup.openai.com/en/latest/algorithms/dqn.html) — 清晰的现代解说
- [Sutton & Barto (2018). Ch. 9 — On-policy Prediction with Approximation](http://incompleteideas.net/book/RLbook2020.pdf) — "致命三角"的教科书处理
- [CleanRL DQN implementation](https://docs.cleanrl.dev/rl-algorithms/dqn/) — 用于消融研究的参考单文件 DQN；与本课的从零实现对照阅读
