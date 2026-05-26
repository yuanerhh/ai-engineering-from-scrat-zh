# 演员-评论家——A2C 与 A3C

> REINFORCE 很嘈杂。添加一个学习 `V̂(s)` 的评论家，从回报中减去它，就得到了期望相同但方差低得多的优势。这就是演员-评论家。A2C 同步运行它；A3C 跨线程运行它。两者都是每个现代深度强化学习方法的思维模型。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 9 · 04（TD 学习）、Phase 9 · 06（REINFORCE）
**时长：** 约 75 分钟

## 问题背景

原始 REINFORCE 有效，但方差很糟糕。蒙特卡洛回报 `G_t` 在回合之间可以波动超过 10 倍。将这种噪声乘以 `∇ log π` 并取平均，产生的梯度估计器需要数千个回合才能使策略移动相同的距离，而 DQN 的更新次数要少得多。

方差来自使用原始回报。如果你减去一个基线 `b(s_t)`——任何状态的函数，包括学习到的价值——期望不变而方差下降。最好的可处理基线是 `V̂(s_t)`。现在乘以 `∇ log π` 的量是*优势（advantage）*：

`A(s, a) = G - V̂(s)`

如果一个动作产生了高于平均的回报，它就是好的；低于平均则是坏的。带有学习评论家的 REINFORCE 就是*演员-评论家（actor-critic）*。评论家给演员一个低方差的指导者。这就是 2015 年后每个深度策略方法（A2C、A3C、PPO、SAC、IMPALA）的本质。

## 核心概念

![演员-评论家：策略网络加价值网络，TD 残差作为优势](../assets/actor-critic.svg)

**两个网络，一个共享损失：**

- **演员（Actor）** `π_θ(a | s)`：策略。通过采样来行动。用策略梯度训练。
- **评论家（Critic）** `V_φ(s)`：估计从状态出发的期望回报。通过最小化 `(V_φ(s) - target)²` 训练。

**优势。** 两种标准形式：

- *MC 优势：* `A_t = G_t - V_φ(s_t)`。无偏，方差较高。
- *TD 优势：* `A_t = r_{t+1} + γ V_φ(s_{t+1}) - V_φ(s_t)`。有偏（使用 `V_φ`），方差低得多。也称为 *TD 残差* `δ_t`。

**n 步优势（n-step advantage）。** 在两者之间插值：

`A_t^{(n)} = r_{t+1} + γ r_{t+2} + … + γ^{n-1} r_{t+n} + γ^n V_φ(s_{t+n}) - V_φ(s_t)`

`n = 1` 是纯 TD。`n = ∞` 是 MC。大多数实现对 Atari 使用 `n = 5`，对 MuJoCo 上的 PPO 使用 `n = 2048`。

**广义优势估计（Generalized Advantage Estimation，GAE）。** Schulman 等（2016）提出所有 n 步优势的指数加权平均：

`A_t^{GAE} = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}`

其中 `λ ∈ [0, 1]`。`λ = 0` 是 TD（低方差，高偏差）。`λ = 1` 是 MC（高方差，无偏）。`λ = 0.95` 是 2026 年的默认值——调整直到偏差/方差平衡达到你想要的位置。

**A2C：同步优势演员-评论家（Synchronous Advantage Actor-Critic）。** 跨 `N` 个并行环境收集 `T` 步。为每一步计算优势。在合并的批量上更新演员和评论家。重复。A3C 更简单、更可扩展的兄弟方法。

**A3C：异步优势演员-评论家（Asynchronous Advantage Actor-Critic）。** Mnih 等（2016）。生成 `N` 个工作线程，每个运行一个环境。每个工作者在自己的展开数据上本地计算梯度，然后异步应用到共享参数服务器。无需回放缓冲区——工作者通过运行不同轨迹去相关。A3C 证明了可以在 CPU 上规模化训练。2026 年，基于 GPU 的 A2C（批量并行环境）占主导，因为 GPU 需要大批量。

**组合损失：**

`L(θ, φ) = -E[ A_t · log π_θ(a_t | s_t) ]  +  c_v · E[(V_φ(s_t) - G_t)²]  -  c_e · E[H(π_θ(·|s_t))]`

三项：策略梯度损失、价值回归、熵奖励。`c_v ~ 0.5`，`c_e ~ 0.01` 是规范的起始点。

## 动手实现

### 步骤一：评论家

线性评论家 `V_φ(s) = w · features(s)`，用 MSE 更新：

```python
def critic_update(w, x, target, lr):
    v_hat = dot(w, x)
    err = target - v_hat
    for j in range(len(w)):
        w[j] += lr * err * x[j]
    return v_hat
```

在表格环境上，评论家在几百个回合内收敛。在 Atari 上，将线性评论家替换为共享 CNN 主干 + 价值头。

### 步骤二：n 步优势

给定长度为 `T` 的展开数据和自举的最终 `V(s_T)`：

```python
def compute_advantages(rewards, values, gamma=0.99, lam=0.95, last_value=0.0):
    advantages = [0.0] * len(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_v = values[t + 1] if t + 1 < len(values) else last_value
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns
```

`returns` 是评论家目标。`advantages` 是乘以 `∇ log π` 的量。

### 步骤三：组合更新

```python
for step_i, (x, a, _r, probs) in enumerate(traj):
    adv = advantages[step_i]
    target_v = returns[step_i]

    # 评论家
    critic_update(w, x, target_v, lr_v)

    # 演员
    for i in range(N_ACTIONS):
        grad_logpi = (1.0 if i == a else 0.0) - probs[i]
        for j in range(N_FEAT):
            theta[i][j] += lr_a * adv * grad_logpi * x[j]
```

在策略，每次更新一次展开，演员和评论家使用单独的学习率。

### 步骤四：并行化（A3C vs A2C）

- **A3C：** 启动 `N` 个线程。每个运行自己的环境和自己的前向传播。定期将梯度更新推送到共享主控。主控上不加锁——竞争是可以的，只是增加噪声。
- **A2C：** 在单个进程中运行 `N` 个环境实例，将观察堆叠成 `[N, obs_dim]` 批量，批量前向传播，批量反向传播。更高的 GPU 利用率，确定性，更易推理。2026 年的默认方式。

我们的玩具代码为清晰起见是单线程的；重写为批量 A2C 只需三行 numpy。

## 常见陷阱

- **演员梯度之前评论家偏差。** 如果评论家是随机的，其基线没有信息量，训练在纯噪声上进行。在启动策略梯度之前，先用几百步热身评论家，或使用较慢的演员学习率。
- **优势归一化。** 将每批优势归一化为零均值/单位标准差。以近乎零代价大幅稳定训练。
- **共享主干。** 对图像输入，为演员和评论家使用共享特征提取器。分离的头。共享特征搭两个损失的便车。
- **在策略契约。** A2C 对每次数据恰好使用一次更新。超过一次，你的梯度就有偏差（重要性采样校正是 PPO 添加的内容）。
- **熵坍缩。** 没有 `c_e > 0`，策略在几百次更新内变得近确定性并停止探索。
- **奖励尺度。** 优势量级依赖于奖励尺度。归一化奖励（如运行标准差除法）以在不同任务间保持一致的梯度量级。

## 生产使用

A2C/A3C 在 2026 年很少是最终选择，但它们是所有后续方法的改进架构：

| 方法 | 与 A2C 的关系 |
|------|-------------|
| PPO | A2C + 截断重要性比率用于多轮更新 |
| IMPALA | A3C + V-trace 离策略校正 |
| SAC（Phase 9 · 07） | 带软价值评论家的离策略 A2C（下一课） |
| GRPO（Phase 9 · 12） | 没有评论家的 A2C——组相对优势 |
| DPO | A2C 折叠为偏好排序损失，无采样 |
| AlphaStar / OpenAI Five | A2C + 联赛训练 + 模仿预训练 |

如果你在 2026 年的论文中看到"优势（advantage）"，就联想到演员-评论家。

## 上手实践

保存为 `outputs/skill-actor-critic-trainer.md`：

```markdown
---
name: actor-critic-trainer
description: Produce an A2C / A3C / GAE configuration for a given environment, with advantage estimation and loss weights specified.
version: 1.0.0
phase: 9
lesson: 7
tags: [rl, actor-critic, gae]
---

给定环境和计算预算，输出：

1. 并行化。A2C（GPU 批量）vs A3C（CPU 异步）以及工作者数量。
2. 展开长度 T。每次更新每个环境的步数。
3. 优势估计器。n 步或 GAE(λ)；指定 λ。
4. 损失权重。`c_v`（价值）、`c_e`（熵）、梯度裁剪。
5. 学习率。演员和评论家（如果分开使用）。

拒绝在视野 > 1000 的环境上使用单工作者 A2C（过于在策略，太慢）。拒绝在没有优势归一化的情况下发布。将任何 `c_e = 0` 且观察熵 < 0.1 的运行标记为熵坍缩。
```

## 练习

1. **简单。** 在 4×4 GridWorld 上用 MC 优势（`G_t - V(s_t)`）训练演员-评论家。比较与第 06 课带滑动均值基线的 REINFORCE 的样本效率。
2. **中等。** 切换到 TD 残差优势（`r + γ V(s') - V(s)`）。测量优势批次的方差。下降了多少？
3. **困难。** 实现 GAE(λ)。扫描 `λ ∈ {0, 0.5, 0.9, 0.95, 1.0}`。绘制最终回报 vs 样本效率。偏差/方差的最优点在哪里？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 演员（Actor） | "策略网络" | `π_θ(a|s)`，通过策略梯度更新。 |
| 评论家（Critic） | "价值网络" | `V_φ(s)`，通过对回报/TD目标的 MSE 回归更新。 |
| 优势（Advantage） | "比平均好多少" | `A(s, a) = Q(s, a) - V(s)` 或其估计器。`∇ log π` 的乘数。 |
| TD 残差（TD residual） | "δ" | `δ_t = r + γ V(s') - V(s)`；单步优势估计。 |
| GAE | "插值旋钮" | n 步优势的指数加权和，由 `λ` 参数化。 |
| A2C | "同步演员-评论家" | 跨环境批量；每次展开一步梯度。 |
| A3C | "异步演员-评论家" | 工作线程将梯度推送到共享参数服务器。原始论文；2026 年较少使用。 |
| 自举（Bootstrap） | "在视野处使用 V" | 截断展开，添加 `γ^n V(s_{t+n})` 来闭合求和。 |

## 延伸阅读

- [Mnih et al. (2016). Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) — A3C，原始异步演员-评论家论文
- [Schulman et al. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) — GAE
- [Sutton & Barto (2018). Ch. 13 — Actor-Critic Methods](http://incompleteideas.net/book/RLbook2020.pdf) — 基础；当评论家是神经网络时，配合第 9 章函数近似阅读
- [Espeholt et al. (2018). IMPALA](https://arxiv.org/abs/1802.01561) — 带 V-trace 离策略校正的可扩展分布式演员-评论家
- [OpenAI Baselines / Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — 值得阅读的生产 A2C/PPO 实现
- [Konda & Tsitsiklis (2000). Actor-Critic Algorithms](https://papers.nips.cc/paper/1786-actor-critic-algorithms) — 两时间尺度演员-评论家分解的基础收敛结果
