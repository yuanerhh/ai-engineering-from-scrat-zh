# 策略梯度——从零实现 REINFORCE

> 停止估计价值。直接参数化策略，计算期望回报的梯度，沿坡向上走。Williams（1992）用一个定理写出了它。这就是 PPO、GRPO 以及每个 LLM 强化学习循环存在的原因。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 3 · 03（反向传播）、Phase 9 · 03（蒙特卡洛）、Phase 9 · 04（TD 学习）
**时长：** 约 75 分钟

## 问题背景

Q 学习和 DQN 对*价值*函数进行参数化。通过 `argmax Q` 选择动作。这在离散动作和离散状态下没问题，但在以下情况下会失效：动作是连续的（对 10 维力矩如何取 `argmax`？），或者你想要一个随机策略（`argmax` 从定义上是确定性的）。

策略梯度（Policy gradients）直接对*策略*进行参数化。`π_θ(a | s)` 是一个神经网络，输出动作上的分布。从中采样来行动。计算期望回报关于 `θ` 的梯度。沿坡向上走。无需 `argmax`，无需贝尔曼递归。只是对 `J(θ) = E_{π_θ}[G]` 做梯度上升。

REINFORCE 定理（Williams 1992）告诉你这个梯度是可计算的：`∇J(θ) = E_π[ G · ∇_θ log π_θ(a | s) ]`。运行一个回合，计算回报，在每一步乘以 `∇ log π_θ(a | s)`，取平均，梯度上升，完毕。

2026 年每个 LLM 强化学习算法——PPO、DPO、GRPO——都是 REINFORCE 的改进。在肌肉记忆层面理解它是本阶段其余部分以及 Phase 10 · 07（RLHF 实现）和 Phase 10 · 08（DPO）的前提条件。

## 核心概念

![策略梯度：softmax 策略，log-π 梯度，回报加权更新](../assets/policy-gradient.svg)

**策略梯度定理。** 对于任何以 `θ` 参数化的策略 `π_θ`：

`∇J(θ) = E_{τ ~ π_θ}[ Σ_{t=0}^{T} G_t · ∇_θ log π_θ(a_t | s_t) ]`

其中 `G_t = Σ_{k=t}^{T} γ^{k-t} r_{k+1}` 是从步骤 `t` 开始的折扣回报。期望是在从 `π_θ` 采样的完整轨迹 `τ` 上的。

**证明很短。** 在期望下对 `J(θ) = Σ_τ P(τ; θ) G(τ)` 求微分。使用对数导数技巧（log-derivative trick）`∇P(τ; θ) = P(τ; θ) ∇ log P(τ; θ)`。分解 `log P(τ; θ) = Σ log π_θ(a_t | s_t) + 不依赖 θ 的环境项`。环境项消失。两行代数给出定理。

**方差减少技巧。** 原始 REINFORCE 方差极大——回报有噪声，`∇ log π` 有噪声，它们的乘积噪声很大。两个标准修复：

1. **基线减法（Baseline subtraction）。** 用 `G_t - b(s_t)` 替换 `G_t`，其中基线 `b(s_t)` 不依赖于 `a_t`。无偏，因为 `E[b(s_t) · ∇ log π(a_t | s_t)] = 0`。典型选择：由评论家（critic）学习的 `b(s_t) = V̂(s_t)` → 演员-评论家（第 07 课）。
2. **未来回报（Reward-to-go）。** 用 `Σ_t G_t^{from t} · ∇ log π_θ(a_t | s_t)` 替换 `Σ_t G_t · ∇ log π_θ(a_t | s_t)`。对于给定动作，只有未来回报有关——过去奖励贡献零均值噪声。

两者结合，得到：

`∇J ≈ (1/N) Σ_{i=1}^{N} Σ_{t=0}^{T_i} [ G_t^{(i)} - V̂(s_t^{(i)}) ] · ∇_θ log π_θ(a_t^{(i)} | s_t^{(i)})`

这是带基线的 REINFORCE——A2C（第 07 课）和 PPO（第 08 课）的直接前身。

**Softmax 策略参数化。** 对于离散动作，标准选择：

`π_θ(a | s) = exp(f_θ(s, a)) / Σ_{a'} exp(f_θ(s, a'))`

其中 `f_θ` 是任何为每个动作输出一个分数的神经网络。梯度有简洁的形式：

`∇_θ log π_θ(a | s) = ∇_θ f_θ(s, a) - Σ_{a'} π_θ(a' | s) ∇_θ f_θ(s, a')`

即所取动作的分数减去其在策略下的期望值。

**连续动作的高斯策略。** `π_θ(a | s) = N(μ_θ(s), σ_θ(s))`。`∇ log N(a; μ, σ)` 有封闭形式。这就是 Phase 9 · 07 的 SAC 所需的全部。

## 动手实现

### 步骤一：Softmax 策略网络

```python
def policy_logits(theta, state_features):
    return [dot(theta[a], state_features) for a in range(N_ACTIONS)]

def softmax(logits):
    m = max(logits)
    exps = [exp(l - m) for l in logits]
    Z = sum(exps)
    return [e / Z for e in exps]
```

对表格环境使用线性策略（每个动作一个权重向量）。对 Atari，换入 CNN 并保留 softmax 头。

### 步骤二：采样与对数概率

```python
def sample_action(probs, rng):
    x = rng.random()
    cum = 0
    for a, p in enumerate(probs):
        cum += p
        if x <= cum:
            return a
    return len(probs) - 1

def log_prob(probs, a):
    return log(probs[a] + 1e-12)
```

### 步骤三：捕获对数概率的展开

```python
def rollout(theta, env, rng, gamma):
    trajectory = []
    s = env.reset()
    while not done:
        logits = policy_logits(theta, s)
        probs = softmax(logits)
        a = sample_action(probs, rng)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r, probs))
        s = s_next
    return trajectory
```

### 步骤四：REINFORCE 更新

```python
def reinforce_step(theta, trajectory, gamma, lr, baseline=0.0):
    returns = compute_returns(trajectory, gamma)
    for (s, a, _, probs), G in zip(trajectory, returns):
        advantage = G - baseline
        grad_log_pi_a = [-p for p in probs]
        grad_log_pi_a[a] += 1.0
        for i in range(N_ACTIONS):
            for j in range(len(s)):
                theta[i][j] += lr * advantage * grad_log_pi_a[i] * s[j]
```

梯度 `∇ log π(a|s) = e_a - π(·|s)`（`a` 的独热向量减去概率）是 softmax 策略梯度的核心。烂熟于胸。

### 步骤五：基线

最近几个回合 `G` 的滑动均值足以减少方差，让 4×4 GridWorld 运行；约 500 个回合收敛。将基线升级为学习到的 `V̂(s)` 就得到了演员-评论家。

## 常见陷阱

- **梯度爆炸。** 回报可能很大。在乘以 `∇ log π` 之前，始终将批次中的 `G` 归一化到 `~N(0, 1)`。
- **熵坍缩（Entropy collapse）。** 策略过早收敛到近确定性动作，停止探索，陷入困境。修复：在目标中添加熵奖励 `β · H(π(·|s))`。
- **高方差。** 原始 REINFORCE 需要数千个回合。评论家基线（第 07 课）或 TRPO/PPO 的信任域（第 08 课）是标准修复。
- **样本效率低。** 在策略意味着每次更新后丢弃所有转移。通过重要性采样的离策略修正可以带回数据，但代价是方差增大（PPO 的比率是截断的 IS 权重）。
- **非平稳梯度。** 100 个回合前的相同梯度使用旧的 `π`。在策略方法因此每隔几次展开就更新一次。
- **信用分配。** 没有未来回报，过去奖励贡献噪声。始终使用未来回报（reward-to-go）。

## 生产使用

2026 年，REINFORCE 很少被直接运行，但其梯度公式无处不在：

| 使用场景 | 派生方法 |
|---------|---------|
| 连续控制 | 带高斯策略的 PPO / SAC |
| LLM RLHF | 带 KL 惩罚的 PPO，在 token 级别策略上运行 |
| LLM 推理（DeepSeek） | GRPO——带组相对基线的 REINFORCE，无评论家 |
| 多智能体 | 中心化评论家 REINFORCE（MADDPG、COMA） |
| 离散动作机器人 | A2C、A3C、PPO |
| 仅偏好设置 | DPO——将 REINFORCE 重写为偏好似然损失，无采样 |

当你在 2026 年的训练脚本中读到 `loss = -advantage * log_prob` 时，那就是带基线的 REINFORCE。整篇论文（DPO、GRPO、RLOO）都是在这一行的基础上进行方差减少的技巧。

## 上手实践

保存为 `outputs/skill-policy-gradient-trainer.md`：

```markdown
---
name: policy-gradient-trainer
description: Produce a REINFORCE / actor-critic / PPO training config for a given task and diagnose variance issues.
version: 1.0.0
phase: 9
lesson: 6
tags: [rl, policy-gradient, reinforce]
---

给定环境（离散 / 连续动作，视野，奖励统计），输出：

1. 策略头。Softmax（离散）或高斯（连续），以及参数数量。
2. 基线。无（原始）、滑动均值、学习到的 `V̂(s)` 或 A2C 评论家。
3. 方差控制。默认开启未来回报，回报归一化，梯度裁剪值。
4. 熵奖励。系数 β 和衰减调度。
5. 批量大小。每次更新的回合数；在策略数据新鲜度契约。

拒绝在视野 > 500 步时使用无基线 REINFORCE。拒绝用 softmax 头进行连续动作控制。将任何 `β = 0` 且观察到策略熵 < 0.1 的运行标记为熵坍缩。
```

## 练习

1. **简单。** 在 4×4 GridWorld 上用线性 softmax 策略实现 REINFORCE。不带基线训练 1,000 个回合。绘制学习曲线；测量方差（回报标准差）。
2. **中等。** 添加滑动均值基线。重新训练。比较样本效率和方差与原始运行的差异。基线将收敛步数减少了多少？
3. **困难。** 添加熵奖励 `β · H(π)`。扫描 `β ∈ {0, 0.01, 0.1, 1.0}`。绘制最终回报和策略熵。在这个任务上，最优点在哪里？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 策略梯度（Policy gradient） | "直接训练策略" | `∇J(θ) = E[G · ∇ log π_θ(a|s)]`；由对数导数技巧推导。 |
| REINFORCE | "原始 PG 算法" | Williams（1992）；蒙特卡洛回报乘以对数策略梯度。 |
| 对数导数技巧（Log-derivative trick） | "得分函数估计器" | `∇P(τ;θ) = P(τ;θ) · ∇ log P(τ;θ)`；使期望的梯度可处理。 |
| 基线（Baseline） | "方差减少" | 从 `G` 中减去的任何 `b(s)`；无偏，因为 `E[b · ∇ log π] = 0`。 |
| 未来回报（Reward-to-go） | "只有未来回报计数" | 用 `G_t^{from t}` 代替完整的 `G_0`；正确且方差更低。 |
| 熵奖励（Entropy bonus） | "鼓励探索" | `+β · H(π(·|s))` 项防止策略坍缩。 |
| 在策略（On-policy） | "从刚看到的训练" | 梯度期望关于当前策略——不能直接重用旧数据。 |
| 优势（Advantage） | "比平均好多少" | `A(s, a) = G(s, a) - V(s)`；带基线 REINFORCE 乘以的有符号量。 |

## 延伸阅读

- [Williams (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/article/10.1007/BF00992696) — 原始 REINFORCE 论文
- [Sutton et al. (2000). Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html) — 带函数近似的现代策略梯度定理
- [Sutton & Barto (2018). Ch. 13 — Policy Gradient Methods](http://incompleteideas.net/book/RLbook2020.pdf) — 教科书介绍
- [OpenAI Spinning Up — VPG / REINFORCE](https://spinningup.openai.com/en/latest/algorithms/vpg.html) — 带 PyTorch 代码的清晰教学解说
- [Peters & Schaal (2008). Reinforcement Learning of Motor Skills with Policy Gradients](https://homes.cs.washington.edu/~todorov/courses/amath579/reading/PolicyGradient.pdf) — 方差减少和自然梯度观点，连接 REINFORCE 到信任域族（TRPO、PPO）
