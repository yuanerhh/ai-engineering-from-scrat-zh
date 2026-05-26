# 游戏强化学习——AlphaZero、MuZero 与 LLM 推理时代

> 1992 年：TD-Gammon 用纯 TD 在西洋双陆棋上击败了人类冠军。2016 年：AlphaGo 击败李世石。2017 年：AlphaZero 从零开始征服国际象棋、将棋和围棋。2024 年：DeepSeek-R1 证明同样的配方——用 GRPO 替换 PPO——在推理上同样有效。游戏是驱动本阶段每个突破的基准。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 9 · 05（DQN）、Phase 9 · 08（PPO）、Phase 9 · 09（RLHF）、Phase 9 · 10（多智能体强化学习）
**时长：** 约 120 分钟

## 问题背景

游戏拥有强化学习想要的一切：干净的奖励（胜/负）、无限回合（自博弈重置）、完美仿真（游戏*就是*模拟器）、离散或小型连续动作空间，以及迫使对抗鲁棒性的多智能体结构。

游戏也是每个重大强化学习突破的测试场。TD-Gammon（西洋双陆棋，1992）。Atari-DQN（2013）。AlphaGo（2016）。AlphaZero（2017）。OpenAI Five（Dota 2，2019）。AlphaStar（星际争霸 II，2019）。MuZero（学习模型，2019）。AlphaTensor（矩阵乘法，2022）。AlphaDev（排序算法，2023）。DeepSeek-R1（数学推理，2025）——游戏强化学习技术适用于文本的最新证明。

本课程总结三个里程碑式架构——AlphaZero、MuZero 和 GRPO——通过一个统一的视角：**自博弈 + 搜索 + 策略改进**。每个都是前一个的推广；GRPO 尤其是将 AlphaZero 的配方应用到语言模型推理上，以 token 为动作，以数学验证为胜利信号。

## 核心概念

![AlphaZero ↔ MuZero ↔ GRPO：相同的循环，不同的环境](../assets/rl-games.svg)

**统一循环：**

```
while True:
    trajectory = self_play(current_policy, search)     # 与自身对弈
    policy_target = search.improved_policy(trajectory) # 搜索改进原始策略
    policy_net.update(policy_target, value_target)     # 在搜索输出上监督学习
```

**AlphaZero（2017）。** Silver 等。给定带已知规则的游戏（国际象棋、将棋、围棋）：

- 策略-价值网络：单一主干 `f_θ(s) → (p, v)`。`p` 是合法走法上的先验。`v` 是预期游戏结果。
- 蒙特卡洛树搜索（MCTS）：每次走棋时，展开可能延续的树。用 `(p, v)` 作为先验 + 自举。通过 UCB（PUCT）选择节点：`a* = argmax Q(s, a) + c · p(a|s) · √N(s) / (1 + N(s, a))`。
- 自博弈：智能体对智能体进行博弈。在第 `t` 步，MCTS 访问分布 `π_t` 成为策略训练目标。
- 损失：`L = (v - z)² - π · log p + c · ||θ||²`。`z` 是游戏结果（+1 / 0 / -1）。

零人类知识。零手工启发式。同一套配方在数千万次自博弈后掌握了国际象棋、将棋和围棋。

**MuZero（2019）。** Schrittwieser 等。去除了规则已知的要求。

- 不是固定环境，而是学习一个*潜在动力学模型* `(h, g, f)`：
  - `h(s)`：将观察编码为潜在状态。
  - `g(s_latent, a)`：预测下一潜在状态 + 奖励。
  - `f(s_latent)`：预测策略先验 + 价值。
- MCTS 在*学习到的潜在空间*中运行。相同的搜索，相同的训练循环。
- 适用于围棋、国际象棋、将棋*以及* Atari——一个算法，无需规则知识。

**随机 MuZero（2022）。** 添加随机动力学和机会节点；扩展到西洋双陆棋类游戏。

**Muesli、Gumbel MuZero（2022-2024）。** 在样本效率和确定性搜索上的改进。

**GRPO（2024-2025）。** DeepSeek-R1 配方。同样的 AlphaZero 形式循环，应用于语言模型推理：

- "游戏"：回答数学/编程/推理问题。"胜利" = 验证器（测试用例通过，数值答案匹配）返回 1。
- 策略：语言模型。动作：token。状态：提示词 + 到目前为止的回复。
- 没有评论家（PPO 风格的 V_φ）。相反，对于每个提示词，从策略中采样 `G` 个补全。计算每个的奖励。使用**组相对优势** `A_i = (r_i - mean_r) / std_r` 作为 REINFORCE 风格更新的信号。
- 对参考策略的 KL 惩罚防止漂移（类似 RLHF）。
- 完整损失：

  `L_GRPO(θ) = -E_{q, {o_i}} [ (1/G) Σ_i A_i · log π_θ(o_i | q) ] + β · KL(π_θ || π_ref)`

无奖励模型，无评论家，无 MCTS。组相对基线取代了三者。在推理基准上以一小部分计算量达到或超过 PPO-RLHF 的质量。

**R1 完整配方。** DeepSeek-R1（DeepSeek 2025）一篇论文中包含两个模型：

- **R1-Zero。** 从 DeepSeek-V3 基础模型开始。没有 SFT。直接应用 GRPO，带两个奖励组件：*准确率奖励*（基于规则——最终答案是否正确解析 / 代码是否通过单元测试）和*格式奖励*（补全是否用 `<think>…</think>` 标签包裹其思维链）。经过数千步，平均回复长度从约 100 增长到约 10,000 个 token，数学基准分数攀升到接近 o1-preview 水平。模型从零开始学会推理。缺点：其思维链通常难以阅读，混合语言，缺乏风格一致性。
- **R1。** 用四阶段流水线修复 R1-Zero 的可读性问题：
  1. **冷启动 SFT。** 收集几千个带清晰格式的长思维链演示。在其上对基础模型进行监督微调。这提供了可读的起点。
  2. **面向推理的 GRPO。** 应用带准确率+格式奖励加*语言一致性*奖励（防止代码切换）的 GRPO。
  3. **拒绝采样 + 第二轮 SFT。** 从强化学习检查点采样约 60 万条推理轨迹，只保留最终答案正确且思维链可读的，与约 20 万条非推理 SFT 样本（写作、问答、自我认知）合并。再次微调基础模型。
  4. **全谱 GRPO。** 覆盖推理（基于规则的奖励）和一般对齐（有用性/无害性偏好奖励）的又一轮强化学习。

结果在开放权重下匹配 o1 在 AIME 和 MATH-500 上的表现，且足够小可以蒸馏。同一篇论文还通过对 R1 推理轨迹进行 SFT 发布了六个蒸馏密集模型（Qwen-1.5B 到 Llama-70B）——学生不需要强化学习。强大强化学习教师的蒸馏在学生规模上一贯优于从零开始的强化学习。

**GRPO 而非 PPO 用于推理的原因。** DeepSeekMath 论文（2024 年 2 月）给出三个原因：(1) 无需训练价值网络，内存减半；(2) 组基线自然处理推理任务产生的稀疏轨迹末端奖励；(3) 按提示词归一化使不同难度问题的优势可比，而 PPO 的单一评论家做不到。

**无搜索 vs 基于搜索。** 游戏已经分叉：

- *具有长视野的完全信息游戏*（围棋、国际象棋）：仍基于搜索。AlphaZero / MuZero 主导。
- *语言模型推理*：生产中尚无 MCTS；对完整展开使用 GRPO，推理计算使用 Best-of-N。过程奖励模型（PRM）暗示步级搜索将被加回。

## 动手实现

`code/main.py` 中的代码实现了**微型 GRPO**——一个带多组样本的赌博机。算法与在语言模型上相同；只是策略和环境更简单。它教授*损失*和*组相对优势*，这是 2025 年的创新。

### 步骤一：微型验证器环境

```python
QUESTIONS = [
    {"prompt": "q1", "correct": 3},
    {"prompt": "q2", "correct": 1},
]

def verify(prompt_idx, answer_token):
    return 1.0 if answer_token == QUESTIONS[prompt_idx]["correct"] else 0.0
```

在真实 GRPO 中，验证器运行单元测试或检查数学等式。

### 步骤二：策略：每个提示词上 K 个答案 token 的 softmax

```python
def policy_probs(theta, p_idx):
    return softmax(theta[p_idx])
```

等价于以提示词为条件的语言模型的最终层输出。

### 步骤三：组采样与组相对优势

```python
def grpo_step(theta, p_idx, G=8, beta=0.01, lr=0.1, rng=None):
    probs = policy_probs(theta, p_idx)
    samples = [sample(probs, rng) for _ in range(G)]
    rewards = [verify(p_idx, s) for s in samples]
    mean_r = sum(rewards) / G
    std_r = stddev(rewards) + 1e-8
    advs = [(r - mean_r) / std_r for r in rewards]

    for a, A in zip(samples, advs):
        grad = onehot(a) - probs
        for i in range(len(probs)):
            theta[p_idx][i] += lr * A * grad[i]
    # KL 惩罚：将 theta 拉向参考
    for i in range(len(probs)):
        theta[p_idx][i] -= beta * (theta[p_idx][i] - reference[p_idx][i])
```

组相对优势是 2024 年 DeepSeek 的技巧。不需要评论家。"基线"是组均值，归一化使用组标准差。

### 步骤四：与 REINFORCE 基线（无价值函数）比较

相同设置，相同计算量，普通 REINFORCE。GRPO 收敛更快、更稳定。

### 步骤五：观察熵和 KL

与 RLHF 相同的诊断：对参考模型的平均 KL、策略熵、随时间变化的奖励。一旦这些稳定，训练就完成了。

## 常见陷阱

- **通过验证器操控进行奖励黑客。** GRPO 继承了 RLHF 的风险：如果验证器是错误的或可利用的，语言模型会找到漏洞。健壮的验证器（多个测试用例，形式证明）很重要。
- **组大小太小。** 组基线的方差约为 `1/√G`。低于 `G = 4`，优势信号嘈杂；标准选择是 `G = 8` 到 `64`。
- **长度偏差。** 不同长度的语言模型补全有不同的对数概率。按 token 数归一化，或使用序列级对数概率，或截断到最大长度。
- **纯自博弈循环。** AlphaZero 风格的训练在一般和游戏中可能陷入支配循环。通过多样化对手池（联赛博弈，第 10 课）缓解。
- **搜索-策略不匹配。** AlphaZero 训练策略来模仿搜索输出。如果策略网络太小无法表示搜索的分布，训练会停滞。
- **计算下限。** MuZero / AlphaZero 需要大量计算。单个消融实验通常需要数百 GPU 小时。存在微型演示（例如，Connect Four 上的 AlphaZero）用于学习。
- **验证器覆盖。** 对有 bug 的解决方案通过的单元测试会强化 bug。设计能捕获边缘情况的验证器。

## 生产使用

2026 年按领域划分的游戏强化学习格局：

| 领域 | 主流方法 |
|------|---------|
| 双人零和棋盘游戏（围棋、国际象棋、将棋） | AlphaZero / MuZero / KataGo |
| 不完全信息牌类游戏（扑克） | CFR + 深度学习（DeepStack、Libratus、Pluribus） |
| Atari / 像素游戏 | Muesli / MuZero / IMPALA-PPO |
| 大型多人策略游戏（Dota、星际争霸） | PPO + 自博弈 + 联赛（OpenAI Five，AlphaStar） |
| 语言模型数学/代码推理 | GRPO（DeepSeek-R1，Qwen-RL，开放复现） |
| 语言模型对齐 | DPO / RLHF-PPO（非 GRPO；验证器是偏好而非可验证） |
| 机器人 | PPO + DR（非游戏强化学习，但使用相同的策略梯度工具） |
| 组合优化问题 | AlphaZero 变体（AlphaTensor，AlphaDev） |

*配方*——自博弈、搜索增强改进、策略蒸馏——跨越文本、像素和物理控制。GRPO 是最新的实例；更多正在到来。

## 上手实践

保存为 `outputs/skill-game-rl-designer.md`：

```markdown
---
name: game-rl-designer
description: Design a game-RL or reasoning-RL training pipeline (AlphaZero / MuZero / GRPO) for a given domain.
version: 1.0.0
phase: 9
lesson: 12
tags: [rl, alphazero, muzero, grpo, self-play]
---

给定目标（完全信息游戏 / 不完全信息 / Atari / 语言模型推理 / 组合优化），输出：

1. 环境适配。规则已知？马尔可夫？随机？多智能体？影响 AlphaZero vs MuZero vs GRPO 的选择。
2. 搜索策略。MCTS（带学习先验的 PUCT）、Gumbel 采样、Best-of-N 或无搜索。
3. 自博弈计划。对称自博弈 / 联赛 / 离线数据 / 验证器生成。
4. 目标信号。游戏结果 / 验证器奖励 / 偏好 / 学习模型。包含鲁棒性计划。
5. 诊断。相对基线的胜率、ELO 曲线、验证器通过率、对参考模型的 KL。

拒绝对不完全信息游戏使用 AlphaZero（转向 CFR）。拒绝没有可信验证器的 GRPO。拒绝任何没有固定基线对手集的游戏强化学习流水线（否则自博弈 ELO 没有校准）。
```

## 练习

1. **简单。** 在 `code/main.py` 中实现 GRPO 赌博机。在 2 个提示词 × 每个 4 个答案 token 上训练。用 `G=8` 在 < 1,000 次更新内收敛。
2. **中等。** 接入 PPO（截断）和普通 REINFORCE。与 GRPO 在同一赌博机上比较样本效率和奖励方差。
3. **困难。** 扩展到长度为 2 的"推理链"：智能体发出两个 token，验证器奖励这对 token。测量 GRPO 如何处理两步序列的信用分配。（提示：对*完整序列*计算组优势，传播到两个 token 位置。）

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| MCTS | "带学习网络的树搜索" | 蒙特卡洛树搜索；带学习 `(p, v)` 先验的 UCB1/PUCT 选择。 |
| AlphaZero | "自博弈 + MCTS" | 策略-价值网络，训练为匹配 MCTS 访问分布和游戏结果。 |
| MuZero | "学习模型的 AlphaZero" | 相同循环但通过学习到的动力学在潜在空间中。 |
| GRPO | "无评论家 PPO" | 组相对策略优化；带组均值基线 + KL 的 REINFORCE。 |
| PUCT | "AlphaZero 的 UCB" | `Q + c · p · √N / (1 + N_a)`——平衡价值估计与先验。 |
| 自博弈（Self-play） | "智能体对抗过去的自己" | 零和标准；对称训练信号。 |
| 联赛博弈（League play） | "基于种群的自博弈" | 过去 + 当前 + 利用者作为对手采样。 |
| 验证器奖励（Verifier reward） | "可验证强化学习" | 奖励来自确定性检查器（测试通过，答案匹配）。 |
| 过程奖励（Process reward） | "PRM" | 给每个推理步骤打分，而非只给最终答案。 |

## 延伸阅读

- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270)
- [Silver et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play (AlphaZero)](https://www.science.org/doi/10.1126/science.aar6404)
- [Schrittwieser et al. (2020). Mastering Atari, Go, chess and shogi by planning with a learned model (MuZero)](https://www.nature.com/articles/s41586-020-03051-4)
- [Vinyals et al. (2019). Grandmaster level in StarCraft II (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z)
- [DeepSeek-AI (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO)](https://arxiv.org/abs/2402.03300) — 引入 GRPO 和组相对基线的论文
- [DeepSeek-AI (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) — 完整的四阶段 R1 配方加 R1-Zero 消融
- [Brown et al. (2019). Superhuman AI for multiplayer poker (Pluribus)](https://www.science.org/doi/10.1126/science.aay2400) — 规模化 CFR + 深度学习
- [Tesauro (1995). Temporal Difference Learning and TD-Gammon](https://dl.acm.org/doi/10.1145/203330.203343) — 这一切的起点
- [Hugging Face TRL — GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) — 应用自定义奖励函数的 GRPO 生产参考
- [Qwen Team (2024). Qwen2.5-Math — GRPO 复现](https://github.com/QwenLM/Qwen2.5-Math) — 多种规模的 R1 配方开放复现
- [Sutton & Barto (2018). Ch. 17 — Frontiers of Reinforcement Learning](http://incompleteideas.net/book/RLbook2020.pdf) — 自博弈、搜索和"设计奖励"的教科书框架，R1 在语言模型规模上实例化了这些
