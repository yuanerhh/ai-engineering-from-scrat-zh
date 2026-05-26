# 仿真到真实迁移（Sim-to-Real Transfer）

> 在模拟器中训练但在硬件上失败的策略，是一个记住了模拟器的策略。域随机化、域适应和系统辨识是让学到的控制器跨越现实差距的三种工具。

**类型：** 学习
**语言：** Python
**前置条件：** Phase 9 · 08（PPO）、Phase 2 · 10（偏差/方差）
**时长：** 约 45 分钟

## 问题背景

训练真实机器人缓慢、危险且昂贵。一个双足机器人需要数百万个训练回合才能学会行走；一个真实的双足机器人哪怕跌倒一次就可能损坏硬件。仿真提供了无限重置、确定性可重复性、并行环境，以及无物理损伤。

但模拟器是错误的。轴承的摩擦力比 MuJoCo 模型中更大。相机有模拟器未包含的镜头畸变。电机有 99% 的仿真模型忽略的延迟、反冲和饱和。风、灰尘和可变光照会破坏在无菌渲染环境中训练的策略。**现实差距（reality gap）**——仿真分布和真实分布之间的系统性差异——是部署强化学习机器人的核心问题。

你需要一个对*仿真到真实分布偏移具有鲁棒性*的策略。三种历史方法：随机化模拟器（域随机化）、用少量真实数据适应策略（域适应/微调），或辨识真实系统的参数并匹配它们（系统辨识）。2026 年，主流配方将三者结合，加上大规模并行仿真（Isaac Sim、Isaac Lab、GPU 上的 Mujoco MJX）。

## 核心概念

![三种仿真到真实制度：域随机化、适应、系统辨识](../assets/sim-to-real.svg)

**域随机化（Domain Randomization，DR）。** Tobin 等 2017，Peng 等 2018。训练期间，随机化可能在真实机器人上不同的每个仿真参数：质量、摩擦系数、电机 PD 增益、传感器噪声、相机位置、光照、纹理、接触模型。策略学习关于"今天处于哪个仿真"的条件分布，并在整个范围内泛化。如果真实机器人落在训练包络内，策略就能工作。

- **优点：** 不需要真实数据。一套配方，多种机器人。
- **缺点：** 过度随机化的训练产生"通用"但过于保守的策略。噪声太多 ≈ 正则化太多。

**系统辨识（System Identification，SI）。** 在训练前将模拟器参数拟合到真实世界数据。如果你能测量真实机器人的臂关节摩擦力，就把它代入仿真。然后训练期望这些值的策略。需要访问真实系统，但直接减少现实差距。

- **优点：** 精确、低噪声的训练目标。
- **缺点：** 残余模型误差对策略不可见；小的未识别效应（如电机死区）仍然会破坏部署。

**域适应（Domain Adaptation）。** 在仿真中训练，用少量真实数据微调。两种形式：

- **Real2Sim2Real：** 用真实展开学习残差模拟器 `f(s, a, z) - f_sim(s, a)`，在修正后的仿真中训练。无需大量真实数据就能缩小差距。
- **观察适应：** 训练将真实观察通过学习到的特征提取器映射到类仿真观察的策略（例如，GAN 像素到像素）。控制器保持在仿真中。

**特权学习 / 教师-学生（Privileged learning / teacher-student）。** Miki 等 2022（ANYmal 四足机器人）。在仿真中训练一个可以访问特权信息（地面真实摩擦力、地形高度、IMU 漂移）的*教师*。蒸馏一个只看真实传感器观察的*学生*。学生学会从历史中推断特权特征，在物理参数间具有鲁棒性。

**大规模并行仿真。** 2024–2026 年。Isaac Lab、Mujoco MJX、Brax 都在单个 GPU 上运行数千个并行机器人。带 4,096 个并行人形机器人的 PPO 在几小时内收集多年的经验。当那 4,096 个环境中每个都有不同随机参数时，"现实差距"随训练分布扩大而缩小；DR 几乎变得免费。

**2026 年真实世界配方（四足机器人行走示例）：**

1. 带域随机化重力、摩擦力、电机增益、有效载荷的大规模并行仿真。
2. 用特权信息（地形图、体速度真值）训练教师策略。
3. 仅使用本体感知（腿关节编码器）从教师蒸馏学生策略。
4. 可选：通过真实 IMU 上的自动编码器进行观察适应。
5. 部署。在 10+ 种环境上零样本。如果失败，用安全约束 PPO 进行几分钟的真实世界微调。

## 动手实现

本课代码是在带*噪声*转移的 GridWorld 上进行域随机化的微型演示。我们训练一个在"仿真"中经历随机滑移概率的策略，并在它从未在训练中见过的滑移水平的"真实"情况下评估。这种形状直接映射到 MuJoCo 到硬件的迁移。

### 步骤一：参数化仿真

```python
def step(state, action, slip):
    if rng.random() < slip:
        action = random_perpendicular(action)
    ...
```

`slip` 是模拟器暴露的参数。在真实机器人中它可以是摩擦力、质量、电机增益——任何在仿真和真实之间偏移的东西。

### 步骤二：用 DR 训练

每个回合开始时，采样 `slip ~ Uniform[0.0, 0.4]`。训练 PPO / Q 学习 / 任何算法。这样做很多回合。

### 步骤三：在"真实"滑移上零样本评估

在 `slip ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7}` 上评估。前四个在训练支持范围内；`0.5` 和 `0.7` 在外面。DR 训练的策略应该在支持范围内接近最优，在外面优雅降级。固定滑移训练的策略在其训练滑移之外会很脆弱。

### 步骤四：与狭窄训练比较

只在 `slip = 0.0` 上训练第二个策略。在同一滑移扫描上评估。一旦真实滑移 > 0，你应该看到灾难性下降。

## 常见陷阱

- **随机化太多。** 在 `slip ∈ [0, 0.9]` 上训练，策略会过于规避风险，从不尝试最优路径。匹配*期望的*真实世界分布，而非"任何事都可能发生"。
- **随机化太少。** 在薄片上训练，策略根本无法泛化。使用自适应课程（自动域随机化）随着策略改进扩大分布。
- **错误识别参数空间。** 随机化错误的东西（当真实差距在电机延迟时随机化相机色调），DR 没有帮助。先对真实机器人进行分析。
- **特权信息泄漏。** 教师使用全局状态作为动作（而非仅观察）产生的学生无法追上。确保教师的策略对学生基于观察历史是可实现的。
- **仿真到仿真迁移失败。** 如果策略对更难的仿真变体不鲁棒，它也不会对真实世界鲁棒。在部署前始终在保留的仿真变体上测试。
- **没有真实世界安全包络。** 在仿真中有效且在真实中"有效"但没有低级安全护盾的策略仍然可能损坏硬件。在非学习控制器中添加速率限制、力矩限制、关节限制。

## 生产使用

2026 年的仿真到真实技术栈：

| 领域 | 技术栈 |
|------|--------|
| 腿式运动（ANYmal、Spot、人形机器人） | Isaac Lab + DR + 特权教师/学生 |
| 操作（灵巧手、拾放） | Isaac Lab + DR + 视觉 DR-GAN |
| 自动驾驶 | CARLA / NVIDIA DRIVE Sim + DR + 真实微调 |
| 无人机竞速 | RotorS / Flightmare + DR + 在线适应 |
| 手指/手内操作 | OpenAI Dactyl（前所未有规模的 DR） |
| 工业机械臂 | MuJoCo-Warp + SI + 小型真实微调 |

对所有规模的控制，工作流是一致的：尽可能拟合仿真，随机化你无法拟合的，训练庞大的策略，蒸馏，带安全护盾部署。

## 上手实践

保存为 `outputs/skill-sim2real-planner.md`：

```markdown
---
name: sim2real-planner
description: Plan a sim-to-real transfer pipeline for a given robot + task, covering DR, SI, and safety.
version: 1.0.0
phase: 9
lesson: 11
tags: [rl, sim2real, robotics, domain-randomization]
---

给定机器人平台、任务和访问真实硬件的时间，输出：

1. 现实差距清单。按预期影响排序的可疑来源（接触、传感、执行延迟、视觉）。
2. DR 参数。精确列表、范围、分布。对照真实测量值证明每个范围的合理性。
3. SI 步骤。要测量哪些参数；测量方法。
4. 教师/学生划分。教师使用的特权信息；学生使用的观察。
5. 安全包络。低级限制、紧急停止、备用控制器。

拒绝在没有以下三项的情况下部署：(a) 零样本仿真变体测试，(b) 安全护盾，(c) 回滚计划。将任何 DR 范围超过测量真实变异性 3 倍的情况标记为可能过度随机化。
```

## 练习

1. **简单。** 在固定滑移 GridWorld（slip=0.0）上训练 Q 学习智能体。在 slip ∈ {0.0, 0.1, 0.3, 0.5} 上评估。绘制回报 vs 滑移曲线。
2. **中等。** 训练采样 `slip ~ Uniform[0, 0.3]` 的 DR Q 学习智能体。评估同样的扫描。在 slip=0.5（分布外）时，DR 带来了多少收益？
3. **困难。** 实现课程：从 slip=0.0 开始，每当策略达到最优的 90% 就扩大 DR 范围。测量达到 slip=0.3 零样本所需的总环境步数，与固定 DR 基线相比。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 现实差距（Reality gap） | "仿真到真实差异" | 训练和部署物理/传感之间的分布偏移。 |
| 域随机化（Domain randomization，DR） | "在随机仿真中训练" | 训练期间随机化仿真参数，使策略泛化。 |
| 系统辨识（System identification，SI） | "测量真实并拟合仿真" | 估计真实物理参数；将仿真设置为匹配。 |
| 域适应（Domain adaptation） | "在真实数据上微调" | 仿真训练后少量真实世界微调；可以适应观察或动力学。 |
| 特权信息（Privileged info） | "教师的真值" | 只有仿真才有的信息；学生必须从观察历史中推断。 |
| 教师/学生（Teacher/student） | "将特权蒸馏为可观察" | 教师用捷径训练；学生学会在没有捷径的情况下模仿。 |
| ADR | "自动域随机化" | 随着策略改进扩大 DR 范围的课程。 |
| Real2Sim | "用真实数据缩小差距" | 学习残差使仿真模仿真实展开。 |

## 延伸阅读

- [Tobin et al. (2017). Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907) — 原始 DR 论文（机器人视觉）
- [Peng et al. (2018). Sim-to-Real Transfer of Robotic Control with Dynamics Randomization](https://arxiv.org/abs/1710.06537) — 动力学 DR，四足机器人运动
- [OpenAI et al. (2019). Solving Rubik's Cube with a Robot Hand](https://arxiv.org/abs/1910.07113) — Dactyl，规模化 ADR
- [Miki et al. (2022). Learning robust perceptive locomotion for quadrupedal robots in the wild](https://www.science.org/doi/10.1126/scirobotics.abk2822) — ANYmal 的教师-学生方法
- [Makoviychuk et al. (2021). Isaac Gym: High Performance GPU Based Physics Simulation for Robot Learning](https://arxiv.org/abs/2108.10470) — 推动 2025–2026 年部署的大规模并行仿真
- [Akkaya et al. (2019). Automatic Domain Randomization](https://arxiv.org/abs/1910.07113) — ADR 课程方法
- [Sutton & Barto (2018). Ch. 8 — Planning and Learning with Tabular Methods](http://incompleteideas.net/book/RLbook2020.pdf) — 支撑现代仿真到真实流水线的 Dyna 框架
- [Zhao, Queralta & Westerlund (2020). Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: a Survey](https://arxiv.org/abs/2009.13303) — 带基准结果的仿真到真实方法分类
