---
name: policy-gradient-trainer
description: 为给定任务生成 REINFORCE / Actor-Critic / PPO 训练配置并诊断方差问题。
version: 1.0.0
phase: 9
lesson: 6
tags: [rl, policy-gradient, reinforce]
---

给定环境（离散/连续动作、时间跨度、奖励统计），输出以下内容：

1. 策略头部。Softmax（离散）或高斯（连续），带参数数量。
2. 基线。无（vanilla）、滚动均值、学习到的 `V̂(s)` 或 A2C 评论家。
3. 方差控制。默认开启"奖励到终点"、回报归一化、梯度裁剪值。
4. 熵奖励。系数 β 及衰减调度。
5. 批次大小。每次更新的回合数；在策略数据新鲜度约定。

拒绝在时间跨度 > 500 步时使用无基线的 REINFORCE。拒绝连续动作控制使用 Softmax 头部。标记任何 `β = 0` 且观测到策略熵 < 0.1 的运行为熵崩溃。
