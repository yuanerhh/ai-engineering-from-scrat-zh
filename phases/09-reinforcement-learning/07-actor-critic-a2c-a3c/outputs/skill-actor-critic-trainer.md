---
name: actor-critic-trainer
description: 为给定环境生成 A2C / A3C / GAE 配置，明确指定优势估计和损失权重。
version: 1.0.0
phase: 9
lesson: 7
tags: [rl, actor-critic, gae]
---

给定环境和计算预算，输出以下内容：

1. 并行化方式。A2C（GPU 批处理）vs A3C（CPU 异步）及工作进程数量。
2. 回合长度 T。每个环境每次更新的步骤数。
3. 优势估计器。n 步或 GAE(λ)；指定 λ。
4. 损失权重。`c_v`（价值）、`c_e`（熵）、梯度裁剪。
5. 学习率。Actor 和 Critic 的学习率（如果分开使用）。

拒绝在时间跨度 > 1000 的环境中使用单工作进程的 A2C（过于在策略，速度太慢）。拒绝在没有优势归一化的情况下发布。标记任何 `c_e = 0` 且观测到熵 < 0.1 的运行为熵崩溃。
