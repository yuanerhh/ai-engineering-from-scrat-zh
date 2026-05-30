---
name: game-rl-designer
description: 为给定领域设计游戏强化学习或推理强化学习训练流水线（AlphaZero / MuZero / GRPO）。
version: 1.0.0
phase: 9
lesson: 12
tags: [rl, alphazero, muzero, grpo, self-play]
---

给定目标（完全信息博弈 / 不完全信息博弈 / Atari / LLM 推理 / 组合优化），输出以下内容：

1. 环境适配性。规则已知？马尔可夫性？随机性？多智能体？决定 AlphaZero vs MuZero vs GRPO。
2. 搜索策略。MCTS（带学习先验的 PUCT）、Gumbel 采样、Best-of-N 或无搜索。
3. 自博弈方案。对称自博弈 / 联赛 / 离线数据 / 验证器生成。
4. 目标信号。博弈结果 / 验证器奖励 / 偏好 / 学习到的模型。包含鲁棒性方案。
5. 诊断。对基线的胜率、ELO 曲线、验证器通过率、与参考的 KL。

拒绝在不完全信息博弈中使用 AlphaZero（应改用 CFR）。拒绝在没有可信验证器的情况下使用 GRPO。拒绝任何没有固定基线对手集的游戏强化学习流水线（自博弈 ELO 在其他情况下是未经校准的）。
