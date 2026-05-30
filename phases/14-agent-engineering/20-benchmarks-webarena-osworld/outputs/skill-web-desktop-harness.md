---
name: web-desktop-harness
description: 构建 WebArena/OSWorld 风格的测试装置，包含基于执行的评估和轨迹效率指标。
version: 1.0.0
phase: 14
lesson: 20
tags: [webarena, osworld, harness, trajectory-efficiency]
---

给定目标应用（Web 或桌面）和一组带有黄金轨迹的任务列表，构建评估测试装置。

输出内容：

1. 任务定义：`(tid, description, gold_steps, success_predicate, state_reset)`。
2. 运行器：运行智能体，捕获每个动作，记录步骤数 + 耗时 + 成功状态。
3. 轨迹效率指标：`agent_steps / gold_steps`。报告每任务和聚合结果。
4. 任务间状态重置——绝不在被上一个任务污染的状态上运行下一个任务。
5. 失败模式分类器：对每次失败，打上标签，区分是定位失误（选错元素）还是规划失误（选错动作）。

硬性拒绝：

- 任务间不重置状态。跨任务污染会使所有分数无效。
- 只报告成功率。轨迹效率是 2026 年的标准。
- 只有截图而没有 DOM 对等的测试装置。某些智能体同时使用 DOM+视觉；除非有意限制接入面，否则两者都要提供。

拒绝规则：

- 如果任务没有黄金轨迹，拒绝。没有黄金轨迹就无法衡量效率。
- 如果应用没有固定到特定版本，拒绝。版本漂移会使跨运行比较失效。
- 如果智能体拥有破坏性工具（删除、发布），要求使用应用的沙箱副本。

输出：`tasks.py`、`runner.py`、`failure_classifier.py`、`report.py`、`README.md`（说明重置策略、黄金轨迹来源及定位与规划失误的区分方式）。末尾附"下一步阅读"，指向第 21 课（计算机使用模型）或第 30 课（评估驱动开发）。
