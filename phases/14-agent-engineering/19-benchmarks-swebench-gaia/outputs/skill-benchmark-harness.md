---
name: benchmark-harness
description: 为代码库构建 SWE-bench 风格的测试装置，包含 FAIL_TO_PASS / PASS_TO_PASS 门控、污染检查和步骤计数指标。
version: 1.0.0
phase: 14
lesson: 19
tags: [swe-bench, gaia, agentbench, harness, evaluation]
---

给定一个代码库和一组（缺陷, 修复）对，构建一个基于真实单元测试进行门控并记录运营指标的基准测试装置。

输出内容：

1. 每个任务的定义：`(tid, description, state_before, fail_to_pass_tests, pass_to_pass_tests, solution)`。
2. 一个运行器：应用智能体补丁，在沙箱中运行代码库的测试套件，并记录：FTP 通过数、PTP 通过数、步骤数、token 数、挂钟时间、成本。
3. 污染检查：将问题文本与生成的补丁进行模式匹配；标记重叠度 >=30% 的情况。
4. 一个报告器：以 JSON 格式输出每任务和聚合分数，以及步骤数和成本的 P50/P75/P95 分位数。
5. 一个 CI 任务：在每次 PR 上运行测试装置，当回归 >=5% 时失败。

硬性拒绝：

- 只报告单一聚合数字的测试装置。必须提供每任务结果和分布数据。
- 不在沙箱中运行测试的测试装置。智能体提供的补丁是不可信代码。
- 没有 PASS_TO_PASS 门控的测试装置。破坏其他测试的补丁会悄无声息地让产品退化。

拒绝规则：

- 如果用户只要求"FAIL_TO_PASS 分数"，拒绝。加上 PASS_TO_PASS；破坏现有测试比没修复缺陷是更严重的退化。
- 如果测试没有固定到特定提交，拒绝。测试漂移会导致跨运行分数无法比较。
- 如果任务与训练期间见过的问题文本有重叠，显式标记。

输出：`tasks.py`、`harness.py`、`contamination.py`、`report.py`、`README.md`（说明沙箱、门控条件和污染策略）。末尾附"下一步阅读"，指向第 30 课，了解在测试装置之上进行评估驱动开发。
