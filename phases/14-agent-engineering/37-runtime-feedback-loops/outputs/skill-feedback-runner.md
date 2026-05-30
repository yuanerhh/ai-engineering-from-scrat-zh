---
name: feedback-runner
description: 以确定性方式捕获 shell 命令的 stdout/stderr/退出码/耗时，每条命令持久化一条 JSONL 记录，并在反馈缺失时拒绝推进 Agent 循环。
version: 1.0.0
phase: 14
lesson: 37
tags: [feedback, subprocess, runner, jsonl, loop-control]
---

给定一个在 Agent 循环中运行 shell 命令的项目，生成反馈运行器及其写入的 JSONL。

需产出：

1. `tools/run_with_feedback.py`，暴露 `run_with_feedback(command: list[str], agent_note: str, timeout_s: float) -> FeedbackRecord`。
2. `feedback_record.jsonl` 位于工作台下，每行一条记录。
3. `tools/feedback_loader.py`，返回当前任务最近的 N 条记录。
4. Agent 循环在声明成功前调用的 `loop_can_advance(record) -> bool` 辅助函数。
5. 覆盖以下场景的测试：成功路径、非零退出、超时、缺失二进制文件、确定性头/尾截断。

强制拒绝：

- 运行器中任何地方使用 `shell=True`。只允许参数列表方式。
- 依赖系统时钟或随机采样的截断方式。相同输入必须产生相同记录。
- 不含 `duration_ms` 的记录。探针速度变慢是工作台卡死的第一个信号。
- 返回无界列表的加载器。限制为最后 N 条或分页。

拒绝规则：

- 如果项目通过 stdout 传递密钥，在没有脱敏步骤的情况下拒绝交付运行器。暴露会被捕获的行。
- 如果项目有可能无限期挂起的命令，在没有默认超时和明确覆盖列表的情况下拒绝交付。
- 如果运行器在有共享状态的 worker 内部运行，拒绝跳过 JSONL 追加操作周围的文件锁。多个写入者会损坏文件。

输出结构：

```
<repo>/
├── feedback_record.jsonl
└── tools/
    ├── run_with_feedback.py
    ├── feedback_loader.py
    └── test_feedback_runner.py
```

最后以"下一步阅读"结尾，指向：

- 第 38 课，了解消费这些记录的验证门控。
- 第 39 课，了解在为运行评分时读取反馈的审查者 Agent。
- 第 23 课，了解在反馈稳定后添加到遥测端的 OTel GenAI 约定。
