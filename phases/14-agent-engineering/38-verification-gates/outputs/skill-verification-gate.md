---
name: verification-gate
description: 生成一个确定性验证门控，将范围、规则和反馈产物合并为每个任务的单一 verification_report.json，并配以 CI 接线，在没有绿色裁决的情况下拒绝合并。
version: 1.0.0
phase: 14
lesson: 38
tags: [verification, gate, deterministic, ci, override-log]
---

给定项目的验收标准和现有工作台产物，生成验证门控和覆盖审计日志。

需产出：

1. `tools/verify_agent.py`，暴露 `verify(task_id, artifacts) -> VerdictReport`。纯函数，确定性，无 LLM 调用。
2. `outputs/verification/<task_id>.json` 作为单一真实来源的裁决。
3. `tools/override.py`，向 `outputs/verification/overrides.jsonl` 追加签名覆盖条目（必须包含原因、用户 ID、时间戳、发现代码）。
4. 在 `passed: false` 时失败并内联展示报告的 CI 工作流。
5. `docs/verification.md`，列出每个检查项、其严重级别、来源产物以及覆盖策略。

强制拒绝：

- 调用 LLM 的检查项。门控是确定性的基础设施；LLM 判断属于审查者。
- Agent 可在没有签名条目的情况下走通的覆盖路径。覆盖仅限人工操作。
- 省略了所消费产物路径的验证报告。报告必须可审计。
- 工作流可以静默降级的阻断级别发现。严重级别在写入时固定，而不是在读取时。

拒绝规则：

- 如果项目没有验收命令，拒绝交付门控，直到存在验收命令为止。什么都证明不了的门控不过是摆设。
- 如果规则报告不存在，拒绝跳过规则检查；失败时关闭。
- 如果反馈日志不存在，拒绝跳过验收检查；缺失日志本身就是阻断项。
- 如果覆盖条目不受版本控制，拒绝接入覆盖路径；不留记录的覆盖会使门控形同虚设。

输出结构：

```
<repo>/
├── tools/
│   ├── verify_agent.py
│   └── override.py
├── outputs/verification/
│   ├── overrides.jsonl
│   └── <task_id>.json
├── docs/verification.md
└── .github/workflows/verify.yml
```

最后以"下一步阅读"结尾，指向：

- 第 39 课，了解在绿色裁决之后接手工作的审查者 Agent。
- 第 40 课，了解将裁决纳入数据包的交接生成器。
- 第 41 课，了解针对真实风格示例应用运行门控的方法。
