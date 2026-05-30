# AGENTS.md

你正在一个运行 Agent 工作台的代码仓库中工作。

在采取任何行动之前，请先阅读以下内容：

1. `agent_state.json` — 上一个会话停止的位置。
2. `task_board.json` — 正在进行的任务以及下一个待处理的任务。
3. `docs/agent-rules.md` — 启动规则、禁止事项、完成条件、不确定情况处理、审批流程。
4. `docs/reliability-policy.md` — 本工作台设计用来吸收的故障模式。
5. `docs/handoff-protocol.md` — 会话结束时必须产出的内容。
6. `docs/reviewer-rubric.md` — 已完成工作的评判标准。

验证命令：请参见任务板上当前活跃任务中的 `acceptance_criteria`。

包版本：1.0.0
