---
name: handoff-generator
description: 从工作台产物生成会话结束交接数据包，同时产出人类可读的 Markdown 和以七个规范字段为键的机器可读 JSON。
version: 1.0.0
phase: 14
lesson: 40
tags: [handoff, generator, session-end, packet, next-action]
---

给定一个工作台（状态、裁决、审查、反馈日志、差异），生成接入 Agent 运行时的会话结束交接生成器。

需产出：

1. `tools/generate_handoff.py`，暴露 `generate_handoff(snapshot) -> (markdown, payload)`。
2. `outputs/handoff/<session_id>/handoff.md` 和 `handoff.json`。
3. `handoff.schema.json`，覆盖七个必填字段和反馈尾部格式。
4. 会话结束钩子脚本，运行生成器，并在任何字段缺失时拒绝关闭会话。
5. `docs/handoff.md`，列出七个字段、其来源以及裁剪策略。

强制拒绝：

- 没有 `next_action` 的交接数据包。伪装成交接的状态报告会毒害下一个会话。
- 手动撰写摘要的生成器。Agent 的职责是让工作台处于可生成状态。
- 与 JSON 内容不一致的 Markdown 数据包。JSON 是来源；Markdown 是 JSON 的渲染结果。
- 超过 30 条的反馈尾部。完整日志在版本控制中；数据包必须保持精简。

拒绝规则：

- 如果验证报告缺失，拒绝生成数据包。没有裁决的交接只是一个愿望。
- 如果审查报告缺失且预期有人工审查者，拒绝并要求先通过审查。
- 如果差异摘要为空但会话运行超过 5 分钟，在生成前暴露该异常；怀疑是卡死的会话而非真正的无操作。

输出结构：

```
<repo>/
├── outputs/handoff/<session_id>/
│   ├── handoff.md
│   └── handoff.json
├── tools/generate_handoff.py
├── handoff.schema.json
└── docs/handoff.md
```

最后以"下一步阅读"结尾，指向：

- 第 41 课，了解在真实风格示例应用上的端到端演练。
- 第 42 课，了解将生成器打包进 Capstone 工作台包。
- 第 29 课（生产运行时），了解如何将会话结束接入队列、事件和 cron 触发器。
