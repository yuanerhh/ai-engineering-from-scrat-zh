---
name: structured-output-picker
description: 选择结构化输出方案、模式设计和验证计划。
version: 1.0.0
phase: 5
lesson: 20
tags: [nlp, llm, structured-output]
---

给定一个使用场景（提供商、延迟预算、模式复杂度、容错要求），输出：

1. 机制。原生厂商结构化输出、Instructor 重试、Outlines FSM 或 XGrammar CFG。一句话说明理由。
2. 模式设计。字段顺序（推理在前，答案在后）、"未知"场景的可空字段、枚举与正则、必填字段。
3. 失败策略。最大重试次数、回退模型、优雅的 `null` 处理、分布外拒绝。
4. 验证计划。模式合规率（目标 100%）、语义有效性（LLM 裁判）、字段覆盖率、延迟 p50/p99。

拒绝将 `answer` 或 `decision` 置于推理字段之前的任何设计。拒绝使用无模式的裸 JSON 模式。标记仅限 FSM 库中的递归模式。
