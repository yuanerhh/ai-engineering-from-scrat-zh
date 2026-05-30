---
name: structured-output-designer
description: 为自由文本抽取目标设计兼容 strict 模式的 JSON Schema 和 Pydantic 模型，并内置类型化拒绝和重试处理存根。
version: 1.0.0
phase: 13
lesson: 04
tags: [structured-output, json-schema, pydantic, strict-mode, extraction]
---

给定一个自由文本抽取目标（发票、简历、支持工单、研究摘要），生成一套生产就绪的抽取契约：JSON Schema 2020-12、Pydantic 模型、拒绝处理程序和重试策略。

输出内容：

1. JSON Schema 2020-12。每个属性均有类型。`required` 列出所有属性。每个对象上都有 `additionalProperties: false`。封闭值集使用枚举。不使用 `$ref`。不使用模糊的 `oneOf` / `anyOf`。通过 OpenAI strict 模式要求验证。
2. Pydantic v2 BaseModel。与 Schema 对应的 Python 类型镜像。`model_json_schema()` 必须生成与（1）等价的 Schema。
3. 拒绝处理程序。类型化的 `Refusal(reason: str, category: str)` 输出。列出分类：`safety`、`input_mismatch`、`insufficient_info`。
4. 重试策略。三种重试形式：(a) 注入验证错误并重试一次（非 strict 模式外部使用）；(b) 接受拒绝为最终结果（strict 模式）；(c) 多次拒绝后升级到更强的模型。
5. 测试向量。十个输入，涵盖正常路径、对抗性字段、部分输入和触发拒绝的情况。每个均附预期结果。

硬性拒绝：
- 任何含有未类型化字段的 Schema。在 strict 模式和验证器中均会失败。
- 任何缺少 `additionalProperties: false` 的 Schema。会泄漏幻觉内容。
- 任何使用没有判别字段的 `oneOf` 的 Schema。解码存在歧义。
- 任何未经 JSON Schema 往返校验的 Pydantic 模型。

拒绝规则：
- 如果目标域包含个人身份数据但没有记录用途，拒绝并路由到 Phase 18（伦理）进行合法依据论证。
- 如果用户要求的 Schema 无法用 JSON Schema 2020-12 表达（例如递归任意图），拒绝并提出最接近的可表达放宽方案。
- 如果抽取目标是"从任意内容中提取结构化数据"，拒绝并要求明确具体领域。

输出：一页契约，包含 Schema JSON、Pydantic 类、拒绝与重试策略，以及十个测试向量。以一条说明首选目标提供商及原因的注释作为结尾。
