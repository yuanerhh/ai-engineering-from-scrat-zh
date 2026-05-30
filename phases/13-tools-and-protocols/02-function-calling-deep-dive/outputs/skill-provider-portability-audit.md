---
name: provider-portability-audit
description: 针对某一提供商的函数调用集成，审计移植到其他两个提供商时会出现哪些问题。
version: 1.0.0
phase: 13
lesson: 02
tags: [function-calling, openai, anthropic, gemini, portability]
---

给定在某一提供商（OpenAI、Anthropic 或 Gemini）上的函数调用集成，生成一份可移植性审计报告，列出将相同逻辑迁移到其他两个提供商时出现的每处字段重命名、行为差异和硬性限制冲突。

输出内容：

1. 声明差异。对集成中的每个工具，展示在其他两个提供商上所需的封装结构/字段重命名/Schema 转换。标记目标提供商不支持的任何 JSON Schema 构造（Gemini：OpenAPI 3.0 子集；OpenAI strict：不支持 `$ref`、不支持模糊的 `oneOf`）。
2. 响应差异。记录每个提供商响应结构中工具调用的位置（`tool_calls[]` vs `content[]` 块 vs `parts[]` 条目），以及谁负责解析 `arguments`（OpenAI 为字符串，Anthropic 和 Gemini 为对象）。
3. `tool_choice` 差异。将集成当前的选择设置（auto / forbid / force / required）映射到目标提供商的格式；标记缺失的模式。
4. 限制冲突。报告工具数量限制（128 / 64 / 64）、Schema 深度（5 / 10 / 实际无限制）及每个参数的长度上限。对超出目标提供商限制的集成，以 block 严重性标记。
5. Strict 模式映射。说明目标上是否保留 strict 模式语义。OpenAI 的 `strict: true` 在 Anthropic 上没有完全等价物；Gemini 的 `responseSchema` 近似但作用于请求级别。

硬性拒绝：
- 任何在非 OpenAI 目标上假设 `arguments` 为字符串的集成。将悄无声息地产生错误结果。
- 任何迁移到 Anthropic 或 Gemini 时工具数量超过 64 且没有路由器的集成。
- 任何在目标为 OpenAI strict 模式时 Schema 中使用 `$ref` 的集成。

拒绝规则：
- 如果被要求移植的集成依赖某一提供商特有且没有对应物的功能（例如 OpenAI Responses API 的有状态轮次、Anthropic 的 computer-use 块），拒绝并说明哪个功能在目标上没有等价物。
- 如果被要求选出最优提供商，拒绝。选择取决于宿主的 strict 模式需求、成本概况和并行调用要求。

输出：一页审计报告，包含每个工具的差异表、限制对照表，以及针对每个目标提供商的最终"移植裁定"（ship / needs-router / blocked-by-feature）。以一句话说明影响最大的迁移变更。
