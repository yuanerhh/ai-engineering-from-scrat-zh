---
name: elicitation-form-designer
description: 为需要在调用过程中进行用户确认或消歧的工具设计 elicitation 表单 Schema 和消息模板。
version: 1.0.0
phase: 13
lesson: 12
tags: [mcp, elicitation, user-input, forms]
---

给定一个在调用过程中可能需要用户输入的工具，设计 elicitation Schema 和消息。

输出内容：

1. 触发条件。说明应触发工具调用 `elicitation/create` 的确切输入或歧义情况。
2. 消息模板。宿主向用户展示的一句话。表达清晰、具体，无技术术语。
3. Schema。带有类型化属性的扁平 JSON Schema，以及用于消歧的 `enum` 列表或用于确认的 `boolean`。不得嵌套。
4. 分支处理。将 `accept` / `decline` / `cancel` 映射到工具行为。
5. 速率限制规则。为每次工具调用设置 elicitation 次数上限；绝不在循环内触发 elicitation。

硬性拒绝：
- 任何嵌套对象的 Schema。Elicitation v1 为扁平结构。
- 任何用于填补 LLM 本可以用自然语言询问的缺失参数的 elicitation。
- 任何高频 elicitation（每次工具调用超过一次）。

拒绝规则：
- 如果工具为只读且低风险，拒绝触发 elicitation，直接返回结果。
- 如果工具具有破坏性且宿主支持 `destructiveHint` 注释，建议使用注释并让客户端原生处理确认。
- 如果需求是 OAuth 登录，推荐使用 URL 模式 elicitation 并标记 SEP-1036 漂移风险。

输出：一页设计文档，包含触发条件、消息模板、Schema、分支处理、速率限制规则，以及关于表单模式和 URL 模式哪种更合适的说明。
