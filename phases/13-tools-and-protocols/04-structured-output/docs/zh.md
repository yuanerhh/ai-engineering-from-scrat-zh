# 结构化输出——JSON Schema、Pydantic、Zod、约束解码

> "礼貌地要求模型返回 JSON"会有 5%-15% 的失败率，即使在前沿模型上也是如此。结构化输出通过约束解码填补了这一差距：模型被从字面上阻止输出会违反 schema 的词元。OpenAI 的严格模式、Anthropic 的 schema 类型化工具使用、Gemini 的 `responseSchema`、Pydantic AI 的 `output_type` 和 Zod 的 `.parse` 是同一个思路的五种外在形式。本课构建 schema 验证器和严格模式契约，学习者将在每个生产提取流水线中使用它们。

**类型：** 构建
**编程语言：** Python（标准库，JSON Schema 2020-12 子集）
**前置知识：** Phase 13 · 02（函数调用深度解析）
**预计时间：** 约 75 分钟

## 学习目标

- 使用正确的约束（enum、min/max、required、pattern）为提取目标编写 JSON Schema 2020-12。
- 解释为何严格模式和约束解码给出的保证与"生成后验证"不同。
- 区分三种失败模式：解析错误、schema 违反、模型拒绝。
- 构建带类型化修复和类型化拒绝处理的提取流水线。

## 问题背景

一个读取采购订单邮件的智能体需要将自由文本转换为 `{customer, line_items, total_usd}`。三种方法：

**方法一：提示输出 JSON。** "以包含 customer、line_items、total_usd 字段的 JSON 格式回复。"在前沿模型上 85%-95% 的时间有效。以六种方式失败：缺少大括号、尾随逗号、错误类型、幻觉字段、在词元限制处截断、泄漏"这是您的 JSON："之类的散文。

**方法二：生成后验证。** 自由生成，解析，根据 schema 验证，失败时重试。可靠但昂贵——每次重试都要付费，截断 bug 每次出现要多花一轮。

**方法三：约束解码。** 提供商在解码时强制执行 schema，无效词元从采样分布中被屏蔽。输出保证可解析，保证通过验证。失败只剩一种模式：拒绝（模型判断输入不符合 schema）。

2026 年每家前沿提供商都提供了某种形式的第三种方法：

- **OpenAI。** `response_format: {type: "json_schema", strict: true}` 加上模型拒绝时响应中的 `refusal`。
- **Anthropic。** 对 `tool_use` 输入的 schema 强制执行；`stop_reason: "refusal"` 不存在，但没有工具调用的 `end_turn` 就是信号。
- **Gemini。** 请求级的 `responseSchema`；2026 年 Gemini 对部分类型提供了词元级语法约束。
- **Pydantic AI。** `output_type=InvoiceModel` 输出类型为 `InvoiceModel` 的结构化 `RunResult`。
- **Zod（TypeScript）。** 根据 Zod schema 验证提供商输出的运行时解析器；与 OpenAI 的 `beta.chat.completions.parse` 配合使用。

共同点：声明一次 schema，端到端强制执行。

## 核心概念

### JSON Schema 2020-12——通用语言

每家提供商都接受 JSON Schema 2020-12。最常用的构造：

- `type`：`object`、`array`、`string`、`number`、`integer`、`boolean`、`null` 之一。
- `properties`：字段名到子 schema 的映射。
- `required`：必须出现的字段名列表。
- `enum`：允许值的封闭集合。
- `minimum`/`maximum`（数字）、`minLength`/`maxLength`/`pattern`（字符串）。
- `items`：应用于每个数组元素的子 schema。
- `additionalProperties`：`false` 禁止额外字段（默认值因模式而异）。

OpenAI 严格模式增加了三个要求：每个属性都必须列在 `required` 中，`additionalProperties: false` 在所有地方，以及没有未解析的 `$ref`。违反这些，API 在请求时返回 400。

### Pydantic——Python 绑定

Pydantic v2 通过 `model_json_schema()` 从数据类形状的模型生成 JSON Schema。Pydantic AI 将其包装，使你可以这样写：

```python
class Invoice(BaseModel):
    customer: str
    line_items: list[LineItem]
    total_usd: Decimal
```

智能体框架在边界将 schema 翻译为 OpenAI 严格模式、Anthropic `input_schema` 或 Gemini `responseSchema`。模型的输出以类型化的 `Invoice` 实例返回，验证错误抛出带类型化错误路径的 `ValidationError`。

### Zod——TypeScript 绑定

Zod（`z.object({customer: z.string(), ...})`）是 TS 的等价物。OpenAI 的 Node SDK 暴露了 `zodResponseFormat(Invoice)`，将其翻译为 API 的 JSON Schema 载荷。

### 拒绝

严格模式无法强迫模型回答。如果输入无法符合 schema（"邮件是一首诗，不是发票"），模型输出包含原因的 `refusal` 字段。你的代码必须将此作为一等结果处理，而非失败。拒绝也是一个有用的安全信号：被要求从受保护内容邮件中提取信用卡号的模型会返回附有安全原因的拒绝。

### 开放环境下的约束解码

开放权重实现使用三种技术：

1. **基于语法的解码**（`outlines`、`guidance`、`lm-format-enforcer`）：从 schema 构建确定性有限自动机；在每一步，屏蔽会违反 FSM 的词元的 logit。
2. **带 JSON 解析器的 logit 掩码**：与模型同步运行流式 JSON 解析器；在每一步计算有效下一词元集合。
3. **带验证器的投机解码**：廉价草稿模型提出词元，验证器强制执行 schema。

商业提供商在幕后选择这些方法之一。2026 年的最新实现对短结构化输出比普通生成更快，对长输出速度大致相同。

### 三种失败模式

1. **解析错误。** 输出不是有效 JSON。严格模式下不可能发生，非严格提供商下仍可能出现。
2. **Schema 违反。** 输出可以解析但违反了 schema。严格模式下不可能发生，在严格模式外很常见。
3. **拒绝。** 模型拒绝。必须作为类型化结果处理。

### 重试策略

当不在严格模式下（Anthropic 工具使用、非严格 OpenAI、较旧的 Gemini），恢复模式是：

```
生成 -> 解析 -> 验证 -> 如果失败，注入错误并重试，最多 3 次
```

通常一次重试就足够。三次重试能捕获弱模型的偶发错误。超过三次则是 schema 设计有问题的迹象：模型在某些输入上无法满足它，提示或 schema 需要修复。

### 小模型支持

约束解码适用于小模型。带语法强制执行的 30 亿参数开放模型，在结构化任务上优于使用原始提示的 700 亿参数模型。这是结构化输出对生产重要的主要原因：它将可靠性与模型大小解耦。

## 动手实践

`code/main.py` 提供了标准库中的最小 JSON Schema 2020-12 验证器（类型、required、enum、min/max、pattern、items、additionalProperties）。它包装了一个 `Invoice` schema，并将假 LLM 输出通过验证器运行，演示解析错误、schema 违反和拒绝路径。在生产中将假输出替换为任何提供商的真实响应。

重点关注：
- 验证器返回带路径和消息的类型化 `[ValidationError]` 列表，这是你想要暴露给重试提示的形式。
- 拒绝分支不重试，记录并返回类型化拒绝。Phase 14 · 09 将拒绝用作安全信号。
- `additionalProperties: false` 检查在对抗性测试输入上触发，显示为什么严格模式能关上幻觉字段的大门。

## 产出技能

本课产出 `outputs/skill-structured-output-designer.md`：给定一个自由文本提取目标（发票、支持工单、简历等），该技能产出一个严格模式兼容的 JSON Schema 2020-12 和一个镜像它的 Pydantic 模型，并带有类型化拒绝和重试处理的存根。

## 练习

1. 运行 `code/main.py`。添加第四个 `total_usd` 为负数的测试用例，确认验证器用 `minimum` 约束路径拒绝它。

2. 扩展验证器以支持带判别器的 `oneOf`。常见情况：`line_item` 是产品或服务，由 `kind` 标记。严格模式在这里有微妙规则；查看 OpenAI 的结构化输出指南。

3. 将相同的 Invoice schema 写成 Pydantic BaseModel，并将 `model_json_schema()` 输出与手写 schema 对比。找出 Pydantic 默认设置但手写版本遗漏的那一个字段。

4. 测量拒绝率。构建十个不应该可提取的输入（歌词、数学证明、空白邮件），通过带严格模式的真实提供商运行它们。统计拒绝数 vs 幻觉输出数。这是感知重试的基准真相。

5. 从头到尾阅读 OpenAI 的结构化输出指南。找出它在严格模式下明确禁止但普通 JSON Schema 允许的一个构造。然后设计一个非必要使用该禁止构造的 schema，并将其重构为严格模式兼容。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| JSON Schema 2020-12 | "Schema 规范" | 每家现代提供商都使用的 IETF 草案 schema 方言 |
| 严格模式（Strict mode） | "保证 schema" | OpenAI 通过约束解码强制执行 schema 的标志 |
| 约束解码（Constrained decoding） | "Logit 掩码" | 解码时屏蔽无效下一词元的强制执行 |
| 拒绝（Refusal） | "模型拒绝" | 输入无法符合 schema 时的类型化结果 |
| 解析错误（Parse error） | "无效 JSON" | 输出无法解析为 JSON；严格模式下不可能 |
| Schema 违反 | "形状错误" | 解析成功但违反了类型/required/enum/范围 |
| `additionalProperties: false` | "不允许额外字段" | 禁止未知字段；OpenAI 严格模式必须设置 |
| Pydantic BaseModel | "类型化输出" | 输出和验证 JSON Schema 的 Python 类 |
| Zod schema | "TypeScript 输出类型" | 用于提供商输出验证的 TS 运行时 schema |
| 语法强制执行 | "开放权重约束解码" | 基于 FSM 的 logit 掩码，如 outlines / guidance |

## 延伸阅读

- [OpenAI — 结构化输出](https://platform.openai.com/docs/guides/structured-outputs) — 严格模式、拒绝和 schema 要求
- [OpenAI — 结构化输出发布公告](https://openai.com/index/introducing-structured-outputs-in-the-api/) — 2024 年 8 月解释解码保证的发布文章
- [Pydantic AI — 输出](https://ai.pydantic.dev/output/) — 序列化到各提供商的类型化 output_type 绑定
- [JSON Schema — 2020-12 发版说明](https://json-schema.org/draft/2020-12/release-notes) — 权威规范
- [Microsoft — Azure OpenAI 中的结构化输出](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/structured-outputs) — 企业部署说明和严格模式注意事项
