---
name: skill-structured-outputs
description: 根据提供商、可靠性和复杂度选择正确结构化输出策略的决策框架
version: 1.0.0
phase: 11
lesson: 03
tags: [structured-output, json, schema, constrained-decoding, pydantic, function-calling]
---

# 结构化输出策略

在构建需要结构化数据的 LLM 应用时，请使用以下决策框架。

## 各方案适用时机

**基于提示词（"返回 JSON"）：** 仅用于原型开发。可以接受偶发解析失败的内部工具中可以使用。添加 try/except 加重试机制。绝不用于生产流水线。

**JSON 模式（API 标志）：** 你需要保证有效 JSON，但 Schema 简单或灵活。在应用侧验证结构时有效。可用提供商：OpenAI、Anthropic（通过工具使用）、Google。

**Schema 模式（约束解码）：** 每次输出都必须匹配特定 Schema 的生产系统。零解析失败，零 Schema 违规。对任何生产提取或分类任务默认使用此方法。可用提供商：OpenAI 结构化输出、Outlines、Guidance。

**函数调用/工具使用：** 模型需要选择调用哪个函数，而不仅仅是填充参数。你有多个 Schema，模型选择合适的一个。也用于与现有工具/函数基础设施集成时。

**Instructor 库：** 你想要跨任何提供商使用 Pydantic 验证和自动重试。Python 项目的最佳开发体验。支持 OpenAI、Anthropic、Google 和开源模型。

## 提供商特定指南

**OpenAI：** 使用 `response_format` 配合 `json_schema` 类型。约束解码内置。Pydantic 模型直接支持。最可靠的结构化输出实现。

**Anthropic：** 使用工具使用实现结构化输出。定义一个具有所需 Schema 的单个工具。模型返回匹配 Schema 的工具调用参数。可靠，但需要工具使用 API 模式。

**开源模型（vLLM、Ollama）：** 使用 Outlines 或 Guidance 进行约束解码。这些库将 JSON Schema 编译为有限状态机，在生成期间屏蔽无效 token。需要在本地运行推理。

## Schema 设计指南

1. 尽量保持 Schema 扁平。嵌套超过 2 层的对象会增加提取错误。
2. 对分类字段使用枚举。不要依赖模型发明正确的字符串。
3. 将不确定的字段设为必填且明确支持 null，而非可选。迫使模型做出决定。
4. 为 Schema 属性添加描述。模型将这些描述作为指令读取。
5. 除非必要，避免联合类型（oneOf/anyOf）。它们增加解码复杂度。
6. 为数字设置 minimum/maximum。捕获幻觉出的极端值。
7. 对数组使用 minItems/maxItems，防止空输出或无界输出。

## 常见失败模式及修复

- **模型用 Markdown 围栏包裹 JSON**：从基于提示词切换到 JSON 模式或 Schema 模式
- **Schema 有效但事实错误**：在提取后添加 LLM 评判验证步骤
- **枚举值不一致**：切换到约束解码，或添加后处理规范化
- **缺少可选字段**：在应用代码中使其必填或添加默认值
- **提取非常缓慢**：约束解码增加 5-15% 延迟，如果对延迟敏感则降低 Schema 复杂度
- **含有多样化条目的大型数组**：分块输入并按块提取，然后合并结果

## 可靠性阶梯

| 方案 | 解析成功率 | Schema 匹配率 | 配置工作量 |
|------|-----------|-------------|----------|
| 基于提示词 | ~90% | ~80% | 1 分钟 |
| JSON 模式 | 100% | ~90% | 5 分钟 |
| Schema 模式 | 100% | ~99% | 15 分钟 |
| 约束解码 | 100% | 100% | 30 分钟 |
| Instructor + 重试 | 100% | ~99.5% | 10 分钟 |
