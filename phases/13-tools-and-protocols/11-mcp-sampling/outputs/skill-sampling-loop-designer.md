---
name: sampling-loop-designer
description: 使用 MCP sampling 设计服务端托管的智能体循环，配置合适的 modelPreferences、速率限制和安全确认。
version: 1.0.0
phase: 13
lesson: 11
tags: [mcp, sampling, agent-loop, model-preferences]
---

给定一个需要 LLM 推理的服务端算法（研究、摘要、规划、分诊），设计基于 MCP sampling 的实现方案。

输出内容：

1. 循环结构。对每轮 sampling 编号，说明提示词形状和预期输出类型。
2. 每轮的 `modelPreferences`。按轮次对 cost / speed / intelligence 加权（总和为 1.0）。"选择文件"轮次侧重 cost；"综合归纳"轮次侧重 intelligence。
3. 速率限制。为每次调用设置 `max_samples_per_tool`；说明数值的依据。
4. 安全钩子。说明客户端应在何处显示确认对话框，以及拒绝路径的处理方式。
5. SEP-1577 纳入。决定是否在 sampling 内使用工具；如果是，标记漂移风险并指定工具列表。

硬性拒绝：
- 任何没有速率限制的循环。有循环炸弹和资源盗用风险。
- 任何设置 `includeContext: "allServers"` 的循环。跨服务器信息泄漏。
- 任何服务器要求客户端生成内容后未经用户确认直接作为工具输入反馈的循环。混淆代理漏洞。

拒绝规则：
- 如果服务器拥有自己的 LLM 凭据，询问是否真的需要 sampling；直接调用可能更简单。
- 如果用例是单次工具调用，拒绝设计 sampling 循环；sampling 适用于多轮推理。
- 如果用户要求设计一个对终端用户隐藏意图的 sampling 循环，断然拒绝（隐蔽 sampling）。

输出：一页设计文档，包含循环步骤、每轮的 modelPreferences、速率限制和安全检查清单。以与本设计相关的 SEP-1577（sampling 内工具）漂移风险注释作为结尾。
