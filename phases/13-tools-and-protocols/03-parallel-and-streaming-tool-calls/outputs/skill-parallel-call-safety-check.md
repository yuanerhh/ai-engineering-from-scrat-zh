---
name: parallel-call-safety-check
description: 审计工具注册表以确保安全并行化。为每个工具标记 parallel_safe，记录排序依赖，并标记下游限速风险。
version: 1.0.0
phase: 13
lesson: 03
tags: [parallel-tool-calls, streaming, correlation, rate-limits]
---

给定一个工具注册表（包含名称、描述和执行器的工具列表），返回一个注释后的副本，每个工具新增 `parallel_safe: bool`、`ordering_deps: [tool_name]` 和 `rate_limit_group: name` 字段。

输出内容：

1. 逐工具分类。对每个工具判断：在同一轮次中是否可以安全并行运行（纯读取、不同资源）；还是不安全（变更操作、共享资源、外部限速）。
2. 依赖图。识别一个工具的输出应作为另一个工具输入的工具对。在同一轮次内不能并行。用 `ordering_deps` 标记。
3. 限速分组。命中同一下游 API 的工具共享一个分组。宿主应按分组限制并发，而非按工具限制。
4. 安全建议。对每个不安全的工具，说明是对该轮次禁用并行、排队，还是按资源分片。
5. 特定提供商标记。当不安全工具在工具集内时，建议在 OpenAI 上使用 `parallel_tool_calls=false`，或在 Anthropic 上使用 `disable_parallel_tool_use=true`。

硬性拒绝：
- 审计后仍有工具未分类。默认拒绝；未知意味着不安全。
- 共享资源上的写路径工具被标记为 `parallel_safe: true`。存在竞争条件。
- 命中限速外部 API 的工具没有 `rate_limit_group`。

拒绝规则：
- 如果被要求不经检查就将所有工具标记为并行安全，拒绝。
- 如果注册表中包含作用于同一资源的后果型工具（如 `delete_file` 和 `write_file` 作用于同一路径），拒绝并行化，并指向 Phase 14 · 09 进行沙箱级别的序列化处理。
- 如果用户声称其工具永远不会发生竞争，拒绝并要求提供证明（测试、日志或形式化论证）。竞争条件在生产中会悄无声息地发生。

输出：一个修订后的注册表 JSON，每个工具包含三个新字段，随后附上一段简短摘要，说明风险最高的并行化选择及推荐的缓解措施。以建议的当前轮次 `tool_choice` 覆盖项作为结尾。
