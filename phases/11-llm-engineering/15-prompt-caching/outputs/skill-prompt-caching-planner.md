---
name: prompt-caching-planner
description: 设计对缓存友好的提示词布局，并为特定提供商选择正确的缓存模式。
version: 1.0.0
phase: 11
lesson: 15
tags: [llm-engineering, caching, cost]
---

给定一个提示词（系统提示 + 工具 + 少样本 + 检索 + 历史 + 用户）和一个使用概况（每小时请求数、所需 TTL、提供商），输出：

1. 布局。重新排序的各节，标注单个缓存断点；解释哪些节是稳定的，哪些是易变的。
2. 提供商模式。Anthropic 的 cache_control、OpenAI 的自动缓存，或 Gemini 的 CachedContent。根据 TTL 和复用模式说明选择理由。
3. 盈亏平衡点。TTL 内每次写入预期的读取次数；与不缓存相比的净成本（含计算过程）。
4. 验证计划。CI 断言：对第二次相同请求，cache_read_input_tokens > 0；按已缓存与未缓存 token 分类的仪表板。
5. 失败模式。列出此配置中缓存最可能未命中的三个原因（动态时间戳、工具重排、近似重复文本），以及如何预防每一个。

拒绝发布在断点上方放置动态字段的缓存方案。如果没有足够的复用次数来抵消 2 倍写入溢价，拒绝启用 1 小时 TTL。
