---
name: inference-platform-picker
description: 根据工作负载、SLA、预算和运营约束，选择推理平台（Fireworks、Together、Baseten、Modal、Replicate、Anyscale 或自定义硅）。标准化每令牌、每分钟和每次预测的定价。
version: 1.0.0
phase: 17
lesson: 02
tags: [inference, fireworks, together, baseten, modal, replicate, anyscale, economics]
---

给定一个工作负载概况（模型、每日令牌数、持续利用率、TTFT SLA、突发因子、合规、Python vs 混合栈），生成平台推荐。

产出内容：

1. **主要平台。** 列出平台名称和具体定价层（无服务器 vs 专用 vs 批量）。用匹配工作负载特征的理由说明——例如，"Fireworks 无服务器，因为 TTFT < 500ms 是 SLA 且流量是突发性的。"
2. **有效成本。** 将所选定价模型标准化为 $/百万输出令牌。与至少两个替代方案比较。说明何时每分钟优于每令牌（持续利用率超过约 30%），反之亦然。
3. **冷启动计划。** 对于无服务器选择（Fireworks、Modal、Replicate），说明预期的冷启动延迟和缓解措施（预热、min_workers=1、实时迁移）。对于专用选择（Baseten、Anyscale），跳过此部分但注意权衡。
4. **备选方案。** 列出第二平台以及切换的明确条件（例如，"如果我们完成需要 HIPAA + 专用 GPU 的企业交易，则迁移到 Baseten"）。
5. **网关层。** 推荐是否在平台前面放置 AI 网关（LiteLLM、Portkey、Kong AI Gateway）以将产品与提供商更替隔离。默认：是，除非规模低于 500 RPS。

硬性拒绝：
- 不标准化就比较每令牌和每分钟。拒绝并坚持有效 $/百万令牌。
- 以"最快"为由选择 Fireworks 而不根据已发布的基准验证 TTFT SLA。
- 为任何非延迟绑定的工作负载推荐自定义硅（Groq、Cerebras、SambaNova）。它们定价有溢价，只在交互式 SLA 上才合理。

拒绝规则：
- 如果工作负载需要受监管框架（SOC 2 Type II、HIPAA）且客户选择了 Modal 或 Replicate，拒绝——两者都没有 Baseten 或 Anyscale 相同的企业足迹。建议 Baseten。
- 如果预期流量低于每日 100k 令牌，拒绝推荐每分钟计费（Baseten、Modal、Anyscale）。经济上不合算——默认使用市场（OpenRouter、DeepInfra）或托管超大规模云。
- 如果客户想要"最便宜的"，拒绝——列出多维成本函数（令牌速率 + 冷启动 + 归因 + 网关 + 开发体验）。

输出：一页推荐，列出主要平台、有效成本、冷启动计划、备选方案、网关姿态。结尾给出将揭示错误选择的单一指标（冷启动 P99、每令牌速率或利用率漂移）。
