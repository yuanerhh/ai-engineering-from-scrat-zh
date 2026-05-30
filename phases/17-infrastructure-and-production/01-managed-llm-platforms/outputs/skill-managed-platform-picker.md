---
name: managed-platform-picker
description: 根据工作负载、SLA 和合规要求选择托管 LLM 平台（Bedrock、Azure OpenAI、Vertex AI）及一个备用平台，然后制定 FinOps 仪器化计划。
version: 1.0.0
phase: 17
lesson: 01
tags: [bedrock, azure-openai, vertex-ai, ptu, finops, managed-platforms]
---

给定一个工作负载概况（所需模型、月度令牌数、P50/P99 TTFT SLA、合规约束、现有云足迹），生成平台推荐。

产出内容：

1. **主要平台。** 列出平台名称、其覆盖的具体模型，以及考虑到利用率是否适合按需或预置吞吐量单位（PTU）/预置吞吐量。引用盈亏平衡数学（PTU 在约 40-60% 持续利用率时合适）。
2. **次要平台。** 列出两提供商最低限度的备用方案。说明配对理由——冗余必须涵盖模型重叠（Bedrock 上的 Claude + Azure OpenAI 上的 GPT 是常见配对）和区域重叠。
3. **FinOps 仪器化。** 指定第一天要启用的内容：Bedrock Application Inference Profiles、Azure 范围 + PTU 预留作为成本对象、Vertex 每团队项目 + BigQuery Billing Export。列出归因维度——每用户、每任务、每租户。
4. **SLA 检查。** 将目标 TTFT P99 与已发布的基准进行比较（Azure OpenAI PTU ≈ 50ms P50；Bedrock 按需 ≈ 75ms P50）。如果 SLA 比按需能够提供的更严格，要求使用 PTU。
5. **合规检查。** 根据需要验证 BAA、SOC 2 Type II、HIPAA、EU 数据驻留。注意三者均满足基线，但保留策略和滥用监控退出选项各有不同。
6. **迁移路径。** 列出团队本周可以采取的一个可逆步骤（例如，通过 AI 网关部署以抽象化提供商；仪器化归因头部），以及一个长期步骤（PTU 承诺；跨区域故障转移）。

硬性拒绝：
- 推荐没有具名备用方案的单一平台。拒绝并坚持两提供商最低限度。
- 没有利用率估算就选择 PTU。拒绝并要求提供持续利用率数据。
- 当归因被列为需求时忽略 Bedrock Application Inference Profiles——它们是最干净的原生接口。

拒绝规则：
- 如果工作负载将 Claude、Gemini 和 GPT 全部视为 P0，列出三平台现实（Bedrock + Vertex + Azure OpenAI 在网关后面），而不是假装一个平台可以服务所有三者。
- 如果 SLA 是 TTFT P99 < 100ms 且预期预算无法支持 PTU，拒绝承诺该 SLA——解释按需方差上限。
- 如果客户要求"使用最便宜的提供商"，拒绝——价格是多维的（令牌速率 + 专用容量 + 归因开销 + 锁定成本）。

输出：一页决策，包含主要平台、次要平台、PTU vs 按需、仪器化列表、SLA/合规验证和两个迁移步骤。结尾给出将检测计划偏差的单一指标（持续利用率、PTU 浪费或归因覆盖率）。
