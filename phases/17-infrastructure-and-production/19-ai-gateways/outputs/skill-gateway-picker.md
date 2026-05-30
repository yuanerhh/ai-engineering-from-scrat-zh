---
name: gateway-picker
description: 根据规模、延迟预算、合规要求、运维立场及定价容忍度，选择 AI 网关（LiteLLM、Portkey、Kong AI、Cloudflare/Vercel）。
version: 1.0.0
phase: 17
lesson: 19
tags: [ai-gateway, litellm, portkey, kong, cloudflare, vercel, bifrost, fallback, rate-limit, guardrails]
---

给定 RPS（当前及未来 12 个月预测）、延迟预算、合规要求（是否需要自托管）、护栏需求（PII 脱敏、越狱检测、审计）及定价容忍度，生成网关推荐方案。

输出内容：

1. 主网关。说明所选工具。结合 RPS 上限、开销及功能契合度加以说明。
2. 回退链。按顺序列出三个提供商；OpenAI → Anthropic → 自托管 是标准配置。计算预期可用性。
3. 限流策略。>500 RPS 推荐滑动窗口；否则令牌桶可接受。按租户分级。
4. 护栏。需要 PII/越狱防护时选 Portkey；需要大规模 + 护栏时选 Kong；仅开发层级选 LiteLLM。
5. 可观测性交接。对接 Phase 17 · 13 的选择；确认 OTel GenAI 规范属性能流经网关。
6. 迁移方案。若从应用层集成迁移，采用分阶段发布（网关 1% 灰度，成功后扩量）。

强制拒绝：
- LiteLLM 在 >2000 RPS 场景下使用。拒绝——Kong 基准测试显示会出现级联故障；先迁移。
- Portkey 在 TTFT P99 < 100 ms SLA 场景下使用。拒绝——30 ms 开销会消耗过多延迟预算。
- Cloudflare AI Gateway 用于受监管的本地化客户。拒绝——仅托管模式，不支持自托管。

拒绝规则：
- 若规模不确定性较大（当前 100 RPS，6 个月内计划达 2K+），在承诺 LiteLLM 之前要求提供迁移方案。
- 若合规要求 SOC 2 Type II 而所选网关仅为无托管 SLA 的 OSS 版本，要求客户提供自身的 SOC 2 认证。
- 若团队没有 Kubernetes 却选择自托管 Kong，拒绝——推荐托管版 Kong 或 Portkey 托管版。

输出：一页决策报告，列明网关、回退链、限流策略、护栏立场、可观测性流向、迁移方案。最后给出唯一指标：过去一小时网关延迟 P99；超出阈值时告警。
