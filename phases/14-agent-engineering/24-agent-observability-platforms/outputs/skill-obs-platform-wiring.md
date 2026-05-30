---
name: obs-platform-wiring
description: 选择可观测性平台（Langfuse、Phoenix、Opik、Datadog），并将链路追踪、评估和提示词版本接入已有的 Agent。
version: 1.0.0
phase: 14
lesson: 24
tags: [observability, langfuse, phoenix, opik, datadog, tracing]
---

给定一个 Agent 运行时和产品需求，选择可观测性平台并搭建接入配置。

决策：

1. 需要在同一处管理提示词 + 会话回放 -> **Langfuse**。
2. 需要深度 RAG 相关性 + 漂移/异常检测 -> **Phoenix**。
3. 需要自动化提示词优化 + PII 防护 -> **Opik**。
4. 已在使用 Datadog -> **Datadog LLM Observability**（v1.37+ 原生支持 GenAI 映射）。
5. 需要无 ELv2 许可证 -> **Langfuse**（MIT）或 **Opik**（Apache 2.0）；纯开源分发场景避免使用 Phoenix。

输出内容：

1. OTel GenAI 插桩（第 23 课）——这是公共基础层。
2. 平台专属 SDK 或 OTel exporter 配置。
3. 针对你所在领域的 LLM 裁判评分标准（事实正确性、范围、语气、拒绝质量）。
4. 将提示词版本与链路绑定（Langfuse）或链路聚类配置（Phoenix）或实验定义（Opik）。
5. 日志内容防护：PII 脱敏、密钥清洗。
6. Dashboard：会话健康度、故障分类、延迟分布、每会话成本。

硬性拒绝：

- 不配评估就上线。单纯的链路追踪是昂贵的日志记录。
- 使用没有外部验证的自研 LLM 裁判。CRITIC 模式（第 05 课）：裁判需要外部工具进行事实核查。
- 将 PII 存储在 span 正文中。始终使用外部存储 + 引用 ID。

拒绝规则：

- 如果用户要求"一个平台搞定所有"，拒绝并给出上述决策建议。没有任何单一平台在三个维度上全面领先。
- 如果产品对每个 Agent 任务都没有验收标准，拒绝交付评估。LLM 裁判需要评分标准，而评分标准需要产品决策。
- 如果用户想要"不采样，全量捕获"，拒绝。链路量随流量线性增长；规模化场景必须进行采样（头部采样或尾部采样）。

输出：`instrumentation.py`、`judge.py`、`dashboards.md`、`README.md`，说明平台选择、评分标准、采样策略和事故响应流程。最后附"下一步阅读"，指向第 30 课（评估驱动开发）或第 26 课（故障模式分类）。
