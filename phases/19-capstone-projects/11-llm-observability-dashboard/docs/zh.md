# 顶点项目 11 — LLM 可观测性与评测仪表板

> Langfuse 转为开放核心模式。Arize Phoenix 发布了 2026 年的 GenAI 语义约定映射。Helicone 和 Braintrust 都在每用户成本归因上加大投入。Traceloop 的 OpenLLMetry 成为事实上的 SDK 埋点标准。生产形态是：ClickHouse 存储追踪，Postgres 存储元数据，Next.js 作为 UI，还有一批评测任务（DeepEval、RAGAS、LLM 评委）运行在采样追踪上。构建一个自托管版本，从至少四个 SDK 家族摄入数据，并展示在五分钟内捕捉到一个注入的回归。

**类型：** 顶点项目  
**语言：** TypeScript（UI），Python/TypeScript（摄入 + 评测），SQL（ClickHouse）  
**前置条件：** 第 11 阶段（LLM 工程），第 13 阶段（工具），第 17 阶段（基础设施），第 18 阶段（安全）  
**涵盖阶段：** P11 · P13 · P17 · P18  
**预计时间：** 25 小时

## 问题背景

2026 年，每个运行生产流量的 AI 团队都在模型旁边维护一个可观测性平面。成本归因。幻觉检测。漂移监控。越狱信号。SLO 仪表板。PII 泄露告警。开源参考——Langfuse、Phoenix、OpenLLMetry——以 OpenTelemetry GenAI 语义约定作为摄入模式达成共识。你现在可以用一个 SDK 对 OpenAI、Anthropic、Google、LangChain、LlamaIndex 和 vLLM 进行埋点，并交付兼容的 span。

你将构建一个自托管仪表板，从至少四个 SDK 家族摄入数据，对采样追踪运行一小批评测任务，检测漂移，并发送告警。衡量标准：给定一个故意注入的回归（一个开始产出 PII 的提示词），仪表板在五分钟内捕捉并发出告警。

## 核心概念

摄入使用 OTLP HTTP。SDK 生成 GenAI 语义约定 span：`gen_ai.system`、`gen_ai.request.model`、`gen_ai.usage.input_tokens`、`gen_ai.response.id`、`llm.prompts`、`llm.completions`。Span 落入 ClickHouse 进行列式分析；元数据（用户、会话、应用）落入 Postgres。

评测以批处理任务的形式在采样追踪上运行。DeepEval 评分忠实度、毒性和答案相关度。当追踪携带检索上下文时，RAGAS 评分检索指标。自定义 LLM 评委运行特定领域检查（PII 泄露、违反策略的回答）。评测运行将结果作为与父追踪关联的评测 span 写回同一个 ClickHouse。

漂移检测随时间监控嵌入空间分布（提示词嵌入上的 PSI 或 KL 散度），以及评测分数趋势。告警馈入 Prometheus Alertmanager，然后到 Slack/PagerDuty。UI 是带 Recharts 的 Next.js 15。

## 架构图

```
production apps:
  OpenAI SDK  +  Anthropic SDK  +  Google GenAI SDK
  LangChain + LlamaIndex + vLLM
       |
       v
  OpenTelemetry SDK with GenAI semconv
       |
       v  OTLP HTTP
  collector (ingest, sample, fan-out)
       |
       +-------------+-----------+
       v             v           v
   ClickHouse    Postgres    S3 archive
   (spans)       (metadata)  (raw events)
       |
       +---> eval jobs (DeepEval, RAGAS, LLM-judge)
       |     sampled or all-trace
       |     write eval spans back
       |
       +---> drift detector (PSI / KL on prompt embeddings)
       |
       +---> Prometheus metrics -> Alertmanager -> Slack / PagerDuty
       |
       v
   Next.js 15 dashboard (Recharts)
```

## 技术栈

- 摄入：OpenTelemetry SDK + GenAI 语义约定；OTLP HTTP 传输
- 采集器：带尾部采样处理器的 OpenTelemetry Collector（用于成本控制）
- 存储：ClickHouse 存储 span，Postgres 存储元数据，S3 存储原始事件归档
- 评测：DeepEval、RAGAS 0.2、Arize Phoenix 评测包、自定义 LLM 评委
- 漂移：每周对汇聚的提示词嵌入（sentence-transformers）计算 PSI/KL
- 告警：Prometheus Alertmanager -> Slack/PagerDuty
- UI：Next.js 15 App Router + Recharts + server actions
- 开箱即支持的 SDK：OpenAI、Anthropic、Google GenAI、LangChain、LlamaIndex、vLLM

## 构建步骤

1. **采集器配置。** OpenTelemetry Collector 配置 OTLP HTTP 接收器、尾部采样器（保留 100% 错误追踪和 10% 成功追踪），以及到 ClickHouse 和 S3 的导出器。

2. **ClickHouse 模式。** 表 `spans`，列镜像 GenAI 语义约定：`gen_ai_system`、`gen_ai_request_model`、`input_tokens`、`output_tokens`、`latency_ms`、`prompt_hash`、`trace_id`、`parent_span_id`，加上长 payload 的 JSON 袋。按 user_id 和 app_id 添加二级索引。

3. **SDK 覆盖率测试。** 用每个 SDK（OpenAI、Anthropic、Google、LangChain、LlamaIndex、vLLM）和 OpenLLMetry 自动埋点编写一个小客户端应用。验证每个都能生成落入 ClickHouse 的规范 GenAI span。

4. **评测任务。** 定时任务读取最近 15 分钟的采样追踪，运行 DeepEval 忠实度、毒性和答案相关度评分。输出是与父追踪关联的评测 span。

5. **自定义 LLM 评委。** PII 泄露评委：给定一个回答，调用防护 LLM 评分 PII 泄露可能性。高分回答进入分诊队列。

6. **漂移检测。** 每周任务计算本周汇聚提示词嵌入与过去 4 周基线之间的 PSI。如果 PSI 超过阈值，发送告警。

7. **仪表板。** Next.js 15，页面包括：概览（span/秒、每用户成本、p95 延迟）、追踪（搜索 + 瀑布图）、评测（忠实度趋势、毒性）、漂移（PSI 随时间变化）、告警。

8. **告警链。** Prometheus 导出器读取评测分数聚合和延迟百分位；Alertmanager 将警告路由到 Slack，将关键违约路由到 PagerDuty。

9. **回归探针。** 注入一个 bug：被评测的聊天机器人开始 1% 的时间泄露假 SSN。测量 MTTR（平均恢复时间）：从 bug 部署到 Slack 告警。

## 使用示例

```
$ curl -X POST https://my-otel-collector/v1/traces -d @trace.json
[collector]  accepted 1 trace, 3 spans
[clickhouse] inserted 3 spans (app=chat, user=u_42)
[eval]       DeepEval faithfulness 0.82, toxicity 0.03
[drift]      weekly PSI 0.08 (below 0.2 threshold)
[ui]         live at https://obs.example.com
```

## 交付物

`outputs/skill-llm-observability.md` 是交付物。给定一个 LLM 应用，仪表板摄入其追踪，运行评测，在漂移时告警，并在 Next.js 中呈现每用户成本分解。

| 权重 | 评分标准 | 衡量方式 |
|:-:|---|---|
| 25 | 追踪模式覆盖 | 生成规范 GenAI span 的 SDK 家族数量（目标：6+） |
| 20 | 评测正确性 | DeepEval/RAGAS 分数 vs 手工标注集 |
| 20 | 仪表板体验 | 注入回归上的 MTTR（目标：五分钟内） |
| 20 | 成本/规模 | 在 1k span/秒持续摄入下无积压 |
| 15 | 告警 + 漂移检测 | Prometheus/Alertmanager 链路端到端演练 |
| **100** | | |

## 练习题

1. 为 Haystack 框架添加自定义埋点。验证规范 span 以忠实的 `gen_ai.*` 属性落入 ClickHouse。

2. 在相同追踪上将 DeepEval 替换为 Phoenix 评测器。测量两个评测引擎之间的分数漂移。

3. 精细化漂移检测器：按 app-id 而不是全局计算 PSI。展示每应用漂移轨迹。

4. 添加"用户影响"页面：带迷你图的每用户成本和每用户失败率。

5. 构建尾部采样策略：保留 100% 毒性 > 0.5 的追踪，加上其余 10% 的分层采样。测量引入的采样偏差。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| GenAI 语义约定 | "OTel LLM 属性" | 2025 年 OpenTelemetry LLM span 属性规范（system、model、tokens） |
| 尾部采样 | "追踪后采样" | 采集器在追踪完成后决定保留或丢弃（可以查看错误） |
| PSI | "群体稳定性指数" | 比较两个分布的漂移指标；> 0.2 通常表示有意义的漂移 |
| LLM 评委 | "模型即评测" | 一个 LLM 用评分标准评测另一个 LLM 的输出（忠实度、毒性、PII） |
| 尾部采样策略 | "保留规则" | 决定哪些追踪保留 vs 丢弃的规则；错误 + 采样率 |
| 评测 span | "关联评测追踪" | 携带与原始 LLM 调用 span 关联的评测分数的子 span |
| 每用户成本 | "单位经济学" | 在一个时间窗口内归因到 user_id 的美元成本；关键产品指标 |

## 延伸阅读

- [Langfuse](https://github.com/langfuse/langfuse) — 参考开放核心可观测性平台
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — 备选参考，漂移支持强大
- [OpenLLMetry（Traceloop）](https://github.com/traceloop/openllmetry) — 自动埋点 SDK 家族
- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 摄入模式
- [Helicone](https://www.helicone.ai) — 备选托管可观测性
- [Braintrust](https://www.braintrust.dev) — 备选评测优先平台
- [ClickHouse 文档](https://clickhouse.com/docs) — 列式 span 存储
- [DeepEval](https://github.com/confident-ai/deepeval) — 评测库
