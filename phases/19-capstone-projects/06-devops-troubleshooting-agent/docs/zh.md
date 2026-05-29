# 顶点项目 06 — Kubernetes DevOps 故障排查智能体

> AWS 的 DevOps Agent 正式发布，Resolve AI 发布了 K8s 运维手册，NeuBird 展示了语义监控，Metoro 将 AI SRE 与每个服务的 SLO 绑定。生产形态已趋于稳定：告警 webhook 触发，智能体读取遥测数据，遍历 K8s 对象图，对根本原因假设进行排序，并在 Slack 上发布附有审批按钮的简报。默认只读。每个修复操作均需人工审批。本顶点项目就是构建这个智能体，在 20 个合成故障场景上进行评测，并与 AWS 智能体在三个共享案例上进行比较。

**类型：** 顶点项目  
**语言：** Python（智能体），TypeScript（Slack 集成）  
**前置条件：** 第 11 阶段（LLM 工程），第 13 阶段（工具与 MCP），第 14 阶段（智能体），第 15 阶段（自主系统），第 17 阶段（基础设施），第 18 阶段（安全）  
**涵盖阶段：** P11 · P13 · P14 · P15 · P17 · P18  
**预计时间：** 30 小时

## 问题背景

2025-2026 年的 SRE 叙事变成了："AI 智能体负责故障分诊，人类负责审批修复操作。" AWS DevOps Agent、Resolve AI、NeuBird、Metoro、PagerDuty AIOps 都在生产中交付了这种形态。智能体读取 Prometheus 指标、Loki 日志、Tempo 追踪、kube-state-metrics 以及 K8s 对象知识图谱。它在五分钟内生成带遥测引用的排序根本原因假设。未经人工通过 Slack 明确审批，它绝不执行破坏性操作。

大部分难点在于范围划定和安全性，而非推理本身。智能体需要一个默认只读的 RBAC 工具表面、一个经过加固的 MCP 工具服务器，以及每条被考虑的命令与被执行的命令的审计日志。它需要知道何时超出能力范围并上报。而且它的运行成本必须足够低，才不至于让 OOM kill 级联事件生成一张 $5000 的智能体账单。

## 核心概念

智能体在知识图谱上运行。节点是 K8s 对象（Pod、Deployment、Service、Node、HPA、PVC）加上遥测源（Prometheus 序列、Loki 流、Tempo 追踪）。边编码了归属关系（Pod -> ReplicaSet -> Deployment）、调度关系（Pod -> Node）和观测关系（Pod -> Prometheus 序列）。图谱通过 kube-state-metrics 同步保持新鲜，并在每次告警时重新采样。

当告警触发时，智能体从受影响的对象开始进行根本原因分析。它遍历边，拉取相关遥测切片（最近 15 分钟），并起草假设。假设按证据排序：有多少遥测引用支持它，多近，多具体。前三个假设连同图路径可视化和修复操作审批按钮一起发送到 Slack。

修复操作受门控。默认允许的操作是只读的。破坏性操作（缩容、回滚、删除 Pod）需要 Slack 审批；ArgoCD 回滚钩子需要智能体永远不持有的授权 token。审计日志记录智能体*考虑*过的每条命令——不仅仅是已执行的——因此审查过程可以发现险些酿成问题的操作。

## 架构图

```
PagerDuty / Alertmanager webhook
           |
           v
     FastAPI receiver
           |
           v
   LangGraph root-cause agent
           |
           +---- read-only MCP tools ----+
           |                             |
           v                             v
   K8s knowledge graph              telemetry slices
     (Neo4j / kuzu)              Prometheus, Loki, Tempo
   ownership + scheduling          last 15m, scoped
           |
           v
   hypothesis ranking (evidence weight)
           |
           v
   Slack brief + approval buttons
           |
           v (approved)
   ArgoCD rollback hook / PagerDuty escalate
           |
           v
   audit log: considered vs executed, every command
```

## 技术栈

- 可观测性来源：Prometheus、Loki、Tempo、kube-state-metrics
- 知识图谱：K8s 对象 + 遥测边的 Neo4j（托管）或 kuzu（嵌入式）
- 智能体：LangGraph，带每工具白名单，默认只读
- 工具传输：FastMCP over StreamableHTTP；破坏性工具的独立服务器在审批门后面
- 模型：Claude Sonnet 4.7 用于根本原因推理，Gemini 2.5 Flash 用于日志摘要
- 修复：ArgoCD 回滚 webhook、PagerDuty 上报、Slack 审批卡片
- 审计：追加式结构化日志（已考虑、已执行、已审批、结果）
- 部署：具有自身窄 RBAC 角色的 K8s 部署；独立命名空间

## 构建步骤

1. **图谱摄入。** 每 30 秒将 kube-state-metrics 同步到 Neo4j/kuzu。节点：Pod、Deployment、Node、Service、PVC、HPA。边：OWNED_BY、SCHEDULED_ON、EXPOSES、MOUNTS、SCALES。遥测叠加边：OBSERVED_BY（一个 Pod 被一个 Prometheus 序列观测）。

2. **告警接收器。** FastAPI 端点接受 PagerDuty 或 Alertmanager webhook。提取受影响的对象和 SLO 违约信息。

3. **只读工具表面。** 通过 FastMCP 封装 kubectl、Prometheus 查询、Loki logql、Tempo traceql。每个工具都有一个窄 RBAC 动词（"get"、"list"、"describe"）。默认服务器中没有"delete"、"exec"、"scale"。

4. **根本原因智能体。** LangGraph 包含三个节点：`sample` 拉取最近 15 分钟遥测切片，`walk` 查询图谱中的相邻对象，`hypothesize` 起草带遥测引用的排序根本原因候选项。

5. **证据评分。** 每个假设的得分 = 时效性 × 具体性 × 图路径长度倒数 × 引用数量。返回前三名。

6. **Slack 简报。** 发布一个附件，包含假设、图路径可视化（服务端渲染的子图图像）以及至多一个修复操作的审批按钮。

7. **修复门控。** 破坏性工具（缩容、回滚、删除）位于第二个 MCP 服务器，需要审批 token。只有在 Slack 卡片被人工审批后，智能体才能调用这些工具。

8. **审计日志。** 追加式 JSONL：对于每个候选命令，记录它是否被考虑过、是否被执行过、是谁审批的。每天发送到 S3。

9. **合成故障场景集。** 构建 20 个场景：OOMKill 级联、DNS 抖动、HPA 震荡、PVC 填满、嘈杂邻居、故障 sidecar、错误 ConfigMap 发布、证书轮换、镜像拉取退避等。对智能体的根本原因准确率和假设生成时间进行评分。

## 使用示例

```
webhook: alert.pagerduty.com -> checkout-api SLO breach, error rate 14%
[graph]   affected: Deployment checkout-api (3 Pods, Node ip-10-2-3-4)
[walk]    neighbors: ReplicaSet checkout-api-abc, Service checkout-api,
           recent rollout 14m ago
[sample]  prometheus error_rate 14%, up-trend; loki 500s on /api/v2/pay
[hypo]    #1 bad rollout: latest image checkout-api:v2.41 fails /healthz
          citations: deploy.yaml (rev 42), prometheus errorRate, loki 500 stack
[slack]   [ROLL BACK to v2.40]  [ESCALATE]  [IGNORE]
          (approval required; agent does not roll back unilaterally)
```

## 交付物

`outputs/skill-devops-agent.md` 是交付物。给定一个 K8s 集群和告警源，智能体生成排序根本原因假设和 Slack 门控的修复流程。

| 权重 | 评分标准 | 衡量方式 |
|:-:|---|---|
| 25 | 故障场景集上的根本原因分析准确率 | 20 个合成故障场景中正确根本原因 ≥ 80% |
| 20 | 安全性 | 审计日志中破坏性操作在没有 Slack 审批的情况下永不触发 |
| 20 | 假设生成时间 | 从告警到 Slack 简报的 p50 低于 5 分钟 |
| 20 | 可解释性 | 每个假设都有图路径和遥测引用 |
| 15 | 集成完整性 | PagerDuty、Slack、ArgoCD、Prometheus 端到端可用 |
| **100** | | |

## 练习题

1. 在 AWS DevOps Agent 演示的同三个故障场景上运行你的智能体。发布并排对比报告。报告智能体在哪里有分歧。

2. 添加"险情"审计，标记智能体*考虑*过但未经审批就会造成破坏性操作的每条命令。测量一周内的险情率。

3. 将假设模型从 Claude Sonnet 4.7 替换为自托管 Llama 3.3 70B。测量根本原因分析准确率差值和每次故障的费用。

4. 构建因果过滤器：区分相关的遥测峰值与真正的根本原因。在 20 个场景标签上训练一个小分类器。

5. 添加回滚演练：在使用相同清单的暂存集群上执行 ArgoCD 回滚。在 Slack 审批按钮出现前验证回滚计划。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| K8s 知识图谱 | "集群图谱" | 节点 = K8s 对象 + 遥测序列；边 = 归属、调度、观测关系 |
| 默认只读 | "范围 RBAC" | 智能体的服务账号只有 get/list/describe 动词；破坏性动词位于审批后的独立服务器 |
| 审计日志 | "已考虑 vs 已执行" | 每条候选命令的追加式记录，包括是否运行、谁审批 |
| 假设排序 | "证据得分" | 时效性 × 具体性 × 图路径长度倒数 × 引用数量 |
| Slack 审批卡片 | "HITL 门控" | 带修复按钮的交互式 Slack 消息；智能体在人工点击前无法继续 |
| 遥测引用 | "证据指针" | 支持某个论断的 Prometheus 查询、Loki 选择器或 Tempo 追踪 URL |
| MTTR | "平均恢复时间" | 从告警触发到 SLO 恢复的墙上时钟时间 |

## 延伸阅读

- [AWS DevOps Agent 正式发布](https://aws.amazon.com/blogs/aws/aws-devops-agent-helps-you-accelerate-incident-response-and-improve-system-reliability-preview/) — 2026 年的标志性参考
- [Resolve AI K8s 故障排查](https://resolve.ai/blog/kubernetes-troubleshooting-in-resolve-ai) — 竞品参考
- [NeuBird 语义监控](https://www.neubird.ai) — 语义图谱方法
- [Metoro AI SRE](https://metoro.io) — SLO 优先的生产框架
- [kube-state-metrics](https://github.com/kubernetes/kube-state-metrics) — 集群状态来源
- [LangGraph](https://langchain-ai.github.io/langgraph/) — 参考智能体编排器
- [FastMCP](https://github.com/jlowin/fastmcp) — Python MCP 服务器框架
- [ArgoCD 回滚](https://argo-cd.readthedocs.io/en/stable/user-guide/commands/argocd_app_rollback/) — 门控修复目标
