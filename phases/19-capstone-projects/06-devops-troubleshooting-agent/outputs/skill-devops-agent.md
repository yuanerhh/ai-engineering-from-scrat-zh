---
name: devops-agent
description: 构建一个 Kubernetes 故障排查智能体，能够遍历集群知识图谱、对根因进行排名，并通过 Slack 审批每项修复操作。
version: 1.0.0
phase: 19
lesson: 06
tags: [capstone, devops, sre, kubernetes, langgraph, fastmcp, aiops]
---

给定一个 K8s 集群和一个告警源（PagerDuty 或 Alertmanager），构建一个能在五分钟内生成排名后的根因假设，并通过 Slack 审批卡片对每项修复操作进行把关的智能体。

构建计划：

1. 每 30 秒将 kube-state-metrics 摄取到 Neo4j 或 kuzu。构建包含 Pod、Deployment、Service、Node、PVC、HPA 的图，并在 Prometheus、Loki 和 Tempo 来源上叠加遥测边。
2. 搭建用于接收 PagerDuty 和 Alertmanager 的 FastAPI Webhook 接收器。
3. 通过 FastMCP（StreamableHTTP 传输）暴露只读工具：kubectl get/describe、promql、logql、traceql。
4. 构建包含三个节点的 LangGraph 根因智能体：`sample`（拉取 15 分钟遥测）、`walk`（遍历图邻居）、`hypothesize`（按时效性 × 特异性 × 引用次数对候选项排名）。
5. 将前 3 个排名假设连同图路径可视化发布到 Slack，附带审批按钮。
6. 将破坏性工具（scale、rollback、delete）放在单独的 FastMCP 服务器上，仅在智能体获得 Slack 签批后才能获取审批 Token。
7. 维护仅追加的审计日志：每条*被考虑*的命令、是否获批、是否执行、谁批准。
8. 构建 20 个合成故障场景（OOMKill、DNS 抖动、HPA 震荡、PVC 填满、噪音邻居、故障 Sidecar、ConfigMap 错误发布、证书轮换、镜像拉取失败、探针失败以及 10 个其他场景）。评估智能体在根因准确率和假设生成时间上的表现。

评估标准：

| 权重 | 评估项 | 度量方式 |
|:-:|---|---|
| 25 | 场景套件的根因准确率 | 在 20 个合成故障中至少 80% 根因正确 |
| 20 | 安全性 | 破坏性操作防护在审计日志中没有未经 Slack 审批就触发的记录 |
| 20 | 假设生成时间 | 从告警到 Slack 简报的 p50 低于 5 分钟 |
| 20 | 可解释性 | 每个假设都有图路径和遥测引用 |
| 15 | 集成完整性 | PagerDuty、Slack、ArgoCD、Prometheus 端到端可用 |

硬性拒绝条件：

- 将只读工具和破坏性工具混在同一个 MCP 服务器上的智能体。
- 任何没有遥测引用的根因分析。未引用的假设必须被拒绝。
- 只记录执行记录的审计日志。必须记录每条被考虑的命令。
- 未在带种子的 20 个场景套件上运行智能体就声称准确率。

拒绝规则：

- 拒绝在没有值班人工操作员通过 Slack 审批的情况下进行修复。即使假设显而易见也不例外。
- 拒绝通过只读 MCP 暴露 `kubectl exec`、`kubectl port-forward` 或任何交互式工具。这些工具实际上具有破坏性。
- 拒绝在没有每个 Deployment 对应审批卡片的情况下批量应用多个 Deployment 的修复操作。

输出：一个包含 FastAPI 接收器、LangGraph 智能体、只读和破坏性 MCP 服务器、Slack 集成、20 个场景测试套件的代码库，与 AWS DevOps Agent 在三个共享故障上的并排对比，以及一份关于一周观察窗口内"近未遂命令"（智能体*考虑过*但未执行的操作）的报告。
