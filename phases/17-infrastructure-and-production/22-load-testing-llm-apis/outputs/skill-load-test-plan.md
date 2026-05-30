---
name: load-test-plan
description: 设计真实的 LLM 压力测试——选择工具（LLMPerf、k6、GenAI-Perf、guidellm），构建四种流量模式（稳态、爬坡、峰值、长跑），并接入 CI 门控。
version: 1.0.0
phase: 17
lesson: 22
tags: [load-testing, llmperf, k6, genai-perf, guidellm, llm-locust, ci-gate]
---

给定工作负载（端点、TTFT/TPOT/错误率的 SLA）、目标规模（并发数、RPS）以及 CI 立场（PR 门控还是仅发布时执行），生成一份压力测试计划。

产出内容：

1. 工具选型。LLMPerf 用于基线运行；k6 + 流式扩展用于 CI 门控；GenAI-Perf 用于 NVIDIA 参考运行；guidellm 用于大规模合成负载。仅在团队已使用 Locust 的情况下选用 LLM-Locust。
2. 提示词分布。使用真实流量中的均值 + 标准差 input token 数（如有），否则使用已发布的分布（ShareGPT / HumanEval）。禁止使用单一提示词循环。
3. 四种流量模式。稳态、爬坡、峰值、长跑。每种模式标注：目标 RPS、持续时间、预期失败模式。
4. CI 门控。明确阈值：TTFT P95 < X、5xx < 5%、TPOT < Y。每次 PR 运行时间：3-5 分钟。
5. 指标对齐。说明所用报告工具的指标定义——GenAI-Perf 风格（ITL 不含 TTFT）或 LLMPerf 风格（ITL 含 TTFT）。选定一种并保持一致。
6. 输出产物。一个脚本文件（k6 JS、LLMPerf CLI），提交至代码仓库。

强制拒绝：
- 使用统一提示词进行压力测试。拒绝——测试数据失真。
- 不支持流式的压力测试。拒绝——LLM 端点默认使用流式传输。
- 跨工具比较数据而不说明指标定义差异。拒绝。

拒绝规则：
- 如果团队计划使用原生 Locust 而不加 LLM-Locust 扩展，拒绝——GIL 陷阱。
- 如果 CI 门控预算 < 60 秒/PR，拒绝完整长跑测试——建议使用快速稳态测试，并单独安排夜间长跑。
- 如果没有提示词分布数据，需记录所使用的已发布分布（ShareGPT）并注明假设前提。

输出：一页计划，包含工具选型、提示词分布、四种流量模式及目标、CI 门控阈值、指标对齐说明。最后附上单一 CI 输出标准：所有阈值通过且三次运行稳定，PR 方可变为绿色。
