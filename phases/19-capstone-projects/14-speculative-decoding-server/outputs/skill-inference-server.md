---
name: inference-server
description: 交付一个使用 EAGLE-3 或 P-EAGLE 草稿的投机解码推理服务器，具备 K8s 自动扩缩容，并提供完整的吞吐量/延迟/成本报告。
version: 1.0.0
phase: 19
lesson: 14
tags: [capstone, inference, vllm, sglang, eagle-3, p-eagle, speculative-decoding, quantization, hpa]
---

给定两个开放目标模型（Llama 3.3 70B 和 Qwen3-Coder-30B MoE 或 GPT-OSS-120B），交付一个带投机解码、量化和 Kubernetes 自动扩缩容的生产服务栈。发布测量的加速比和尾延迟数据。

构建计划：

1. 在 vLLM 0.7（或 SGLang 0.4）下部署目标模型，使用 FP8 Marlin 量化。
2. 从 Red Hat Speculators 加载对齐的 EAGLE-3 草稿（或通过 SpecForge 训练一个）。
3. 基线数据：在无投机的情况下，batch 1/8/32 时的 tokens/s 和 p50/p99 延迟。
4. 启用 EAGLE-3。重新运行相同基准。报告加速比、接受率、p99 尾延迟变化。
5. 启用 P-EAGLE 并行投机；报告更深树结构有利与有害的拐点。
6. 跨分布运行基准：ShareGPT、HumanEval、领域数据。发布接受率漂移。
7. 在第二个目标模型（MoE）上重复；识别草稿接受中的路由噪声敏感性。
8. 在 Kubernetes 上部署，HPA 追踪 `queue_wait_ms`。演示负载三倍时的扩展能力。
9. 在匹配评估上与 Anthropic Claude Sonnet 4.7 和 OpenAI GPT-5.4 比较每百万 Token 成本。

评估标准：

| 权重 | 评估项 | 度量方式 |
|:-:|---|---|
| 25 | 相对基线的测量加速比 | 在两个模型上，相同质量下 2.5 倍以上吞吐量 |
| 20 | 真实流量接受率 | 每分布的接受率报告 |
| 20 | p99 尾延迟规范 | batch 1/8/32 时有无投机的 p99 |
| 20 | 运维 | K8s 部署、基于队列等待的 HPA、平滑发布、先排水再升级 |
| 15 | 报告与方法论 | 指标的清晰推导、匹配的基线 |

硬性拒绝条件：

- 报告稳态吞吐量而不含尾延迟。
- HPA 基于 CPU 而非队列等待。在 GPU 饱和时会产生振荡。
- 忽略草稿-目标版本对齐。版本漂移的草稿比不用投机解码成本更高。
- 省略托管 API 提示缓存折扣的成本对比。

拒绝规则：

- 拒绝在没有发布排水的情况下提供服务。在请求处理中原地升级是取消资格的条件。
- 拒绝报告跨分布聚合的接受率。按分布报告是强制要求。
- 拒绝在没有对照非投机数字的情况下声称投机解码在 bs=32 时胜出。

输出：一个包含 vLLM / SGLang 配置、EAGLE-3 草稿下载脚本、K8s 部署清单、基于队列等待的 HPA 配置、ShareGPT / HumanEval / 领域数据基准框架、每百万 Token 成本对比表的代码库，以及一份说明投机解码引入的三大尾延迟回退以及修复每个问题的缓解措施（批量门控、n-gram 回退、量化调整）的报告。
