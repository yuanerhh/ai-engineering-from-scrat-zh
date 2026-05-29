# 顶点项目 14 — 投机解码推理服务器

> vLLM 0.7 中的 EAGLE-3 在真实流量上实现了 2.5-3 倍的吞吐量。P-EAGLE（AWS 2026）将并行投机进一步推进。SGLang 的 SpecForge 在规模上训练草稿头。Red Hat 的 Speculators hub 为常见开源模型发布了对齐草稿。TensorRT-LLM 将投机解码作为 NVIDIA 的一等公民特性。2026 年的生产服务栈是：配合 EAGLE 系列草稿的 vLLM 或 SGLang、FP8 或 INT4 量化，以及按队列等待时间扩缩容的 HPA。这个顶点项目要求以完整的尾延迟报告，在两个开源模型上实现 2.5 倍以上基线吞吐量。

**类型：** 顶点项目  
**语言：** Python（服务），C++/CUDA（内核检查），YAML（配置）  
**前置条件：** 第 3 阶段（深度学习），第 7 阶段（Transformer），第 10 阶段（从零构建 LLM），第 17 阶段（基础设施）  
**涵盖阶段：** P3 · P7 · P10 · P17  
**预计时间：** 30 小时

## 问题背景

2026 年，投机解码成为了一项商品技术。EAGLE-3 草稿头在目标模型的隐藏状态上训练，提前预测 N 个 token；目标模型在一次前向传播中验证所有 token。60-80% 的接受率转化为 2-3 倍的端到端吞吐量。vLLM 0.7 原生集成了这一功能。SGLang + SpecForge 提供了训练流水线。Red Hat 的 Speculators 为 Llama 3.3 70B、Qwen3-Coder-30B MoE、GPT-OSS-120B 发布了对齐草稿。

精髓在于服务运营，而非模型本身。接受率随流量分布而漂移（ShareGPT vs 代码 vs 领域数据）。拒绝时的尾延迟比没有投机时更差——你必须在多个批次大小下报告 p99，而不仅仅是稳态 tokens/s。相比 Anthropic/OpenAI API 的每百万 token 成本是可信度杠杆。

## 核心概念

投机解码有两层。**草稿**模型（EAGLE-3 头、ngram 或更小的目标对齐模型）在每一步提出 k 个候选 token。**目标**模型在一次前向传播中验证所有 k 个；任何被接受的前缀替换贪心路径。接受率取决于草稿-目标对齐程度和输入分布。

EAGLE-3 在大多数流量上优于 ngram 草稿。P-EAGLE 为更深的草稿树运行并行投机。权衡是：拒绝时的 P99 延迟更高，因为验证通道更大。服务配置必须报告按批次大小分桶的延迟以揭示这一点。

部署使用 Kubernetes。vLLM 0.7 每 GPU 或张量并行分片运行一个副本。HPA 按队列等待时间而非 CPU 进行自动扩缩容。FP8（Marlin）和 INT4（AWQ）量化将 GPU 内存控制在 H100/H200 信封内。端到端报告包括吞吐量、接受率、batch 1/8/32 时的 p50/p99，以及每百万 token 成本。

## 架构图

```
request ingress
    |
    v
vLLM server (0.7) or SGLang (0.4)
    |
    +-- draft: EAGLE-3 heads | P-EAGLE parallel | ngram fallback
    +-- target: Llama 3.3 70B | Qwen3-Coder-30B | GPT-OSS-120B
    |     quantized FP8-Marlin or INT4-AWQ
    |
    v
verify pass: batch k draft tokens through target
    |
    v (accept prefix; resample for rejected suffix)
    v
token stream back to client
    |
    v
Prometheus metrics: throughput, acceptance rate, queue wait, latency p50/p99
    |
    v
HPA on queue-wait metric
```

## 技术栈

- 服务：vLLM 0.7 或 SGLang 0.4
- 投机方法：EAGLE-3 草稿头、P-EAGLE 并行投机、ngram 回退
- 草稿训练：SpecForge（SGLang）或 Red Hat Speculators
- 目标模型：Llama 3.3 70B、Qwen3-Coder-30B MoE、GPT-OSS-120B
- 量化：FP8（Marlin）、INT4 AWQ
- 部署：Kubernetes + NVIDIA 设备插件；HPA 按队列等待指标
- 评测：ShareGPT、MT-Bench-v2、GSM8K、HumanEval 用于领域覆盖接受率测量
- 参考：TensorRT-LLM 投机解码作为供应商基线

## 构建步骤

1. **目标模型准备。** 选择 Llama 3.3 70B。通过 Marlin 量化为 FP8。在 1xH100（或 2x 张量并行）上部署于 vLLM 0.7。

2. **草稿来源。** 从 Red Hat Speculators 拉取对齐的 EAGLE-3 草稿头（或通过 SpecForge 训练一个）。加载到 vLLM 的投机解码配置中。

3. **基线数据。** 在启用投机解码之前：batch 1/8/32 时的 tokens/s，p50/p99 延迟，GPU 利用率。发布。

4. **启用 EAGLE-3。** 切换配置；重新运行相同基准。报告加速比、接受率、p99 尾延迟差值。

5. **P-EAGLE。** 启用并行投机；测量更深草稿树 vs 串行 EAGLE-3。报告 P-EAGLE 有帮助 vs 有害的拐点。

6. **领域流量。** 对同一服务器运行 ShareGPT vs HumanEval vs 领域特定流量。按分布测量接受率。找出草稿何时漂移。

7. **第二个目标模型。** 对 Qwen3-Coder-30B MoE 运行相同流水线。草稿更难（MoE 路由噪声）。报告结果。

8. **K8s HPA。** 在 K8s 下部署，HPA 跟踪 `queue_wait_ms`。在负载三倍时演示弹性扩容。

9. **成本对比。** 计算每百万 token 成本 vs Anthropic Claude Sonnet 4.7 和 OpenAI GPT-5.4 在相同评测上的成本。发布。

## 使用示例

```
$ curl https://infer.example.com/v1/chat/completions -d '{"messages":[...]}'
[serve]     vLLM 0.7, Llama 3.3 70B FP8, EAGLE-3 active
[decode]    bs=8, accepted_tokens_per_step=3.2, acceptance_rate=0.76
[latency]   first-token 42ms, full-response 980ms (620 tokens)
[cost]      $0.34 per 1M output tokens at sustained throughput
```

## 交付物

`outputs/skill-inference-server.md` 描述了交付物。一个带投机解码的可测量服务栈，完整基准报告，以及 K8s 部署。

| 权重 | 评分标准 | 衡量方式 |
|:-:|---|---|
| 25 | 相比基线的可测加速 | 两个模型上在相同质量下 2.5 倍以上吞吐量 |
| 20 | 真实流量的接受率 | 按分布的接受率报告 |
| 20 | p99 尾延迟纪律 | 有无投机解码时 batch 1/8/32 的 p99 |
| 20 | 运维 | K8s 部署、按队列等待的 HPA、平滑滚动发布 |
| 15 | 报告与方法论 | 清晰解释什么改变了以及为什么 |
| **100** | | |

## 练习题

1. 测量草稿落后目标一个版本时的接受率退化（例如，Llama 3.3 -> 3.4 漂移）。构建监控告警。

2. 实现 ngram 回退：如果 EAGLE-3 接受率低于阈值，切换到 ngram 草稿。报告可靠性改善情况。

3. 运行受控 MoE 实验：注入路由噪声 vs 未注入的 Qwen3-Coder-30B。测量草稿接受率灵敏度。

4. 扩展到 H200（141 GB）。报告每副本模型规模余量，以及是否可以服务未量化的 Llama 3.3 70B。

5. 在相同 H100 硬件上基准测试 TensorRT-LLM 投机解码。报告它在哪里赢过 vLLM。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 草稿模型 | "投机器" | 为目标验证提出 N 个 token 的小模型 |
| EAGLE-3 | "2026 年草稿架构" | 在目标隐藏状态上训练的草稿头；约 75% 接受率 |
| P-EAGLE | "并行投机" | 在一次目标前向传播中验证的草稿分支树 |
| 接受率 | "命中率" | 无需重采样即被接受的草稿 token 比例 |
| 量化 | "FP8/INT4" | 降低精度的权重，以在 GPU 内存中容纳更大的模型 |
| 队列等待 | "HPA 指标" | 请求在开始推理前在待处理队列中等待的时间 |
| Speculators hub | "对齐草稿" | Red Hat Neural Magic 为常见开源模型提供的 EAGLE 草稿 hub |

## 延伸阅读

- [vLLM EAGLE 和 P-EAGLE 文档](https://docs.vllm.ai) — 参考服务栈
- [P-EAGLE（AWS 2026）](https://aws.amazon.com/blogs/machine-learning/p-eagle-faster-llm-inference-with-parallel-speculative-decoding-in-vllm/) — 并行投机解码论文 + 集成
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — 草稿头训练流水线
- [Red Hat Speculators](https://github.com/neuralmagic/speculators) — 对齐草稿 hub
- [TensorRT-LLM 投机解码](https://nvidia.github.io/TensorRT-LLM/) — 供应商备选
- [Fireworks.ai 服务架构](https://fireworks.ai/blog) — 商业参考
- [EAGLE-3 论文（arXiv:2503.01840）](https://arxiv.org/abs/2503.01840) — 方法论文
- [vLLM 仓库](https://github.com/vllm-project/vllm) — 代码与基准
