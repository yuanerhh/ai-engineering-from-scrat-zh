# 顶点项目 07 — 端到端微调流水线（从数据到 SFT 到 DPO 到部署）

> 一个用自有数据训练的 8B 模型，用自有偏好数据进行 DPO 对齐，量化，投机解码，并以可测量的每百万 token 成本提供服务。2026 年的开源技术栈是：Axolotl v0.8、TRL 0.15、Unsloth 用于快速迭代、GPTQ/AWQ/GGUF 用于量化、vLLM 0.7 配合 EAGLE-3 用于服务。这个顶点项目要求你可复现地运行整条流水线——YAML 配置进，服务端点出——并在 2026 年模型开放度框架下发布模型卡片。

**类型：** 顶点项目  
**语言：** Python（流水线），YAML（配置），Bash（脚本）  
**前置条件：** 第 2 阶段（机器学习），第 3 阶段（深度学习），第 7 阶段（Transformer），第 10 阶段（从零构建 LLM），第 11 阶段（LLM 工程），第 17 阶段（基础设施），第 18 阶段（安全）  
**涵盖阶段：** P2 · P3 · P7 · P10 · P11 · P17 · P18  
**预计时间：** 35 小时

## 问题背景

2026 年每个认真的 AI 团队都随时保有一条微调流水线。不是因为他们要发布前沿基础模型，而是因为下游适配——领域 SFT、对标注偏好的 DPO、用于投机解码的蒸馏草稿、用 EAGLE-3 提供服务——才是可测量收益的所在。Axolotl v0.8 处理多 GPU SFT 配置。TRL 0.15 处理 DPO 和 GRPO。Unsloth 实现快速单 GPU 迭代。vLLM 0.7 配合 EAGLE-3 在不损失质量的情况下将解码吞吐量提升 2-3 倍。工具链已经成熟；精髓在于 YAML 配置、数据卫生和评测规范。

你将把一个 8B 基础模型（Llama 3.3、Qwen3 或 Gemma 3）经过 SFT 再经 DPO 在任务特定数据上进行训练，量化后部署，并在 lm-evaluation-harness、RewardBench-2、MT-Bench-v2 和 MMLU-Pro 上测量提升效果。你将在 2026 年模型开放度框架（MOF）下发布一张模型卡片。重点是可复现性——一条命令就能端到端重新运行整条流水线。

## 核心概念

流水线有五个阶段。**数据**：去重（MinHash / Datatrove）、质量过滤（Nemotron-CC 风格分类器）、PII 清洗、针对公开基准的数据污染分割卫生检查。**SFT**：Axolotl YAML，8xH100 上的 ZeRO-3，余弦调度，打包序列，2-3 个 epoch。**DPO 或 GRPO**：TRL 配置，1 个 epoch，偏好对（人工标注或模型评判），beta 调优。**量化**：GPTQ + AWQ + GGUF，灵活部署。**服务**：vLLM 0.7 配合 EAGLE-3 投机解码头（或 SGLang + SpecForge），K8s 部署，HPA 按队列等待时间扩缩容。

消融实验是交付物：SFT-only vs SFT+DPO vs SFT+GRPO 在三个任务特定基准上的对比。服务指标：batch 1/8/32 时的 tokens/s，EAGLE-3 接受率，每百万 token 成本。安全评测：Llama Guard 4 通过率。模型卡片：偏差评测、可复现性种子、数据授权。

## 架构图

```
raw data (HF datasets + internal)
    |
    v
Datatrove dedup + Nemotron-CC quality filter + PII scrub
    |
    v
split hygiene (MMLU-Pro contamination check)
    |
    v
Axolotl SFT config (YAML)  ---> 8xH100, ZeRO-3
    |
    v
TRL DPO / GRPO config       ---> 4xH100, 1 epoch
    |
    v
GPTQ + AWQ + GGUF quantize
    |
    v
vLLM 0.7 + EAGLE-3 speculative decoding
    |
    v
K8s deployment, HPA on queue-wait
    |
    v
lm-eval-harness + RewardBench-2 + MT-Bench-v2 + MMLU-Pro
    |
    v
model card (2026 MOF) + safety eval (Llama Guard 4)
```

## 技术栈

- 数据：Datatrove 用于去重，Nemotron-CC 分类器用于质量过滤，Presidio 用于 PII 清洗
- 基础模型：Llama 3.3 8B、Qwen3 14B 或 Gemma 3 12B
- SFT：Axolotl v0.8，ZeRO-3，Flash Attention 3，打包序列
- 偏好调优：TRL 0.15 用于 DPO 或 GRPO；Unsloth 用于单 GPU 迭代
- 量化：GPTQ（Marlin）、AWQ、通过 llama.cpp 的 GGUF
- 服务：vLLM 0.7 配合 EAGLE-3 投机解码（或 SGLang 0.4 + SpecForge）
- 评测：lm-evaluation-harness、RewardBench-2、MT-Bench-v2、MMLU-Pro
- 安全评测：Llama Guard 4、ShieldGemma-2
- 基础设施：Kubernetes + NVIDIA 设备插件，HPA 按队列等待指标
- 可观测性：W&B 用于训练，Langfuse 用于推理

## 构建步骤

1. **数据流水线。** 对原始语料库运行 Datatrove 去重。应用 Nemotron-CC 风格质量分类器。Presidio 清洗 PII。用显式种子写入训练/验证分割。

2. **污染检查。** 对每个验证分割，对 MMLU-Pro、MT-Bench-v2、RewardBench-2 测试集计算 MinHash。拒绝任何重叠。

3. **Axolotl SFT。** 带 ZeRO-3、FA3、序列打包的 YAML。8xH100 上 2-3 个 epoch。日志记录到 W&B。

4. **TRL DPO / GRPO。** 取 SFT 检查点，在偏好对上运行一个 epoch 的 DPO（或带可验证奖励的 GRPO，用于数学/代码）。调优 beta。

5. **量化。** 生成三种量化版本：GPTQ-INT4-Marlin、AWQ-INT4、GGUF-Q4_K_M（用于 llama.cpp）。记录大小和名义吞吐量。

6. **用投机解码提供服务。** vLLM 0.7 配置，通过 Red Hat Speculators 训练的 EAGLE-3 草稿头。测量 batch 1/8/32 下的接受率和尾延迟。报告每百万 token 成本 vs Anthropic/OpenAI 在相同评测上的成本。

7. **评测矩阵。** 对基础模型、SFT-only、SFT+DPO、SFT+GRPO 运行 lm-eval-harness、RewardBench-2、MT-Bench-v2、MMLU-Pro。生成表格。

8. **安全评测。** Llama Guard 4 在开发集上的通过率。ShieldGemma-2 输出过滤器。

9. **模型卡片。** MOF 2026 模板：数据、训练、评测、安全、许可证、带 YAML 和提交 SHA 的可复现性部分。

## 使用示例

```
$ ./pipeline.sh config/llama3.3-8b-domainX.yaml
[data]    300k deduped, 12k filtered, 280k accepted (seed=7)
[SFT]     3 epochs, 8xH100, 6h12m, val loss 1.42 -> 1.03
[DPO]     1 epoch, beta=0.08, 4xH100, 1h40m
[quant]   GPTQ-INT4 4.6 GB, AWQ-INT4 4.8 GB, GGUF-Q4_K_M 5.1 GB
[serve]   vLLM 0.7, EAGLE-3 acceptance 0.74, p99 126ms @ bs=8
[eval]    MMLU-Pro +3.2, MT-Bench-v2 +0.41, RewardBench-2 +0.08
[card]    model-card.md generated under 2026 MOF
```

## 交付物

`outputs/skill-finetuning-pipeline.md` 描述了交付物。单条命令将数据经过 SFT、DPO、量化、部署、评测跑完，并输出模型卡片和服务端点。

| 权重 | 评分标准 | 衡量方式 |
|:-:|---|---|
| 25 | 评测指标 vs 基础模型的提升 | 目标任务（MMLU-Pro、MT-Bench-v2、任务特定）的可测量提升 |
| 20 | 流水线可复现性 | 相同种子的单命令端到端重新运行 |
| 20 | 数据卫生 | 去重率、PII 清洗覆盖率、污染检查通过 |
| 20 | 服务效率 | bs=1/8/32 时的 tokens/s，EAGLE-3 接受率，每百万 token 成本 |
| 15 | 模型卡片 + 安全评测 | 2026 MOF 完整性 + Llama Guard 4 通过率 |
| **100** | | |

## 练习题

1. 在相同任务特定基准上比较 SFT-only、SFT+DPO 和 SFT+GRPO。报告哪种偏好方法胜出以及胜出多少。

2. 将 Llama 3.3 8B 替换为 Qwen3 14B。测量相同质量下的每百万 token 成本。

3. 测量 EAGLE-3 在领域数据与通用 ShareGPT 上的接受率。报告差值及其对延迟预算的影响。

4. 向训练数据中注入 1% 的污染（将 MMLU-Pro 答案泄露进训练数据），然后重新运行评测。观察 MMLU-Pro 精度不切实际地跳升。构建一个可以捕捉到这一情况的污染检查 CI 门控。

5. 添加 LoRA SFT 作为全量微调的替代方案。测量内存减少 10 倍时的质量差距。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Axolotl | "SFT 训练器" | 统一的 YAML 驱动训练器，支持 SFT、DPO 和蒸馏 |
| TRL | "偏好调优器" | Hugging Face 用于 DPO、GRPO、PPO 微调 LLM 的库 |
| GRPO | "组相对策略优化" | DeepSeek R1 的带可验证奖励的 RL 方案 |
| EAGLE-3 | "投机解码草稿" | 提前预测 N 个 token 的草稿头；vLLM 用目标模型验证 |
| MOF | "模型开放度框架" | 2026 年在数据、代码、许可证方面对模型发布进行评级的标准 |
| 污染检查 | "分割卫生" | 基于 MinHash 检测测试集泄露进训练数据 |
| 接受率 | "EAGLE/MTP 指标" | 目标模型接受的草稿 token 比例 |

## 延伸阅读

- [Axolotl 文档](https://axolotl-ai-cloud.github.io/axolotl/) — 参考 SFT/DPO 训练器
- [TRL 文档](https://huggingface.co/docs/trl) — DPO 和 GRPO 参考实现
- [Unsloth](https://github.com/unslothai/unsloth) — 单 GPU 快速迭代参考
- [DeepSeek R1 论文（arXiv:2501.12948）](https://arxiv.org/abs/2501.12948) — GRPO 方法论
- [vLLM + EAGLE-3 文档](https://docs.vllm.ai) — 参考服务栈
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — 备选投机解码训练器
- [模型开放度框架 2026](https://isocpp.org/) — 开放发布评级标准
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — 标准评测运行器
