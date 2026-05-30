---
name: finetuning-pipeline
description: 运行一个从数据到 SFT 再到 DPO 再到部署的可复现微调管道，包含消融实验、量化和 2026 年模型开放框架模型卡。
version: 1.0.0
phase: 19
lesson: 07
tags: [capstone, fine-tuning, axolotl, trl, dpo, grpo, vllm, eagle-3, mof]
---

给定一个基础模型（Llama 3.3 8B、Qwen3 14B 或 Gemma 3 12B）和一个特定任务数据集，构建一个单命令管道，生成一个可提供服务的端点和一张可复现的模型卡。

构建计划：

1. 数据阶段：Datatrove 去重、Nemotron-CC 风格质量过滤、Presidio PII 脱敏、带种子的训练/验证集划分。
2. 污染检查：使用 MinHashLSH 对 MMLU-Pro、MT-Bench-v2、RewardBench-2 进行比对。有重叠则拒绝。
3. SFT：在 8xH100 上使用 Axolotl v0.8，配置 ZeRO-3、Flash Attention 3、序列打包，训练 2-3 个 epoch。
4. 偏好调优：TRL 0.15 DPO（或使用可验证奖励的 GRPO）训练 1 个 epoch，进行 beta 扫描。
5. 量化：GPTQ-INT4-Marlin + AWQ-INT4 + GGUF-Q4_K_M。
6. 部署：使用带 EAGLE-3 投机解码（通过 Red Hat Speculators 或 SGLang SpecForge 的草稿头）的 vLLM 0.7。K8s 部署，HPA 基于队列等待时间。
7. 评估：在 base/仅SFT/SFT+DPO/SFT+GRPO 版本上运行 lm-evaluation-harness、RewardBench-2、MT-Bench-v2、MMLU-Pro。
8. 安全性：Llama Guard 4 通过率、ShieldGemma-2 输出过滤。
9. 依据 2026 年模型开放框架撰写模型卡，包含数据、训练、评估、安全性、可复现性各节。

评估标准：

| 权重 | 评估项 | 度量方式 |
|:-:|---|---|
| 25 | 相对基础模型的评估提升 | 在 MMLU-Pro、MT-Bench-v2 及特定任务基准上的测量收益 |
| 20 | 管道可复现性 | 使用相同种子的单命令重新运行产生相同哈希值 |
| 20 | 数据卫生 | 去重率、PII 脱敏覆盖率、污染检查通过 |
| 20 | 服务效率 | batch 1/8/32 时的 tokens/s、EAGLE-3 接受率、每百万 Token 成本 |
| 15 | 模型卡 + 安全性评估 | 2026 年 MOF 完整性 + Llama Guard 4 通过率 |

硬性拒绝条件：

- 跳过 MinHash 污染检查的管道。将 MMLU-Pro 泄漏到训练集是经典的评估作弊失败模式。
- 没有附加种子或 YAML 的训练运行。可复现性是硬性要求。
- 没有 EAGLE-3 或等效投机解码配置的部署。基线 tokens/s 不是 2026 年的标准。
- 缺少安全性评估。每个微调模型都需附带 Llama Guard 4 通过率。

拒绝规则：

- 拒绝发布声称基准分数却未附加 lm-eval-harness commit SHA 的模型卡。
- 拒绝在许可证禁止衍生模型的数据上进行微调。MOF 会对数据许可进行评级。
- 拒绝在未在评估矩阵上测量质量损失的情况下发布量化模型。

输出：一个包含管道编排器、Llama 3.3 8B + 一个备选基础模型的 YAML 配置、SFT 和 DPO W&B 运行日志、量化产物、已部署端点、三项基准评估矩阵、安全性评估、2026 年 MOF 模型卡的代码库，以及一份说明你发现并修复的三大数据卫生问题的报告。
