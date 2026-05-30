---
name: engine-picker
description: 根据硬件、规模和工作负载选择自托管 LLM 推理引擎（llama.cpp、Ollama、TGI、vLLM、SGLang）。将 2026 年 TGI 进入维护模式作为迁移触发条件。
version: 1.0.0
phase: 17
lesson: 28
tags: [self-hosted, vllm, sglang, llama-cpp, ollama, tgi, trt-llm, engine-selection]
---

给定硬件（CPU / Apple Silicon / AMD / NVIDIA Hopper / NVIDIA Blackwell）、规模（单用户 / 小团队 / 生产环境 / 企业级）以及工作负载（通用对话 / 智能体 / RAG / 长上下文 / 代码），给出推理引擎推荐。

产出内容：

1. 引擎选型。指定具体引擎。引用"硬件优先、规模次之、工作负载第三"的决策树。
2. 为何不选其他方案。对每个备选引擎，说明未被选中的原因（TGI 维护模式、AMD 不支持 TRT-LLM、Ollama 仅适用于开发环境）。
3. 流水线设计。若为生产环境，指定流水线模式（开发用 Ollama → 预发布用 llama.cpp → 生产用 vLLM/SGLang），并确认权重格式（GGUF 或 HF）可贯通流程。
4. 生产环境堆叠。在生产规模下，参考第 17 阶段 · 第 18 课（生产栈）、第 17 课（分解式推理）、第 11 课（缓存感知路由器）进行组合。
5. TGI 迁移。如果现有方案使用 TGI，制定迁移计划和时间线——不紧急，但应在 6 个月内启动。
6. 硬件注意事项。指出两个硬约束：仅 CPU → 选 llama.cpp；AMD → 不支持 TRT-LLM。

强制拒绝：
- 2026 年新项目默认选择 TGI。拒绝——已进入维护模式。
- Ollama 用于并发用户 > 1 的共享生产环境。拒绝——吞吐量差距过大。
- 在未确认 NVIDIA 专属环境的情况下推荐 TRT-LLM。拒绝——AMD / 非 NVIDIA 硬件是硬性阻断条件。

拒绝规则：
- 如果硬件混合部署（部分 AMD、部分 NVIDIA），需按集群分别做引擎决策；不得强制统一使用一种引擎。
- 如果工作负载在生产规模下属于"未知/通用"，默认选择 vLLM 并计划在积累 3 个月流量数据后重新评估。
- 如果团队希望"在没有 Blackwell 可用的情况下追求每 GPU 最高性能"并坚持只用 Hopper，予以确认——TRT-LLM 或 vLLM 均可接受。

输出：一页推荐方案，包含引擎选型、备选方案淘汰理由、流水线设计、生产环境堆叠、TGI 迁移态势。最后附上单一季度评审：当工作负载形态发生重大变化时重新评估引擎选型。
