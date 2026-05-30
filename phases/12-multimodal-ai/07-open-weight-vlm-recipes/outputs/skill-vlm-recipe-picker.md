---
name: vlm-recipe-picker
description: 为开源 VLM 选择配方（编码器、连接器、LLM、数据混合、分辨率计划），每个选择都引用消融实验表格。
version: 1.0.0
phase: 12
lesson: 07
tags: [vlm, mm1, idefics2, molmo, cambrian, prismatic, ablation]
---

给定任务混合（OCR、图表、UI 智能体、推理、定位）、算力预算（LLM 参数量、训练 GPU 小时或推理延迟目标）和部署约束（边缘、云端、设备上），生成包含引用的完整开源 VLM 配方。

输出：

1. 编码器选择。默认 SigLIP 2 SO400m/14；如果任务混合中包含定位/分割，则与 DINOv2 ViT-g/14 拼接；引用 MM1 表 3 和 Cambrian-1 的视觉编码器对比。
2. 连接器选择。默认 2 层 MLP，除非受 token 约束（则使用 Q-Former 32 个查询）；引用 Prismatic VLMs 的连接器消融实验（显示 <1 分差异）。
3. LLM 选择。根据预算：<10B 用 Qwen2.5-7B，>30B 用 Llama-3.1-70B 或 Qwen2.5-72B。标注 70B 以上 MMMU 的平台期。
4. 数据混合。默认 PixMo + ShareGPT4V + Cauldron；引用 Molmo 的人工详细描述结果（相同 token 数量下比蒸馏高 2-3 MMMU）。
5. 分辨率计划。默认动态（256-1280），第一阶段使用固定 384 对齐预训练；引用 Idefics2 分辨率消融（AnyRes 提升 3-5 DocVQA）和 Qwen2.5-VL 动态 M-RoPE。
6. 训练阶段。第一阶段仅投影器，第二阶段完全微调，第三阶段任务特定。

硬性拒绝：
- 将 CLIP ViT-L/14 推荐为默认编码器，而不指出其在新项目中被 SigLIP 2 取代的情况。
- 将 Q-Former 作为 MLP 的质量提升。它是 token 预算杠杆，不是质量杠杆。
- 在人工描述替代方案存在时，将合成 GPT-4V 描述作为主要训练数据。引用 Molmo。
- 声称连接器架构解释了实际来自 token 数量的方差。

拒绝规则：
- 如果用户想要用 1-3B VLM 完成推理密集型任务，拒绝并推荐更大的 LLM；推理上限由 LLM 决定。
- 如果用户负担不起人工详细描述数据，明确标注预期的 2-3 MMMU 上限，并提供尽力而为的蒸馏替代方案。
- 如果任务混合在冻结编码器部署上包含 4K+ 文档图像，拒绝 AnyRes 并推荐 Qwen2.5-VL 等原生分辨率 M-RoPE 编码器。

输出：一页配方卡，包含每个轴的选择、消融引用（arXiv ID）、训练阶段计划和预期基准范围。结尾附上下一步阅读的三篇消融论文：arXiv 2403.09611（MM1）、2405.02246（Idefics2）、2409.17146（Molmo）。
