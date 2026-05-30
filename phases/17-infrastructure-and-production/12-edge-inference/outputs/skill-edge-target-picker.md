---
name: edge-target-picker
description: 根据设备、模型和延迟预算，选择边缘推理目标（Apple ANE、Qualcomm Hexagon、WebGPU/WebLLM、NVIDIA Jetson）及匹配的量化格式。
version: 1.0.0
phase: 17
lesson: 12
tags: [edge, ane, hexagon, webgpu, webllm, jetson, core-ml, qnn, nvfp4]
---

给定部署平台（iOS、Android、浏览器、机器人/汽车/边缘服务器）、模型及延迟/内存预算，生成边缘目标推荐方案。

输出内容：

1. 目标。说明具体的 NPU/GPU（ANE、Hexagon、WebGPU、Jetson Orin Nano / AGX / Thor）。结合平台和 2026 年运行时覆盖情况加以说明。
2. 带宽上限。计算理论解码上限：`bandwidth_GB_s / model_size_GB`。与用户的 tok/s 需求进行比较。若上限低于需求，拒绝或建议使用更小的模型/更紧的量化。
3. 量化格式。选择 Q4 GGUF（浏览器/边缘 CPU）、Core ML INT4 + FP16（ANE）、QNN INT8/INT4（Hexagon）或 NVFP4 + FP8 KV（Jetson Thor / Edge-LLM）。
4. 转换流水线。说明具体的转换工具（Core ML converter、Qualcomm AI Hub、MLC-LLM for WebLLM、TensorRT-LLM Edge compiler）。
5. 上下文预算。说明设备 RAM 在容纳权重的同时能支持的最大上下文长度。对于长上下文场景，指定 KV 量化（Q4 KV）或拒绝。
6. 回退方案。当设备不支持或 WebGPU 不可用（Firefox Android、旧版浏览器）时，指定使用相同 OpenAI 兼容接口的服务端 API 回退方案。

强制拒绝：
- 承诺超过带宽上限的 tok/s。拒绝——物理限制不可逾越。
- 在 2026 年通过非 Core ML 运行时直接访问 ANE。只有 Core ML 能原生暴露 ANE。
- 假设所有浏览器都支持 WebGPU。2026 年移动端覆盖率约为 70-75%；必须始终指定回退方案。

拒绝规则：
- 若模型 >6 GB 且目标设备为手机（4-8 GB RAM），拒绝——优先建议使用更小的模型或激进量化。
- 若请求是在 iPhone 上以 7B 模型支持 128K 上下文，拒绝——设备 RAM 无法容纳，除非使用 Q4 KV 加滑动窗口注意力。
- 若部署需要通过 WebGPU 在 Android 上进行长上下文流式推理，且用户要求支持 Firefox，拒绝并要求改用 Chrome 或服务端回退方案。

输出：一页方案，列明目标、上限、量化格式、转换工具、上下文预算、回退方案。最后给出唯一指标：目标机队中最差设备上的实测 tok/s。
