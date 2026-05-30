---
name: vllm-scheduler-reader
description: 通过读取调度器级别的旋钮来诊断 vLLM 服务配置，并识别 PagedAttention、连续批处理和分块预填充中哪个是瓶颈。
version: 1.0.0
phase: 17
lesson: 04
tags: [vllm, paged-attention, continuous-batching, chunked-prefill, serving, scheduler]
---

给定一个 vLLM 服务配置（模型、dtype、硬件、`--gpu-memory-utilization`、`--max-num-batched-tokens`、`--enable-chunked-prefill`、`--speculative-model` 或 `--speculative-config`、最大并发，以及 TTFT 均值/P99、ITL 均值/P99、吞吐量 tok/s 的观测指标集），生成调度器级别诊断。

产出内容：

1. **配置解读。** 对于每个标志，列出其控制的调度器行为和 2026 年默认值。标记设置为非默认值的任何标志并说明原因。
2. **瓶颈识别。** 将瓶颈分类为以下之一：PagedAttention 配置不足（KV 块饥饿）、连续批处理停顿（WAITING 队列增长）、分块预填充大小不当（TTFT 尾部峰值）、解码计算绑定（ITL 下限）或 HBM 绑定（无法容纳批次）。用报告的指标说明理由。
3. **旋钮建议。** 具体的有序操作——要翻转哪个标志、要尝试哪个值，以及要观察哪个指标。在用尽调度器级别调优之前，不要建议"尝试更多 GPU"。
4. **兼容性检查。** 特别针对 vLLM v0.18.0：将 `--enable-chunked-prefill` + `--speculative-model` 组合标记为硬性不兼容。如果两者都需要，推荐 V1 中的 N-gram GPU 推测解码作为记录在案的例外。
5. **下一步阅读。** 根据诊断结果，指向 vLLM v0.18.0 发布说明、PagedAttention 论文或 Aleksa Gordic V1 调度器演练中的一个。

硬性拒绝：
- 没有四个核心指标（TTFT、ITL、吞吐量、并发）就进行诊断。拒绝并要求提供指标集。
- 不检查推测解码配置就推荐 `--enable-chunked-prefill`。
- 将 `DCGM_FI_DEV_GPU_UTIL` 视为扩缩信号。vLLM 预先分配 KV；占空比数字会产生误导。

拒绝规则：
- 如果报告的吞吐量在 H100 上低于 100 tok/s，瓶颈可能不是 vLLM——检查客户端侧的分词器、Python GIL 或请求级串行化。
- 如果 `--gpu-memory-utilization` 设置低于 0.7，拒绝进一步调优——运营商选择了放弃 HBM，修复方法是在翻转调度器标志之前提高上限。
- 如果运营商要求在草稿模型推测上使用推测解码 + 分块预填充的配方，拒绝并列出 v0.18.0 不兼容性。指向 Phase 17 · 05 的 EAGLE-3。

输出：一页调度器诊断，列出标志、瓶颈、有序建议、兼容性注意事项和下一步阅读指针。结尾给出"下一步要测量什么"段落，根据识别的瓶颈列出 P99 ITL、块分配率或 WAITING 队列深度中的一个。
