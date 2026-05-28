# Agno 与 Mastra：生产运行时

> Agno（Python）和 Mastra（TypeScript）是 2026 年生产运行时的配对组合。Agno 的目标是微秒级智能体实例化和无状态 FastAPI 后端。Mastra 在 Vercel AI SDK 基础上提供智能体、工具、工作流、统一模型路由和复合存储。

**类型：** 学习
**编程语言：** Python、TypeScript
**前置知识：** Phase 14 · 01（智能体循环）、Phase 14 · 13（LangGraph）
**预计时间：** 约 45 分钟

## 学习目标

- 识别 Agno 的性能目标以及它们在什么时候重要。
- 列出 Mastra 的三个原语——Agents（智能体）、Tools（工具）、Workflows（工作流）——以及支持的服务器适配器。
- 解释为什么无状态的会话范围 FastAPI 后端是推荐的 Agno 生产路径。
- 针对给定的技术栈（Python 优先 vs TypeScript 优先）在 Agno 和 Mastra 之间做出选择。

## 问题背景

LangGraph、AutoGen、CrewAI 是重型框架。想要"仅仅是智能体循环、快速、在我的运行时中"的团队会选择 Agno（Python）或 Mastra（TypeScript）。两者都用一些框架拥有的原语换取了原始速度和对周围技术栈更紧密的适配。

## 核心概念

### Agno

- Python 运行时，前身为 Phi-data。
- "没有图、链或复杂模式——只有纯 Python。"
- 来自文档的性能目标：约 2μs 智能体实例化，约 3.75 KiB 每智能体内存，约 23 个模型提供商。
- 生产路径：无状态的会话范围 FastAPI 后端。每个请求启动一个新鲜智能体；会话状态存在数据库中。
- 原生多模态（文本、图像、音频、视频、文件）和智能体 RAG。

当每秒有数千个短暂智能体时（聊天扇入、评估流水线），速度目标很重要。当一个智能体运行 10 分钟时，它们就不那么重要了。

### Mastra

- TypeScript，构建在 Vercel AI SDK 之上。
- 三个原语：**Agents（智能体）**、**Tools（工具）**（Zod 类型化）、**Workflows（工作流）**。
- 统一模型路由器——跨 94 个提供商的 3,300+ 个模型（2026 年 3 月）。
- 复合存储：记忆、工作流、可观测性到不同后端；规模化可观测性推荐 ClickHouse。
- Apache 2.0，`ee/` 目录采用源码可用企业许可证。
- 支持 Express、Hono、Fastify、Koa 的服务器适配器；一流的 Next.js 和 Astro 集成。
- 自带 Mastra Studio（localhost:4111）用于调试。
- 1.0 版（2026 年 1 月）时 22k+ GitHub stars，每周 npm 下载量 30 万+。

### 定位

两者都不试图成为 LangGraph。它们在以下方面竞争：

- **语言适配。** Agno 用于 Python 优先的团队；Mastra 用于 TypeScript 优先的团队。
- **运行时人机工程学。** Agno = 接近零开销；Mastra = 与 Vercel 生态系统集成。
- **可观测性。** 两者都与 Langfuse/Phoenix/Opik（第 24 课）集成，但 Mastra Studio 是第一方的。

### 何时选择各自

- **Agno**——Python 后端、大量短暂智能体、强性能要求、FastAPI 技术栈。
- **Mastra**——TypeScript 后端、Next.js/Vercel 部署、统一多提供商模型路由、Zod 类型化工具。
- **LangGraph**（第 13 课）——当持久化状态和显式图推理比原始速度更重要时。
- **OpenAI / Claude Agent SDK**——当你想要提供商产品化形态时（第 16-17 课）。

### 这个模式在哪里出错

- **为了性能而性能。** 因为"2μs"听起来好就选择 Agno，但工作负载是每个请求一次慢速智能体调用。开销不是瓶颈。
- **生态系统锁定。** Mastra 的 Vercel 风格集成在 Vercel 上是优点，在其他地方是缺点。
- **企业许可证混淆。** Mastra 的 `ee/` 目录是源码可用的，而非 Apache 2.0。如果你计划 fork，请仔细阅读许可证。

## 动手实践

这节课主要是比较性的——没有单一的代码制品能对两个框架都公平。参见 `code/main.py` 中的并排玩具：一个最小的"运行智能体、流式输出、持久化会话"流程，实现了两次（一次 Agno 形态，一次 Mastra 形态）。

运行：

```
python3 code/main.py
```

两条结构上不同但功能上等价的轨迹。

## 使用建议

- **Agno**——需要速度和 FastAPI 形态的 Python 后端。
- **Mastra**——带多提供商和工作流原语的 TypeScript 后端。
- 两者都提供第一方可观测性钩子。两者都与 Langfuse 集成。

## 产出技能

`outputs/skill-runtime-picker.md` 根据技术栈、延迟预算和操作形态选择 Agno、Mastra、LangGraph 或提供商 SDK。

## 练习

1. 阅读 Agno 文档。将标准库 ReAct 循环（第 01 课）移植到 Agno。什么消失了？什么留下了？
2. 阅读 Mastra 文档。将同样的循环移植到 Mastra。工具类型化有什么变化（Zod vs 无）？
3. 基准测试：测量你技术栈上的智能体实例化延迟。Agno 的 2μs 对你的工作负载重要吗？
4. 设计迁移方案：如果你一直在 Python 中运行 CrewAI，移动到 Agno 会有什么问题？
5. 阅读 Mastra 的 `ee/` 许可证条款。哪些限制会影响开源 fork？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Agno | "快速 Python 智能体" | 无状态会话范围的智能体运行时 |
| Mastra | "Vercel AI SDK 上的 TypeScript 智能体" | 智能体 + 工具 + 工作流 + 模型路由器 |
| 统一模型路由器 | "多提供商访问" | 跨 94 个提供商 3,300+ 个模型的单一客户端 |
| 复合存储 | "多后端" | 记忆/工作流/可观测性各到不同的存储 |
| Mastra Studio | "本地调试器" | localhost:4111 的智能体自省 UI |
| 源码可用 | "非开源" | 许可证允许读取源码但限制商业使用 |

## 延伸阅读

- [Agno 智能体框架文档](https://www.agno.com/agent-framework) — 性能目标、FastAPI 集成
- [Mastra 文档](https://mastra.ai/docs) — 原语、服务器适配器、模型路由器
- [LangGraph 概述](https://docs.langchain.com/oss/python/langgraph/overview) — 有状态图的替代方案
- [Comet Opik](https://www.comet.com/site/products/opik/) — Mastra 集成引用的可观测性比较
