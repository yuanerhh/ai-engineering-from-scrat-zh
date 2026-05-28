# 浏览器智能体与长时间网络任务

> ChatGPT 智能体（2025 年 7 月）将 Operator 和深度研究合并为一个浏览器/终端智能体，并在 BrowseComp 上创下 68.9% 的 SOTA。OpenAI 于 2025 年 8 月 31 日关闭了 Operator——产品层的整合。Anthropic 的 Vercept 收购将 Claude Sonnet 在 OSWorld 上的成绩从 15% 以下提升至 72.5%。WebArena-Verified（ServiceNow，ICLR 2026）修复了原始 WebArena 中 11.3 个百分点的假阴性率，并发布了 258 个任务的 Hard 子集。这些数字是真实的。攻击面也是真实的：OpenAI 准备主管公开表示，针对浏览器智能体的间接提示词注入"不是一个可以完全修补的漏洞"。2025-2026 年已记录的攻击：Tainted Memories（Atlas CSRF）、HashJack（Cato Networks）和 Perplexity Comet 中的一键劫持。

**类型：** 学习
**编程语言：** Python（标准库，间接提示词注入攻击面模型）
**前置知识：** Phase 15 · 10（权限模式）、Phase 15 · 01（长时间智能体）
**预计时间：** 约 45 分钟

## 问题背景

浏览器智能体是一个读取不可信内容并采取后果性操作的长时间智能体。智能体访问的每个页面都是用户没有编写的输入。每个页面上的每个表单都是潜在的命令渠道。2025-2026 年的攻击语料库表明这不是假设的：Tainted Memories 让攻击者通过精心制作的页面将恶意指令绑定到智能体的内存；HashJack 在智能体访问的 URL 片段中隐藏命令；Perplexity Comet 劫持在单击时发生。

防御图景令人不安。OpenAI 准备主管说出了大家想说但没说的话：间接提示词注入"不是一个可以完全修补的漏洞"。这是因为攻击存在于智能体的读取与操作边界中，而这个边界在架构上是模糊的——模型读取的每个 token 原则上都可以被读作指令。

本课命名攻击面，命名基准格局（BrowseComp、OSWorld、WebArena-Verified），并模拟最小间接提示词注入场景，以便你能在第 14 课和第 18 课中推理真实防御。

## 核心概念

### 2026 年格局，每个系统一段话

**ChatGPT 智能体（OpenAI）。** 2025 年 7 月推出。统一了 Operator（浏览）和深度研究（多小时研究）。2025 年 8 月 31 日关闭了独立的 Operator。BrowseComp SOTA 为 68.9%；OSWorld 和 WebArena-Verified 上的数字表现强劲。

**Claude Sonnet + Vercept（Anthropic）。** Anthropic 的 Vercept 收购专注于计算机使用能力。将 Claude Sonnet 在 OSWorld 上的成绩从 <15% 提升至 72.5%。Claude Computer Use 作为工具 API 发布。

**Gemini 3 Pro 带浏览器使用（DeepMind）。** Browser Use 集成提供计算机使用控制；FSF v3（2026 年 4 月，第 20 课）专门跟踪 ML R&D 领域的自主性。

**WebArena-Verified（ServiceNow，ICLR 2026）。** 修复了一个有据可查的问题：原始 WebArena 有约 11.3% 的假阴性率（标记为失败但实际上已解决的任务）。Verified 版本用人工策划的成功标准重新评分，并添加了 258 个任务的 Hard 子集（ICLR 2026 论文，openreview.net/forum?id=94tlGxmqkN）。

### BrowseComp vs OSWorld vs WebArena

| 基准 | 测量内容 | 时间视野 |
|------|---------|---------|
| BrowseComp | 在时间压力下在开放网络上找到特定事实 | 分钟 |
| OSWorld | 智能体操作完整桌面（鼠标、键盘、shell） | 数十分钟 |
| WebArena-Verified | 模拟站点中的事务性网络任务 | 分钟 |
| Hard 子集 | 带多页状态转换的 WebArena-Verified 任务 | 数十分钟 |

不同的轴。高 BrowseComp 分数说明智能体能找到事实；它不说明智能体能预订航班。OSWorld 分数更接近"它在我的桌面上能工作"。WebArena-Verified 更接近"它能完成一个流程"。任何生产决策都需要与任务分布匹配的基准。

### 攻击面，已命名

1. **间接提示词注入。** 不可信的页面内容包含指令。智能体读取它们。智能体执行它们。公开例子：2024 年 Kai Greshake 等人、2025 年 Tainted Memories 论文、2026 年 HashJack（Cato Networks）。
2. **URL 片段/查询注入。** 爬取的 URL 的 `#fragment` 或查询字符串包含命令。从不可见地渲染；但仍在智能体的上下文中。
3. **内存绑定攻击。** 页面指示智能体写入持久内存（第 12 课涵盖持久状态）。下一个会话，内存触发有效载荷，没有可见触发器。
4. **针对已认证会话的 CSRF 形状攻击。** Tainted Memories 类：智能体在某处已登录；攻击者的页面发出智能体使用用户 cookie 执行的状态更改请求。
5. **一键劫持。** 视觉上无害的按钮携带智能体跟随的有效载荷。Comet 类。
6. **智能体主机面中的内容安全策略漏洞。** 渲染和工具层本身可以是攻击向量；浏览器中的浏览器智能体堆栈很宽广。

### 为什么"无法完全修补"

攻击与智能体的能力是同构的。智能体必须读取不可信内容才能完成工作。智能体读取的任何内容都可能包含指令。智能体遵循的任何指令都可能与用户的实际请求不一致。防御（信任边界、分类器、工具允许列表、后果性操作上的 HITL）提高了攻击的成本并减少了其爆炸半径。它们不能关闭这个类别。

这与 Lob 定理（第 8 课）的推理模式相同：智能体不能证明下一个 token 是安全的；它只能建立一个不安全 token 更可检测的系统。

### 实际部署的防御态势

- **读取/写入边界。** 读取从不具有后果性。写入（提交表单、发布内容、调用有副作用的工具）如果启动内容来自信任边界之外，则需要新的人工审批。
- **每任务工具允许列表。** 智能体可以浏览；它不能发起电汇，除非该工具已为该任务明确启用。第 13 课涵盖预算。
- **会话隔离。** 浏览器智能体会话仅使用范围凭据运行。没有生产认证，没有个人电子邮件。保留每个 HTTP 请求的日志以供审计。
- **内容消毒器。** 获取的 HTML 在连接到模型上下文之前去除已知不良模式。（减少简单攻击；不能阻止复杂有效载荷。）
- **后果性操作上的 HITL。** 提议-然后-提交模式（第 15 课）。
- **内存上的金丝雀令牌。** 如果内存条目触发，用户会看到它（第 14 课）。

## 动手实践

`code/main.py` 对三个合成页面模拟一个小型浏览器智能体运行。一个页面是无害的，一个在可见文本中有直接提示词注入块，一个有 URL 片段注入（不可见但在智能体上下文中）。脚本显示 (a) 朴素智能体会做什么，(b) 读取/写入边界捕获什么，(c) 消毒器捕获什么，(d) 两者都无法捕获什么。

## 产出技能

`outputs/skill-browser-agent-trust-boundary.md` 确定提议的浏览器智能体部署的范围：它触及哪些信任区域，被授权写入什么，以及在第一次运行之前必须到位哪些防御。

## 练习

1. 运行 `code/main.py`。找出消毒器捕获但读取/写入边界捕获不到的攻击，以及只有读取/写入边界捕获的攻击。

2. 扩展消毒器以检测一类 HashJack 风格的 URL 片段注入。测量良性 URL（带合法片段）上的误报率。

3. 选择你知道的一个真实浏览器智能体工作流（例如"预订航班"）。列出每次读取和每次写入。标记哪些写入需要 HITL 以及原因。

4. 阅读 WebArena-Verified ICLR 2026 论文。找出原始 WebArena 评分不可靠的一类任务，并解释 Verified 子集如何解决它。

5. 为浏览器智能体设置设计一个内存金丝雀。你会存储什么，存储在哪里，什么触发警报？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 间接提示词注入 | "坏页面文本" | 智能体读取的页面中的不可信内容包含智能体执行的指令 |
| Tainted Memories | "内存攻击" | 智能体将攻击者提供的指令写入持久内存；下一个会话触发 |
| HashJack | "URL 片段攻击" | 隐藏在 URL 片段/查询字符串中的有效载荷在智能体上下文中但不可见地渲染 |
| 一键劫持 | "坏按钮" | 可见的可操作元素携带智能体执行的后续有效载荷 |
| BrowseComp | "网络搜索基准" | 在开放网络上找到特定事实；分钟级时间视野 |
| OSWorld | "桌面基准" | 完整操作系统控制；多步 GUI 任务 |
| WebArena-Verified | "固定网络任务基准" | ServiceNow 的重新评分 WebArena，带 Hard 子集 |
| 读取/写入边界 | "副作用门" | 读取从不具有后果性；写入如果内容来自不可信源则需要新批准 |

## 延伸阅读

- [OpenAI — 介绍 ChatGPT 智能体](https://openai.com/index/introducing-chatgpt-agent/) — Operator 和深度研究的合并；BrowseComp SOTA
- [OpenAI — 计算机使用智能体](https://openai.com/index/computer-using-agent/) — Operator 谱系和成为 ChatGPT 智能体的架构
- [Zhou 等人 — WebArena](https://webarena.dev/) — 原始基准
- [WebArena-Verified（OpenReview）](https://openreview.net/forum?id=94tlGxmqkN) — ICLR 2026 固定子集论文
- [Anthropic — 实践中测量智能体自主性](https://www.anthropic.com/research/measuring-agent-autonomy) — 包含计算机使用智能体的攻击面讨论
