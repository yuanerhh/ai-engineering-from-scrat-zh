# 基准测试：WebArena 与 OSWorld

> WebArena 测试跨四个自托管应用的网页智能体能力。OSWorld 测试跨 Ubuntu、Windows、macOS 的桌面智能体能力。发布时（2023-2024 年）两者都显示出最先进智能体与人类之间的巨大差距。差距正在缩小；失败模式没有变化。

**类型：** 学习
**编程语言：** Python（标准库）
**前置知识：** Phase 14 · 19（SWE-bench、GAIA）
**预计时间：** 约 60 分钟

## 学习目标

- 描述 WebArena 的四个自托管应用以及为什么基于执行的评估很重要。
- 解释为什么 OSWorld 使用真实的操作系统截图而不是无障碍 API。
- 列出两种主要的 OSWorld 失败模式：GUI 定位和操作知识。
- 总结 OSWorld-G 和 OSWorld-Human 在基础基准之上添加了什么。

## 问题背景

通用智能体可以调用工具。它们能在浏览器中经过 20 次点击完成购物结账吗？它们能仅使用键盘和鼠标配置一台 Linux 机器吗？这些是 WebArena 和 OSWorld 回答的问题。

## 核心概念

### WebArena（Zhou 等人，ICLR 2024）

- 跨四个自托管网页应用的 812 个长视野任务：一个购物网站、一个论坛、一个类 GitLab 的开发工具、一个业务 CMS。
- 加上工具：地图、计算器、草稿本。
- 评估基于执行，通过 gym API——订单是否下了，问题是否关闭，CMS 页面是否更新了？
- 发布时：最佳 GPT-4 智能体达到 14.41%，而人类为 78.24%。

自托管框架很重要——因为目标应用被固定且可复现，基准不是脆弱的。

### 扩展版本

- **VisualWebArena**——视觉定位任务，成功取决于解读图像（截图作为一等观察）。
- **TheAgentCompany**（2024 年 12 月）——添加终端 + 编码；更像真实的远程工作环境。

### OSWorld（Xie 等人，NeurIPS 2024）

- 跨 Ubuntu、Windows、macOS 的 369 个真实计算机任务。
- 对真实应用的自由形式键盘和鼠标控制。
- 1920×1080 截图作为观察。
- 发布时：最佳模型 12.24%，而人类 72.36%。

### 主要失败模式

1. **GUI 定位（GUI grounding）。** 像素 → 元素映射。模型在 1920×1080 中可靠定位 UI 元素存在困难。
2. **操作知识（Operational knowledge）。** 哪个菜单有该设置，哪个键盘快捷键，哪个偏好面板。人类多年积累的知识长尾。

### 后续版本

- **OSWorld-G**——564 样本定位套件 + Jedi 训练集。将定位与规划分解，使你可以分别测量它们。
- **OSWorld-Human**——手工策划的黄金动作轨迹。显示顶级智能体使用比必要多 1.4-2.7 倍的步骤（轨迹效率差距）。

### 为什么这很重要

Claude computer use、OpenAI CUA、Gemini 2.5 Computer Use（第 21 课）都在以 WebArena 和 OSWorld 为形态的工作负载上训练。这些基准是目标；生产模型是发布的答案。

### 基准测试哪里出错

- **仅截图评估。** OSWorld 是截图驱动的；在 OSWorld 上评估使用 DOM 或无障碍 API 的智能体会错过定位挑战。
- **忽略轨迹长度。** 只对成功率评分会遗漏 OSWorld-Human 揭示的 1.4-2.7 倍步骤低效。
- **陈旧的自托管应用。** WebArena 的应用固定到特定版本；在不重新策划的情况下更新会破坏可比性。

## 动手实践

`code/main.py` 实现了一个玩具网页智能体框架：

- 最小的"购物应用"状态机：list_items、add_to_cart、checkout。
- 3 个任务的黄金轨迹。
- 一个尝试每个任务的脚本化智能体。
- 基于执行的评估器（状态检查）和轨迹效率指标（步骤数 vs 黄金）。

运行：

```
python3 code/main.py
```

输出：每任务的成功率和轨迹效率，镜像 OSWorld-Human 的方法论。

## 使用建议

- **WebArena Verified** 在内部集群上自托管，用于持续评估。
- **OSWorld** 在 VM 集群中用于桌面智能体。
- **计算机使用智能体**（第 21 课）——Claude、OpenAI CUA、Gemini——都在类似这些的工作负载上训练。
- **你自己的产品流程**——为你的前 20 个任务捕获黄金轨迹；每周针对它们运行智能体。

## 产出技能

`outputs/skill-web-desktop-harness.md` 构建一个带基于执行评估和轨迹效率指标的网页/桌面智能体框架。

## 练习

1. 用第二个应用（一个论坛）扩展玩具框架。写 3 个任务加上黄金轨迹。
2. 添加每任务的轨迹效率报告。在你的玩具上，智能体是黄金的 1x、2x 还是 3x？
3. 实现一个"干扰"工具——黄金轨迹从不使用的工具。脚本化智能体会被诱惑吗？
4. 阅读 OSWorld-G。你如何在自己的评估中将定位失败与规划失败分开？
5. 阅读 WebArena 的应用 README。升级某个固定应用版本时会有什么问题？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| WebArena | "网页智能体基准" | 跨 4 个自托管应用的 812 个任务；gym 风格评估 |
| VisualWebArena | "视觉 WebArena" | 视觉定位的 WebArena；截图是观察 |
| OSWorld | "桌面智能体基准" | 真实 Ubuntu/Windows/macOS 上的 369 个任务 |
| GUI 定位 | "像素到元素映射" | 模型在 1920×1080 中定位 UI 元素 |
| 操作知识 | "操作系统知识" | 哪个菜单、哪个快捷键、哪个偏好面板 |
| OSWorld-G | "定位套件" | 564 个纯定位样本 + 训练集 |
| OSWorld-Human | "黄金轨迹" | 手工专家动作序列，用于测量效率 |
| 轨迹效率 | "步骤数 vs 黄金" | 智能体步骤数除以人类最小步骤数 |

## 延伸阅读

- [Zhou 等人，WebArena（arXiv:2307.13854）](https://arxiv.org/abs/2307.13854) — 四应用网页基准
- [Xie 等人，OSWorld（arXiv:2404.07972）](https://arxiv.org/abs/2404.07972) — 跨操作系统桌面基准
- [Anthropic，介绍 computer use](https://www.anthropic.com/news/3-5-models-and-computer-use) — Claude 的基准形态能力
- [OpenAI，计算机使用智能体](https://openai.com/index/computer-using-agent/) — OSWorld 和 WebArena 数字
