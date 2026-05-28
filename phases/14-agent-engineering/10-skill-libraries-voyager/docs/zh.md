# 技能库与终身学习（Voyager）

> Voyager（Wang 等人，TMLR 2024）将可执行代码视为技能。技能是可命名、可检索、可组合的，并由环境反馈精炼。这是 Claude Agent SDK 技能、skillkit 和 2026 年技能库模式的参考架构。

**类型：** 构建
**编程语言：** Python（标准库）
**前置知识：** Phase 14 · 07（MemGPT）、Phase 14 · 08（Letta 块）
**预计时间：** 约 75 分钟

## 学习目标

- 说出 Voyager 的三个组件——自动课程、技能库、迭代提示词机制——及各自的作用。
- 解释为什么 Voyager 将代码而非原始命令作为行动空间。
- 用标准库实现一个带注册、检索、组合和失败驱动精炼的技能库。
- 将 Voyager 的模式映射到 2026 年的 Claude Agent SDK 技能和 skillkit 生态系统。

## 问题背景

每次会话都从头重建所有能力的智能体会犯三个错误：

1. **浪费 token。** 每个任务都重新引发相同的推理。
2. **失去进度。** 在会话 A 中学到的纠正不会转移到会话 B。
3. **在长视野组合上失败。** 复杂任务需要能力层级；单次提示词无法表达。

Voyager 的答案：将每个可复用能力视为存储在库中的命名代码块，可通过相似度检索，可与其他技能组合，并由执行反馈精炼。

## 核心概念

### 三个组件

Voyager（arXiv:2305.16291）围绕以下三者构建智能体：

1. **自动课程（Automatic curriculum）。** 一个好奇心驱动的提议者根据智能体当前的技能集和环境状态选择下一个任务。探索是自下而上的。
2. **技能库（Skill library）。** 每个技能是可执行代码。当任务成功时添加新技能。技能通过查询与描述的相似度检索。
3. **迭代提示词机制（Iterative prompting mechanism）。** 失败时，智能体接收执行错误、环境反馈和自我验证输出，然后精炼技能。

Minecraft 评估（Wang 等人，2024）：独特物品数量提升 3.3 倍，石制工具速度提升 8.5 倍，铁制工具速度提升 6.4 倍，地图穿越距离提升 2.3 倍，均优于基准。数字是 Minecraft 特定的，但模式可以迁移。

### 行动空间 = 代码

大多数智能体输出原始命令。Voyager 输出 JavaScript 函数。一个技能是：

```javascript
async function craftIronPickaxe(bot) {
  await mineIron(bot, 3);
  await mineStick(bot, 2);
  await placeCraftingTable(bot);
  await craft(bot, 'iron_pickaxe');
}
```

由子技能组合而成。以描述和嵌入为键存储。作为程序检索，而非提示词。

这就是 2026 年的 Claude Agent SDK 技能：一个命名的、可检索的代码块加上智能体按需加载的指令。

### 技能检索

新任务"制作钻石镐"。智能体：

1. 嵌入任务描述。
2. 在技能库中查询 top-k 相似技能。
3. 检索 `craftIronPickaxe`、`mineDiamond`、`placeCraftingTable` 等。
4. 从检索到的原语 + 新逻辑中组合新技能。

这就是 MCP 资源（Phase 13）和 Agent SDK 技能实现的模式：在知识/代码表面上进行检索，范围限定在当前任务。

### 迭代精炼

Voyager 的反馈循环：

1. 智能体写一个技能。
2. 技能在环境中运行。
3. 三种信号之一返回：`success`、`error`（带堆栈跟踪）、`self-verification failure`（自我验证失败）。
4. 智能体使用该信号作为上下文重写技能。
5. 循环直到成功或达到最大轮次。

这是 Self-Refine（第 05 课）应用于代码生成，带有环境驱动的验证。CRITIC（第 05 课）是相同的模式，以外部工具作为验证器。

### 课程与探索

Voyager 的课程模块根据智能体拥有的内容和尚未完成的内容提出"在湖边建造庇护所"之类的任务。提议者使用环境状态 + 技能清单来选择刚好超出当前能力的任务——探索的甜蜜点。

对于生产智能体，这转化为"缺少什么"运算符：给定当前技能库和一个领域，我们还没有覆盖哪些技能？团队通常手动将其实现为课程审查。

### 这个模式在哪里出错

- **技能库腐化。** 相同技能以略微不同的描述被添加了 10 次。写入时添加去重；检索只返回一个。
- **组合技能漂移。** 父技能依赖于已被精炼的子技能。对技能进行版本控制；固定到 v1 的父技能不会自动获取 v3。
- **检索质量。** 对技能描述的向量检索在库超过几百个后退化。用标签过滤器和硬约束（"仅具有 `category=tooling` 的技能"）作为补充。

## 动手实践

`code/main.py` 实现了一个标准库技能库：

- `Skill`——名称、描述、代码（字符串形式）、版本、标签、依赖项。
- `SkillLibrary`——注册、搜索（token 重叠）、组合（依赖项的拓扑排序）和精炼（更新时版本提升）。
- 一个脚本化智能体，注册三个原始技能，组合第四个，遇到失败，然后精炼。

运行：

```
python3 code/main.py
```

轨迹显示库写入、检索、组合、失败的执行和 v2 精炼——端到端的 Voyager 循环。

## 使用建议

- **Claude Agent SDK 技能**（Anthropic）——2026 年的参考：每个技能有描述、代码和指令；在智能体会话中按需加载。
- **skillkit**（npm: skillkit）——用于 32+ AI 编码智能体的跨智能体技能管理。
- **自定义技能库**——特定领域的（数据智能体的 SQL 技能，基础设施智能体的 Terraform 技能）。Voyager 模式可以规模缩减。
- **OpenAI Agents SDK `tools`**——在低端；每个工具是一个轻量级技能。

## 产出技能

`outputs/skill-skill-library.md` 为任何目标运行时生成 Voyager 形态的技能库，内置注册、检索、版本控制和精炼。

## 练习

1. 向 `compose()` 添加依赖循环检测器。当技能 A 依赖 B 而 B 依赖 A 时会发生什么？报错还是警告？
2. 实现每技能版本固定。当父技能组合子 `crafting@1` 时，对 `crafting@2` 的精炼不得悄悄升级父技能。
3. 用 sentence-transformers 嵌入（或标准库 BM25 实现）替换 token 重叠检索。在 50 技能玩具库上测量 retrieval@5。
4. 添加"课程"智能体：给定当前库和领域描述，提出 5 个缺失的技能。每周调用一次。
5. 阅读 Anthropic 的 Claude Agent SDK 技能文档。将玩具库移植到 SDK 的技能 schema。可发现性有什么变化？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 技能（Skill） | "可复用能力" | 命名的代码块 + 描述，可通过相似度检索 |
| 技能库（Skill library） | "智能体的方法记忆" | 技能的持久化存储，可搜索和可组合 |
| 课程（Curriculum） | "任务提议者" | 由当前能力差距驱动的自下而上目标生成器 |
| 组合（Composition） | "技能 DAG" | 技能调用技能；执行时进行拓扑排序 |
| 迭代精炼 | "自我纠错循环" | 环境反馈 + 错误 + 自我验证折回下一版本 |
| 代码作为行动空间 | "程序化行动" | 输出函数而非原始命令，用于时间扩展行为 |
| 写入时去重 | "技能折叠" | 近似重复的描述折叠为一个标准技能 |

## 延伸阅读

- [Wang 等人，Voyager（arXiv:2305.16291）](https://arxiv.org/abs/2305.16291) — 原始技能库论文
- [Claude Agent SDK 概述](https://platform.claude.com/docs/en/agent-sdk/overview) — 技能作为 2026 年的产品化
- [Anthropic，使用 Claude Agent SDK 构建智能体](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) — 实践中的技能和子智能体
- [Madaan 等人，Self-Refine（arXiv:2303.17651）](https://arxiv.org/abs/2303.17651) — Voyager 底层的精炼循环
