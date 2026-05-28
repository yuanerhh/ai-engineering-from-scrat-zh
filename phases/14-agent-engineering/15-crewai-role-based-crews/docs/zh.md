# CrewAI：基于角色的 Crew 与 Flow

> CrewAI 是 2026 年基于角色的多智能体框架。四个原语：Agent（智能体）、Task（任务）、Crew（团队）、Process（流程）。两种顶层形态：Crew（自主、基于角色的协作）和 Flow（事件驱动、确定性的）。文档明确指出："对于任何生产就绪的应用，从 Flow 开始。"

**类型：** 学习 + 构建
**编程语言：** Python（标准库）
**前置知识：** Phase 14 · 12（工作流模式）、Phase 14 · 14（Actor 模型）
**预计时间：** 约 75 分钟

## 学习目标

- 列出 CrewAI 的四个原语（Agent、Task、Crew、Process）及各自负责的内容。
- 区分顺序（Sequential）、层次（Hierarchical）和共识（Consensual）流程；按工作负载选择一种。
- 区分 Crew（自主基于角色）与 Flow（事件驱动确定性），并解释文档的生产建议。
- 使用 `@tool` 装饰器和 `BaseTool` 子类连接工具；权衡结构化输出与自由文本。
- 列出四种 CrewAI 记忆类型以及各自何时有价值。
- 实现一个产出简报的标准库三智能体 crew（研究员、写作者、编辑）。
- 识别三种 CrewAI 失败模式：提示词膨胀、管理器 LLM 税、脆弱的交接。

## 问题背景

采用多智能体框架的团队撞上了同一堵墙。"自主协作"在演示中听起来很棒。然后客户提交了一个 bug，你需要确定性的重放。或者财务问每次运行 LLM 路由的 crew 花了多少钱。或者值班人员需要知道凌晨 3 点哪个智能体卡住了。

自由形式的 LLM 路由 crew 对这些问题都没有清晰的答案。纯 DAG 能回答所有问题，但失去了头脑风暴智能体所需的探索形态。

CrewAI 的拆分对这个权衡是诚实的。Crew 用于协作、基于角色、探索性的工作。Flow 用于事件驱动、代码拥有、可审计的生产环境。同一个框架，两种形态，按场景选择。

## 核心概念

### 四个原语

CrewAI 的接口很小。记住这些，其余的都是配置。

- **Agent（智能体）。** `role + goal + backstory + tools + (可选) llm`。backstory（背景故事）是关键所在。它塑造语气、判断力、智能体何时停止。工具是智能体可以调用的函数（详见下文）。
- **Task（任务）。** `description + expected_output + agent + (可选) context + (可选) output_pydantic`。可复用的工作单元。`expected_output` 是契约。`context` 列出上游任务，其输出会被传入。`output_pydantic` 强制指定结构化形态。
- **Crew（团队）。** 容器。拥有 `agents` 列表、`tasks` 列表、`process`，以及可选的 `memory`、`verbose` 和 `manager_llm` 设置。
- **Process（流程）。** 执行策略。顺序、层次、共识。选择运行的形态。

智能体不能直接相互看到。任务引用智能体。Crew 对任务排序。Process 决定谁选择下一个任务。这就是整个心智模型。

### 顺序 vs 层次 vs 共识

- **顺序（Sequential）。** 任务按声明顺序运行。任务 N 的输出作为 `context` 可用于任务 N+1。成本最低。最可预测。当顺序固定时使用。
- **层次（Hierarchical）。** 一个管理器 Agent（单独的 LLM 调用）在专业智能体之间路由。CrewAI 从你的 `manager_llm` 配置或默认值中生成管理器。管理器每轮选择下一个任务，可以拒绝或重新路由。当你有四个或更多专业智能体且顺序真正取决于先前输出时使用。
- **共识（Consensual）。** 测试版。智能体对下一步投票。在研究之外很少值得这些来回。

层次在每个专业智能体调用之上增加了每轮的 LLM 调用（管理器）。在五步运行中，token 成本可能翻三倍。只有在你需要路由时才付出这个代价。

### Crew vs Flow

这是 2026 年文档开头强调的框架。

- **Crew。** LLM 驱动的自主性。框架在运行时选择形态。适合：研究、头脑风暴、初稿、任何路径是答案一部分的场景。难以重放。难以测试。易于原型设计。
- **Flow。** 你拥有的事件驱动图。`@start` 标记入口。`@listen(topic)` 标记当另一个步骤发出该主题时触发的步骤。每个步骤是普通 Python（可以在内部调用 Crew）。适合：生产环境。可观察。可测试。确定性。

文档 2026 年的生产建议：从 Flow 开始。当自主性值得付出成本时，在 Flow 步骤内通过 `Crew.kickoff()` 调用折叠 Crew。Flow 提供审计轨迹，Crew 提供探索。组合使用，不要二选一。

### 工具集成

给 Agent 工具的三种方式。选择最简单适合的。

1. **`@tool` 装饰器。** 纯函数变成工具。签名是 schema；文档字符串是 LLM 看到的描述。最适合一次性辅助工具。

   ```python
   from crewai.tools import tool

   @tool("搜索网络")
   def search(query: str) -> str:
       """返回查询的最高结果。"""
       return run_search(query)
   ```

2. **`BaseTool` 子类。** 带显式参数 schema、异步支持、重试的基于类的工具。当工具有状态（客户端、缓存）或需要结构化参数时使用。

   ```python
   from crewai.tools import BaseTool
   from pydantic import BaseModel

   class SearchArgs(BaseModel):
       query: str
       limit: int = 10

   class SearchTool(BaseTool):
       name = "web_search"
       description = "搜索网络并返回最高结果。"
       args_schema = SearchArgs

       def _run(self, query: str, limit: int = 10) -> str:
           return self.client.search(query, limit=limit)
   ```

3. **内置工具包。** CrewAI 自带适配器：`SerperDevTool`、`FileReadTool`、`DirectoryReadTool`、`CodeInterpreterTool`、`RagTool`、`WebsiteSearchTool`。一次导入即可连接。

结构化输出使用 Pydantic。在 Task 上传递 `output_pydantic=MyModel`。CrewAI 对照模型验证 LLM 响应，强制转换或重试。配合严格的 `expected_output` 字符串使用。自由文本输出适合草稿；结构化输出是下游 Flow 可以消费的内容。

### 记忆钩子

CrewAI 自带四种记忆类型。它们可以组合：一个 Crew 可以同时启用全部四种。

- **短期（Short-term）。** 单次运行内的对话缓冲区。运行结束时清除。
- **长期（Long-term）。** 跨运行持久化。存储在向量数据库中（默认 Chroma，可替换）。通过与当前任务的相似度检索。
- **实体（Entity）。** 每实体事实。"客户 X 使用企业计划。"以实体为键，而非相似度。跨运行存活。
- **上下文（Contextual）。** 组装时检索。在 Agent 需要时拉取相关记忆，而非预加载。

在 Crew 上用 `memory=True` 或每类型配置启用。由你配置的嵌入提供商支持（默认 OpenAI，可替换为本地）。记忆是 CrewAI 相对于更薄框架的优势之一；纯 LangGraph 需要你自己连接这些。

### CrewAI 适合的场景

- 三到六个有命名角色和协作工作流的智能体。起草、审查、规划、头脑风暴。
- LLM 关于下一步的判断是价值的一部分的路由（层次化）。
- 团队更喜欢读 `role + goal + backstory` 而非图定义的任何地方。

### CrewAI 不适合的场景

- 带严格顺序的确定性 DAG。使用 LangGraph（第 13 课）。图形态是正确的抽象；CrewAI 的角色框架是摩擦。
- 亚秒延迟预算。层次化增加了来回次数。即使是顺序化也会序列化包含背景故事和先前输出的提示词。
- 单智能体循环。跳过框架；一个智能体循环（第 01 课）加上工具注册表代码更短。

第 17 课（智能体框架权衡）以矩阵形式展示这些。简版：CrewAI 坐落在"协作基于角色"的角落。

### 依赖形态

独立于 LangChain。Python 3.10 到 3.13。使用 `uv`。2026 年初 30k+ GitHub stars。AWS Bedrock 集成有文档；其基准引用了相比 LangGraph 在 QA 任务上 5.76 倍的加速。将框架厂商的数字视为方向性参考。

### 这个模式在哪里出错

- **背景故事导致的提示词膨胀。** 每个智能体 2000 字的背景故事加上五个智能体，在第一次工具调用之前就耗尽了上下文预算。背景故事控制在 200 字以内。在智能体之间复用短语；不要重复五次写作风格。
- **管理器 LLM token 税。** 层次化流程在每次专业智能体调用之前增加一次管理器 LLM 调用。五任务 crew 是六次 LLM 调用而非五次，管理器调用携带完整任务列表和先前输出。除非路由取决于输出，否则切换到顺序化。
- **脆弱的交接。** 任务 N 的 `expected_output` 是"一个大纲"。任务 N+1 将其读取为 `context` 并尝试解析三个部分。LLM 产出了四个。下游 Agent 即兴发挥。用任务 N 上的 `output_pydantic` 修复，使任务 N+1 读取类型化对象，而非自由文本。
- **Crew 作为生产。** 自由形式的 Crew 没有 Flow 包装就发布到生产。输出变化大；重放不可能；值班人员无法将坏运行与好运行对比。用 Flow 包装。

## 动手实践

`code/main.py` 实现了两种形态的标准库版本加上一个三智能体 crew。

形态：

- 与 CrewAI 接口匹配的 `Agent`、`Task` 数据类。
- `SequentialCrew.kickoff(inputs)` 按声明顺序运行任务，将输出作为 `context` 传递。
- `HierarchicalCrew.kickoff(topic)` 添加一个管理器 Agent，每轮选择下一个专业智能体，在"完成"时停止。
- 带 `@start` 和 `@listen(topic)` 装饰器的 `Flow`，一个小型事件循环，以及轨迹。
- 镜像 CrewAI `@tool` 形态的 `tool(name)` 装饰器。
- 带 `short_term`、`long_term`、`entity` 存储的 `Memory`；模拟相似度使用 numpy。
- 模拟 LLM 响应是以角色加输入前缀为键的硬编码字符串。无网络。确定性。

具体演示：研究员、写作者、编辑 crew 产出一份关于"2026 年智能体工程"的简报。研究员拉取（模拟的）来源。写作者起草。编辑精炼。同一 crew 通过 Flow 运行以展示确定性形态。

运行：

```bash
python3 code/main.py
```

轨迹涵盖：顺序 crew 通过 `context` 传递输出、带管理器选择的层次 crew（研究员、写作者、编辑，然后"完成"）、同三步通过显式主题（`researched`、`drafted`、`edited`）运行的 flow、通过 `@tool` 路由的工具调用，以及跨两次 kickoff 存活的长期记忆。

Crew 轨迹是流动的；管理器原则上可以重新排序。Flow 轨迹是固定的。这个选择就是本课的要点。

## 使用建议

- **CrewAI Flow** 用于生产。即使 Flow 只有一个调用 `Crew.kickoff()` 的步骤。Flow 提供审计边界。
- **CrewAI Crew（顺序）** 用于顺序明确的协作工作，尤其是初稿和审查循环。
- **CrewAI Crew（层次）** 当路由取决于输出且有四个或更多专业智能体时。
- **LangGraph**（第 13 课）用于显式状态机、持久化恢复、严格顺序。
- **AutoGen v0.4**（第 14 课）用于 Actor 模型并发和故障隔离。
- **OpenAI Agents SDK**（第 16 课）用于带 handoffs 和防护栏的 OpenAI 优先产品。
- **Claude Agent SDK**（第 17 课）用于带子智能体和会话存储的 Claude 优先产品。

## 产出技能

`outputs/skill-crew-or-flow.md` 为任务选择 Crew 或 Flow 并搭建最小实现。对无背景故事的 Crew、无显式主题的 Flow、少于三个专业智能体的层次化给出硬拒绝。

## 陷阱

- **背景故事当作风味。** 它塑造输出。每个智能体测试三个变体；差异是真实存在的。选一个，固定下来。
- **跳过 `expected_output`。** 没有每任务的契约，下游任务会拾取 LLM 产出的任何内容。Crew 运行了；审计失败了。
- **记忆总是开启。** 长期记忆每次运行都写。向量数据库增长。检索变得嘈杂。将写入范围限制在事实是持久的任务。
- **管理器提示词漂移。** 层次化的管理器提示词是隐式的。如果路由变得奇怪，在 verbose 模式下转储并阅读。
- **Crew 工具中的副作用。** Crew 可能调用工具的次数超出预期。POST、DELETE、支付操作属于 Flow 步骤，永远不要放在 Crew 工具中。

## 练习

1. 将顺序 crew 转换为 Flow。数一数变化性降低的接触点。注意可读性降低的地方。
2. 向 crew 添加实体记忆：关于客户的事实跨 kickoff 持久化。验证检索拉取了正确的实体。
3. 实现层次化流程，管理器拒绝路由到编辑，直到写作者的输出至少有三段。跟踪重试。
4. 为（模拟的）网络搜索连接一个 `BaseTool` 子类。比较与 `@tool` 装饰器版本的轨迹形态。
5. 向编辑任务添加 `output_pydantic=Brief`，其中 `Brief` 有 `title`、`summary`、`sections`。让写作者任务一次输出格式错误的 JSON；验证 CrewAI 在轨迹中的重试行为。
6. 阅读 CrewAI 的文档简介。将玩具移植到真实的 `crewai` API。标准库版本跳过了哪些保证？
7. 将 AgentOps 或 Langfuse（第 24 课）连接到真实运行。标准库版本遗漏了哪些追踪？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Agent（智能体） | "人物角色" | 角色 + 目标 + 背景故事 + 工具 |
| Task（任务） | "工作单元" | 描述 + 预期输出 + 负责人 + 可选结构化输出 |
| Crew（团队） | "智能体团队" | 包含智能体 + 任务 + 流程的容器 |
| Process（流程） | "执行策略" | 顺序 / 层次 / 共识 |
| Flow（流）| "确定性工作流" | 事件驱动、代码拥有、可测试的 |
| Backstory（背景故事） | "人物提示词" | 智能体的语气和判断力塑造器 |
| `@tool` | "函数工具" | 将函数变成智能体可调用工具的装饰器 |
| `BaseTool` | "类工具" | 带参数 schema、重试、异步支持的基于类的工具 |
| 实体记忆 | "每实体事实" | 限定在客户/账户/问题的记忆 |
| 长期记忆 | "跨运行记忆" | 在 kickoff 之间存活的向量支持记忆 |
| 上下文记忆 | "即时检索" | 在 Agent 需要时拉取的记忆 |
| 管理器 LLM | "路由器智能体" | 层次化流程中选择下一个任务的额外 LLM |
| `expected_output` | "任务契约" | 告诉 Agent（和审计）返回什么形态的字符串 |

## 延伸阅读

- [CrewAI 文档简介](https://docs.crewai.com/en/introduction)——概念和推荐的生产路径
- [CrewAI Flow 指南](https://docs.crewai.com/en/concepts/flows)——事件驱动形态，`@start`，`@listen`
- [CrewAI 工具参考](https://docs.crewai.com/en/concepts/tools)——`@tool`，`BaseTool`，内置工具包
- [CrewAI 记忆](https://docs.crewai.com/en/concepts/memory)——短期、长期、实体、上下文
- [Anthropic，构建有效智能体](https://www.anthropic.com/research/building-effective-agents)——多智能体何时有帮助，何时没有
- [LangGraph 概述](https://docs.langchain.com/oss/python/langgraph/overview)——状态机替代方案
