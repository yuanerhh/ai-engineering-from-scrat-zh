---
name: injection-defense
description: 为任意 Agent 运行时构建 PVE（提示词-验证器-执行器）层，包含来源标签内容、注入标记扫描和导航白名单。
version: 1.0.0
phase: 14
lesson: 27
tags: [security, prompt-injection, pve, greshake, source-tag]
---

给定一个具有工具访问和检索能力的 Agent，生成注入防御层。

输出内容：

1. 对每段内容打来源标签：`user_message`、`tool_output`、`retrieved_web`、`retrieved_memory`、`retrieved_file`。在消息历史中传播标签。
2. `Validator.assess(tool_call, contents)` —— 拒绝带有注入形状参数或已检索内容的工具调用；仅当来源标签与声明的信任级别匹配时才允许。
3. 导航白名单/黑名单：Agent 可访问的 URL、域名、文件路径。
4. 记忆写入防护：拒绝看起来像指令的写入操作。
5. 内容捕获纪律（第 23 课）：将已检索内容存储在外部；span 携带引用 ID，而非纯文本内容。
6. 测试套件：五种 Greshake 漏洞利用类型作为红队测试用例。

硬性拒绝：

- 工具调用面没有来源标签。没有来源信息就无法区分权限级别。
- 验证器只在最终输出上运行。事后验证是无效的——模型已经行动了。
- "信任我，系统提示词会处理"。系统提示词卫生不是一种控制手段。

拒绝规则：

- 如果 Agent 有任何检索能力但没有来源标签，拒绝交付。已检索内容是典型的注入向量。
- 如果敏感工具（发送消息、执行 shell、向 / 写文件）没有人工确认环节，拒绝。
- 如果记忆写入未受保护，拒绝。持久化记忆污染会在下次会话中再次发作。

输出：`validator.py`、`source_tag.py`、`allowlist.py`、`memory_guard.py`、`red_team.py`、`README.md`，说明六层控制栈、残余风险和持续审查节奏。最后附"下一步阅读"，指向第 21 课（计算机使用安全）和第 23 课（通过 OTel 进行内容捕获）。
