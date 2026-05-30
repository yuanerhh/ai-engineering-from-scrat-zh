---
name: stategraph-designer
description: 将智能体任务转化为带有命名节点、类型化状态、归约器、检查点和人工中断的 LangGraph StateGraph。
version: 1.0.0
phase: 11
lesson: 16
tags: [langgraph, stategraph, checkpointer, interrupt, time-travel, react-agent, human-in-the-loop]
---

给定智能体任务（面向用户的目标、可用工具、预期轮次数、带安全影响半径的副作用、持久性要求、目标延迟预算），输出：

1. 节点列表。命名每个离散步骤：LLM 思考器、每个工具运行器、每个人工审查步骤、任何摘要器或批评者、任何检索器。如果任何节点涉及多个关注点，拒绝该设计；拆分它。
2. 状态 Schema。TypedDict（或 Pydantic）字段，每个列表都有归约器。消息日志始终使用 Annotated[list, add_messages]。将任何任务特定列表（计划、预算计数器、检索文档列表）从消息中提取出来，以便归约器在并行更新下保持正确。
3. 边的映射。下一步是确定性的地方使用静态边。只有在模型选择下一步的地方才使用带有命名路由函数的条件边。拒绝任何路由函数依赖于你在先前节点中尚未进行的全新 LLM 调用的图。
4. 中断放置。在每个具有不可逆副作用的节点之前使用 interrupt_before（写入、删除、支付、有成本的外部 API 调用）。当输出验证在单独进程中运行时，在模型节点之后使用 interrupt_after。拒绝在任何有副作用的节点上使用 interrupt_after；那时副作用已经发生了。
5. 检查点器。仅在测试中使用 MemorySaver。对任何必须在重启后存活的环境，从 PostgresSaver、SQLiteSaver、RedisSaver 中选择。确认 thread_id 策略（按用户、按会话、按对话）和检查点 TTL。

拒绝发布没有检查点器的 LangGraph。没有检查点器意味着没有恢复、没有时间旅行、没有人在循环中的重放。拒绝发布没有 add_messages 的消息字段；第二次写入会静默覆盖第一次，导致半段对话消失。拒绝每个转换都是由规划器 LLM 路由的条件边的图；这是带有额外步骤的 AutoGen，每轮都会消耗 token。

示例输入：「基于 Anthropic Claude 的退款处理智能体，有三个工具（lookup_order、issue_refund、send_email），超过 100 美元的退款前必须暂停等待人工确认，服务器重启后必须恢复，P95 延迟预算 8 秒。」

示例输出：
- 节点：agent（LLM 调用）、lookup_tool、refund_tool、email_tool、human_review。
- 状态：带 add_messages 的 messages，order_context（覆写），refund_amount（覆写），reviewer_decision（覆写）。
- 边：agent 到 should_continue 路由器，分支为 lookup_tool、refund_tool、email_tool、human_review、END。工具节点返回到 agent。
- 中断：当 refund_amount > 100 时，在 refund_tool 之前使用 interrupt_before。lookup_tool 或 email_tool 上没有中断。
- 检查点器：PostgresSaver，thread_id 为「user:{user_id}:case:{case_id}」，TTL 为 30 天。
