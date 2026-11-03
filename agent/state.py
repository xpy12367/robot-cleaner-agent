from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


# 限定专业 Agent 的名称，后面路由时统一使用这些名字。
AgentName = Literal["knowledge", "environment", "report"]
# Literal 类型，只能取指定的值。

class MultiAgentState(TypedDict):
    """一次多 Agent 协作过程中，各节点共享的数据。"""

    # 用户和助手的对话消息。
    messages: Annotated[list[AnyMessage], add_messages]
    # Annotated 类型，用于添加元数据。这里用于添加 add_messages 元数据，
    # add_messages: 用于在节点执行时，将当前节点的输出添加到 messages 中。

    # 本轮用户原始问题。
    user_query: str

    # 协调 Agent 为专业 Agent 整理的、可以独立理解的任务。
    task_query: str

    # 本轮需要执行哪些专业 Agent。
    selected_agents: list[AgentName]

    # 各专业 Agent 的处理结果。
    knowledge_result: str
    environment_result: str
    report_result: str

    # 汇总节点生成的最终回答。
    final_answer: str


class AgentContext(TypedDict):
    """由服务端传入的运行上下文，与图的任务状态分开。"""

    # 当前项目使用页面选择的演示用户；None 表示未选择用户。
    # 模型不能通过生成工具参数来设置这个身份。
    user_id: str | None
