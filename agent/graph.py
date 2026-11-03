from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from agent.environment_agent import environment_node
from agent.knowledge_agent import knowledge_node
from agent.report_agent import report_node
from agent.state import AgentContext, MultiAgentState
from agent.summary import summary_node
from agent.supervisor import supervisor_node


def route_to_agents(state: MultiAgentState) -> list[str]:
    """读取协调结果，返回下一步要执行的节点名称。"""

    # 返回多个名称时，LangGraph 会安排这些节点并行执行。
    # 这里只负责路线选择，不调用模型，也不修改状态。
    selected_agents = state["selected_agents"]

    if selected_agents:
        return list(selected_agents)

    # 问候、能力介绍等任务，直接进入汇总节点。
    return ["summary"]


@lru_cache(maxsize=1)
# lru_cache: 缓存函数调用结果，避免重复计算。
# maxsize=1 表示只缓存最近一次调用结果。
def get_multi_agent_graph():
    """构建并缓存多 Agent 总图。"""

    # MultiAgentState 定义各节点之间传递的数据。
    # AgentContext 定义服务端传入的上下文。
    # 声明后，报告节点才能通过 Runtime 获取当前用户身份。
    builder = StateGraph(
        MultiAgentState,
        context_schema=AgentContext,
    )

    # 注册节点：“节点名称”与“执行函数”建立对应关系。
    # 专业节点的名称与 AgentName 中约定的名称保持一致。
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("knowledge", knowledge_node)
    builder.add_node("environment", environment_node)
    builder.add_node("report", report_node)
    builder.add_node("summary", summary_node)

    # START 是图的入口标记，第一步执行协调节点。
    builder.add_edge(START, "supervisor")

    # 条件边：协调完成后，根据状态选择下一步。
    # 映射表明确列出可能到达的节点，也方便生成图示。
    builder.add_conditional_edges(
        "supervisor",
        route_to_agents,
        {
            "knowledge": "knowledge",
            "environment": "environment",
            "report": "report",
            "summary": "summary",
        },
    )

    # 每个专业节点完成后都指向汇总节点。
    #
    # 当前结构中，选中的专业节点位于同一个执行步：
    # LangGraph 等这一批节点完成并合并更新后，
    # 再进入下一步的 summary，汇总节点只执行一次。
    #
    # 子 Agent 内部即使调用多次工具，
    # 在外层图看来仍是一次专业节点执行。
    builder.add_edge("knowledge", "summary")
    builder.add_edge("environment", "summary")
    builder.add_edge("report", "summary")

    # 汇总完成后结束本轮运行。
    builder.add_edge("summary", END)

    # compile 将图定义转换成可 invoke / stream 的执行对象。
    # 当前不配置 checkpointer，历史仍由调用方传入。
    return builder.compile()
