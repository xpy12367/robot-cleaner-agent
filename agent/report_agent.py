from functools import lru_cache
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime

from agent.state import AgentContext, MultiAgentState
from model.factory import chat_model
from utils.path_tool import get_abs_path


@lru_cache(maxsize=1)
# lru_cache: 缓存函数调用结果，避免重复计算。
# maxsize=1 表示只缓存最近一次调用结果。
def get_report_agent():
    """创建并缓存报告 Agent 的执行图。"""

    from agent.tools.agent_tools import fetch_usage_report
    from agent.tools.middleware import monitor_tool, log_before_model

    prompt = Path(get_abs_path("prompts/report_agent_prompt.txt")).read_text(
        encoding="utf-8"
    )

    return create_agent(
        model=chat_model,
        tools=[fetch_usage_report],
        system_prompt=prompt,
        # 声明这个子 Agent 接收的运行上下文结构。
        context_schema=AgentContext,
        middleware=[monitor_tool, log_before_model],
        name="report_agent",
    )


def report_node(
    state: MultiAgentState,
    config: RunnableConfig,
    runtime: Runtime[AgentContext],
    #Runtime: 用于在节点执行时访问上下文、日志、追踪等运行信息。
) -> dict:
    """读取服务端上下文，执行报告任务，返回 report_result。"""

    # runtime 与 config 一样，由 LangGraph 调用节点时注入。
    # 单独手动检查节点时，需要自行构造 Runtime。
    context = runtime.context or {}
    user_id = context.get("user_id")

    # 未选择用户时，无需让模型尝试查询，直接返回业务提示。
    if not user_id:
        return {"report_result": "当前未选择用户，无法查询使用记录，请先选择演示用户。"}

    task_query = state["task_query"].strip()
    if not task_query:
        raise ValueError("报告 Agent 收到的 task_query 为空")

    result = get_report_agent().invoke(
        {"messages": [HumanMessage(content=task_query)]},
        config=config,
        # 显式把用户身份传给子 Agent。
        # 它的工具通过 ToolRuntime.context 读取身份，
        # 而不是让模型在工具参数里填写用户ID。
        #
        # 创建新字典，也避免工具修改上下文时影响外层上下文。
        context={"user_id": user_id},
    )

    final_message = result["messages"][-1]

    if not isinstance(final_message, AIMessage) or final_message.tool_calls:
        raise RuntimeError("报告 Agent 未正常完成回答")

    answer = final_message.text.strip()
    if not answer:
        raise RuntimeError("报告 Agent 返回了空回答")

    return {"report_result": answer}
