from functools import lru_cache
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from agent.state import MultiAgentState
from model.factory import chat_model
from utils.path_tool import get_abs_path


@lru_cache(maxsize=1)
# lru_cache: 缓存函数调用结果，避免重复计算。
# maxsize=1 表示只缓存最近一次调用结果。
def get_environment_agent():
    """首次使用时创建环境 Agent，之后复用执行图。"""

    # 延迟导入工具。
    # 注意：当前 agent_tools 模块还会初始化 RAG 服务，
    # 所以第一次使用环境 Agent 时也可能看到知识库加载日志。
    # 这是现有工具模块的初始化方式造成的，不代表环境 Agent 调用了 RAG。
    from agent.tools.agent_tools import get_weather, get_user_location
    from agent.tools.middleware import monitor_tool, log_before_model

    prompt = Path(get_abs_path("prompts/environment_prompt.txt")).read_text(
        encoding="utf-8"
    )

    # 只开放定位和天气工具，明确这个 Agent 的能力范围。
    # 不挂载 report_prompt_switch，避免覆盖环境助手的系统提示词。
    return create_agent(
        model=chat_model,
        tools=[get_user_location, get_weather],
        system_prompt=prompt,
        middleware=[monitor_tool, log_before_model],
        name="environment_agent",
    )


def environment_node(
    state: MultiAgentState,
    config: RunnableConfig,
    # RunnableConfig 用于传递 callbacks、tags、metadata、recursion_limit 等运行配置。
    # 它本身不自动提供通用的超时或重试机制。
) -> dict:
    """执行环境任务，只更新 environment_result。"""

    # 与知识节点保持相同接口，后面方便连接到同一个 StateGraph。
    task_query = state["task_query"].strip()
    if not task_query:
        raise ValueError("环境 Agent 收到的 task_query 为空")

    # 只传入本轮完整任务。
    # config 向下传递回调、追踪等配置，便于观察工具执行。
    result = get_environment_agent().invoke(
        {"messages": [HumanMessage(content=task_query)]},
        config=config,
    )

    final_message = result["messages"][-1]

    # 确认拿到的是完成工具调用后的助手回复。
    if not isinstance(final_message, AIMessage) or final_message.tool_calls:
        raise RuntimeError("环境 Agent 未正常完成回答")

    answer = final_message.text.strip()
    if not answer:
        raise RuntimeError("环境 Agent 返回了空回答")

    # 查询失败也可以是一条有效结果：
    # 例如“定位失败，需要提供城市”应交给汇总节点继续处理，
    # 而不是伪造数据，或把正常的业务失败当作程序异常。
    return {"environment_result": answer}
