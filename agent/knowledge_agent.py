from functools import lru_cache
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig

from agent.state import MultiAgentState
from model.factory import chat_model
from utils.path_tool import get_abs_path
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

@lru_cache(maxsize=1)
# lru_cache: 缓存函数调用结果，避免重复计算。
# maxsize=1 表示只缓存最近一次调用结果。
def get_knowledge_agent():
    """首次使用时创建知识库 Agent，之后复用这个实例。"""

    # 放在函数内部导入，是因为当前 agent_tools.py 在导入时
    # 就会创建 RagSummarizeService，并检查索引、初始化检索组件。
    # 延迟导入可以避免仅导入本模块时就加载这些资源。
    from agent.tools.agent_tools import rag_summarize
    from agent.tools.middleware import monitor_tool, log_before_model

    prompt = Path(get_abs_path("prompts/knowledge_prompt.txt")).read_text(
        encoding="utf-8"
    )

    # 继续使用熟悉的 create_agent，但只开放知识库工具。
    # 模型可以选择检索词、调用工具，并根据结果继续处理任务。
    #
    # 不挂载 report_prompt_switch：
    # 该中间件会动态替换系统提示词，这里应始终使用知识专家提示词。
    return create_agent(
        model=chat_model,
        tools=[rag_summarize],
        system_prompt=prompt,
        middleware=[monitor_tool, log_before_model],
        name="knowledge_agent",
    )


def knowledge_node(
    state: MultiAgentState,
    config: RunnableConfig,  
    # RunnableConfig 用于传递 callbacks、tags、metadata、recursion_limit 等运行配置。
    # 它本身不自动提供通用的超时或重试机制。
) -> dict:
    """总图中的知识节点：调用专业 Agent，返回知识结果。"""

    # 只接收协调员整理的任务，避免把其他 Agent 的内部消息混进来。
    # 指代补全由 supervisor_node 负责，因此这里不重复传入历史对话。
    task_query = state["task_query"].strip()
    if not task_query:
        raise ValueError("知识库 Agent 收到的 task_query 为空")

    # 每次传入新的消息列表。
    # get_knowledge_agent 缓存的是执行图；这里没有配置 checkpointer，
    # 因此不会自动把上一次调用的对话保留到本次调用。
    #
    # config 由 LangGraph 调用节点时注入。
    # 继续传给子 Agent，让回调、追踪等运行配置可以向下传递。
    result = get_knowledge_agent().invoke(
        {"messages": [HumanMessage(content=task_query)]},
        config=config,
    )

    # create_agent 返回的是包含 messages 的状态字典，
    # 不是直接返回字符串。最后一条通常是 Agent 的最终回复。
    final_message = result["messages"][-1]

    # 工具调用消息也是 AIMessage，所以还要确认没有待执行的工具调用。
    if not isinstance(final_message, AIMessage) or final_message.tool_calls:
        raise RuntimeError("知识库 Agent 未正常完成回答")

    # 交给汇总节点的依据，直接取自 RAG 工具结果。
    # 不使用 Agent 最后的自然语言改写，避免把该阶段补充的常识
    # 当成检索依据继续传下去。
    evidence_parts = []

    for message in result["messages"]:
        # Agent 内部还包含用户消息、模型消息等，
        # 这里只提取指定 RAG 工具的返回消息。
        if (
            isinstance(message, ToolMessage)
            and message.name == "rag_summarize"
            and message.status == "success"
        ):
            text = message.text.strip()
            if text:
                evidence_parts.append(text)

    # 如果模型没有真正调用 RAG，不能把它自行生成的回答
    # 当作知识库查询成功。这里明确报错，方便定位。
    if not evidence_parts:
        raise RuntimeError("知识库 Agent 没有返回有效的 RAG 工具结果")

    # 同一任务可能检索多个主题，按调用结果顺序保留资料。
    # 这里的编号用于区分工具返回，不表示原始文档的引用编号。
    evidence = "\n\n".join(
        f"【检索结果{index}】\n{text}"
        for index, text in enumerate(evidence_parts, start=1)
    )  # enumerate: 遍历列表并返回索引和值，start=1 表示索引从 1 开始。

    return {"knowledge_result": evidence} 
