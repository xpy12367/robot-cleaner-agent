import json
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from agent.state import MultiAgentState
from model.factory import chat_model
from utils.path_tool import get_abs_path

SUMMARY_PROMPT = Path(get_abs_path("prompts/summary_prompt.txt")).read_text(
    encoding="utf-8"
)


# 将协调节点使用的 Agent 名称映射到对应结果字段。
# 后面只读取本轮选中的 Agent，避免误用其他结果字段。
RESULT_FIELDS = {
    "knowledge": "knowledge_result",
    "environment": "environment_result",
    "report": "report_result",
}


def summary_node(
    state: MultiAgentState,
    config: RunnableConfig,
) -> dict:
    """汇总本轮专业结果，并生成最终助手消息。"""

    selected_agents = state["selected_agents"]

    # 根据本轮分配读取结果。
    # 未选中的 Agent 不会被放进模型输入。
    agent_results = {name: state[RESULT_FIELDS[name]] for name in selected_agents}

    payload = {
        "user_query": state["user_query"],
        "task_query": state["task_query"],
        "selected_agents": selected_agents,
        "agent_results": agent_results,
    }

    # 使用 JSON 清楚区分各结果字段。
    # ensure_ascii=False 保留中文，便于阅读和排查。
    # JSON 只是数据组织方式，不能替代提示词中的依据约束。
    result_message = HumanMessage(
        content=(
            "请结合前面的对话及以下本轮任务数据，生成最终回答：\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
        )  # ensure_ascii=False: 保留非 ASCII 字符，如中文。indent=2: 美化输出，缩进 2 个空格。 
    )

    response = chat_model.invoke(
        [
            SystemMessage(content=SUMMARY_PROMPT),
            *state["messages"],
            result_message,
        ],
        config=config,
    )

    # 汇总节点不绑定业务工具，只负责生成回答。
    if not isinstance(response, AIMessage) or response.tool_calls:
        raise RuntimeError("汇总节点未正常完成回答")

    answer = response.text.strip()
    if not answer:
        raise RuntimeError("汇总节点返回了空回答")

        # final_answer 供前端或调用方直接读取。
        # messages 只追加最终回答，不追加临时构造的结果消息。
        #
        # state.py 中的 add_messages 会负责合并这条新消息，
        # 所以这里不能把全部历史再返回一遍。
    return {
        "final_answer": answer,
        # 保留模型回复的原始消息 ID，让流式处理识别为同一条消息，
        # 避免先输出文本片段，又重复输出完整回答。
        #
        # model_copy 创建副本，并将正文统一为整理后的 answer；
        # 没有指定更新的字段（包括 id）会保留。
        "messages": [response.model_copy(update={"content": answer})],
    }
