from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agent.state import AgentName, MultiAgentState
from model.factory import chat_model
from utils.path_tool import get_abs_path


class SupervisorDecision(BaseModel):
    """约束模型返回的任务分配结果，并在解析时校验字段。"""

    # AgentName 来自 state.py，避免在多个文件里重复定义名称。
    # 空列表表示无需专业 Agent，后续直接交给汇总节点。
    # max_length=3 限制列表长度；重复名称会在节点函数里去掉。
    selected_agents: list[AgentName] = Field(
        description="本轮需要的专业 Agent；无需专业处理时返回空列表",
        max_length=3,
    ) #Field: 用于描述字段的元数据，如描述、最小值、最大值等。

    # min_length=1 不允许空字符串，但是否补全正确仍需实际测试。
    task_query: str = Field(
        description="结合历史补全指代后的完整任务，保留用户条件，不编造信息",
        min_length=1,
    )


# 使用绝对路径，避免从不同目录启动时找不到提示词文件。
SUPERVISOR_PROMPT = Path(get_abs_path("prompts/supervisor_prompt.txt")).read_text(
    encoding="utf-8"
) #path: 用于表示文件路径的类，支持跨平台操作。read_text: 读取文件内容为字符串，指定编码为 utf-8。


# 创建一个带结构化输出规则的模型调用对象。
# function_calling 通过工具调用格式传递结果，不会执行天气或 RAG 工具。
# 传入 Pydantic 类后，invoke 成功返回的是 SupervisorDecision 对象。
# 当前使用兼容接口，是否支持该方式仍需后续真实调用确认。
supervisor_model = chat_model.with_structured_output(
    SupervisorDecision,
    method="function_calling",
) # with_structured_output: 用于将模型输出转换为结构化数据。method: 指定转换方式，这里使用 function_calling。


def supervisor_node(state: MultiAgentState) -> dict:
    """读取图状态，返回本节点需要更新的字段。"""

    # 约定：messages 包含历史对话和本轮用户消息。
    # list(...) 复制列表，避免向模型准备输入时修改原来的状态。
    conversation = list(state["messages"])

    # 方便单独测试：如果没有传入消息，就用 user_query 构造一条。
    # 已有消息时不再追加，防止本轮问题被重复发送。
    if not conversation:
        conversation = [HumanMessage(content=state["user_query"])]

    decision = supervisor_model.invoke(
        [
            SystemMessage(content=SUPERVISOR_PROMPT),
            *conversation,  # * 将消息列表展开，按原顺序放入新列表。
        ]
    )

    # 去重并保持模型给出的顺序。
    # 比如 ["knowledge", "knowledge"] 变为 ["knowledge"]。
    selected_agents = list(dict.fromkeys(decision.selected_agents))
    # dict.fromkeys() 去重，保持顺序。list() 转换为列表。

    # 只返回需要更新的字段，LangGraph 会将它们合并到当前状态。
    # 分配结果不写进 messages，避免内部调度信息混入用户聊天记录。
    return {
        "selected_agents": selected_agents,
        "task_query": decision.task_query,
        # 每轮重新分配任务时清空结果，防止后续误用上一轮的数据。
        "knowledge_result": "",
        "environment_result": "",
        "report_result": "",
        "final_answer": "",
    }


