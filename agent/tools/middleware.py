from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langgraph.prebuilt.tool_node import ToolCallRequest #一次工具调用请求的包装盒，里面有 tool_call（大模型说"调 get_weather，参数…"这个决定）
from langchain_core.messages import ToolMessage #ToolMessage：工具执行完后回传给模型的"结果消息"
from langgraph.types import Command #langgraph 里用于"发指令"（跳转/中断）的东西。这里只是类型注解，表示"函数可能返回消息，也可能返回指令"。
from typing import Callable #表示"一个可调用函数"的类型注解
from utils.logger_handler import logger
from langgraph.runtime import Runtime  #运行时上下文对象——它里面有个 context 字典，是各中间件之间传话的"公共黑板"。
from langchain.agents import AgentState #整个 Agent 的状态，本质是个字典，装着 messages（到目前为止所有对话/工具消息）
from utils.prompt_loader import load_report_prompts,load_system_prompts

@wrap_tool_call   # 每一次工具调用前后
def monitor_tool(
    # 请求的数据封装
    request:ToolCallRequest,
    # 执行的函数本身
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
)-> ToolMessage | Command:  # 每一次工具调用前后
    logger.info(f"[tool monitor]执行工具: {request.tool_call['name']}")
    logger.info(f"[tool monitor]传入参数: {request.tool_call['args']}")

    try:
        result = handler(request)  #调用底层真正的工具函数。此刻才真正执行 agent_tools.py 里那些函数体。
        logger.info(f"[tool monitor]工具{request.tool_call['name']}调用成功")

        if request.tool_call['name'] == "fill_context_for_report":
            request.runtime.context["report"] = True

        return result
    except Exception as e:
        logger.error(f"工具{request.tool_call['name']}调用失败, 原因: {str(e)}")
        raise e

@before_model    # 在模型执行前调用
def log_before_model(
    state: AgentState,          # 整个Agent智能体中的状态记录
    runtime: Runtime,           # 记录了整个执行过程中的上下文信息
):  # 在模型执行前输出日志
    logger.info(f"[log_before_model]即将调用模型，带有{len(state['messages'])}条消息。")

    logger.debug(f"[log_before_model]{type(state['messages'][-1]).__name__} | {state['messages'][-1].content.strip()}")
    return None

@dynamic_prompt     # 每一次在生成提示词之前，调用此函数
def report_prompt_switch(request:ModelRequest) -> str:
    is_report = request.runtime.context.get("report", False)   #"report"一般为False,如果为True则表示是报告生成场景
    if is_report:
        return load_report_prompts()  # 是报告生成场景，返回报告生成提示词内容
    return load_system_prompts()