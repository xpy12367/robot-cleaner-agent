from langchain.agents.middleware import before_model, dynamic_prompt, ModelRequest
from langgraph.runtime import Runtime  #运行时上下文对象——它里面有个 context 字典，是各中间件之间传话的"公共黑板"。
from langchain.agents import AgentState #整个 Agent 的状态，本质是个字典，装着 messages（到目前为止所有对话/工具消息）
from utils.prompt_loader import load_report_prompts,load_system_prompts
from utils.logger_handler import logger

# 导入增强版监控中间件（替代原有的简单monitor_tool）
from agent.tools.monitor_enhanced import monitor_tool_enhanced as monitor_tool

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
