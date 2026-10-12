from langchain.agents import create_agent
from utils.prompt_loader import load_system_prompts
from model.factory import chat_model
from agent.tools.agent_tools import (rag_summarize, get_weather, get_user_location, get_user_id,
    get_current_month, fetch_external_data, fill_context_for_report)
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch

class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[rag_summarize, get_weather, get_user_location, get_user_id,
                    get_current_month, fetch_external_data, fill_context_for_report],
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
        )

    def execute_stream(self,query:str):
        input_dict={
            "messages": [
                {"role": "user", "content": query},
            ]
        }  #把用户问题包成 Agent 认识的"对话消息"格式

        # 第三个参数context就是上下文runtime中的信息，就是我们做提示词切换的标记
        #.stream(...) —— 流式执行方法   input_dict —— 输入数据  stream_mode="values" —— 流式输出的模式   context={"report": False} —— 传给运行时的自定义上下文（关键）
        for chunk in self.agent.stream(input_dict, stream_mode="values", context={"report": False}):
            latest_message = chunk["messages"][-1]
            if latest_message.content:
                yield latest_message.content.strip() + "\n"
                #yield:多次返回，实现流式输出   return:一次返回，实现普通输出

    def execute_chat_stream(self, query: str):
        """
        面向 Web 前端(Streamlit 页面)的流式输出：
        只产出最终回答的文本增量，自动过滤模型思考过程、工具内部输出等中间消息，
        以便前端实现逐字输出的打字机效果
        :param query: 用户问题
        :return: 生成器，逐段产出最终回答文本
        """
        input_dict = {
            "messages": [
                {"role": "user", "content": query},
            ]
        }

        # stream_mode="messages"：token 级流式输出，每次产出 (消息块, 元数据)
        # 元数据中的 langgraph_node 标识消息来源节点：model=主模型节点  tools=工具内部节点
        for chunk, metadata in self.agent.stream(input_dict, stream_mode="messages", context={"report": False}):
            # 主模型的思考过程 content 为空，不产出；只有最终回答才有文本内容
            if metadata.get("langgraph_node") == "model" and getattr(chunk, "content", None):
                yield chunk.content

if __name__ == '__main__':
    agent = ReactAgent()

    for chunk in agent.execute_stream("扫地机器人在我所在的地区的气温下如何保养"):
        print(chunk, end="", flush=True)
