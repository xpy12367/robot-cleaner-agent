from langchain.agents import create_agent
from model.factory import chat_model
from agent.tools.agent_tools import (rag_summarize, get_weather, get_user_location, get_user_id,
    get_current_month, fetch_external_data, fill_context_for_report)
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch
from utils.prompt_loader import load_system_prompts
from agent.memory_manager import MemoryManager
from utils.logger_handler import logger

class ReactAgent:
    def __init__(self, session_id: str = "default", enable_memory: bool = True, max_history_turns: int = 5):
        """
        初始化ReactAgent

        Args:
            session_id: 会话唯一标识符，不同用户应使用不同ID
            enable_memory: 是否启用会话记忆
            max_history_turns: 最大记忆轮次（一问一答算一轮）
        """
        self.session_id = session_id
        self.enable_memory = enable_memory
        self.max_history_turns = max_history_turns

        # 创建底层LangChain Agent
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[rag_summarize, get_weather, get_user_location, get_user_id,
                    get_current_month, fetch_external_data, fill_context_for_report],
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
        )

        # 初始化记忆管理器
        if self.enable_memory:
            self.memory = MemoryManager.get_memory(
                session_id=session_id,
                max_turns=max_history_turns,
                max_tokens=4000,
                enable_persistence=False  # 默认不持久化，避免隐私问题
            )
            logger.info(f"[ReactAgent] 已启用会话记忆 (session_id={session_id}, max_turns={max_history_turns})")
        else:
            self.memory = None
            logger.info(f"[ReactAgent] 未启用会话记忆")
        
    def execute_stream(self, query: str):
        """
        执行流式输出：
        :param query: 用户问题
        :return: 生成器，逐段产出模型回答文本
        """
        # 1. 构建消息列表：历史对话 + 当前问题
        messages = []
        if self.enable_memory and self.memory:
            # 获取历史对话（最近N轮）
            history_messages = self.memory.get_recent_messages(num_turns=self.max_history_turns)
            messages.extend(history_messages)  # 扩展消息列表
            logger.debug(f"[ReactAgent] 加载 {len(history_messages)} 条历史消息")

        # 添加当前用户问题
        messages.append({"role": "user", "content": query})

        input_dict = {"messages": messages}

        # 2. 执行流式推理
        full_response = ""
        for chunk in self.agent.stream(input_dict, stream_mode="values", context={"report": False}):
            latest_message = chunk["messages"][-1]
            if latest_message.content:
                content = latest_message.content.strip() + "\n"
                full_response += content
                yield content

        # 3. 保存到记忆
        if self.enable_memory and self.memory:
            self.memory.add_user_message(query)
            self.memory.add_assistant_message(full_response.strip())
            logger.info(f"[ReactAgent] 已保存对话到记忆 (当前轮次: {self.memory.get_turn_count()})")

    def execute_chat_stream(self, query: str):
        """
        面向 Web 前端(Streamlit 页面)的流式输出：
        只产出最终回答的文本增量，自动过滤模型思考过程、工具内部输出等中间消息，
        以便前端实现逐字输出的打字机效果
        :param query: 用户问题
        :return: 生成器，逐段产出最终回答文本
        """
        # 1. 构建消息列表：历史对话 + 当前问题
        messages = []
        if self.enable_memory and self.memory:
            # 获取历史对话（最近N轮）
            history_messages = self.memory.get_recent_messages(num_turns=self.max_history_turns)
            messages.extend(history_messages)
            logger.debug(f"[ReactAgent.execute_chat_stream] 加载 {len(history_messages)} 条历史消息")

        # 添加当前用户问题
        messages.append({"role": "user", "content": query})

        input_dict = {"messages": messages}

        # 2. 执行流式推理（token级）
        full_response = ""
        for chunk, metadata in self.agent.stream(input_dict, stream_mode="messages", context={"report": False}):
            # 主模型的思考过程 content 为空，不产出；只有最终回答才有文本内容
            if metadata.get("langgraph_node") == "model" and getattr(chunk, "content", None):
                full_response += chunk.content
                yield chunk.content

        # 3. 保存到记忆
        if self.enable_memory and self.memory:
            self.memory.add_user_message(query)
            self.memory.add_assistant_message(full_response.strip())
            logger.info(f"[ReactAgent.execute_chat_stream] 已保存对话到记忆 (当前轮次: {self.memory.get_turn_count()})")

    def clear_memory(self) -> None:
        """清空当前会话的记忆"""
        if self.enable_memory and self.memory:
            self.memory.clear()
            logger.info(f"[ReactAgent] 已清空会话记忆 (session_id={self.session_id})")

    def get_memory_summary(self) -> str:
        """获取当前记忆摘要（用于调试或展示）"""
        if not self.enable_memory or not self.memory:
            return "记忆功能未启用"

        turn_count = self.memory.get_turn_count()
        summary = self.memory.get_summary_context(max_chars=300)

        return f"当前记忆轮次: {turn_count}\n最近对话摘要:\n{summary}"

if __name__ == '__main__':
    agent = ReactAgent()

    for chunk in agent.execute_stream("扫地机器人在我所在的地区的气温下如何保养"):
        print(chunk, end="", flush=True)
