from langchain.agents import create_agent
from model.factory import chat_model
from agent.tools.agent_tools import (
    rag_summarize,
    get_weather,
    get_user_location,
    fetch_usage_report,
)
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch
from utils.prompt_loader import load_system_prompts
from agent.memory_manager import ConversationMemory, MemoryManager
from utils.logger_handler import logger

DEFAULT_SESSION_ID = "default"


class ReactAgent:
    def __init__(
        self,
        enable_memory: bool = True,
        max_history_turns: int = 5,
    ):
        """
        初始化ReactAgent

        Agent 实例本身不保存会话状态，可以被多个会话共用；
        会话记忆由 MemoryManager 按 session_id 隔离，每次调用时传入 session_id。

        Args:
            enable_memory: 是否启用会话记忆
            max_history_turns: 最大记忆轮次（一问一答算一轮）
        """
        self.enable_memory = enable_memory
        self.max_history_turns = max_history_turns

        # 创建底层LangChain Agent
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[
                rag_summarize,
                get_weather,
                get_user_location,
                fetch_usage_report,
            ],
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
        )

        logger.info(
            f"[ReactAgent] 已初始化 (会话记忆={enable_memory}, max_turns={max_history_turns})"
        )

    def get_memory(self, session_id: str = DEFAULT_SESSION_ID) -> ConversationMemory | None:
        """获取指定会话的记忆；未启用记忆时返回 None"""
        if not self.enable_memory:
            return None
        return MemoryManager.get_memory(
            session_id=session_id,
            max_turns=self.max_history_turns,
            max_tokens=4000,
            enable_persistence=False,  # 默认不持久化，避免隐私问题
        )

    def _build_input(self, query: str, memory: ConversationMemory | None) -> dict:
        """把历史对话 + 当前问题包成 Agent 认识的"对话消息"格式"""
        messages = []
        if memory:
            history_messages = memory.get_recent_messages(num_turns=self.max_history_turns)
            messages.extend(history_messages)
            logger.debug(f"[ReactAgent] 加载 {len(history_messages)} 条历史消息")

        messages.append({"role": "user", "content": query})
        return {"messages": messages}

    def _save_turn(self, memory: ConversationMemory | None, query: str, answer: str) -> None:
        """把本轮问答写入会话记忆"""
        if not memory:
            return
        memory.add_user_message(query)
        memory.add_assistant_message(answer.strip())
        logger.info(f"[ReactAgent] 已保存对话到记忆 (当前轮次: {memory.get_turn_count()})")

    def execute_stream(
        self,
        query: str,
        session_id: str = DEFAULT_SESSION_ID,
        user_id: str | None = None,
    ):
        """
        执行流式输出：
        :param query: 用户问题
        :param session_id: 会话ID，不同会话的记忆互相隔离
        :param user_id: 当前用户ID，由服务端传入，工具通过 runtime.context 读取
        :return: 生成器，逐段产出模型回答文本
        """
        memory = self.get_memory(session_id)
        input_dict = self._build_input(query, memory)

        # context —— 传给运行时的自定义上下文：report 用于提示词切换，user_id 供工具识别用户
        full_response = ""
        for chunk in self.agent.stream(
            input_dict,
            stream_mode="values",
            context={"report": False, "user_id": user_id},
        ):
            latest_message = chunk["messages"][-1]
            if latest_message.content:
                content = latest_message.content.strip() + "\n"
                full_response += content
                yield content

        self._save_turn(memory, query, full_response)

    def execute_chat_stream(
        self,
        query: str,
        session_id: str = DEFAULT_SESSION_ID,
        user_id: str | None = None,
    ):
        """
        面向 Web 前端(Streamlit 页面)的流式输出：
        只产出最终回答的文本增量，自动过滤模型思考过程、工具内部输出等中间消息，
        以便前端实现逐字输出的打字机效果
        :param query: 用户问题
        :param session_id: 会话ID，不同会话的记忆互相隔离
        :param user_id: 当前用户ID，由服务端传入，工具通过 runtime.context 读取
        :return: 生成器，逐段产出最终回答文本
        """
        memory = self.get_memory(session_id)
        input_dict = self._build_input(query, memory)

        # stream_mode="messages"：token 级流式输出，每次产出 (消息块, 元数据)
        full_response = ""
        for chunk, metadata in self.agent.stream(
            input_dict,
            stream_mode="messages",
            context={"report": False, "user_id": user_id},
        ):
            # 主模型的思考过程 content 为空，不产出；只有最终回答才有文本内容
            if metadata.get("langgraph_node") == "model" and getattr(
                chunk, "content", None
            ):
                full_response += chunk.content
                yield chunk.content

        self._save_turn(memory, query, full_response)

    def clear_memory(self, session_id: str = DEFAULT_SESSION_ID) -> None:
        """清空指定会话的记忆"""
        if self.enable_memory:
            MemoryManager.clear_memory(session_id)
            logger.info(f"[ReactAgent] 已清空会话记忆 (session_id={session_id})")

    def get_memory_summary(self, session_id: str = DEFAULT_SESSION_ID) -> str:
        """获取指定会话的记忆摘要（用于调试或展示）"""
        memory = self.get_memory(session_id)
        if not memory:
            return "记忆功能未启用"

        summary = memory.get_summary_context(max_chars=300)
        return f"当前记忆轮次: {memory.get_turn_count()}\n最近对话摘要:\n{summary}"


if __name__ == "__main__":
    agent = ReactAgent()

    for chunk in agent.execute_stream("扫地机器人在我所在的地区的气温下如何保养", user_id="1001"):
        print(chunk, end="", flush=True)
