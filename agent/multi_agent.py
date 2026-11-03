from langchain_core.messages import HumanMessage

from agent.graph import get_multi_agent_graph
from agent.memory_manager import ConversationMemory, MemoryManager
from agent.state import MultiAgentState
from collections.abc import Iterator

class MultiAgent:
    """对外提供多 Agent 调用和会话记忆管理。"""

    def __init__(
        self,
        enable_memory: bool = True,
        max_history_turns: int = 5,
    ):
        self.enable_memory = enable_memory
        self.max_history_turns = max_history_turns

        # 执行图可以复用，每次调用的状态单独构造。
        self.graph = get_multi_agent_graph()

    def get_memory(
        self,
        session_id: str,
    ) -> ConversationMemory | None:
        """沿用现有管理器，按 session_id 隔离历史。"""

        if not self.enable_memory:
            return None

        return MemoryManager.get_memory(
            session_id=session_id,
            max_turns=self.max_history_turns,
            max_tokens=4000,
            enable_persistence=False,
        )

    def _build_input(
        self,
        query: str,
        memory: ConversationMemory | None,
    ) -> MultiAgentState:
        """历史对话加本轮问题，构成一次新的图输入。"""

        messages = []

        if memory is not None:
            messages.extend(
                memory.get_recent_messages(num_turns=self.max_history_turns)
            )  # 最多取最近 max_history_turns 轮，一轮通常包含两条消息。
            # extend(): 将一个可迭代对象的元素添加到列表末尾，保持原有顺序。

        messages.append(HumanMessage(content=query))

        # 专业结果每轮重新初始化，不能沿用上一轮的结果。
        # 历史只通过 messages 传入，由协调节点补全本轮任务。
        return {
            "messages": messages,
            "user_query": query,
            "task_query": "",
            "selected_agents": [],
            "knowledge_result": "",
            "environment_result": "",
            "report_result": "",
            "final_answer": "",
        }

    def _save_turn(
        self,
        memory: ConversationMemory | None,
        query: str,
        answer: str,
    ) -> None:
        """只保存用户问题和最终回答，不保存内部工具消息。"""

        if memory is None:
            return

        memory.add_user_message(query)
        memory.add_assistant_message(answer)

    def execute(
        self,
        query: str,
        session_id: str,
        user_id: str | None = None,
    ) -> str:
        """执行一次完整任务，返回最终回答字符串。"""

        query = query.strip()
        if not query:
            raise ValueError("用户问题不能为空")

        memory = self.get_memory(session_id)
        state = self._build_input(query, memory)

        result = self.graph.invoke(
            state,
            config={"recursion_limit": 30},# 递归调用限制，防止无限循环。
            context={"user_id": user_id},
        )

        answer = result["final_answer"].strip()
        if not answer:
            raise RuntimeError("多 Agent 总图返回了空回答")

        # 图执行成功且取得有效回答后，才保存本轮。
        # 中途抛出异常时，不把不完整结果写入历史。
        self._save_turn(memory, query, answer)
        return answer

    def execute_chat_stream(
        self,
        query: str,
        session_id: str,
        user_id: str | None = None,
    ) -> Iterator[str]:
        """执行总图，只逐段输出汇总节点的回答。"""

        query = query.strip()
        if not query:
            raise ValueError("用户问题不能为空")

        memory = self.get_memory(session_id)
        state = self._build_input(query, memory)

        answer_parts = []

        for event in self.graph.stream(
            state,
            config={"recursion_limit": 30},# 递归调用限制，防止无限循环。
            context={"user_id": user_id},
            stream_mode="messages",
            # 明确使用 v2，每个事件都是包含 type、ns、data 的字典。
            version="v2",
        ):
            if event["type"] != "messages":
                continue

            # messages 事件的数据是：消息片段、运行元数据。
            chunk, metadata = event["data"]

            # 只展示总图 summary 节点调用模型时产生的文本。
            # 协调结果、专业 Agent 的回答和 RAG 内部输出都跳过。
            if metadata.get("langgraph_node") != "summary":
                continue

            # text 只提取文本，不展示工具调用参数或推理内容块。
            text = chunk.text
            if not text:
                continue

            # 每段原样保留，不能逐段 strip，
            # 否则 Markdown 换行或词语间的空格可能丢失。
            answer_parts.append(text)

            # yield 把当前文本交给调用方，随后继续读取下一段。
            yield text

        answer = "".join(answer_parts).strip()
        if not answer:
            raise RuntimeError("多 Agent 总图没有输出有效的最终回答")

        # 只有流被完整消费、且图正常结束后，才保存本轮。
        # 中途报错或调用方提前关闭生成器时，不保存部分回答。
        self._save_turn(memory, query, answer)

    def clear_memory(self, session_id: str) -> None:
        """清空指定会话，不影响其他会话。"""

        if self.enable_memory:
            MemoryManager.clear_memory(session_id)
