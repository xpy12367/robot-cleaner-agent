"""手动检查知识库节点。

在项目根目录运行：python -m agent.check_knowledge

会真实调用模型、检索知识库。
首次初始化可能检查或补建索引，并加载重排模型。
"""

from langchain_core.callbacks import BaseCallbackHandler

from agent.knowledge_agent import knowledge_node


class ToolCallRecorder(BaseCallbackHandler):
    """通过回调观察工具执行，不修改 Agent 的业务逻辑。"""

    def __init__(self):
        self.started_tools = []
        self.finished_count = 0

    def on_tool_start(self, serialized, input_str, **kwargs):
        # 工具开始执行时，框架会调用这个方法。
        # 只记录名称，不打印完整参数或内部资料。
        name = (serialized or {}).get("name", "unknown")
        self.started_tools.append(name)
        print("开始调用工具：", name)

    def on_tool_end(self, output, **kwargs):
        # 工具正常返回时调用。
        self.finished_count += 1

        # LangChain 工具通常返回 ToolMessage，
        # 其中 content 才是交给 Agent 阅读的工具结果。
        # getattr 提供回退，兼容直接返回字符串的情况。
        content = getattr(output, "content", output)

        # 临时展示中间结果，用于区分：
        # 是 RAG 总结时补充了资料外内容，
        # 还是知识库 Agent 二次整理时补充了内容。
        print("\n--- 工具返回内容（诊断用）---")
        print(content)
        print("--- 工具返回内容结束 ---\n")


def main():
    recorder = ToolCallRecorder()

    # 直接给知识节点一条完整任务，
    # 先单独检查它，之后再与协调节点连接。
    task = "扫地机器人的滚刷缠了头发，应该如何安全清理？"

    print("任务：", task)

    result = knowledge_node(
        {"task_query": task},
        config={
            # callbacks 会随 config 传入子 Agent，观察内部工具调用。
            "callbacks": [recorder],
            # 限制图执行步数，避免模型不断调用工具。
            # 它限制的是执行步数，不是秒数，也不是工具调用次数。
            "recursion_limit": 20,
        },
    )

    print("\n调用过的工具：", recorder.started_tools)
    print("工具正常返回次数：", recorder.finished_count)
    print("是否调用 RAG：", "rag_summarize" in recorder.started_tools)
    print("返回字段：", list(result))
    print("\n知识库 Agent 的结果：")
    print(result["knowledge_result"])


if __name__ == "__main__":
    main()
