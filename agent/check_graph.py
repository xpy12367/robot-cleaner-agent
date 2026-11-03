"""多 Agent 总图的手动检查。

运行：python -m agent.check_graph

会真实调用模型、天气接口和 RAG。
共检查问候、双 Agent 协作、报告月份回退三个场景。
"""

from langchain_core.messages import HumanMessage

from agent.graph import get_multi_agent_graph
from agent.state import MultiAgentState
# 复用工具记录器，展示 RAG 等工具实际返回的内容。
from agent.check_environment import ToolCallRecorder

recorder = ToolCallRecorder()

def run_case(
    name: str,
    query: str,
    expected_agents: set[str],
    user_id: str | None = None,
):
    print(f"\n用例：{name}", flush=True)
    print("问题：", query, flush=True)

    # 每个用例创建独立状态，避免混入其他用例的历史和结果。
    state: MultiAgentState = {
        "messages": [HumanMessage(content=query)],
        "user_query": query,
        "task_query": "",
        "selected_agents": [],
        "knowledge_result": "",
        "environment_result": "",
        "report_result": "",
        "final_answer": "",
    }

    result = get_multi_agent_graph().invoke(
        state,
        config={
            "recursion_limit": 30,
            # 将回调传入总图，继续向子 Agent 和工具传播。
            # 可以对照“工具结果 → 专业结果 → 最终回答”定位内容变化。
            "callbacks": [recorder],
        },
        # 总图先接收身份，报告节点再显式传给报告子 Agent。
        context={"user_id": user_id},
    )

    print("实际分配：", result["selected_agents"])
    print(
        "分配匹配：",
        set(result["selected_agents"]) == expected_agents,
    )
    print("完整任务：", result["task_query"])

    for agent_name in result["selected_agents"]:
        field = f"{agent_name}_result"
        print(f"\n--- {field} ---")
        print(result[field])
        print("--- 专业结果结束 ---")
        # 与上面的 for 循环同级，只打印一次最终结果。
    print("\n最终消息数：", len(result["messages"]))
    print(
        "末条消息与最终回答一致：",
        result["messages"][-1].text == result["final_answer"],
    )
    print("最终回答：")
    print(result["final_answer"])


def main():
    run_case(
        name="问候直接汇总",
        query="你好，你能帮我做什么？",
        expected_agents=set(),
    )

    run_case(
        name="环境与知识协作",
        query=(
            "查询上海当前天气和湿度，"
            "并结合这些信息说明潮湿天气使用扫地机器人时的保养注意事项。"
        ),
        expected_agents={"environment", "knowledge"},
    )

    run_case(
        name="报告身份传递与月份回退",
        query="生成2026年7月的扫地机器人使用报告。",
        expected_agents={"report"},
        user_id="1001",
    )


if __name__ == "__main__":
    main()
