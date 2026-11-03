"""汇总节点的手动检查。

运行：python -m agent.check_summary
共调用汇总模型3次，专业结果全部使用手动准备的数据。
"""

from langchain_core.messages import HumanMessage

from agent.state import MultiAgentState
from agent.summary import summary_node


def main():
    cases = [
        {
            "name": "天气失败时仍保留知识建议",
            "query": "查询上海当前天气，并告诉我潮湿天气如何保养机器人。",
            "agents": ["environment", "knowledge"],
            "environment": "上海实时天气查询失败，当前温度和湿度未知。",
            "knowledge": (
                "潮湿环境下，清洗后的拖布应完全晾干。"
                "本次资料没有提供设备适用的湿度阈值。"
            ),
            "report": "",
        },
        {
            "name": "报告月份回退",
            "query": "生成2026年7月的使用报告。",
            "agents": ["report"],
            "environment": "",
            "knowledge": "",
            "report": (
                "未找到2026-07的记录，以下报告基于最近可用月份2026-06。"
                "覆盖率85%，日均清扫面积45m²，主刷剩余寿命60天。"
                "基于项目演示数据生成，不代表真实设备采集记录。"
            ),
        },
        {
            "name": "无需专业Agent的问候",
            "query": "你好，你能帮我做什么？",
            "agents": [],
            "environment": "",
            "knowledge": "",
            "report": "",
        },
    ]

    for case in cases:
        print(f"\n用例：{case['name']}")

        # 手动准备完整状态，模拟专业 Agent 已经完成处理。
        # 这里不调用 supervisor，也不调用任何专业 Agent。
        state: MultiAgentState = {
            "messages": [HumanMessage(content=case["query"])],
            "user_query": case["query"],
            "task_query": case["query"],
            "selected_agents": case["agents"],
            "knowledge_result": case["knowledge"],
            "environment_result": case["environment"],
            "report_result": case["report"],
            "final_answer": "",
        }

        result = summary_node(state, config={})

        # 单独调用节点只返回更新字典，不自动合并历史。
        # 因此返回消息应只有最终回答这一条。
        print("返回消息数：", len(result["messages"]))
        print(
            "消息与最终回答一致：",
            result["messages"][0].text == result["final_answer"],
        )
        print("最终回答：")
        print(result["final_answer"])


if __name__ == "__main__":
    main()
