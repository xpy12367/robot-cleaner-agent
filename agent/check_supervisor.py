"""手动检查协调节点。

运行方式：在项目根目录执行 python -m agent.check_supervisor
注意：会使用项目现有配置，真实调用模型，共调用 4 次。
"""

from langchain_core.messages import AIMessage, HumanMessage

from agent.supervisor import supervisor_node


def main():
    # 每个用例包含：名称、消息列表、期望选择的 Agent。
    # 最后一条消息始终是本轮用户问题，前面的消息代表历史。
    cases = [
        (
            "普通知识咨询",
            [HumanMessage(content="扫地机器人的尘盒应该怎么清理？")],
            {"knowledge"},
        ),
        (
            "实时环境与知识组合",
            [
                HumanMessage(
                    content="查询上海当前天气和湿度，"
                    "结合这些信息告诉我今天使用扫地机器人要注意什么。"
                )
            ],
            {"environment", "knowledge"},
        ),
        (
            "个人使用报告",
            [HumanMessage(content="帮我生成2026年6月的机器人使用报告")],
            {"report"},
        ),
        (
            "结合历史补全指代",
            [
                HumanMessage(content="我想了解扫地机器人的滚刷怎么保养。"),
                AIMessage(content="可以，我可以介绍滚刷的日常清理和保养。"),
                HumanMessage(content="那它缠了头发该怎么清理？"),
            ],
            {"knowledge"},
        ),
    ]

    for name, messages, expected_agents in cases:
        print(f"\n用例：{name}")

        # 此处只单独调用协调节点，因此只提供它实际读取的两个字段。
        # 完整图运行时，我们会统一初始化 MultiAgentState 的全部字段。
        state = {
            "messages": messages,
            "user_query": messages[-1].content,
        }

        try:
            result = supervisor_node(state)
        except Exception as exc:
            # 一个用例失败后仍继续检查其他用例。
            # 不打印完整异常，避免兼容接口错误中可能包含连接信息。
            print(f"调用失败：{type(exc).__name__}")
            continue

        actual_agents = set(result["selected_agents"])
        #set() 转换为集合，方便比较。去重。

        # 使用集合比较，允许模型返回不同顺序。
        print("实际分配：", result["selected_agents"])
        print("期望分配：", sorted(expected_agents))
        print("分配匹配：", actual_agents == expected_agents)
        print("完整任务：", result["task_query"])


if __name__ == "__main__":
    # 只有作为入口运行时才执行检查；被其他模块导入时不会调用模型。
    main()
