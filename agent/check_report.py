"""报告 Agent 的手动检查。

运行：python -m agent.check_report

前两个用例会真实调用模型和报告工具。
第三个用例应在节点内直接返回，不调用模型或工具。
"""

from langgraph.runtime import Runtime

# 复用已经写好的工具记录器，展示工具名称和返回内容。
# check_environment 的 main 有入口保护，导入它不会运行环境检查。
from agent.check_environment import ToolCallRecorder
from agent.report_agent import report_node


def run_case(
    name: str,
    task: str,
    user_id: str | None,
    expected_tools: list[str],
):
    recorder = ToolCallRecorder()
    print(f"\n用例：{name}")
    print("任务：", task)

    # 手动调用节点时，框架不会自动注入 Runtime，
    # 因此这里自行构造，模拟页面传入当前演示用户。
    runtime = Runtime(context={"user_id": user_id})

    result = report_node(
        {"task_query": task},
        config={
            "callbacks": [recorder],
            "recursion_limit": 20,
        },
        runtime=runtime,
    )

    print("实际工具：", recorder.started_tools)
    print("期望工具：", expected_tools)
    print("路径匹配：", recorder.started_tools == expected_tools)
    print("返回字段：", list(result))
    print("报告结果：")
    print(result["report_result"])


def main():
    # 当前演示数据中，用户1001有2026-06的记录。
    # 身份通过 Runtime 提供，不写进给模型的任务文字。
    run_case(
        name="正常月份",
        task="生成2026年6月的扫地机器人使用报告。",
        user_id="1001",
        expected_tools=["fetch_usage_report"],
    )

    # 请求没有记录的月份，检查工具的回退标记和报告说明。
    run_case(
        name="月份回退",
        task="生成2026年7月的扫地机器人使用报告。",
        user_id="1001",
        expected_tools=["fetch_usage_report"],
    )

    # 未选择用户：应直接返回提示，不调用工具。
    run_case(
        name="未选择用户",
        task="生成2026年6月的扫地机器人使用报告。",
        user_id=None,
        expected_tools=[],
    )


if __name__ == "__main__":
    main()
