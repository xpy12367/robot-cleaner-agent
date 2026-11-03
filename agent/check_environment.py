"""环境 Agent 的手动检查。

运行：python -m agent.check_environment

两组用例都会真实调用模型。
第一组使用真实天气接口；第二组只模拟定位失败，不发送定位请求。
"""

from unittest.mock import patch

from langchain_core.callbacks import BaseCallbackHandler

from agent.environment_agent import environment_node


class ToolCallRecorder(BaseCallbackHandler):
    """记录工具调用，检查模型是否走了预期路径。"""

    def __init__(self):
        self.started_tools = []

    def on_tool_start(self, serialized, input_str, **kwargs):
        name = (serialized or {}).get("name", "unknown")
        self.started_tools.append(name)
        print("开始调用工具：", name)

    def on_tool_end(self, output, **kwargs):
        # 同时展示工具原始结果，便于核对最终回答是否忠实。
        content = getattr(output, "content", output)
        print("工具返回：", content)


def run_case(task: str, expected_tools: list[str]):
    """运行一个用例，并检查工具调用顺序。"""

    recorder = ToolCallRecorder()
    print("\n任务：", task)

    result = environment_node(
        {"task_query": task},
        config={
            "callbacks": [recorder],
            "recursion_limit": 20,
        },
    )

    print("实际工具：", recorder.started_tools)
    print("期望工具：", expected_tools)
    print("路径匹配：", recorder.started_tools == expected_tools)
    print("返回字段：", list(result))
    print("环境结果：", result["environment_result"])


def main():
    # 城市明确：应直接查天气，不额外定位。
    # 如果没有配置高德 Key，工具会返回查询失败。
    # 这时应检查 Agent 是否如实说明失败，而不是编造天气。
    run_case(
        "查询上海当前的天气、温度和湿度。",
        ["get_weather"],
    )

    # 临时把底层定位函数替换成“返回空字符串”。
    # get_user_location 工具仍按原代码执行，
    # 接收到空字符串后会返回“定位失败”提示。
    #
    # patch 仅在 with 代码块内有效：
    # 不修改源文件、不删除 Key，离开代码块后自动恢复。
    with patch(
        "agent.tools.agent_tools.get_user_city_by_ip",
        return_value="",
    ):
        run_case(
            "查询我所在城市当前的天气和湿度。",
            ["get_user_location"],
        )


if __name__ == "__main__":
    main()
