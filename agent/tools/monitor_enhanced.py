"""
增强的工具调用监控中间件
提供重试、耗时监控、告警和链路追踪功能
"""
import time
import uuid
from typing import Callable, Optional, Dict, Any
from datetime import datetime
from collections import defaultdict

from langchain.agents.middleware import wrap_tool_call
from langgraph.prebuilt.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from utils.logger_handler import logger


# ==================== 全局监控指标 ====================
class ToolMetrics:
    """工具调用指标收集器"""

    def __init__(self):
        # 工具调用次数统计 {tool_name: count}
        self.call_counts: Dict[str, int] = defaultdict(int)

        # 工具调用总耗时统计 {tool_name: total_seconds}
        self.total_durations: Dict[str, float] = defaultdict(float)

        # 工具失败次数统计 {tool_name: count}
        self.failure_counts: Dict[str, int] = defaultdict(int)

        # 最近的工具调用记录（最多保留100条）
        self.recent_calls: list[Dict[str, Any]] = []
        self.max_recent = 100

    def record_call(
        self,
        tool_name: str,
        duration: float,
        success: bool,
        trace_id: str,
        error: Optional[str] = None
    ):
        """记录一次工具调用"""
        self.call_counts[tool_name] += 1
        self.total_durations[tool_name] += duration

        if not success:
            self.failure_counts[tool_name] += 1

        # 记录详细信息
        record = {
            "timestamp": datetime.now().isoformat(),  # 时间戳 isoformat(): 格式为YYYY-MM-DDTHH:MM:SS.SSSZ
            "trace_id": trace_id,
            "tool_name": tool_name,
            "duration": round(duration, 3),
            "success": success,
            "error": error,
        }

        self.recent_calls.append(record)

        # 保持最近调用记录在限制内
        if len(self.recent_calls) > self.max_recent:
            self.recent_calls = self.recent_calls[-self.max_recent:]

    def get_tool_stats(self, tool_name: str) -> Dict[str, Any]:
        """获取指定工具的统计信息"""
        call_count = self.call_counts.get(tool_name, 0)
        total_duration = self.total_durations.get(tool_name, 0)
        failure_count = self.failure_counts.get(tool_name, 0)

        avg_duration = total_duration / call_count if call_count > 0 else 0
        success_rate = (call_count - failure_count) / call_count * 100 if call_count > 0 else 0

        return {
            "tool_name": tool_name,
            "call_count": call_count,
            "failure_count": failure_count,
            "success_rate": round(success_rate, 2),
            "avg_duration": round(avg_duration, 3),
            "total_duration": round(total_duration, 3)
        }

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取所有工具的统计信息"""
        return {
            tool_name: self.get_tool_stats(tool_name)
            for tool_name in self.call_counts.keys()
        }

    def get_recent_calls(self, limit: int = 10) -> list[Dict[str, Any]]:
        """获取最近的工具调用记录"""
        return self.recent_calls[-limit:]

    def clear(self):
        """清空所有统计数据"""
        self.call_counts.clear()
        self.total_durations.clear()
        self.failure_counts.clear()
        self.recent_calls.clear()


# 全局指标实例
_metrics = ToolMetrics()


def get_metrics() -> ToolMetrics:
    """获取全局指标实例"""
    return _metrics


# ==================== 配置 ====================
class MonitorConfig:
    """监控配置"""

    # 重试配置
    max_retries: int = 2  # 最大重试次数
    retry_delay: float = 1.0  # 重试延迟（秒）

    # 耗时告警阈值（秒）
    slow_threshold: float = 5.0  # 慢调用阈值
    critical_threshold: float = 10.0  # 严重超时阈值

    # 需要重试的工具列表（None表示所有工具都重试）
    retryable_tools: Optional[list[str]] = ["get_weather", "get_user_location", "rag_summarize"]

    # 不需要重试的错误类型
    non_retryable_errors: list[type] = [ValueError, TypeError, KeyError]


# ==================== 增强监控中间件 ====================
@wrap_tool_call
# 工具调用监控中间件
def monitor_tool_enhanced(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    """
    增强的工具调用监控中间件

    功能：
    - 链路追踪：为每次调用生成trace_id
    - 耗时监控：记录执行时间并告警
    - 失败重试：自动重试可恢复的错误
    - 指标收集：统计调用次数、成功率、平均耗时
    """
    tool_name = request.tool_call['name']
    tool_args = request.tool_call['args']

    # 生成链路追踪ID
    trace_id = uuid.uuid4().hex[:8]

    # 记录开始
    start_time = time.time()
    logger.info(
        f"[{trace_id}] 开始执行工具: {tool_name} "
        f"| 参数: {_format_args(tool_args)}"
    )

    # 判断是否可重试
    is_retryable = (
        MonitorConfig.retryable_tools is None
        or tool_name in MonitorConfig.retryable_tools
    )

    # 执行工具（带重试）
    result = None
    last_error = None
    attempt = 0

    while attempt <= MonitorConfig.max_retries:
        attempt += 1

        try:
            result = handler(request)
            duration = time.time() - start_time

            # 成功执行
            logger.info(
                f"[{trace_id}] 工具执行成功: {tool_name} "
                f"| 耗时: {duration:.3f}秒"
            )

            # 检查耗时告警
            _check_duration_warning(trace_id, tool_name, duration)

            # 记录指标
            _metrics.record_call(
                tool_name=tool_name,
                duration=duration,
                success=True,
                trace_id=trace_id
            )

            # 特殊逻辑：报告生成上下文切换
            if tool_name == "fill_context_for_report":
                request.runtime.context["report"] = True
                logger.debug(f"[{trace_id}] 已切换到报告生成模式")

            return result

        except Exception as e:
            duration = time.time() - start_time
            last_error = e

            # 判断是否应该重试
            should_retry = (
                is_retryable
                and attempt <= MonitorConfig.max_retries
                and not isinstance(e, tuple(MonitorConfig.non_retryable_errors))  # 检查是否为不可重试错误类型
            )

            if should_retry:
                logger.warning(
                    f"[{trace_id}] 工具调用失败（尝试 {attempt}/{MonitorConfig.max_retries + 1}）: "
                    f"{tool_name} | 错误: {type(e).__name__}: {str(e)} "
                    f"| 将在 {MonitorConfig.retry_delay}秒后重试"
                )

                # 等待后重试
                time.sleep(MonitorConfig.retry_delay)
                continue
            else:
                # 不重试，直接失败
                logger.error(
                    f"[{trace_id}] 工具调用失败（不可重试或已达最大重试次数）: "
                    f"{tool_name} | 错误: {type(e).__name__}: {str(e)} "
                    f"| 耗时: {duration:.3f}秒"
                )

                # 记录失败指标
                _metrics.record_call(
                    tool_name=tool_name,
                    duration=duration,
                    success=False,
                    trace_id=trace_id,
                    error=f"{type(e).__name__}: {str(e)}"
                )

                # 抛出原始异常
                raise

    # 理论上不会到达这里，但为了类型安全
    raise last_error if last_error else RuntimeError("Unknown error")


def _format_args(args: dict) -> str:
    """格式化参数用于日志输出（限制长度）"""
    args_str = str(args)
    max_len = 200
    if len(args_str) > max_len:
        return args_str[:max_len] + "..."
    return args_str


def _check_duration_warning(trace_id: str, tool_name: str, duration: float):
    """检查耗时并发出告警"""
    if duration > MonitorConfig.critical_threshold:
        logger.error(
            f"[{trace_id}] ⚠️ 严重超时告警: {tool_name} "
            f"耗时 {duration:.3f}秒 (阈值: {MonitorConfig.critical_threshold}秒)"
        )
    elif duration > MonitorConfig.slow_threshold:
        logger.warning(
            f"[{trace_id}] ⏱ 慢调用告警: {tool_name} "
            f"耗时 {duration:.3f}秒 (阈值: {MonitorConfig.slow_threshold}秒)"
        )


# ==================== 监控报告生成 ====================
def generate_monitoring_report() -> str:
    """生成监控报告（供调试和展示使用）"""
    stats = _metrics.get_all_stats()

    if not stats:
        return "暂无工具调用记录"

    report_lines = ["=== 工具调用监控报告 ===\n"]

    # 总体统计
    total_calls = sum(s["call_count"] for s in stats.values())
    total_failures = sum(s["failure_count"] for s in stats.values())
    overall_success_rate = (
        (total_calls - total_failures) / total_calls * 100
        if total_calls > 0
        else 0
    )

    report_lines.append(f"总调用次数: {total_calls}")
    report_lines.append(f"总失败次数: {total_failures}")
    report_lines.append(f"总体成功率: {overall_success_rate:.2f}%\n")

    # 各工具详细统计
    report_lines.append("各工具统计:")
    for tool_name, stat in sorted(stats.items(), key=lambda x: -x[1]["call_count"]):
        report_lines.append(
            f"  * {tool_name}:"
            f" 调用{stat['call_count']}次"
            f" | 成功率{stat['success_rate']:.1f}%"
            f" | 平均耗时{stat['avg_duration']:.3f}秒"
        )

    # 最近调用记录
    recent = _metrics.get_recent_calls(limit=5)
    if recent:
        report_lines.append("\n最近5次调用:")
        for call in recent:
            status = "[OK]" if call["success"] else "[FAIL]"
            report_lines.append(
                f"  {status} [{call['trace_id']}] {call['tool_name']}: "
                f"{call['duration']:.3f}秒"
            )
            if call.get("error"):
                report_lines.append(f"      错误: {call['error']}")

    return "\n".join(report_lines)


if __name__ == "__main__":
    # 简单测试
    print("工具监控增强模块加载成功")
    print(f"配置: 最大重试{MonitorConfig.max_retries}次, 慢调用阈值{MonitorConfig.slow_threshold}秒")
