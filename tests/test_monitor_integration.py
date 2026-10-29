"""
工具监控增强 - 集成测试
直接测试指标收集功能（不测试装饰器）
"""
import sys
import os
import time

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent.tools.monitor_enhanced import (
    ToolMetrics,
    MonitorConfig,
    generate_monitoring_report,
    get_metrics
)


def test_metrics_basic():
    """测试基础指标收集"""
    print("=" * 60)
    print("测试1: 基础指标收集")
    print("=" * 60)

    metrics = ToolMetrics()

    # 记录一些成功调用
    metrics.record_call("tool_a", 1.5, True, "trace_1")
    metrics.record_call("tool_a", 2.0, True, "trace_2")
    metrics.record_call("tool_b", 0.5, True, "trace_3")

    # 记录一次失败
    metrics.record_call("tool_a", 1.0, False, "trace_4", error="TestError")

    # 获取统计
    stats_a = metrics.get_tool_stats("tool_a")
    print(f"\ntool_a 统计:")
    print(f"  调用次数: {stats_a['call_count']}")
    print(f"  失败次数: {stats_a['failure_count']}")
    print(f"  成功率: {stats_a['success_rate']:.1f}%")
    print(f"  平均耗时: {stats_a['avg_duration']:.3f}秒")

    assert stats_a['call_count'] == 3
    assert stats_a['failure_count'] == 1
    assert stats_a['success_rate'] == 66.67

    print("\n[PASS] 测试1通过")


def test_recent_calls():
    """测试最近调用记录"""
    print("\n" + "=" * 60)
    print("测试2: 最近调用记录")
    print("=" * 60)

    metrics = ToolMetrics()
    metrics.max_recent = 5

    # 记录10次调用
    for i in range(10):
        metrics.record_call("test_tool", 1.0, True, f"trace_{i}")

    recent = metrics.get_recent_calls(limit=5)
    print(f"\n最近5次调用:")
    for call in recent:
        print(f"  [{call['trace_id']}] {call['tool_name']}: {call['duration']}秒")

    # 应该只保留最近5条
    assert len(metrics.recent_calls) == 5
    assert metrics.recent_calls[-1]['trace_id'] == "trace_9"

    print("\n[PASS] 测试2通过")


def test_monitoring_report():
    """测试监控报告生成"""
    print("\n" + "=" * 60)
    print("测试3: 监控报告生成")
    print("=" * 60)

    # 使用全局指标实例
    metrics = get_metrics()
    metrics.clear()  # 清空旧数据

    # 模拟一些调用
    metrics.record_call("rag_summarize", 2.5, True, "trace_a")
    metrics.record_call("rag_summarize", 3.0, True, "trace_b")
    metrics.record_call("get_weather", 1.2, True, "trace_c")
    metrics.record_call("get_weather", 0.8, False, "trace_d", error="NetworkError")

    # 生成报告
    report = generate_monitoring_report()
    print("\n" + report)

    assert "总调用次数: 4" in report
    assert "总失败次数: 1" in report
    assert "rag_summarize" in report
    assert "get_weather" in report

    print("\n[PASS] 测试3通过")


def test_config():
    """测试配置"""
    print("\n" + "=" * 60)
    print("测试4: 监控配置")
    print("=" * 60)

    print(f"\n当前配置:")
    print(f"  最大重试次数: {MonitorConfig.max_retries}")
    print(f"  重试延迟: {MonitorConfig.retry_delay}秒")
    print(f"  慢调用阈值: {MonitorConfig.slow_threshold}秒")
    print(f"  严重超时阈值: {MonitorConfig.critical_threshold}秒")
    print(f"  可重试工具: {MonitorConfig.retryable_tools}")

    assert MonitorConfig.max_retries >= 0
    assert MonitorConfig.retry_delay > 0
    assert MonitorConfig.slow_threshold > 0
    assert MonitorConfig.critical_threshold > MonitorConfig.slow_threshold

    print("\n[PASS] 测试4通过")


def test_clear_metrics():
    """测试清空指标"""
    print("\n" + "=" * 60)
    print("测试5: 清空指标")
    print("=" * 60)

    metrics = ToolMetrics()

    # 记录一些数据
    metrics.record_call("tool_a", 1.0, True, "trace_1")
    metrics.record_call("tool_b", 2.0, True, "trace_2")

    print(f"\n清空前: {len(metrics.call_counts)} 个工具, {len(metrics.recent_calls)} 条记录")

    # 清空
    metrics.clear()

    print(f"清空后: {len(metrics.call_counts)} 个工具, {len(metrics.recent_calls)} 条记录")

    assert len(metrics.call_counts) == 0
    assert len(metrics.recent_calls) == 0

    print("\n[PASS] 测试5通过")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("工具监控增强 - 集成测试")
    print("=" * 60 + "\n")

    start_time = time.time()

    try:
        test_metrics_basic()
        test_recent_calls()
        test_monitoring_report()
        test_config()
        test_clear_metrics()

        duration = time.time() - start_time

        print("\n" + "=" * 60)
        print("所有测试通过!")
        print(f"总耗时: {duration:.3f}秒")
        print("=" * 60)

        return True

    except AssertionError as e:
        print(f"\n测试失败: {e}")
        return False
    except Exception as e:
        print(f"\n测试出错: {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
