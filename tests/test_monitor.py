"""
测试增强版工具监控中间件
"""
import unittest
import time
from unittest.mock import MagicMock, patch

from agent.tools.monitor_enhanced import (
    ToolMetrics,
    MonitorConfig,
    monitor_tool_enhanced,
    generate_monitoring_report,
    get_metrics
)


class TestToolMetrics(unittest.TestCase):
    """测试工具指标收集器"""

    def setUp(self):
        """每个测试前清空指标"""
        self.metrics = ToolMetrics()

    def test_record_successful_call(self):
        """测试记录成功调用"""
        self.metrics.record_call(
            tool_name="test_tool",
            duration=1.5,
            success=True,
            trace_id="abc123"
        )

        self.assertEqual(self.metrics.call_counts["test_tool"], 1)
        self.assertEqual(self.metrics.total_durations["test_tool"], 1.5)
        self.assertEqual(self.metrics.failure_counts["test_tool"], 0)
        self.assertEqual(len(self.metrics.recent_calls), 1)

    def test_record_failed_call(self):
        """测试记录失败调用"""
        self.metrics.record_call(
            tool_name="test_tool",
            duration=0.5,
            success=False,
            trace_id="def456",
            error="TestError: Something went wrong"
        )

        self.assertEqual(self.metrics.call_counts["test_tool"], 1)
        self.assertEqual(self.metrics.failure_counts["test_tool"], 1)
        self.assertIsNotNone(self.metrics.recent_calls[0].get("error"))

    def test_get_tool_stats(self):
        """测试获取工具统计信息"""
        # 记录3次成功，1次失败
        for i in range(3):
            self.metrics.record_call("tool_a", 1.0, True, f"trace_{i}")

        self.metrics.record_call("tool_a", 2.0, False, "trace_fail", error="Error")

        stats = self.metrics.get_tool_stats("tool_a")

        self.assertEqual(stats["call_count"], 4)
        self.assertEqual(stats["failure_count"], 1)
        self.assertEqual(stats["success_rate"], 75.0)  # 3/4 = 75%
        self.assertAlmostEqual(stats["avg_duration"], 1.25, places=2)  # (3*1 + 2)/4

    def test_get_all_stats(self):
        """测试获取所有工具统计"""
        self.metrics.record_call("tool_a", 1.0, True, "trace_1")
        self.metrics.record_call("tool_b", 2.0, True, "trace_2")

        all_stats = self.metrics.get_all_stats()

        self.assertEqual(len(all_stats), 2)
        self.assertIn("tool_a", all_stats)
        self.assertIn("tool_b", all_stats)

    def test_recent_calls_limit(self):
        """测试最近调用记录的限制"""
        self.metrics.max_recent = 5  # 设置为5条

        # 记录10次调用
        for i in range(10):
            self.metrics.record_call("test_tool", 1.0, True, f"trace_{i}")

        # 应该只保留最近5条
        self.assertEqual(len(self.metrics.recent_calls), 5)
        # 最后一条应该是trace_9
        self.assertEqual(self.metrics.recent_calls[-1]["trace_id"], "trace_9")

    def test_get_recent_calls(self):
        """测试获取最近N条调用记录"""
        for i in range(10):
            self.metrics.record_call("test_tool", 1.0, True, f"trace_{i}")

        recent_3 = self.metrics.get_recent_calls(limit=3)
        self.assertEqual(len(recent_3), 3)
        self.assertEqual(recent_3[-1]["trace_id"], "trace_9")

    def test_clear_metrics(self):
        """测试清空指标"""
        self.metrics.record_call("test_tool", 1.0, True, "trace_1")
        self.metrics.clear()

        self.assertEqual(len(self.metrics.call_counts), 0)
        self.assertEqual(len(self.metrics.total_durations), 0)
        self.assertEqual(len(self.metrics.failure_counts), 0)
        self.assertEqual(len(self.metrics.recent_calls), 0)


class TestMonitorEnhanced(unittest.TestCase):
    """测试增强版监控中间件（集成测试）"""

    def setUp(self):
        """每个测试前清空全局指标"""
        get_metrics().clear()

    @patch('agent.tools.monitor_enhanced.logger')
    def test_successful_tool_call(self, mock_logger):
        """测试成功的工具调用"""
        # 创建mock请求和处理器
        mock_request = MagicMock()
        mock_request.tool_call = {
            'name': 'test_tool',
            'args': {'arg1': 'value1'}
        }
        mock_request.runtime.context = {}

        mock_handler = MagicMock(return_value="success_result")

        # 执行监控中间件
        result = monitor_tool_enhanced(mock_request, mock_handler)

        # 验证结果
        self.assertEqual(result, "success_result")
        mock_handler.assert_called_once_with(mock_request)

        # 验证指标记录
        metrics = get_metrics()
        self.assertEqual(metrics.call_counts["test_tool"], 1)
        self.assertEqual(metrics.failure_counts["test_tool"], 0)

    @patch('agent.tools.monitor_enhanced.time.sleep')  # Mock sleep避免真实等待
    @patch('agent.tools.monitor_enhanced.logger')
    def test_tool_call_with_retry(self, mock_logger, mock_sleep):
        """测试带重试的工具调用"""
        # 配置：第一次失败，第二次成功
        mock_request = MagicMock()
        mock_request.tool_call = {
            'name': 'get_weather',  # 在retryable_tools中
            'args': {'city': '深圳'}
        }
        mock_request.runtime.context = {}

        call_count = 0

        def handler_with_first_failure(req):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Network error")
            return "success_on_retry"

        mock_handler = MagicMock(side_effect=handler_with_first_failure)

        # 执行
        result = monitor_tool_enhanced(mock_request, mock_handler)

        # 验证重试了一次
        self.assertEqual(mock_handler.call_count, 2)
        self.assertEqual(result, "success_on_retry")

        # 验证sleep被调用（重试延迟）
        mock_sleep.assert_called_once_with(MonitorConfig.retry_delay)

    @patch('agent.tools.monitor_enhanced.logger')
    def test_tool_call_non_retryable_error(self, mock_logger):
        """测试不可重试的错误"""
        mock_request = MagicMock()
        mock_request.tool_call = {
            'name': 'test_tool',
            'args': {}
        }
        mock_request.runtime.context = {}

        # ValueError是不可重试的错误
        mock_handler = MagicMock(side_effect=ValueError("Invalid input"))

        # 应该直接抛出异常，不重试
        with self.assertRaises(ValueError):
            monitor_tool_enhanced(mock_request, mock_handler)

        # 只调用了一次
        self.assertEqual(mock_handler.call_count, 1)

        # 记录了失败
        metrics = get_metrics()
        self.assertEqual(metrics.failure_counts["test_tool"], 1)

    @patch('agent.tools.monitor_enhanced.time.sleep')
    @patch('agent.tools.monitor_enhanced.logger')
    def test_tool_call_max_retries_exceeded(self, mock_logger, mock_sleep):
        """测试超过最大重试次数"""
        mock_request = MagicMock()
        mock_request.tool_call = {
            'name': 'get_weather',
            'args': {}
        }
        mock_request.runtime.context = {}

        # 始终失败
        mock_handler = MagicMock(side_effect=ConnectionError("Always fail"))

        # 应该在达到max_retries后抛出异常
        with self.assertRaises(ConnectionError):
            monitor_tool_enhanced(mock_request, mock_handler)

        # 调用了 max_retries + 1 次（初始1次 + 重试2次）
        self.assertEqual(mock_handler.call_count, MonitorConfig.max_retries + 1)

    @patch('agent.tools.monitor_enhanced.logger')
    def test_report_context_switch(self, mock_logger):
        """测试报告生成上下文切换"""
        mock_request = MagicMock()
        mock_request.tool_call = {
            'name': 'fill_context_for_report',
            'args': {}
        }
        mock_request.runtime.context = {}

        mock_handler = MagicMock(return_value="report_mode_activated")

        # 执行
        monitor_tool_enhanced(mock_request, mock_handler)

        # 验证context被设置
        self.assertTrue(mock_request.runtime.context.get("report"))


class TestMonitoringReport(unittest.TestCase):
    """测试监控报告生成"""

    def setUp(self):
        """清空指标"""
        get_metrics().clear()

    def test_empty_report(self):
        """测试空报告"""
        report = generate_monitoring_report()
        self.assertIn("暂无工具调用记录", report)

    def test_report_with_data(self):
        """测试有数据的报告"""
        metrics = get_metrics()

        # 模拟一些调用
        metrics.record_call("tool_a", 1.0, True, "trace_1")
        metrics.record_call("tool_a", 2.0, True, "trace_2")
        metrics.record_call("tool_b", 0.5, False, "trace_3", error="Error")

        report = generate_monitoring_report()

        # 验证报告内容
        self.assertIn("工具调用监控报告", report)
        self.assertIn("总调用次数: 3", report)
        self.assertIn("总失败次数: 1", report)
        self.assertIn("tool_a", report)
        self.assertIn("tool_b", report)


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    suite = unittest.TestSuite()

    # 添加测试
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestToolMetrics))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestMonitorEnhanced))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestMonitoringReport))

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
