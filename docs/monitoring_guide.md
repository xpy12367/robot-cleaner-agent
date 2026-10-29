# 工具调用监控增强 - 使用指南

## 概述

工具调用监控增强模块为Agent系统提供了完整的工具调用可观测性功能，包括：

- ✅ **链路追踪** - 为每次调用生成唯一trace_id
- ✅ **自动重试** - 网络错误等可恢复错误自动重试
- ✅ **耗时监控** - 记录调用时长并触发告警
- ✅ **指标统计** - 自动收集调用次数、成功率、平均耗时等
- ✅ **实时展示** - 在Streamlit界面实时查看监控数据

## 核心组件

### 1. ToolMetrics - 指标收集器

负责收集和统计工具调用的各项指标。

```python
from agent.tools.monitor_enhanced import get_metrics

# 获取全局指标实例
metrics = get_metrics()

# 获取所有工具的统计信息
all_stats = metrics.get_all_stats()
# 返回: {
#   "tool_name": {
#     "call_count": 10,
#     "failure_count": 1,
#     "success_rate": 90.0,
#     "avg_duration": 1.234,
#     "total_duration": 12.34
#   }
# }

# 获取单个工具的统计
tool_stats = metrics.get_tool_stats("rag_summarize")

# 获取最近的调用记录
recent_calls = metrics.get_recent_calls(limit=10)

# 清空统计数据
metrics.clear()
```

### 2. MonitorConfig - 监控配置

控制重试、告警等行为的配置类。

```python
from agent.tools.monitor_enhanced import MonitorConfig

# 重试配置
MonitorConfig.max_retries = 2  # 最大重试次数（默认2次）
MonitorConfig.retry_delay = 1.0  # 重试延迟秒数（默认1秒）

# 耗时告警阈值（秒）
MonitorConfig.slow_threshold = 5.0  # 慢调用警告（默认5秒）
MonitorConfig.critical_threshold = 10.0  # 严重超时（默认10秒）

# 可重试的工具列表（None表示所有工具都重试）
MonitorConfig.retryable_tools = ["get_weather", "rag_summarize"]

# 不重试的错误类型
MonitorConfig.non_retryable_errors = [ValueError, TypeError, KeyError]
```

### 3. monitor_tool_enhanced - 增强监控中间件

核心中间件，已在 `middleware.py` 中自动启用。

## 功能详解

### 链路追踪

每次工具调用都会生成8位唯一的trace_id，方便日志关联：

```
[a1b2c3d4] 开始执行工具: get_weather | 参数: {'city': '深圳'}
[a1b2c3d4] 工具执行成功: get_weather | 耗时: 1.234秒
```

### 自动重试机制

**触发条件：**
1. 工具在 `retryable_tools` 列表中（或retryable_tools为None）
2. 错误类型不在 `non_retryable_errors` 中
3. 未超过 `max_retries` 次数

**重试流程：**

```
第1次调用 → 失败 (ConnectionError)
    ↓ 等待1秒
第2次调用 → 失败 (Timeout)
    ↓ 等待1秒
第3次调用 → 成功 ✓
```

**日志示例：**

```
[abc12345] 工具调用失败（尝试 1/3）: get_weather | 错误: ConnectionError: Network error | 将在 1秒后重试
[abc12345] 工具调用失败（尝试 2/3）: get_weather | 错误: TimeoutError: Request timeout | 将在 1秒后重试
[abc12345] 工具执行成功: get_weather | 耗时: 3.567秒
```

### 耗时监控与告警

**告警级别：**

| 耗时范围 | 级别 | 日志级别 |
|---------|------|---------|
| < 5秒 | 正常 | INFO |
| 5-10秒 | 慢调用 | WARNING |
| > 10秒 | 严重超时 | ERROR |

**告警日志示例：**

```
[def67890] ⏱ 慢调用告警: rag_summarize 耗时 6.234秒 (阈值: 5秒)
[ghi12345] ⚠️ 严重超时告警: get_weather 耗时 12.567秒 (阈值: 10秒)
```

### 指标统计

自动统计每个工具的：

- **调用次数** (call_count)
- **失败次数** (failure_count)
- **成功率** (success_rate) = (总次数 - 失败次数) / 总次数 × 100%
- **平均耗时** (avg_duration)
- **总耗时** (total_duration)

### 最近调用记录

保留最近100次工具调用的详细记录，每条记录包含：

```python
{
    "timestamp": "2026-10-29T15:30:45.123456",  # 调用时间
    "trace_id": "abc12345",  # 链路追踪ID
    "tool_name": "get_weather",  # 工具名称
    "duration": 1.234,  # 耗时（秒）
    "success": True,  # 是否成功
    "error": None  # 错误信息（失败时）
}
```

## 使用场景

### 场景1：调试工具调用问题

当某个工具频繁失败时：

```python
from agent.tools.monitor_enhanced import get_metrics

metrics = get_metrics()

# 查看该工具的统计
stats = metrics.get_tool_stats("get_weather")
print(f"成功率: {stats['success_rate']}%")
print(f"失败次数: {stats['failure_count']}")

# 查看最近的失败调用
recent = metrics.get_recent_calls(limit=20)
failures = [r for r in recent if not r["success"] and r["tool_name"] == "get_weather"]

for f in failures:
    print(f"[{f['trace_id']}] {f['error']}")
```

### 场景2：性能分析

识别慢调用并优化：

```python
all_stats = metrics.get_all_stats()

# 按平均耗时排序
slow_tools = sorted(
    all_stats.items(),
    key=lambda x: x[1]["avg_duration"],
    reverse=True
)

print("最慢的3个工具:")
for tool_name, stats in slow_tools[:3]:
    print(f"{tool_name}: 平均{stats['avg_duration']:.2f}秒")
```

### 场景3：生成监控报告

```python
from agent.tools.monitor_enhanced import generate_monitoring_report

# 生成文本报告
report = generate_monitoring_report()
print(report)

# 输出示例:
# === 工具调用监控报告 ===
#
# 总调用次数: 25
# 总失败次数: 2
# 总体成功率: 92.00%
#
# 各工具统计:
#   • rag_summarize: 调用10次 | 成功率100.0% | 平均耗时1.234秒
#   • get_weather: 调用8次 | 成功率87.5% | 平均耗时0.567秒
#   • get_user_location: 调用7次 | 成功率100.0% | 平均耗时0.234秒
#
# 最近5次调用:
#   ✓ [abc12345] rag_summarize: 1.234秒
#   ✗ [def67890] get_weather: 2.345秒
#       错误: ConnectionError: Network timeout
```

## Streamlit界面集成

监控数据已集成到Streamlit侧边栏的"工具调用监控"面板中。

**展示内容：**

1. **总体统计**
   - 总调用次数
   - 总体成功率

2. **各工具详情**
   - 工具名称
   - 调用次数
   - 平均耗时
   - 成功率（带颜色标识：绿色≥90%，橙色<90%）

3. **最近调用**
   - 最近3次调用的工具、耗时和状态

**访问方式：**

运行应用后，在左侧边栏展开"工具调用监控"即可查看实时数据。

## 配置建议

### 开发环境

```python
# 较快的重试，详细的日志
MonitorConfig.max_retries = 3
MonitorConfig.retry_delay = 0.5
MonitorConfig.slow_threshold = 3.0
MonitorConfig.critical_threshold = 5.0
```

### 生产环境

```python
# 适度重试，合理的阈值
MonitorConfig.max_retries = 2
MonitorConfig.retry_delay = 1.0
MonitorConfig.slow_threshold = 5.0
MonitorConfig.critical_threshold = 10.0

# 只对网络相关工具重试
MonitorConfig.retryable_tools = [
    "get_weather",
    "get_user_location",
    "rag_summarize"
]
```

## 性能影响

监控功能对性能的影响微乎其微：

- **时间开销**: 每次调用增加约1-2毫秒（time()调用和简单计算）
- **内存开销**: 最多保留100条调用记录，约50KB内存
- **无额外IO**: 所有数据在内存中，不涉及文件或网络IO

## 注意事项

1. **线程安全**: 当前实现未加锁，在多线程环境下可能有数据竞争
2. **记录限制**: 最多保留100条最近调用，超出自动清理
3. **重试策略**: 重试会增加总耗时，需权衡用户体验和成功率
4. **内存清理**: 长时间运行建议定期调用 `metrics.clear()` 清理统计数据

## 扩展集成

### 与Prometheus集成

```python
from prometheus_client import Counter, Histogram

# 定义指标
tool_calls_total = Counter(
    'agent_tool_calls_total',
    'Total tool calls',
    ['tool_name', 'status']
)

tool_duration_seconds = Histogram(
    'agent_tool_duration_seconds',
    'Tool execution duration',
    ['tool_name']
)

# 在monitor_enhanced.py的_metrics.record_call中上报
tool_calls_total.labels(
    tool_name=tool_name,
    status='success' if success else 'failure'
).inc()

tool_duration_seconds.labels(tool_name=tool_name).observe(duration)
```

### 导出到日志文件

```python
import json
from datetime import datetime

def export_metrics_to_file(filename: str):
    """导出监控指标到JSON文件"""
    metrics = get_metrics()

    data = {
        "export_time": datetime.now().isoformat(),
        "stats": metrics.get_all_stats(),
        "recent_calls": metrics.get_recent_calls(limit=50)
    }

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# 定期导出
export_metrics_to_file("metrics_2026-10-29.json")
```

## 常见问题

**Q: 为什么有些错误不重试？**

A: ValueError、TypeError、KeyError等通常是代码逻辑错误，重试无意义。只有网络错误、超时等临时性错误才重试。

**Q: 如何禁用某个工具的重试？**

A: 设置 `MonitorConfig.retryable_tools` 为不包含该工具的列表。

**Q: 监控数据会持久化吗？**

A: 不会。所有数据都在内存中，应用重启后清零。如需持久化请使用上述导出方法。

**Q: 如何调整告警阈值？**

A: 修改 `MonitorConfig.slow_threshold` 和 `MonitorConfig.critical_threshold`。

---

## 下一步计划

- [ ] 添加更细粒度的重试策略（指数退避）
- [ ] 支持自定义重试判断函数
- [ ] 集成分布式追踪系统（OpenTelemetry）
- [ ] 添加告警通知机制（邮件/企业微信）
- [ ] 支持监控数据持久化到数据库

**更新时间**: 2026-10-29  
**版本**: v1.0
