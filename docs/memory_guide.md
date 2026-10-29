# 会话记忆管理系统 - 使用文档

## 📖 功能概述

会话记忆管理系统为扫地机器人智能客服提供了**上下文感知**能力,使Agent能够:

- 🧠 记住用户在当前会话中的历史对话
- 💬 理解多轮对话中的指代关系（如"它"、"刚才说的"）
- 🔄 在连续提问中保持话题连贯性
- ⚡ 自动管理记忆长度,避免token溢出

---

## 🏗️ 架构设计

### 核心组件

```
┌─────────────────────────────────────────────────────┐
│                   Streamlit App                      │
│  (前端对话界面 - st.session_state.messages)          │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│                  ReactAgent                          │
│  • session_id: 会话唯一标识                          │
│  • enable_memory: 是否启用记忆                        │
│  • max_history_turns: 最大记忆轮次                    │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│              ConversationMemory                      │
│  • messages: 对话历史列表                            │
│  • max_turns: 最大轮次限制                            │
│  • max_tokens: token数量限制                          │
│  • get_recent_messages(): 获取最近N轮对话            │
│  • add_user_message() / add_assistant_message()      │
└──────────────────────────────────────────────────────┘
```

### 双层记忆系统

1. **Streamlit Session State (前端)**
   - 存储完整的可视化对话历史
   - 用于界面渲染和用户浏览
   - 在"清空对话"时一同清理

2. **ConversationMemory (后端)**
   - 存储精简的对话历史（仅保留最近N轮）
   - 传递给LLM作为上下文
   - 自动管理token限制

---

## 🚀 快速开始

### 1. 基础使用（Web界面）

启动应用后,记忆功能**自动启用**:

```bash
streamlit run app.py
```

- 默认保留最近 **5轮对话** 作为上下文
- 超过限制时自动清理最早的对话
- 在侧边栏"系统信息"可查看当前记忆轮次

### 2. 编程式使用

```python
from agent.react_agent import ReactAgent

# 创建带记忆的Agent
agent = ReactAgent(
    session_id="user_001",          # 唯一会话标识
    enable_memory=True,              # 启用记忆
    max_history_turns=5              # 保留最近5轮
)

# 第一轮对话
for chunk in agent.execute_chat_stream("小户型适合哪种扫地机器人?"):
    print(chunk, end="", flush=True)

# 第二轮对话 - Agent会记住第一轮的内容
for chunk in agent.execute_chat_stream("那它的价格大概多少?"):
    # "它" 指代第一轮提到的扫地机器人
    print(chunk, end="", flush=True)

# 清空记忆
agent.clear_memory()
```

---

## 🔧 配置参数

### ReactAgent 初始化参数

```python
ReactAgent(
    session_id: str = "default",      # 会话ID
    enable_memory: bool = True,       # 是否启用记忆
    max_history_turns: int = 5        # 最大记忆轮次
)
```

#### session_id
- **类型**: `str`
- **默认**: `"default"`
- **说明**: 会话唯一标识符,不同用户应使用不同ID
- **示例**:
  - Web应用: 自动生成的UUID（如 `3fa9c1e8`）
  - API服务: 使用 `user_{user_id}` 或 token hash

#### enable_memory
- **类型**: `bool`
- **默认**: `True`
- **说明**: 是否启用会话记忆
- **禁用场景**: 单次查询、无状态API、批量处理

#### max_history_turns
- **类型**: `int`
- **默认**: `5`
- **建议范围**: 3-10轮
- **说明**: 最大保留对话轮次（1轮 = 1条用户消息 + 1条助手回复）
- **选择建议**:
  - 3轮: 节省token,适合简短对话
  - 5轮: 默认值,平衡记忆与成本
  - 10轮: 长上下文任务,如诊断复杂故障

---

## 💡 使用场景

### 场景1: 连续追问

**用户**: "小户型适合哪种扫地机器人?"

**客服**: "小户型推荐选择2000Pa吸力以下、直径小于35cm的机型..."

**用户**: "那它的续航如何?"  ← **"它"指代前文的推荐机型**

✅ **有记忆**: Agent理解"它"的指代,给出准确回答
❌ **无记忆**: Agent不知道"它"是什么,回答偏离主题

---

### 场景2: 故障诊断

**用户**: "我的扫地机器人突然不工作了"

**客服**: "请先检查电量和开关状态..."

**用户**: "电量是满的"

**客服**: "那请检查尘盒是否装好..." ← **记住"电量满"这个信息**

**用户**: "尘盒也没问题"

**客服**: "综合您提供的信息（电量满、尘盒正常），可能是..." ← **记住所有已排查项**

✅ **有记忆**: 逐步排查,不重复询问
❌ **无记忆**: 每次都从头问起,用户体验差

---

### 场景3: 报告生成

**用户**: "帮我生成6月的使用报告"

**客服**: "正在查询您的6月数据...已生成报告"

**用户**: "能不能再加上清洁效率对比?"  ← **"报告"指代刚生成的报告**

✅ **有记忆**: 在已有报告基础上补充内容
❌ **无记忆**: 重新生成完整报告,浪费时间

---

## ⚙️ 进阶配置

### 启用持久化存储

```python
from agent.memory_manager import ConversationMemory

memory = ConversationMemory(
    session_id="user_001",
    max_turns=10,
    max_tokens=4000,
    enable_persistence=True,         # 启用持久化
    storage_path="logs/conversations"  # 存储路径
)
```

**注意事项**:
- 持久化会将对话保存到磁盘的 JSON 文件
- 默认**关闭**,避免隐私泄露
- 生产环境需配置文件加密和定期清理策略

---

### Token限制自动裁剪

```python
memory = ConversationMemory(
    session_id="user_001",
    max_turns=10,       # 轮次限制
    max_tokens=4000     # token限制（约8000汉字）
)
```

**裁剪规则**:
1. 优先按 `max_turns` 限制轮次
2. 如果总token仍超限,从最旧的对话开始删除
3. 每次删除一对消息（user + assistant）

---

### 多会话管理

```python
from agent.memory_manager import MemoryManager

# 为不同用户创建独立记忆
memory_user1 = MemoryManager.get_memory("user_001")
memory_user2 = MemoryManager.get_memory("user_002")

# 查看所有活跃会话
sessions = MemoryManager.list_sessions()  # ["user_001", "user_002"]

# 清除特定会话
MemoryManager.clear_memory("user_001")
```

---

## 📊 监控与调试

### 1. 查看记忆状态

**Web界面**: 侧边栏 → 系统信息 → "记忆轮次: X/5 轮"

**编程式**:
```python
agent = ReactAgent(session_id="user_001")

# 获取当前轮次
turn_count = agent.memory.get_turn_count()

# 获取摘要
summary = agent.get_memory_summary()
print(summary)
```

输出示例:
```
当前记忆轮次: 3
最近对话摘要:
用户: 小户型适合哪种扫地机器人?
客服: 推荐2000Pa吸力、直径35cm以下的机型...
用户: 那它的续航如何?
客服: 续航时间约90-120分钟,可覆盖60㎡...
```

---

### 2. 日志监控

```python
# 查看日志文件
tail -f logs/agent_*.log | grep "\[Memory\]"
```

关键日志:
```
[Memory] 添加用户消息: 小户型适合哪种扫地机器人?...
[Memory] 添加助手消息: 推荐2000Pa吸力、直径35cm以下的机型...
[ReactAgent] 加载 4 条历史消息
[ReactAgent] 已保存对话到记忆 (当前轮次: 3)
[Memory] 按轮次裁剪，删除 2 条旧消息
```

---

## 🧪 测试与验证

### 运行单元测试

```bash
# 运行所有记忆测试
python -m unittest tests.test_memory -v

# 运行特定测试
python -m unittest tests.test_memory.TestConversationMemory.test_max_turns_limit -v
```

### 测试覆盖

- ✅ 消息添加与检索
- ✅ 最大轮次限制
- ✅ Token自动裁剪
- ✅ 摘要生成
- ✅ 会话隔离
- ✅ 单例模式
- ✅ 清空功能

---

## 🔒 安全与隐私

### 1. 默认不持久化

```python
# 默认配置 - 内存存储,进程结束后自动清空
enable_persistence=False
```

### 2. 建议的安全措施

- **数据加密**: 持久化时使用 AES-256 加密
- **定期清理**: 设置过期时间,自动删除旧会话
- **脱敏处理**: 对敏感信息（电话、地址）进行脱敏
- **访问控制**: 记忆文件设置严格的文件权限

---

## 📈 性能优化

### Token消耗对比

假设单轮对话 = 200 tokens（包括问题+回答）

| 场景 | 无记忆 | 5轮记忆 | 10轮记忆 |
|-----|--------|---------|----------|
| 第1轮 | 200 tokens | 200 tokens | 200 tokens |
| 第2轮 | 200 tokens | 400 tokens | 400 tokens |
| 第3轮 | 200 tokens | 600 tokens | 600 tokens |
| 第6轮 | 200 tokens | 1200 tokens | 1400 tokens |
| **总计(10轮)** | **2000 tokens** | **~8000 tokens** | **~15000 tokens** |

### 优化建议

1. **根据场景调整轮次**
   - 简单咨询: 3轮
   - 一般对话: 5轮
   - 复杂诊断: 7-10轮

2. **使用摘要压缩**（后续改进）
   - 对超过5轮的对话生成摘要
   - 只保留摘要+最近3轮完整对话

3. **智能裁剪**（后续改进）
   - 识别重要信息（如故障描述、用户需求）
   - 优先保留重要消息,删除闲聊内容

---

## 🐛 常见问题

### Q1: 为什么清空前端对话后,Agent还记得之前的内容?

A: 需要同时清空 Streamlit Session State 和 Agent 的 ConversationMemory。现在"清空对话"按钮已经同步清理两者。

---

### Q2: 多个浏览器标签页共享记忆吗?

A: **不共享**。每个标签页有独立的 `session_id`,记忆完全隔离。

---

### Q3: 记忆会不会越来越多,导致成本失控?

A: 不会。系统有**双重限制**:
- 轮次限制（默认5轮）
- Token限制（默认4000 tokens）

任一达到上限就会自动裁剪最早的对话。

---

### Q4: 如何禁用记忆功能?

```python
# 方式1: 初始化时禁用
agent = ReactAgent(enable_memory=False)

# 方式2: 修改 app.py 的 get_agent() 函数
return ReactAgent(
    session_id=session_id,
    enable_memory=False,  # 改为 False
    max_history_turns=5
)
```

---

## 🔮 后续优化方向

1. **记忆摘要功能**
   - 对超长对话生成结构化摘要
   - 保留关键信息,压缩token消耗

2. **重要性评分**
   - 自动识别重要消息（故障描述、需求信息）
   - 裁剪时优先保留高分消息

3. **长期记忆**
   - 提取用户偏好（如"喜欢静音机型"）
   - 存入知识图谱或向量库
   - 跨会话复用

4. **记忆搜索**
   - 支持在历史对话中语义搜索
   - "你上次推荐的机型是哪个?"

---

## 📝 更新日志

### v1.0.0 (2024-10-29)
- ✅ 实现基础会话记忆功能
- ✅ 轮次与token双重限制
- ✅ 多会话隔离
- ✅ 自动清理与裁剪
- ✅ 集成到Streamlit前端
- ✅ 完整单元测试覆盖

---

## 📧 联系与反馈

如有问题或建议,请通过以下方式反馈:

- 项目Issues: https://github.com/xpy12367/robot-cleaner-agent/issues
- 邮箱: [your-email@example.com]

---

**Happy Chatting! 🤖💬**
