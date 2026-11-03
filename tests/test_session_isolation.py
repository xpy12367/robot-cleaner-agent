"""
测试会话记忆隔离功能
验证多个会话的记忆互不干扰
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent.react_agent import ReactAgent
from agent.memory_manager import MemoryManager


def test_session_isolation():
    """测试多会话记忆隔离"""
    print("=" * 70)
    print("测试: 会话记忆隔离")
    print("=" * 70)

    # 创建一个共享的Agent实例
    agent = ReactAgent(enable_memory=True, max_history_turns=3)

    # 模拟两个不同的用户会话
    session_a = "user_alice_session"
    session_b = "user_bob_session"

    # 会话A添加消息
    memory_a = agent.get_memory(session_a)
    memory_a.add_user_message("我想买扫地机器人")
    memory_a.add_assistant_message("好的，我可以帮您推荐")

    # 会话B添加不同的消息
    memory_b = agent.get_memory(session_b)
    memory_b.add_user_message("如何清理扫地机器人？")
    memory_b.add_assistant_message("清理步骤如下...")

    # 验证：两个会话的记忆应该是独立的
    messages_a = memory_a.get_recent_messages()
    messages_b = memory_b.get_recent_messages()

    print(f"\n会话A ({session_a[:20]}...) 的消息数: {len(messages_a)}")
    print(f"  内容: {messages_a[0]['content'][:30]}")

    print(f"\n会话B ({session_b[:20]}...) 的消息数: {len(messages_b)}")
    print(f"  内容: {messages_b[0]['content'][:30]}")

    # 断言：会话A和B的消息应该不同
    assert len(messages_a) == 2, "会话A应该有2条消息"
    assert len(messages_b) == 2, "会话B应该有2条消息"
    assert messages_a[0]['content'] != messages_b[0]['content'], "两个会话的内容应该不同"

    print("\n[PASS] 会话记忆隔离正确")

    # 清理
    MemoryManager.clear_memory(session_a)
    MemoryManager.clear_memory(session_b)


def test_session_memory_persistence():
    """测试同一会话的记忆持久性"""
    print("\n" + "=" * 70)
    print("测试: 同一会话记忆持久性")
    print("=" * 70)

    agent = ReactAgent(enable_memory=True)
    session_id = "test_persistence"

    # 第1次交互
    memory = agent.get_memory(session_id)
    memory.add_user_message("第一个问题")
    memory.add_assistant_message("第一个回答")

    # 第2次交互：应该还能看到第1次的内容
    memory = agent.get_memory(session_id)
    memory.add_user_message("第二个问题")
    memory.add_assistant_message("第二个回答")

    messages = memory.get_recent_messages()

    print(f"\n会话 {session_id} 的消息数: {len(messages)}")
    print(f"  第1条: {messages[0]['content']}")
    print(f"  第4条: {messages[3]['content']}")

    assert len(messages) == 4, "应该累积了4条消息"
    assert messages[0]['content'] == "第一个问题", "第1条消息应该保留"
    assert messages[3]['content'] == "第二个回答", "第4条消息应该正确"

    print("\n[PASS] 同一会话记忆持久正确")

    # 清理
    MemoryManager.clear_memory(session_id)


def test_memory_max_turns():
    """测试记忆轮次限制"""
    print("\n" + "=" * 70)
    print("测试: 记忆轮次限制")
    print("=" * 70)

    agent = ReactAgent(enable_memory=True, max_history_turns=2)
    session_id = "test_max_turns"

    memory = agent.get_memory(session_id)

    # 添加3轮对话（超过max_turns=2）
    for i in range(1, 4):
        memory.add_user_message(f"问题{i}")
        memory.add_assistant_message(f"回答{i}")

    messages = memory.get_recent_messages(num_turns=10)  # 请求所有消息

    print(f"\n添加了3轮对话，最大轮次为2")
    print(f"实际保留的消息数: {len(messages)}")
    print(f"最早的消息: {messages[0]['content']}")

    # 应该只保留最近2轮（4条消息）
    assert len(messages) == 4, "应该只保留最近2轮（4条消息）"
    assert messages[0]['content'] == "问题2", "最早的消息应该是'问题2'（问题1应被清理）"
    assert messages[-1]['content'] == "回答3", "最新的消息应该是'回答3'"

    print("\n[PASS] 记忆轮次限制正确")

    # 清理
    MemoryManager.clear_memory(session_id)


def test_clear_specific_session():
    """测试清空特定会话不影响其他会话"""
    print("\n" + "=" * 70)
    print("测试: 清空特定会话")
    print("=" * 70)

    agent = ReactAgent(enable_memory=True)
    session_a = "session_to_clear"
    session_b = "session_to_keep"

    # 两个会话都添加消息
    memory_a = agent.get_memory(session_a)
    memory_a.add_user_message("会话A的消息")

    memory_b = agent.get_memory(session_b)
    memory_b.add_user_message("会话B的消息")

    print(f"\n清空前:")
    print(f"  会话A消息数: {len(memory_a.get_recent_messages())}")
    print(f"  会话B消息数: {len(memory_b.get_recent_messages())}")

    # 清空会话A
    agent.clear_memory(session_a)

    # 重新获取
    memory_a = agent.get_memory(session_a)
    memory_b = agent.get_memory(session_b)

    print(f"\n清空会话A后:")
    print(f"  会话A消息数: {len(memory_a.get_recent_messages())}")
    print(f"  会话B消息数: {len(memory_b.get_recent_messages())}")

    assert len(memory_a.get_recent_messages()) == 0, "会话A应该被清空"
    assert len(memory_b.get_recent_messages()) == 1, "会话B应该保留"

    print("\n[PASS] 清空特定会话正确")

    # 清理
    MemoryManager.clear_memory(session_b)


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("会话记忆隔离测试 - 开始")
    print("=" * 70 + "\n")

    try:
        test_session_isolation()
        test_session_memory_persistence()
        test_memory_max_turns()
        test_clear_specific_session()

        print("\n" + "=" * 70)
        print("所有测试通过!")
        print("=" * 70)
        return True

    except AssertionError as e:
        print(f"\n测试失败: {e}")
        return False
    except Exception as e:
        print(f"\n测试出错: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
