"""
会话记忆功能测试
验证ConversationMemory和ReactAgent的记忆管理功能
"""
import unittest
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agent.memory_manager import ConversationMemory, MemoryManager


class TestConversationMemory(unittest.TestCase):
    """测试会话记忆管理器"""

    def setUp(self):
        """每个测试前清理状态"""
        self.memory = ConversationMemory(
            session_id="test_session",
            max_turns=3,
            max_tokens=1000,
            enable_persistence=False
        )

    def tearDown(self):
        """清理测试数据"""
        self.memory.clear()

    def test_add_and_retrieve_messages(self):
        """测试添加和检索消息"""
        # 添加一轮对话
        self.memory.add_user_message("扫地机器人怎么保养?")
        self.memory.add_assistant_message("扫地机器人需要定期清理尘盒、滤网和滚刷。")

        # 验证消息数量
        self.assertEqual(len(self.memory.messages), 2)
        self.assertEqual(self.memory.get_turn_count(), 1)

        # 验证消息内容
        messages = self.memory.get_recent_messages()
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"], "扫地机器人怎么保养?")

    def test_max_turns_limit(self):
        """测试最大轮次限制"""
        # 添加4轮对话(超过max_turns=3)
        for i in range(4):
            self.memory.add_user_message(f"问题{i+1}")
            self.memory.add_assistant_message(f"回答{i+1}")

        # 应该只保留最近3轮(6条消息)
        self.assertEqual(len(self.memory.messages), 6)
        self.assertEqual(self.memory.get_turn_count(), 3)

        # 验证最早的对话被删除
        messages = self.memory.get_recent_messages()
        self.assertNotIn("问题1", [m["content"] for m in messages])
        self.assertIn("问题4", [m["content"] for m in messages])

    def test_get_recent_messages_limit(self):
        """测试获取指定数量的最近消息"""
        # 添加3轮对话
        for i in range(3):
            self.memory.add_user_message(f"问题{i+1}")
            self.memory.add_assistant_message(f"回答{i+1}")

        # 只获取最近1轮
        messages = self.memory.get_recent_messages(num_turns=1)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["content"], "问题3")

    def test_summary_context(self):
        """测试摘要上下文生成"""
        self.memory.add_user_message("扫地机器人怎么选?")
        self.memory.add_assistant_message("需要考虑户型大小、预算和功能需求。")

        summary = self.memory.get_summary_context(max_chars=100)

        # 验证摘要包含关键信息
        self.assertIn("用户", summary)
        self.assertIn("客服", summary)
        self.assertTrue(len(summary) <= 100 + 10)  # 允许一些截断误差

    def test_clear_memory(self):
        """测试清空记忆"""
        self.memory.add_user_message("测试问题")
        self.memory.add_assistant_message("测试回答")

        self.assertEqual(len(self.memory.messages), 2)

        self.memory.clear()

        self.assertEqual(len(self.memory.messages), 0)
        self.assertEqual(self.memory.get_turn_count(), 0)


class TestMemoryManager(unittest.TestCase):
    """测试记忆管理器工厂"""

    def tearDown(self):
        """清理所有会话"""
        for session_id in MemoryManager.list_sessions():
            MemoryManager.clear_memory(session_id)

    def test_get_memory_singleton(self):
        """测试单例模式"""
        memory1 = MemoryManager.get_memory("session1")
        memory2 = MemoryManager.get_memory("session1")

        # 应该返回同一个实例
        self.assertIs(memory1, memory2)

    def test_multiple_sessions(self):
        """测试多会话隔离"""
        memory_a = MemoryManager.get_memory("session_a")
        memory_b = MemoryManager.get_memory("session_b")

        # 添加不同的消息
        memory_a.add_user_message("会话A的问题")
        memory_b.add_user_message("会话B的问题")

        # 验证会话隔离
        messages_a = memory_a.get_recent_messages()
        messages_b = memory_b.get_recent_messages()

        self.assertEqual(len(messages_a), 1)
        self.assertEqual(len(messages_b), 1)
        self.assertNotEqual(messages_a[0]["content"], messages_b[0]["content"])

    def test_list_sessions(self):
        """测试列出所有会话"""
        MemoryManager.get_memory("session1")
        MemoryManager.get_memory("session2")
        MemoryManager.get_memory("session3")

        sessions = MemoryManager.list_sessions()

        self.assertEqual(len(sessions), 3)
        self.assertIn("session1", sessions)
        self.assertIn("session2", sessions)

    def test_clear_specific_session(self):
        """测试清除指定会话"""
        memory1 = MemoryManager.get_memory("session1")
        memory2 = MemoryManager.get_memory("session2")

        memory1.add_user_message("测试")
        memory2.add_user_message("测试")

        # 清除session1
        MemoryManager.clear_memory("session1")

        sessions = MemoryManager.list_sessions()
        self.assertNotIn("session1", sessions)
        self.assertIn("session2", sessions)


class TestMemoryWithLongText(unittest.TestCase):
    """测试长文本处理"""

    def setUp(self):
        self.memory = ConversationMemory(
            session_id="test_long",
            max_turns=10,
            max_tokens=200,  # 很小的token限制
            enable_persistence=False
        )

    def tearDown(self):
        self.memory.clear()

    def test_token_limit_trimming(self):
        """测试token限制自动裁剪"""
        # 添加多条长消息
        long_text = "这是一条很长的消息" * 50  # ~500字符

        for i in range(5):
            self.memory.add_user_message(f"{long_text} 问题{i}")
            self.memory.add_assistant_message(f"{long_text} 回答{i}")

        # 由于token限制，应该自动裁剪早期消息
        total_chars = sum(len(m["content"]) for m in self.memory.messages)
        estimated_tokens = total_chars // 2

        # 应该大致保持在token限制附近
        self.assertLess(estimated_tokens, 400)  # 允许一些误差


if __name__ == "__main__":
    # 运行测试
    unittest.main(verbosity=2)
