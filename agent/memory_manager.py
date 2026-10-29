"""
会话记忆管理模块
提供短期记忆（当前会话）和可选的长期记忆持久化功能
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import os
from pathlib import Path
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


class ConversationMemory:
    """会话记忆管理器 - 管理单个会话的对话历史"""

    def __init__(
        self,
        session_id: str = "default",
        max_turns: int = 10,
        max_tokens: int = 4000,
        enable_persistence: bool = False,
        storage_path: Optional[str] = None
    ):
        """
        初始化会话记忆管理器

        Args:
            session_id: 会话唯一标识符
            max_turns: 最大保留对话轮次（一问一答算一轮）
            max_tokens: 最大token数量估算（中文按2字符=1token粗估）
            enable_persistence: 是否启用持久化存储
            storage_path: 持久化存储路径（相对于项目根目录）
        """
        self.session_id = session_id
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.enable_persistence = enable_persistence

        # 对话历史: [{"role": "user"|"assistant", "content": "...", "timestamp": "..."}]
        self.messages: List[Dict[str, Any]] = []

        # 持久化配置
        if enable_persistence:
            if storage_path:
                self.storage_dir = get_abs_path(storage_path)
            else:
                self.storage_dir = get_abs_path("logs/conversations")

            Path(self.storage_dir).mkdir(parents=True, exist_ok=True)
            self.storage_file = os.path.join(
                self.storage_dir,
                f"{self.session_id}.json"
            ) # 会话记忆文件路径

            # 尝试从文件恢复历史记录
            self._load_from_file()

    def add_user_message(self, content: str) -> None:
        """添加用户消息"""
        message = {
            "role": "user",
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        self.messages.append(message)
        logger.debug(f"[Memory] 添加用户消息: {content[:50]}...")

        self._trim_if_needed()
        self._save_if_enabled()

    def add_assistant_message(self, content: str) -> None:
        """添加助手回复"""
        message = {
            "role": "assistant",
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        self.messages.append(message)
        logger.debug(f"[Memory] 添加助手消息: {content[:50]}...")

        self._trim_if_needed()
        self._save_if_enabled()

    def get_recent_messages(
        self,
        num_turns: Optional[int] = None,
        include_system: bool = False
    ) -> List[Dict[str, str]]:
        """
        获取最近N轮对话（适配LangChain消息格式）

        Args:
            num_turns: 获取最近N轮，None则使用max_turns
            include_system: 是否包含系统消息（system_prompt另外处理，这里通常为False）

        Returns:
            消息列表，格式: [{"role": "user", "content": "..."}, ...]
        """
        turns = num_turns if num_turns is not None else self.max_turns

        # 一轮对话 = 1条user + 1条assistant，所以要取 turns * 2 条消息
        recent_messages = self.messages[-(turns * 2):] if turns > 0 else []

        # 移除timestamp字段，只保留role和content
        formatted_messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in recent_messages
        ]

        logger.debug(f"[Memory] 获取最近 {len(formatted_messages)} 条消息")
        return formatted_messages

    def get_summary_context(self, max_chars: int = 500) -> str:
        """
        获取会话摘要上下文（用于提示词注入）

        Args:
            max_chars: 最大字符数

        Returns:
            摘要文本
        """
        if not self.messages:
            return ""

        # 提取最近3轮对话的关键信息
        recent = self.messages[-6:] if len(self.messages) >= 6 else self.messages

        summary_parts = []
        for msg in recent:
            role_label = "用户" if msg["role"] == "user" else "客服"
            content_snippet = msg["content"][:100]  # 每条消息最多100字符
            summary_parts.append(f"{role_label}: {content_snippet}")

        summary = "\n".join(summary_parts)

        # 截断到最大字符数
        if len(summary) > max_chars:
            summary = summary[:max_chars] + "..."

        return summary

    def clear(self) -> None:
        """清空当前会话记忆"""
        self.messages.clear()
        logger.info(f"[Memory] 清空会话 {self.session_id} 的记忆")

        # 如果启用持久化，删除文件
        if self.enable_persistence and os.path.exists(self.storage_file):
            try:
                os.remove(self.storage_file)
                logger.info(f"[Memory] 删除持久化文件: {self.storage_file}")
            except Exception as e:
                logger.error(f"[Memory] 删除文件失败: {e}")

    def get_turn_count(self) -> int:
        """获取当前对话轮次数"""
        # 一轮对话 = 1条user + 1条assistant
        return len(self.messages) // 2

    def _trim_if_needed(self) -> None:
        """根据轮次和token限制裁剪历史记录"""
        # 1. 按轮次裁剪
        max_messages = self.max_turns * 2
        if len(self.messages) > max_messages:
            removed_count = len(self.messages) - max_messages
            self.messages = self.messages[-max_messages:]
            logger.info(f"[Memory] 按轮次裁剪，删除 {removed_count} 条旧消息")

        # 2. 按token估算裁剪（粗估：中文2字符≈1token，英文1单词≈1token）
        total_chars = sum(len(msg["content"]) for msg in self.messages)
        estimated_tokens = total_chars // 2  # 粗略估算

        if estimated_tokens > self.max_tokens:
            # 从最旧的消息开始删除，每次删除一对（user+assistant）
            while estimated_tokens > self.max_tokens and len(self.messages) >= 2:
                removed = self.messages[:2]
                self.messages = self.messages[2:]

                removed_chars = sum(len(msg["content"]) for msg in removed)
                estimated_tokens -= removed_chars // 2

                logger.debug(f"[Memory] 按token裁剪，删除1轮对话")

    def _save_if_enabled(self) -> None:
        """如果启用持久化，保存到文件"""
        if not self.enable_persistence:
            return

        try:
            data = {
                "session_id": self.session_id,
                "updated_at": datetime.now().isoformat(),
                "messages": self.messages
            }

            with open(self.storage_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.debug(f"[Memory] 已保存到: {self.storage_file}")
        except Exception as e:
            logger.error(f"[Memory] 保存失败: {e}")

    def _load_from_file(self) -> None:
        """从文件加载历史记录"""
        if not os.path.exists(self.storage_file):
            logger.debug(f"[Memory] 持久化文件不存在，创建新会话")
            return

        try:
            with open(self.storage_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.messages = data.get("messages", [])
            logger.info(
                f"[Memory] 从文件恢复 {len(self.messages)} 条消息 "
                f"(最后更新: {data.get('updated_at', 'unknown')})"
            )
        except Exception as e:
            logger.error(f"[Memory] 加载文件失败: {e}，使用空记忆")
            self.messages = []


class MemoryManager:
    """记忆管理器工厂类 - 管理多个会话的记忆实例"""

    _instances: Dict[str, ConversationMemory] = {}

    @classmethod
    def get_memory(
        cls,
        session_id: str = "default",
        **kwargs
    ) -> ConversationMemory:
        """
        获取或创建会话记忆实例（单例模式）

        Args:
            session_id: 会话ID
            **kwargs: 传递给ConversationMemory的参数

        Returns:
            会话记忆实例
        """
        if session_id not in cls._instances:
            cls._instances[session_id] = ConversationMemory(
                session_id=session_id,
                **kwargs
            )
            logger.info(f"[MemoryManager] 创建新的记忆实例: {session_id}")

        return cls._instances[session_id]

    @classmethod
    def clear_memory(cls, session_id: str) -> None:
        """清除指定会话的记忆"""
        if session_id in cls._instances:
            cls._instances[session_id].clear()
            del cls._instances[session_id]
            logger.info(f"[MemoryManager] 移除记忆实例: {session_id}")

    @classmethod
    def list_sessions(cls) -> List[str]:
        """列出所有活跃会话ID"""
        return list(cls._instances.keys())
