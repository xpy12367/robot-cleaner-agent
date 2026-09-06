import os
from abc import ABC, abstractmethod
from typing import Optional
from urllib.parse import urlparse
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from utils.config_handler import agent_config


def get_required_env(config_key: str) -> str:
    """根据 YAML 中保存的环境变量名读取模型连接配置。"""
    env_name = agent_config[config_key]
    value = os.getenv(env_name)
    if not value:
        raise RuntimeError(f"缺少模型连接环境变量: {env_name}")
    value = value.strip()
    if config_key == "base_url":
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise RuntimeError(f"模型连接地址格式无效: {env_name}")
        return value.rstrip("/")
    return value

class BaseModelFactory(ABC):
    """
        @abstractmethod
        表示 generator() 是一个抽象方法。
        继承 BaseModelFactory 的子类必须实现这个方法
    """
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass

class ChatModelFactory(BaseModelFactory):
    """
    因为前面的@abstractmethod抽象类，所以这里需要实现generator方法
    """
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return ChatOpenAI(model=agent_config["chat_model_name"],
                          api_key=get_required_env("api_key"),
                          base_url=get_required_env("base_url"),
                          )

class EmbeddingsFactory(BaseModelFactory):
    """
    因为前面的@abstractmethod抽象类，所以这里需要实现generator方法
    """
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return OpenAIEmbeddings(model=agent_config["embedding_model_name"],
                                api_key=get_required_env("api_key"),
                                base_url=get_required_env("base_url"),
                                check_embedding_ctx_length=False,  # 阿里云兼容模式不接受token数组，必须发原始字符串
                                chunk_size=8,  # ① v4 单次上限10条，8留余量
                                dimensions=1536,  # ③ 直接传参，不再用 model_kwargs
                                )

chat_model = ChatModelFactory().generator()
embeddings_model = EmbeddingsFactory().generator()
