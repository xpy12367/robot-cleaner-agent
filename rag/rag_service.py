"""
总结服务类：用户提问，搜索参考资料，将提问和参考资料提交给模型，让模型总结回复
"""

from langchain_core.output_parsers import StrOutputParser
from model.factory import chat_model
from rag.vector_store import VectorStoreService
from rag.query_rewriter import get_rewriter
from utils.prompt_loader import load_rag_prompts
from utils.config_handler import rag_config
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from utils.logger_handler import logger


def print_prompt(prompt):
    print("=" * 20)
    print(prompt.to_string())
    print("=" * 20)
    return prompt


class RagSummarizeService:
    def __init__(self):
        self.vector_store = VectorStoreService()

        index_count = self.vector_store.ensure_index()
        if index_count == 0:
            raise RuntimeError(
                "知识库索引为空，请检查 data 目录、Embedding 配置和网络后重试"
            )

        self.retriever = self.vector_store.get_hybrid_retriever()  # 获取混合检索器

        # 查询改写器（可选）
        enable_rewrite = rag_config.get("enable_query_rewrite", False)
        self.rewriter = get_rewriter(enabled=enable_rewrite)

        self.prompt_text = load_rag_prompts()  # 加载提示词
        self.prompt_template = PromptTemplate.from_template(
            self.prompt_text
        )  # from_template(): 从模板中创建提示词
        self.model = chat_model  # 获取模型
        self.chain = self._init_chain()  # 初始化链式调用

        logger.info(
            f"[RagSummarizeService] 已初始化 | 查询改写: {enable_rewrite}"
        )

    def _init_chain(self):
        """
        初始化链式调用
        """
        chain = self.prompt_template | print_prompt | self.model | StrOutputParser()
        return chain

    def retriever_docs(self, query: str) -> list[Document]:
        """
        根据问题检索参考资料
        :param query: 用户问题
        :return: 参考资料列表
        """
        # 如果启用查询改写，使用多查询融合检索
        if self.rewriter:
            queries = self.rewriter.rewrite(query)
            logger.debug(f"[RagSummarizeService] 查询改写: 1 → {len(queries)}个")
            return self.retriever.invoke_multi(queries)
        else:
            # 未启用改写，使用单查询检索
            return self.retriever.invoke(query)

    def Rag_summarize(self, query: str) -> str:
        """
        根据问题检索参考资料，并生成总结回复
        :param query: 用户问题
        :return: 总结回复
        """
        context_docs = self.retriever_docs(query)

        context = ""
        counter = 0

        for doc in context_docs:
            counter += 1
            context += f"【参考资料{counter}】：参考资料：{doc.page_content}  |  参考元数据：{doc.metadata}\n"
            # .page_content：文档的内容   .metadata：文档来源路径

        return self.chain.invoke({"input": query, "context": context})


if __name__ == "__main__":
    rag_service = RagSummarizeService()
    print(rag_service.Rag_summarize("小户型适合哪些扫地机器人"))
