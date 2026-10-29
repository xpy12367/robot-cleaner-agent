"""
总结服务类：用户提问，搜索参考资料，将提问和参考资料提交给模型，让模型总结回复
"""
from langchain_core.output_parsers import StrOutputParser
from model.factory import chat_model
from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompts
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

def print_prompt(prompt):
    print("=" * 20)
    print(prompt.to_string())
    print("="*20)
    return prompt

class RagSummarizeService:
    def __init__(self):
        self.vector_store = VectorStoreService()

        index_count = self.vector_store.ensure_index()
        if index_count == 0:
            raise RuntimeError(
                "知识库索引为空，请检查 data 目录、Embedding 配置和网络后重试"
            )

        self.retriever = self.vector_store.get_hybrid_retriever()
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)  #from_template(): 从模板中创建提示词
        self.model = chat_model
        self.chain = self._init_chain()

    def _init_chain(self):
        """
        初始化链式调用
        """
        chain = self.prompt_template | print_prompt | self.model | StrOutputParser()
        return chain

    def retriever_docs(self, query: str)->list[Document]:
        """
        根据问题检索参考资料
        :param query: 用户问题
        :return: 参考资料列表
        """
        return self.retriever.invoke(query)  # invoke() 接收字符串，不能包列表，否则embedding会把input多套一层导致400

    def Rag_summarize(self, query: str)->str:
        """
        根据问题检索参考资料，并生成总结回复
        :param query: 用户问题
        :return: 总结回复
        """
        context_docs = self.retriever_docs(query)

        context = ""
        counter=0

        for doc in context_docs:
            counter+=1
            context += f"【参考资料{counter}】：参考资料：{doc.page_content}  |  参考元数据：{doc.metadata}\n"
            # .page_content：文档的内容   .metadata：文档来源路径

        return self.chain.invoke(
            {
            "input": query,
            "context": context
            }
        )

if __name__ == "__main__":
    rag_service = RagSummarizeService()
    print(rag_service.Rag_summarize("小户型适合哪些扫地机器人"))
