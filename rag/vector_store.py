import os
from langchain_chroma import Chroma
from langchain_core.documents import Document
from utils.config_handler import chroma_config
from model.factory import embeddings_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.path_tool import get_abs_path
from utils.file_handle import txt_loader, pdf_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.logger_handler import logger


class VectorStoreService:
    def __init__(self):
        # 向量数据库
        self.vector_store=Chroma(
            collection_name=chroma_config["collection_name"],
            embedding_function=embeddings_model,
            persist_directory=get_abs_path(chroma_config["persist_directory"])  # 用绝对路径，避免工作目录不同导致数据分家
        )
        # 文本分片器
        self.spliter=RecursiveCharacterTextSplitter(
            chunk_size=chroma_config["chunk_size"],
            chunk_overlap=chroma_config["chunk_overlap"],
            separators=chroma_config["separators"],
            length_function=len
        )

    def get_retriever(self):
        """
        获取向量检索器:
            as_retriever()：把 Chroma 向量库包装成 LangChain 标准的 Retriever 对象，
        使上层（Agent / RAG 链）可以统一通过 retriever.invoke(question) 检索。
            search_kwargs={"k": 3}：相似度检索时返回最相关的 3 个文档块，对应 chroma.yaml 中的 k: 3。
            典型用法：在 RAG 服务中 retriever = VectorStoreService().get_retriever()，
        之后对用户提问做向量检索，把命中片段拼进 Prompt 再交给大模型回答。
        """
        return self.vector_store.as_retriever(search_kwargs={"k":chroma_config["k"]})

    def get_all_documents(self) -> list[Document]:
        """读取 Chroma 中的全部文本块，用于建立 BM25 索引。"""
        data = self.vector_store.get(include=["documents", "metadatas"])

        ids = data.get("ids") or []
        contents = data.get("documents") or []
        metadatas = data.get("metadatas") or []

        return [
            Document(
                page_content=content,
                metadata=metadata or {},
                id=doc_id,
            )
            for doc_id, content, metadata in zip(ids, contents, metadatas)
            if content
        ]

    def get_hybrid_retriever(self):
        """创建向量 + BM25 的混合检索器。"""
        from rag.hybrid_retriever import HybridRetriever

        return HybridRetriever(
            vector_store=self.vector_store,
            documents=self.get_all_documents(),
            k=chroma_config["k"],
            candidate_k=chroma_config.get("hybrid_candidate_k", 10),
            vector_weight=chroma_config.get("hybrid_vector_weight", 0.5),
            bm25_weight=chroma_config.get("hybrid_bm25_weight", 0.5),
            rrf_k=chroma_config.get("hybrid_rrf_k", 60),
        )

    def document_count(self) -> int:
        """返回当前向量库中的文本块数量。"""
        return self.vector_store._collection.count()

    def ensure_index(self) -> int:
        """
        确保知识库索引存在：
        - 索引为空时，强制扫描全部文件并构建
        - 索引已有内容时，只处理新增或内容变化的文件
        """
        if self.document_count() == 0:
            logger.info("[加载知识库]索引为空，开始首次构建")
            self.load_document(force=True)
        else:
            self.load_document()

        count = self.document_count()
        logger.info(f"[加载知识库]当前索引共 {count} 个文本块")
        return count

    def load_document(self, force: bool = False):
        """
        从数据文件夹内读取数据文件，转为向量存入向量库
        要计算文件的MD5做去重
        :return: None
        """
        def check_md5_hex(md5_for_check:str):
            """
            检查md5文件
            :param md5_for_check:md5文件
            :return: bool
            """
            if not os.path.exists(get_abs_path(chroma_config["md5_hex_store"])):
                # 创建文件
                open(get_abs_path(chroma_config["md5_hex_store"]),"w",encoding="utf-8").close()
                return False  #md5没处理过

            with open(get_abs_path(chroma_config["md5_hex_store"]),"r",encoding="utf-8") as f:
                for line in f.readlines():   # 遍历文件的每一行
                    line=line.strip()  # 去掉行首行尾的空白字符
                    if line==md5_for_check:
                        return True  #md5处理过
            return False  #md5没处理过

        def save_md5_hex(md5_for_check:str):
            """
            保存md5文件
            :param md5_for_check:md5文件
            :return: None
            """
            with open(get_abs_path(chroma_config["md5_hex_store"]),"a",encoding="utf-8") as f:
                f.write(md5_for_check+"\n")

        def get_file_documents(read_path:str):
            """
            获取文件文档
            :param read_path:文件路径
            :return:文档列表
            """
            if read_path.endswith(".txt"):
                return txt_loader(read_path)
            elif read_path.endswith(".pdf"):
                return pdf_loader(read_path)
            return []

        if force:
            md5_path = get_abs_path(chroma_config["md5_hex_store"])
            with open(md5_path, "w", encoding="utf-8"):
                pass
        # 如果向量库明明是空的，但 md5.text 还残留旧记录，就先清空旧记录，确保文件会被重新写入索引。

        allowed_files_path:list[str]=listdir_with_allowed_type(
            get_abs_path(chroma_config["data_path"]),   # 数据文件夹路径
            tuple(chroma_config["allow_knowledge_file_type"])  # 允许的文件类型  tuple():转元组类型
        )   #这里返回的是文件路径列表

        for path in allowed_files_path:
            # 获取文件的MD5
            md5_hex = get_file_md5_hex(path)

            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path}内容已经存在知识库内，跳过")
                continue

            try:
                documents: list[Document] = get_file_documents(path)   # 获取文件文档的内容

                if not documents:
                    logger.warning(f"[加载知识库]{path}内没有有效文本内容，跳过")
                    continue
                # split_documents(): 分片
                split_document: list[Document] = self.spliter.split_documents(documents)

                if not split_document:
                    logger.warning(f"[加载知识库]{path}分片后没有有效文本内容，跳过")
                    continue

                # 将内容存入向量库
                self.vector_store.add_documents(split_document) #add_documents():存入向量库

                # 记录这个已经处理好的文件的md5，避免下次重复加载
                save_md5_hex(md5_hex)

                logger.info(f"[加载知识库]{path} 内容加载成功")

            except Exception as e:
                # exc_info为True会记录详细的报错堆栈，如果为False仅记录报错信息本身
                logger.error(f"[加载知识库]{path}加载失败：{str(e)}", exc_info=True)
                continue


if __name__ == "__main__":
    vs = VectorStoreService()
    count = vs.ensure_index()

    if count == 0:
        raise SystemExit("知识库索引为空，请检查 data 目录、Embedding 配置和网络")

    print(f"知识库索引已就绪，共 {count} 个文本块")
