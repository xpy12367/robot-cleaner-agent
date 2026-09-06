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

    def load_document(self):
        """
        从数据文件夹内读取数据文件，转为向量存入向量库
        要计算文件的MD5做去重
        :return: None
        """
        def check_md5_hex(md5_for_check:str):
            """
            检查md5文件
            :param md5_for_check:md5文件
            :return:
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
            :param md5_for_check:
            :return:
            """
            with open(get_abs_path(chroma_config["md5_hex_store"]),"a",encoding="utf-8") as f:
                f.write(md5_for_check+"\n")

        def get_file_documents(read_path:str):
            """
            获取文件文档
            :param read_path:
            :return:
            """
            if read_path.endswith(".txt"):
                return txt_loader(read_path)
            elif read_path.endswith(".pdf"):
                return pdf_loader(read_path)
            return []

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

    vs.load_document()

    retriever = vs.get_retriever()

    res = retriever.invoke("扫地")

    for r in res:
        print(r.page_content)    #page_content: 文本内容
        print("-" * 20)
