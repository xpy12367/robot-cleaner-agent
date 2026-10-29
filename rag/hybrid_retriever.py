"""Chroma 向量检索与 BM25 关键词检索的混合召回。"""

import re
from collections import defaultdict
import jieba
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from utils.logger_handler import logger

_TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]+")
"""
\u4e00-\u9fff:CJK 统一汉字区间，即常用中文字符
A-Za-z:英文字母
0-9:阿拉伯数字
+:前面的字符类出现一次或多次
"""


def tokenize_for_search(text: str) -> list[str]:
    """使用 jieba 把中文、英文和数字切成 BM25 可用的词。"""
    tokens: list[str] = []

    for token in jieba.cut_for_search((text or "").lower()):
        # jieba.cut_for_search():搜索引擎模式分词：对长词再做一次细粒度切分，提高召回率
        token = token.strip()
        if token and _TOKEN_PATTERN.fullmatch(token):
            # fullmatch():整个 token 都符合中文、英文和数字的模式
            tokens.append(token)

    return tokens


def document_key(document: Document) -> str:
    """为融合结果生成唯一键，优先使用 Chroma 文档 ID。"""
    if document.id:
        return str(document.id)

    source = str(document.metadata.get("source", "")) # 文档来源
    return f"{source}|{document.page_content}"


class HybridRetriever:
    """把向量召回和 BM25 召回按排名融合。"""

    def __init__(
        self,
        vector_store,
        documents: list[Document],
        *,
        k: int = 3,
        candidate_k: int = 10,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        rrf_k: int = 60,
    ):
        if not documents:
            raise ValueError("知识库文档为空，无法建立 BM25 索引")

        self.vector_store = vector_store
        self.documents = documents
        self.k = k
        self.candidate_k = max(candidate_k, k)
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k

        tokenized_documents = [
            tokenize_for_search(document.page_content) or ["_empty_"]
            for document in documents
        ]
        self.bm25 = BM25Okapi(tokenized_documents)  # BM25Okapi():BM25Okapi 索引，用于 BM25 检索。

    def search_bm25(
        self,
        query: str,
        k: int | None = None,
    ) -> list[Document]:
        """只执行 BM25 检索，便于单独测试。"""
        query_tokens = tokenize_for_search(query)
        if not query_tokens:
            return []

        top_n = min(k or self.candidate_k, len(self.documents))
        scores = self.bm25.get_scores(query_tokens)
        # get_scores():返回BM25 索引中每个文档的 BM25 分数。

        ranked_indices = sorted(
            range(len(scores)),  # 生成文档索引列表
            key=lambda index: scores[index],  # 按 BM25 分数排序
            reverse=True,  # 从高到低排序
        )[:top_n]

        return [self.documents[index] for index in ranked_indices if scores[index] > 0]

    def invoke(self, query: str) -> list[Document]:
        """执行混合检索，接口与原来的 retriever.invoke() 保持一致。"""
        candidate_k = min(self.candidate_k, len(self.documents))

        try:
            vector_documents = self.vector_store.similarity_search(
                query,
                k=candidate_k,
            ) 
        except Exception as exc:
            logger.error(f"[混合检索]向量召回失败，降级为 BM25：{exc}")
            vector_documents = []

        bm25_documents = self.search_bm25(query, candidate_k)

        fused_scores: dict[str, float] = defaultdict(float)
        documents_by_key: dict[str, Document] = {}

        for weight, documents in (
            (self.vector_weight, vector_documents),
            (self.bm25_weight, bm25_documents),
        ):
            for rank, document in enumerate(documents, start=1):  # enumerate():返回文档索引和文档对象,从1开始
                key = document_key(document)
                documents_by_key[key] = document
                fused_scores[key] += weight / (self.rrf_k + rank)

        ranked_keys = sorted(
            fused_scores,  # 融合分数
            key=lambda key: fused_scores[key],  # 按融合分数排序
            reverse=True,  # 从高到低排序
        )

        return [documents_by_key[key] for key in ranked_keys[: self.k]]
