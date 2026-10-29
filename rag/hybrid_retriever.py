"""Chroma 向量检索与 BM25 关键词检索的混合召回（增强版）。

包含以下改进：
1. 交叉编码器精排：在混合检索结果上进行语义精排
2. 动态权重调整：根据问题类型动态调整向量/BM25权重
"""

import re
import time
from collections import defaultdict
import jieba
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from utils.logger_handler import logger
from rag.cross_encoder_reranker import get_reranker
from rag.dynamic_weight_adjuster import get_adjuster

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
    """把向量召回和 BM25 召回按排名融合，并进行重排和权重优化。

    改进：
    1. 动态权重：根据问题类型自动调整向量/BM25的权重
    2. 交叉编码器重排：使用Cross-Encoder对结果进行精准重排
    """

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
        enable_reranking: bool = True,
        enable_dynamic_weight: bool = True,
        reranker_top_k: int = 20  # 重排前的候选数量
    ):
        if not documents:
            raise ValueError("知识库文档为空，无法建立 BM25 索引")

        self.vector_store = vector_store
        self.documents = documents
        self.k = k  # 最终返回的文档数
        self.candidate_k = max(candidate_k, k)
        self.default_vector_weight = vector_weight
        self.default_bm25_weight = bm25_weight
        self.rrf_k = rrf_k
        self.enable_reranking = enable_reranking
        self.enable_dynamic_weight = enable_dynamic_weight
        self.reranker_top_k = max(reranker_top_k, k)

        tokenized_documents = [
            tokenize_for_search(document.page_content) or ["_empty_"]
            for document in documents
        ]
        self.bm25 = BM25Okapi(tokenized_documents)

        # 初始化增强模块
        self.adjuster = get_adjuster(enable_dynamic_weight) if enable_dynamic_weight else None
        self.reranker = get_reranker() if enable_reranking else None

        logger.info(
            f"[HybridRetriever] 已初始化 "
            f"| 动态权重: {enable_dynamic_weight} "
            f"| 交叉编码器重排: {enable_reranking}"
        )

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
        """
        执行混合检索，包含重排和动态权重。

        流程：
        1. 根据查询类型调整权重（如启用）
        2. 执行混合检索（向量 + BM25）
        3. 使用交叉编码器重排（如启用）
        4. 返回top-k结果
        """
        # 第1步：调整权重
        vector_weight = self.default_vector_weight
        bm25_weight = self.default_bm25_weight

        if self.adjuster:
            vector_weight, bm25_weight = self.adjuster.adjust_weights(query)

        candidate_k = min(self.candidate_k, len(self.documents))

        # 第2步：执行混合检索
        try:
            vector_documents = self.vector_store.similarity_search(
                query,
                k=candidate_k,
            )
        except Exception as exc:
            logger.error(f"[混合检索]向量召回失败，降级为 BM25：{exc}")
            vector_documents = []

        bm25_documents = self.search_bm25(query, candidate_k)

        # 融合向量和BM25结果
        fused_scores: dict[str, float] = defaultdict(float)
        documents_by_key: dict[str, Document] = {}

        for weight, documents in (
            (vector_weight, vector_documents),
            (bm25_weight, bm25_documents),
        ):
            for rank, document in enumerate(documents, start=1):
                key = document_key(document)
                documents_by_key[key] = document
                fused_scores[key] += weight / (self.rrf_k + rank)

        ranked_keys = sorted(
            fused_scores,
            key=lambda key: fused_scores[key],
            reverse=True,
        )

        # 获取融合结果（候选集）
        fusion_candidates = [documents_by_key[key] for key in ranked_keys[:self.reranker_top_k]]

        # 第3步：交叉编码器重排（可选）
        if self.reranker and fusion_candidates:
            reranked_docs, scores = self.reranker.rerank(
                query,
                fusion_candidates,
                top_k=self.k
            )

            logger.debug(
                f"[混合检索] 重排完成: {len(fusion_candidates)} → {len(reranked_docs)} | "
                f"权重: vector={vector_weight:.2f}, bm25={bm25_weight:.2f}"
            )

            return reranked_docs
        else:
            # 如果未启用重排，直接返回融合结果的top-k
            return fusion_candidates[:self.k]
