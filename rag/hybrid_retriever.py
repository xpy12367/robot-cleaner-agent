"""Chroma 向量检索与 BM25 关键词检索的混合召回（增强版）。

包含以下改进：
1. 交叉编码器精排：在混合检索结果上进行语义精排
2. 动态权重调整：根据问题类型动态调整向量/BM25权重
3. 同义词扩展：自动补充领域同义词，提升BM25召回率
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
from rag.synonym_expander import get_expander

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
    3. 同义词扩展：自动补充领域同义词，提升BM25召回率
    """

    def __init__(
        self,
        vector_store,
        documents: list[Document],
        *,
        k: int = 3,  # 最终返回的文档数
        candidate_k: int = 10,  # 初始候选文档数
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        rrf_k: int = 60,
        enable_reranking: bool = True,
        enable_dynamic_weight: bool = True,
        enable_synonym_expansion: bool = True,
        reranker_top_k: int = 20
    ):
        if not documents:
            raise ValueError("知识库文档为空，无法建立 BM25 索引")

        self.vector_store = vector_store  # 向量存储
        self.documents = documents  # 知识库文档
        self.k = k  # 最终返回的文档数
        self.candidate_k = max(candidate_k, k)  # 初始候选文档数
        self.default_vector_weight = vector_weight  # 向量权重默认值
        self.default_bm25_weight = bm25_weight  # BM25 权重默认值
        self.rrf_k = rrf_k
        self.enable_reranking = enable_reranking  # 是否启用交叉编码器重排
        self.enable_dynamic_weight = enable_dynamic_weight  # 是否启用动态权重调整
        self.enable_synonym_expansion = enable_synonym_expansion  # 是否启用同义词扩展
        self.reranker_top_k = max(reranker_top_k, k)   # 重排前的候选数量

        tokenized_documents = [
            tokenize_for_search(document.page_content) or ["_empty_"]
            for document in documents
        ]
        self.bm25 = BM25Okapi(tokenized_documents)  # 初始化 BM25 索引

        # 初始化增强模块
        self.adjuster = get_adjuster(enable_dynamic_weight) if enable_dynamic_weight else None
        self.reranker = get_reranker() if enable_reranking else None
        self.expander = get_expander() if enable_synonym_expansion else None

        logger.info(
            f"[HybridRetriever] 已初始化 "
            f"| 动态权重: {enable_dynamic_weight} "
            f"| 交叉编码器重排: {enable_reranking} "
            f"| 同义词扩展: {enable_synonym_expansion}"
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

        # 同义词扩展（可选）
        if self.expander:
            query_tokens = self.expander.expand(query_tokens)

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
            for rank, document in enumerate(documents, start=1):  # 从1开始编号
                key = document_key(document)
                documents_by_key[key] = document
                fused_scores[key] += weight / (self.rrf_k + rank)  # 计算融合分数

        ranked_keys = sorted(
            fused_scores,  # ① 传入字典
            key=lambda key: fused_scores[key],  # ② 排序依据：用 key 去查分数
            reverse=True,  # ③ 降序
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

    def invoke_multi(self, queries: list[str]) -> list[Document]:
        """
        执行多查询融合检索

        对多个查询分别执行混合检索，然后对所有结果再次融合排序。

        Args:
            queries: 查询列表（通常来自查询改写）

        Returns:
            融合后的top-k文档列表

        流程：
        1. 对每个查询执行混合检索
        2. 收集所有候选文档
        3. 按出现频次和平均排名融合
        4. 使用交叉编码器重排（以第一个查询为基准）
        """
        if not queries:
            return []

        # 单查询降级为普通invoke
        if len(queries) == 1:
            return self.invoke(queries[0])

        # 第1步：对每个查询执行混合检索，收集候选文档
        all_candidates: dict[str, list[int]] = defaultdict(list)  # {doc_key: [rank1, rank2, ...]}
        documents_by_key: dict[str, Document] = {}

        for query in queries:
            # 调用invoke获取该查询的top-k结果
            query_results = self.invoke(query)

            # 记录每个文档的排名（1-based）
            for rank, doc in enumerate(query_results, start=1):
                # key: 文档唯一标识
                key = document_key(doc)
                documents_by_key[key] = doc
                all_candidates[key].append(rank)

        # 第2步：按出现频次和平均排名融合排序
        # 计算每个文档的融合分数：出现次数 × 平均倒数排名
        fusion_scores: dict[str, float] = {}
        for key, ranks in all_candidates.items():
            # 出现频次
            frequency = len(ranks)
            # 平均倒数排名（RRF风格）
            avg_reciprocal_rank = sum(1.0 / (self.rrf_k + r) for r in ranks) / len(ranks)
            # 融合分数
            fusion_scores[key] = frequency * avg_reciprocal_rank

        # 按融合分数排序
        ranked_keys = sorted(
            fusion_scores,
            key=lambda k: fusion_scores[k],
            reverse=True,
        )  # ranked_keys: 按融合分数排序的文档key列表

        # 取top-reranker_top_k作为候选
        fusion_candidates = [
            documents_by_key[key] for key in ranked_keys[:self.reranker_top_k]
        ]

        # 第3步：交叉编码器重排（以第一个查询为基准）
        if self.reranker and fusion_candidates:
            reranked_docs, scores = self.reranker.rerank(
                queries[0],  # 使用原始查询作为重排基准
                fusion_candidates,
                top_k=self.k
            )

            logger.debug(
                f"[多查询融合] 完成: {len(queries)}个查询 → {len(fusion_candidates)}候选 → {len(reranked_docs)}结果"
            )

            return reranked_docs
        else:
            # 如果未启用重排，直接返回融合结果的top-k
            return fusion_candidates[:self.k]
