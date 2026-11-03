"""
查询改写模块 - Query Rewriting

将用户单一问题改写为多个角度的查询变体,提升检索召回全面性。

目标场景:
  用户问："宠物家庭适合什么扫地机？"

  改写输出:
    1. 宠物家庭适合什么扫地机？         (原查询)
    2. 扫地机器人如何清理宠物毛发？     (聚焦功能)
    3. 宠物养家推荐哪种扫地机？         (同义表达)

实现策略:
  1. 少样本提示词 (Few-shot Prompt): 给出改写规则和示例
  2. LLM 生成多个改写查询(默认3个)
  3. 自动过滤无效、重复或与原查询相似度过高的变体
"""
import os
import sys
import time
from typing import List, Optional

# 支持直接运行测试
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from model.factory import chat_model
from utils.logger_handler import logger
from utils.config_handler import rag_config


class QueryRewriter:
    """查询改写器 - 将单一问题改写为多个检索变体"""

    def __init__(
        self,
        rewrite_count: int = 3,
        enable_similarity_filter: bool = True,
        similarity_threshold: float = 0.85,
    ):
        """
        初始化查询改写器

        Args:
            rewrite_count: 改写数量(默认3个)
            enable_similarity_filter: 是否启用相似度过滤
            similarity_threshold: 相似度阈值,超过此值的变体会被过滤
        """
        self.rewrite_count = rewrite_count
        self.enable_similarity_filter = enable_similarity_filter
        self.similarity_threshold = similarity_threshold
        self.model = chat_model
        self.prompt_template = self._build_prompt()
        self.chain = self.prompt_template | self.model | StrOutputParser()

        logger.info(
            f"[QueryRewriter] 已初始化 | 改写数量: {rewrite_count} | "
            f"相似度过滤: {enable_similarity_filter}"
        )

    def _build_prompt(self) -> PromptTemplate:
        """构建改写提示词"""
        # 从配置读取改写提示词路径
        prompt_path = rag_config.get("query_rewrite_prompt_path")
        if not prompt_path:
            raise RuntimeError(
                "[QueryRewriter] 配置文件中缺少 query_rewrite_prompt_path"
            )

        # 获取绝对路径
        if not os.path.isabs(prompt_path):  # os.path.isabs() 检查路径是否为绝对路径
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            prompt_path = os.path.join(project_root, prompt_path)

        # 读取提示词模板
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                template_text = f.read()
        except Exception as e:
            raise RuntimeError(f"[QueryRewriter] 读取改写提示词失败: {e}")

        return PromptTemplate.from_template(template_text)

    def _parse_llm_output(self, llm_output: str) -> List[str]:
        """
        解析 LLM 输出的改写结果

        期望格式:
        1. 查询1
        2. 查询2
        3. 查询3

        Returns:
            改写查询列表
        """
        queries = []
        for line in llm_output.strip().split("\n"):
            line = line.strip()
            # 跳过空行
            if not line:
                continue

            # 移除序号前缀 (1. / 1、/ 1) / 一、等)
            if line[0].isdigit() or line[0] in "一二三四五六七八九十":  #isdigit() 检查字符是否为数字
                # 找到第一个非序号字符
                for i, ch in enumerate(line):  # enumerate() 拿到每个字符和它的位置
                    if ch not in "0123456789一二三四五六七八九十.、) ":
                        line = line[i:]
                        break

            if line and len(line) >= 2:  # 过滤过短的结果
                queries.append(line)

        return queries

    def _calculate_similarity(self, query1: str, query2: str) -> float:
        """
        计算两个查询的字符级相似度 (Jaccard相似度)

        Args:
            query1: 查询1
            query2: 查询2

        Returns:
            相似度分数 [0, 1]
        """
        set1 = set(query1)  # set() 转换为集合, 去重
        set2 = set(query2)

        if not set1 or not set2:
            return 0.0

        intersection = len(set1 & set2)  # & 求交集
        union = len(set1 | set2)  # | 求并集

        return intersection / union if union > 0 else 0.0

    def _filter_similar_queries(
        self, original_query: str, rewritten_queries: List[str]
    ) -> List[str]:
        """
        过滤与原查询过于相似的改写结果

        Args:
            original_query: 原始查询
            rewritten_queries: 改写查询列表

        Returns:
            过滤后的查询列表
        """
        if not self.enable_similarity_filter:
            return rewritten_queries

        filtered = []
        for query in rewritten_queries:
            similarity = self._calculate_similarity(original_query, query)

            if similarity < self.similarity_threshold:
                filtered.append(query)
            else:
                logger.debug(
                    f"[QueryRewriter] 过滤高相似度变体(相似度{similarity:.2f}): {query}"
                )

        return filtered

    def rewrite(self, query: str) -> List[str]:
        """
        改写查询

        Args:
            query: 原始查询

        Returns:
            包含原查询和改写变体的列表

        示例:
            输入: "宠物家庭适合什么扫地机？"
            输出: [
                "宠物家庭适合什么扫地机？",
                "扫地机器人如何清理宠物毛发？",
                "宠物养家推荐哪种扫地机？"
            ]
        """
        start_time = time.time()

        try:
            # 调用 LLM 改写
            llm_output = self.chain.invoke(
                {"query": query, "count": self.rewrite_count}
            )

            # 解析输出
            rewritten_queries = self._parse_llm_output(llm_output)

            # 过滤相似查询
            rewritten_queries = self._filter_similar_queries(query, rewritten_queries)

            # 组合: 原查询 + 改写变体
            all_queries = [query] + rewritten_queries

            # 去重 (保持顺序)
            seen = set()  # 用于记录已见查询, 避免重复添加
            unique_queries = []
            for q in all_queries:
                if q not in seen:
                    seen.add(q)
                    unique_queries.append(q)

            duration = time.time() - start_time

            logger.debug(
                f"[QueryRewriter] 改写完成: 1 → {len(unique_queries)}个查询 | "
                f"耗时{duration:.3f}秒"
            )

            return unique_queries

        except Exception as e:
            logger.error(f"[QueryRewriter] 改写失败: {e}, 返回原查询")
            return [query]  # 降级:返回原查询

    def get_stats(self) -> dict:
        """获取改写器统计信息"""
        return {
            "rewrite_count": self.rewrite_count,
            "similarity_filter": self.enable_similarity_filter,
            "similarity_threshold": self.similarity_threshold,
            "model": (str(self.model.model_name) if hasattr(self.model, "model_name") else "unknown"),
            }  # hasattr() 检查模型是否有model_name属性


# 全局单例
_rewriter_instance: Optional[QueryRewriter] = None  
# Optional[QueryRewriter] 表示可选的QueryRewriter实例, 初始值为None


def get_rewriter(enabled: bool = True) -> Optional[QueryRewriter]:
    """
    获取全局查询改写器实例 (单例模式)

    Args:
        enabled: 是否启用改写

    Returns:
        改写器实例,如果禁用则返回None
    """
    global _rewriter_instance

    if not enabled:
        return None

    if _rewriter_instance is None:
        _rewriter_instance = QueryRewriter(
            rewrite_count=rag_config.get("query_rewrite_count", 3),
            enable_similarity_filter=rag_config.get(
                "query_rewrite_similarity_filter", True
            ),
            similarity_threshold=rag_config.get("query_rewrite_similarity_threshold", 0.85),
        )

    return _rewriter_instance


if __name__ == "__main__":
    # 测试
    rewriter = get_rewriter()

    if rewriter:
        stats = rewriter.get_stats()
        print("=" * 60)
        print("查询改写器统计")
        print("=" * 60)
        for key, value in stats.items():
            print(f"{key}: {value}")

        print("\n" + "=" * 60)
        print("改写测试")
        print("=" * 60)

        test_queries = [
            "宠物家庭适合什么扫地机？",
            "扫地机噪音大怎么办？",
            "如何清理尘盒？",
        ]

        for query in test_queries:
            print(f"\n原始查询: {query}")
            rewritten = rewriter.rewrite(query)
            for i, q in enumerate(rewritten, 1):
                print(f"  {i}. {q}")
    else:
        print("改写器未启用")
