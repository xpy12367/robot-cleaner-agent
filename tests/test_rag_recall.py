"""
RAG 召回率对比测试

对比启用/禁用查询改写的召回效果
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.rag_service import RagSummarizeService
from rag.query_rewriter import get_rewriter
from utils.logger_handler import logger


def test_recall_comparison():
    """对比测试：启用 vs 禁用查询改写"""
    print("=" * 80)
    print("RAG 召回率对比测试")
    print("=" * 80)

    # 测试问题集（涵盖不同类型）
    test_questions = [
        "宠物家庭适合什么扫地机？",
        "扫地机噪音突然变大怎么办？",
        "拖布清洗后有异味如何处理？",
        "小户型怎样选择扫地机器人？",
        "机器原地打转是什么原因？",
        "续航时间明显缩短怎么办？",
        "地毯上拖地要注意什么？",
        "悬崖传感器报错如何处理？",
    ]

    # 初始化服务
    rag_service = RagSummarizeService()

    print(f"\n查询改写状态: {'✓ 已启用' if rag_service.rewriter else '✗ 未启用'}")
    print(f"测试问题数量: {len(test_questions)}")

    # 执行测试
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'=' * 80}")
        print(f"[问题 {i}/{len(test_questions)}] {question}")
        print("=" * 80)

        try:
            # 检索文档
            docs = rag_service.retriever_docs(question)

            print(f"\n召回文档数: {len(docs)}")

            # 显示前3个文档的摘要
            for j, doc in enumerate(docs[:3], 1):
                content = doc.page_content[:100].replace("\n", " ")
                source = doc.metadata.get("source", "未知")[-30:]
                print(f"\n  [{j}] 来源: {source}")
                print(f"      内容: {content}...")

                if hasattr(doc, "metadata") and "score" in doc.metadata:
                    print(f"      分数: {doc.metadata['score']:.3f}")

        except Exception as e:
            print(f"\n❌ 检索失败: {e}")
            logger.error(f"检索失败: {question} - {e}")

    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)


def test_full_rag_pipeline():
    """完整RAG流程测试（包含生成）"""
    print("\n" + "=" * 80)
    print("完整 RAG 流程测试")
    print("=" * 80)

    test_questions = [
        "宠物家庭推荐什么扫地机？",
        "扫地机噪音大怎么办？",
        "拖布怎样清洗？",
    ]

    rag_service = RagSummarizeService()

    for i, question in enumerate(test_questions, 1):
        print(f"\n{'=' * 80}")
        print(f"[问题 {i}] {question}")
        print("=" * 80)

        try:
            answer = rag_service.Rag_summarize(question)
            print(f"\n回答:\n{answer}")
        except Exception as e:
            print(f"\n❌ 生成失败: {e}")
            logger.error(f"生成失败: {question} - {e}")


if __name__ == "__main__":
    # 召回率对比测试
    test_recall_comparison()

    # 完整流程测试（可选，会消耗 API tokens）
    print("\n是否执行完整 RAG 流程测试（包含 LLM 生成）? [y/N]: ", end="")
    choice = input().strip().lower()
    if choice == "y":
        test_full_rag_pipeline()
    else:
        print("\n跳过完整流程测试")
