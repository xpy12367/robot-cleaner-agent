"""
查询改写功能测试
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.query_rewriter import get_rewriter


def test_query_rewrite():
    """测试查询改写功能"""
    print("=" * 80)
    print("查询改写功能测试")
    print("=" * 80)

    rewriter = get_rewriter()

    if not rewriter:
        print("[错误] 查询改写器未启用")
        return

    # 获取统计信息
    stats = rewriter.get_stats()
    print(f"\n配置信息:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # 测试用例
    test_queries = [
        "宠物家庭适合什么扫地机？",
        "扫地机噪音大怎么办？",
        "尘盒怎样清理？",
        "小户型推荐哪种型号？",
        "拖布有异味怎么处理？",
    ]

    print("\n" + "=" * 80)
    print("改写结果测试")
    print("=" * 80)

    for i, query in enumerate(test_queries, 1):
        print(f"\n[测试 {i}] 原始查询: {query}")
        rewritten = rewriter.rewrite(query)

        print(f"  改写结果 ({len(rewritten)} 个):")
        for j, q in enumerate(rewritten, 1):
            marker = "[原]" if j == 1 else "[改]"
            print(f"    {marker} {j}. {q}")


if __name__ == "__main__":
    test_query_rewrite()
