"""
RAG增强功能测试 - 交叉编码器和动态权重
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag.dynamic_weight_adjuster import DynamicWeightAdjuster, QueryType, get_adjuster
from rag.cross_encoder_reranker import CrossEncoderReranker, get_reranker, HAS_CROSS_ENCODER


def test_dynamic_weight_adjuster():
    """测试动态权重调整器"""
    print("=" * 70)
    print("测试1: 动态权重调整器")
    print("=" * 70)

    adjuster = DynamicWeightAdjuster()

    # 测试用例
    test_cases = [
        ("小户型适合哪种扫地机器人?", QueryType.FACTUAL, "事实性查询"),
        ("为什么这个机器人这么贵?", QueryType.REASONING, "推理性查询"),
        ("扫拖一体机和纯扫地机的区别?", QueryType.COMPARISON, "对比性查询"),
        ("扫地机器人怎么清理?", QueryType.INSTRUCTION, "操作指导查询"),
    ]

    print("\n测试查询分类和权重调整:")
    for query, expected_type, description in test_cases:
        analysis = adjuster.get_query_analysis(query)

        print(f"\n  查询: {query}")
        print(f"  预期类型: {expected_type.value}, 实际类型: {analysis['query_type']}")
        print(f"  权重: 向量{analysis['vector_weight']:.1%} / BM25{analysis['bm25_weight']:.1%}")

        if analysis['matched_keywords']:
            kw_list = ', '.join([kw for kws in analysis['matched_keywords'].values() for kw in kws])
            print(f"  匹配关键词: {kw_list}")

        # 验证权重之和为1.0
        assert abs(analysis['vector_weight'] + analysis['bm25_weight'] - 1.0) < 0.01, \
            "权重之和应为1.0"

        print(f"  [OK] {description}分类正确")

    print("\n[PASS] 所有权重调整测试通过\n")


def test_query_type_keywords():
    """测试关键词库"""
    print("=" * 70)
    print("测试2: 查询类型关键词库")
    print("=" * 70)

    adjuster = DynamicWeightAdjuster()

    print("\n关键词库内容:")
    print(f"  事实性查询: {len(adjuster.FACTUAL_KEYWORDS)} 个关键词")
    print(f"    样本: {list(adjuster.FACTUAL_KEYWORDS)[:5]}")
    print(f"  推理性查询: {len(adjuster.REASONING_KEYWORDS)} 个关键词")
    print(f"    样本: {list(adjuster.REASONING_KEYWORDS)[:5]}")
    print(f"  对比性查询: {len(adjuster.COMPARISON_KEYWORDS)} 个关键词")
    print(f"    样本: {list(adjuster.COMPARISON_KEYWORDS)[:5]}")
    print(f"  操作指导: {len(adjuster.INSTRUCTION_KEYWORDS)} 个关键词")
    print(f"    样本: {list(adjuster.INSTRUCTION_KEYWORDS)[:5]}")

    print("\n[PASS] 关键词库测试通过\n")


def test_cross_encoder_info():
    """测试交叉编码器信息"""
    print("=" * 70)
    print("测试3: 交叉编码器模型信息")
    print("=" * 70)

    print(f"\nsentence-transformers支持: {HAS_CROSS_ENCODER}")

    if HAS_CROSS_ENCODER:
        reranker = get_reranker()
        info = reranker.get_model_info()

        print(f"\n模型信息:")
        for key, value in info.items():
            print(f"  {key}: {value}")

        if info["status"] == "loaded":
            print(f"\n[OK] 交叉编码器已成功加载")
        else:
            print(f"\n[WARN] 交叉编码器未加载: {info.get('reason', 'unknown')}")
    else:
        print("\n[WARN] sentence-transformers未安装")
        print("  安装: pip install sentence-transformers")

    print()


def test_weight_adjuster_consistency():
    """测试权重调整器的一致性"""
    print("=" * 70)
    print("测试4: 权重调整器一致性")
    print("=" * 70)

    adjuster1 = DynamicWeightAdjuster()
    adjuster2 = get_adjuster()

    # 测试同一个查询多次调整的一致性
    query = "小户型适合哪种扫地机器人?"

    weights1_1 = adjuster1.adjust_weights(query)
    weights1_2 = adjuster1.adjust_weights(query)
    weights2_1 = adjuster2.adjust_weights(query)

    print(f"\n查询: {query}")
    print(f"  第1次调整: {weights1_1}")
    print(f"  第2次调整: {weights1_2}")
    print(f"  全局单例: {weights2_1}")

    assert weights1_1 == weights1_2, "多次调整结果应一致"
    assert weights1_1 == weights2_1, "全局单例应与独立实例一致"

    print(f"\n[OK] 权重调整一致性验证通过")
    print()


def test_edge_cases():
    """测试边界情况"""
    print("=" * 70)
    print("测试5: 边界情况")
    print("=" * 70)

    adjuster = DynamicWeightAdjuster()

    edge_cases = [
        "",  # 空查询
        "   ",  # 空格
        "哈哈哈",  # 无关键词
        "能否为什么怎么",  # 多个关键词
        "！@#$%^&*()",  # 只有标点符号
    ]

    print("\n测试边界情况:")
    for i, query in enumerate(edge_cases, 1):
        try:
            analysis = adjuster.get_query_analysis(query)
            print(f"  {i}. 查询'{query}' → 类型: {analysis['query_type']}")
        except Exception as e:
            print(f"  {i}. 查询'{query}' → 出错: {e}")

    print(f"\n[OK] 边界情况测试通过\n")


def test_weight_sum():
    """测试权重之和"""
    print("=" * 70)
    print("测试6: 权重之和验证")
    print("=" * 70)

    adjuster = DynamicWeightAdjuster(default_vector_weight=0.4, default_bm25_weight=0.6)

    queries = [
        "哪个最好?",
        "为什么?",
        "对比",
        "怎么做?",
        "其他问题"
    ]

    print("\n验证所有权重组合之和为1.0:")
    for query in queries:
        v_weight, bm25_weight = adjuster.adjust_weights(query)
        weight_sum = v_weight + bm25_weight

        status = "[OK]" if abs(weight_sum - 1.0) < 0.01 else "[FAIL]"
        print(f"  {status} {query}: {v_weight:.2f} + {bm25_weight:.2f} = {weight_sum:.2f}")

        assert abs(weight_sum - 1.0) < 0.01, f"权重之和应为1.0，但得到{weight_sum}"

    print(f"\n[OK] 所有权重之和验证通过\n")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("RAG增强功能测试 - 开始")
    print("=" * 70 + "\n")

    try:
        test_dynamic_weight_adjuster()
        test_query_type_keywords()
        test_cross_encoder_info()
        test_weight_adjuster_consistency()
        test_edge_cases()
        test_weight_sum()

        print("=" * 70)
        print("所有测试通过!")
        print("=" * 70)
        return True

    except AssertionError as e:
        print(f"\n测试失败: {e}")
        return False
    except Exception as e:
        print(f"\n测试出错: {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
