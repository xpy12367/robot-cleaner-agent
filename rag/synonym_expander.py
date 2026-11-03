"""
同义词扩展模块 - BM25查询增强

在BM25分词后自动补充同义词，提升召回率。
解决用户用词与知识库用词不一致的问题。

例如：
  用户问："抹布怎么清洗？"
  分词：["抹布", "怎么", "清洗"]
  扩展：["抹布", "拖布", "拖地布", "怎么", "清洗"]  # 补充了"拖布"和"拖地布"
  检索：能命中知识库中"拖布清洗方法"的文档
"""
import os
import sys
from typing import List, Set

# 支持直接运行测试
if __name__ == "__main__":
    # 添加项目根目录到路径
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger_handler import logger


class SynonymExpander:
    """同义词扩展器"""

    def __init__(self, synonym_file: str = None):
        """
        初始化同义词扩展器

        Args:
            synonym_file: 同义词文件路径，默认为 config/synonyms.txt
        """
        if synonym_file is None:
            # 默认路径：项目根目录/config/synonyms.txt
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            synonym_file = os.path.join(project_root, "config", "synonyms.txt")

        self.synonym_file = synonym_file
        # 同义词映射：{词: [该词所在的所有同义词组]}
        self.synonym_map: dict[str, List[str]] = {}
        self._load_synonyms()

    def _load_synonyms(self):
        """从文件加载同义词表"""
        if not os.path.exists(self.synonym_file):
            logger.warning(
                f"[SynonymExpander] 同义词文件不存在: {self.synonym_file}，同义词扩展将被禁用"
            )
            return

        try:
            with open(self.synonym_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):  # 从1开始编号
                    line = line.strip()

                    # 跳过空行和注释
                    if not line or line.startswith("#"):
                        continue

                    # 解析同义词组
                    synonyms = [word.strip() for word in line.split(",") if word.strip()]

                    if len(synonyms) < 2:
                        logger.warning(
                            f"[SynonymExpander] 第{line_num}行同义词少于2个，已跳过: {line}"
                        )
                        continue

                    # 每个词都映射到整个同义词组
                    for word in synonyms:
                        if word not in self.synonym_map:
                            self.synonym_map[word] = synonyms  # 每个词 → 整组
                        else:
                            # 合并：如果一个词出现在多个同义词组，取并集
                            existing = set(
                                self.synonym_map[word]  # set(): 转换为集合，去重
                            )  # ② 以前见过：取出旧组
                            existing.update(synonyms)  #    和新组合并（取并集）
                            merged = list(existing)
                            for w in merged:
                                self.synonym_map[w] = merged  # ③ 把并集写回组内每一个词

            logger.info(
                f"[SynonymExpander] 已加载 {len(self.synonym_map)} 个词的同义词映射"
            )

        except Exception as e:
            logger.error(f"[SynonymExpander] 加载同义词文件失败: {e}")

    def expand(self, tokens: List[str]) -> List[str]:
        """
        扩展分词结果，补充同义词

        Args:
            tokens: 原始分词列表

        Returns:
            扩展后的分词列表（去重）

        示例：
            ["抹布", "清洗"] → ["抹布", "拖布", "拖地布", "清洗"]
        """
        if not self.synonym_map:
            return tokens

        expanded: Set[str] = set(tokens)  # 用集合自动去重

        for token in tokens:
            if token in self.synonym_map:
                # 添加该词的所有同义词
                expanded.update(self.synonym_map[token])

        result = list(expanded)

        if len(result) > len(tokens):
            logger.debug(
                f"[SynonymExpander] 扩展: {len(tokens)}词 → {len(result)}词 "
                f"(新增: {', '.join(set(result) - set(tokens))})"
            )

        return result

    def get_synonyms(self, word: str) -> List[str]:
        """获取指定词的同义词组（包含该词自己）"""
        return self.synonym_map.get(word, [word])

    def get_stats(self) -> dict:
        """获取同义词表统计信息"""
        if not self.synonym_map:
            return {"status": "disabled", "word_count": 0, "group_count": 0}

        # 统计同义词组数量（通过去重 id(list) 来计数不同的组）
        unique_groups = {id(synonyms): synonyms for synonyms in self.synonym_map.values()}

        return {
            "status": "loaded",
            "word_count": len(self.synonym_map),  # 覆盖的词数
            "group_count": len(unique_groups),  # 同义词组数
            "file": self.synonym_file,
        }


# 全局单例
_expander_instance: SynonymExpander = None


def get_expander() -> SynonymExpander:
    """获取全局同义词扩展器实例（单例模式）"""
    global _expander_instance
    if _expander_instance is None:
        _expander_instance = SynonymExpander()
    return _expander_instance


if __name__ == "__main__":
    # 测试
    expander = get_expander()
    stats = expander.get_stats()

    print("=" * 60)
    print("同义词扩展器统计")
    print("=" * 60)
    for key, value in stats.items():
        print(f"{key}: {value}")

    print("\n" + "=" * 60)
    print("扩展测试")
    print("=" * 60)

    test_cases = [
        ["抹布", "怎么", "清洗"],
        ["噪音", "大"],
        ["卡住", "怎么办"],
        ["电量", "不足"],
        ["没有", "匹配"],  # 无同义词的情况
    ]

    for tokens in test_cases:
        expanded = expander.expand(tokens)
        print(f"\n原始: {tokens}")
        print(f"扩展: {expanded}")
