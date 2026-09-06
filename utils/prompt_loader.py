from utils.config_handler import prompts_config
from utils.path_tool import get_abs_path
from utils.logger_handler import logger


def load_system_prompts():
    """
    加载系统提示词
    """
    # 获取系统提示词路径
    try:
        system_prompt_path = get_abs_path(prompts_config["main_prompt_path"])   #prompts_config["main_prompt_path"]  # 获取key("main_prompt_path")对应的value(路径)
    except KeyError as e:
        logger.error(f"[load_system_prompts]在yaml配置项中没有main_prompt_path配置项")
        raise e   #raise: 抛出异常

    # 读取系统提示词
    try:
        return open(system_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_system_prompts]解析系统提示词出错，{str(e)}")
        raise e

def load_rag_prompts():
    """
    加载RAG总结提示词
    """
    # 获取系统提示词路径
    try:
        rag_prompt_path = get_abs_path(prompts_config["rag_summarize_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_rag_prompts]在yaml配置项中没有main_prompt_path配置项")
        raise e   #raise: 抛出异常

    # 读取系统提示词
    try:
        return open(rag_prompt_path, "r", encoding="utf-8").read()  #read():
    except Exception as e:
        logger.error(f"[load_rag_prompts]解析RAG总结提示词出错，{str(e)}")
        raise e

def load_report_prompts():
    """
    加载报告提示词
    """
    # 获取系统提示词路径
    try:
        report_prompt_path = get_abs_path(prompts_config["report_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_report_prompts]在yaml配置项中没有report_prompt_path配置项")
        raise e   #raise: 抛出异常

    # 读取系统提示词
    try:
        return open(report_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_report_prompts]解析报告提示词出错，{str(e)}")
        raise e

if __name__=="__main__":
    # print(load_report_prompts())
    print(load_rag_prompts())