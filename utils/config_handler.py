"""
yaml
key:value
"""
import yaml
from utils.path_tool import get_abs_path
from dotenv import load_dotenv

load_dotenv(dotenv_path=get_abs_path(".env"), override=False)

def load_rag_config(config_path:str=get_abs_path("config/rag.yaml"),encoding:str="utf-8"):
    """
    加载rag配置文件
    :param config_path: 配置文件路径
    :param encoding: 编码方式
    :return: 配置文件内容
    """
    with open(config_path,"r",encoding="utf-8") as f:
        return yaml.load(f,Loader=yaml.FullLoader)  #加载整个配置文件内容

def load_chroma_config(config_path:str=get_abs_path("config/chroma.yaml"),encoding:str="utf-8"):
    """
    加载chroma配置文件
    :param config_path: 配置文件路径
    :param encoding: 编码方式
    :return: 配置文件内容
    """
    with open(config_path,"r",encoding="utf-8") as f:
        return yaml.load(f,Loader=yaml.FullLoader)  #加载整个配置文件内容

def load_prompts_config(config_path:str=get_abs_path("config/prompts.yaml"),encoding:str="utf-8"):
    """
        加载prompts配置文件
        :param config_path: 配置文件路径
        :param encoding: 编码方式
        :return: 配置文件内容
        """
    with open(config_path,"r",encoding="utf-8") as f:
        return yaml.load(f,Loader=yaml.FullLoader)  #加载整个配置文件内容

def load_agent_config(config_path:str=get_abs_path("config/agent.yaml"),encoding:str="utf-8"):
    """
    加载agent配置文件
    :param config_path: 配置文件路径
    :param encoding: 编码方式
    :return: 配置文件内容
    """
    with open(config_path,"r",encoding="utf-8") as f:
        return yaml.load(f,Loader=yaml.FullLoader)  #加载整个配置文件内容

rag_config=load_rag_config()
chroma_config=load_chroma_config()
prompts_config=load_prompts_config()
agent_config=load_agent_config()

if __name__ == "__main__":
    print(agent_config["chat_model_name"])
