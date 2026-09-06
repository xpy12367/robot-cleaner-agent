import os
import hashlib
from utils.logger_handler import logger
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader,TextLoader


def get_file_md5_hex(filepath):    # 获取文件的md5的十六进制字符串
    """
    获取文件的md5的十六进制字符串
    :param filepath: 文件路径
    :return: md5的十六进制字符串
    """
    if not os.path.exists(filepath):  # os.path.exists() 判断文件是否存在
        logger.error(f"[md5计算]文件{filepath}不存在")
        return
    if not os.path.isfile(filepath):  # os.path.isfile() 判断是否是文件
        logger.error(f"[md5计算]路径{filepath}不是文件")
        return

    md5_obj = hashlib.md5()   # hashlib.md5(): 创建一个md5对象
    chunk_size = 4096  # 4KB分片，避免文件过大爆内存
    try:
        with open(filepath, "rb") as f:  # 必须二进制读取   "rb": 二进制读取
            while chunk := f.read(chunk_size):  #  := -> 先赋值，后判断
                md5_obj.update(chunk)    #update(): 更新md5对象的内容

        """
        chunk = f.read(chunk_size)
        while chunk:
            md5_obj.update(chunk)
            chunk = f.read(chunk_size)
        """
        md5_hex = md5_obj.hexdigest()   #hexdigest(): 获取md5的十六进制字符串
        return md5_hex
    except Exception as e:
        logger.error(f"计算文件{filepath}md5失败，{str(e)}")
        return None

def listdir_with_allowed_type(path:str, allowed_types:tuple[str]):    # 返回文件夹内的文件列表（允许的文件后缀）
    """
    返回文件夹内的文件列表（允许的文件后缀）
    :param path: 文件夹路径
    :param allowed_types: 允许的文件后缀
    :return: 文件列表
    """
    files = []

    if not os.path.isdir(path):  # os.path.isdir() 判断是否是文件夹
        logger.error(f"[listdir_with_allowed_type]{path}不是文件夹")
        return allowed_types

    for f in os.listdir(path):   # os.listdir() 获取文件夹内所有文件和文件夹的名称
        if f.endswith(allowed_types):   # f.endswith() 判断文件名是否以 allowed_types 结尾
            files.append(os.path.join(path, f))   #例如 path:data f:name.txt  相加获取完整文件路径

    return tuple(files)   #转成元组,不允许修改


def pdf_loader(filepath:str,password=None)->list[Document]:
    """
    pdf加载器
    :param filepath: 文件路径
    :param password: 密码
    :return: 文档列表
    """
    return PyPDFLoader(filepath, password).load()  # PDFLoader: 加载pdf文件

def txt_loader(filepath:str)->list[Document]:
    """
    txt加载器
    :param filepath: 文件路径
    :return: 文档列表
    """
    return TextLoader(filepath,encoding="utf-8").load()
