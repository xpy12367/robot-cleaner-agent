"""
为整个工程提供统一的绝对路径   可以通过get_abs_path获取绝对路径,从而调用其它文件的方法
"""

import os

def get_project_root() -> str:
    """
    获取工程所在的根目录
    :return: 字符串根目录
    """
    # 当前文件的绝对路径
    current_file = os.path.abspath(__file__)   #F:\PycharmProjects\Agent项目1\utils\path_tool.py
    # 获取工程的根目录，先获取文件所在的文件夹绝对路径
    current_dir = os.path.dirname(current_file)   #F:\PycharmProjects\Agent项目1\utils
    # 获取工程根目录
    project_root=os.path.dirname(current_dir)   #F:\PycharmProjects\Agent项目1

    return project_root

def get_abs_path(relative:str)->str:
    """
    传递相对路径，得到绝对路径
    :param relative_path: 相对路径
    :return: 绝对路径
    """
    object_root = get_project_root()   #F:\PycharmProjects\Agent项目1
    abs_path = os.path.join(object_root, relative)
    return abs_path

if __name__ == '__main__':
    print(get_abs_path('data\\asame'))
