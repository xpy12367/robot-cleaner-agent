import logging
import os
from datetime import datetime

from utils.path_tool import get_abs_path

#日志保存的根目录
LOG_ROOT = get_abs_path('logs')

#确保日志保存目录存在
os.makedirs(LOG_ROOT, exist_ok=True)  # 确保日志保存目录存在，如果存在则不执行，避免报错

#配置日志格式  error,warning,info,debug
DEFAULT_LOG_FORMAT = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)

def get_logger(
        name:str="agent",
        console_level=logging.INFO,
        file_level=logging.DEBUG,
        log_file=None
) -> logging.Logger:
    """
    获取日志器
    :param name: 日志器名称
    :param console_level: 控制台日志级别
    :param file_level: 文件日志级别
    :param log_file: 日志文件路径
    :return: 日志器
    """
    logger=logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    #避免重复添加handle
    if logger.handlers:
        return logger

    #控制台handle
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(DEFAULT_LOG_FORMAT)

    logger.addHandler(console_handler)
    #文件handle
    if not log_file:    #日志文件存放路径
        log_file = os.path.join(LOG_ROOT, f"{name}_{datetime.now().strftime('%Y%m%d')}.log")

    file_handle=logging.FileHandler(log_file,encoding="utf-8")  #FileHandler: 文件处理
    file_handle.setLevel(file_level)
    file_handle.setFormatter(DEFAULT_LOG_FORMAT)

    logger.addHandler(file_handle)

    return logger

#快捷获取日志器
logger=get_logger()

if __name__ == '__main__':
    logger.info("信息日志")
    logger.error("错误日志")
    logger.warning("警告日志")
    logger.debug("调试日志")