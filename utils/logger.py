"""日志配置模块

这个模块实现了统一的日志记录功能。
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional

def setup_logger(
    name: str,
    log_level: Optional[str] = None,
    log_format: Optional[str] = None,
    log_file: Optional[str] = None
) -> logging.Logger:
    """设置日志记录器

    Args:
        name (str): 日志记录器名称
        log_level (Optional[str]): 日志级别，默认为 INFO
        log_format (Optional[str]): 日志格式，默认为标准格式
        log_file (Optional[str]): 日志文件路径，默认为 None（仅控制台输出）

    Returns:
        logging.Logger: 配置好的日志记录器
    """
    # 获取日志级别
    level = getattr(logging, log_level or os.getenv("LOG_LEVEL", "INFO"))

    # 获取日志格式
    fmt = log_format or os.getenv(
        "LOG_FORMAT",
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 创建格式化器
    formatter = logging.Formatter(fmt)

    # 添加控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 如果指定了日志文件，添加文件处理器
    if log_file:
        # 确保日志目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # 创建文件处理器
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# 创建默认日志记录器
default_logger = setup_logger(
    name="satellite_design",
    log_file="logs/satellite_design.log"
) 