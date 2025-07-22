"""
Utils包 - 论文分析工具集
提供各种实用工具模块
"""

import logging
import os
from datetime import datetime

# 配置统一的日志格式
def setup_logging(level=logging.INFO, log_file=None):
    """
    设置统一的日志配置
    
    Args:
        level: 日志级别
        log_file: 日志文件路径，如果为None则只输出到控制台
    """
    # 创建日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # 清除现有的处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 如果指定了日志文件，添加文件处理器
    if log_file:
        # 确保日志目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

# 默认设置日志
setup_logging()

# 导入子模块
from . import pdf_downloader
from . import pdf_extractor

__all__ = [
    'pdf_downloader',
    'pdf_extractor',
    'setup_logging'
]

__version__ = "1.0.0" 
