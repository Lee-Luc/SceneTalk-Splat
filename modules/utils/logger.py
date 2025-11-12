"""
日志系统
提供统一的日志记录功能
"""

import sys
from loguru import logger
from pathlib import Path
from typing import Optional


def setup_logger(
    log_file: Optional[str] = None,
    level: str = "INFO",
    rotation: str = "100 MB",
    retention: str = "10 days"
) -> logger:
    """
    配置日志系统
    
    Args:
        log_file: 日志文件路径，如果为None则只输出到控制台
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        rotation: 日志轮转大小
        retention: 日志保留时间
        
    Returns:
        配置好的logger对象
    """
    # 移除默认的handler
    logger.remove()
    
    # 添加控制台输出
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True
    )
    
    # 如果指定了日志文件，添加文件输出
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=level,
            rotation=rotation,
            retention=retention,
            encoding="utf-8"
        )
    
    return logger


# 创建默认logger实例
default_logger = setup_logger()

