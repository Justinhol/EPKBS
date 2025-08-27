"""
日志配置模块
"""
import sys
from pathlib import Path
from loguru import logger
from config.settings import settings


def setup_logger():
    """配置日志系统"""
    
    # 移除默认的日志处理器
    logger.remove()
    
    # 控制台日志
    logger.add(
        sys.stdout,
        format=settings.LOG_FORMAT,
        level=settings.LOG_LEVEL,
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # 文件日志 - 所有日志
    logger.add(
        settings.LOGS_DIR / "app.log",
        format=settings.LOG_FORMAT,
        level="DEBUG",
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        backtrace=True,
        diagnose=True
    )
    
    # 错误日志
    logger.add(
        settings.LOGS_DIR / "error.log",
        format=settings.LOG_FORMAT,
        level="ERROR",
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        backtrace=True,
        diagnose=True
    )
    
    # RAG相关日志
    logger.add(
        settings.LOGS_DIR / "rag.log",
        format=settings.LOG_FORMAT,
        level="INFO",
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        filter=lambda record: "rag" in record["name"].lower(),
        backtrace=True,
        diagnose=True
    )
    
    return logger


# 初始化日志
app_logger = setup_logger()


def get_logger(name: str = None):
    """获取日志实例"""
    if name:
        return logger.bind(name=name)
    return logger
