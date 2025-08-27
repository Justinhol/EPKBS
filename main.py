#!/usr/bin/env python3
"""
企业私有知识库系统主程序入口
"""
import asyncio
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logger import get_logger
from config.settings import settings

logger = get_logger("main")


async def main():
    """主程序入口"""
    logger.info(f"启动 {settings.PROJECT_NAME} v{settings.PROJECT_VERSION}")
    logger.info(f"调试模式: {settings.DEBUG}")
    
    try:
        # 这里将来会启动各个服务
        logger.info("系统初始化完成")
        logger.info(f"API服务将在 http://{settings.API_HOST}:{settings.API_PORT} 启动")
        
        # 暂时只是打印配置信息
        logger.info("当前配置:")
        logger.info(f"  - 数据目录: {settings.DATA_DIR}")
        logger.info(f"  - 日志目录: {settings.LOGS_DIR}")
        logger.info(f"  - 数据库: {settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}")
        logger.info(f"  - Redis: {settings.REDIS_HOST}:{settings.REDIS_PORT}")
        logger.info(f"  - Milvus: {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")
        
        logger.info("系统准备就绪，等待进一步开发...")
        
    except Exception as e:
        logger.error(f"系统启动失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
