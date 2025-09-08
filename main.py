#!/usr/bin/env python3
"""
企业私有知识库系统主程序入口 - 统一架构版本
"""
from config.settings import settings
from src.utils.cache_manager import get_cache_manager
from src.utils.model_manager import get_model_pool
from src.agent.core import get_agent_core
from src.utils.logger import get_logger
import asyncio
import sys
import signal
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))


logger = get_logger("main")

# 全局资源引用
agent_core = None
model_pool = None
cache_manager = None


async def initialize_system():
    """初始化系统"""
    global agent_core, model_pool, cache_manager

    logger.info(f"启动 {settings.project_name} v{settings.project_version}")
    logger.info(f"环境: {settings.environment}")
    logger.info(f"调试模式: {settings.debug}")

    try:
        # 1. 初始化模型池
        logger.info("初始化模型池...")
        model_pool = get_model_pool()
        await model_pool.start_cleanup_task()

        # 2. 初始化缓存管理器
        logger.info("初始化缓存管理器...")
        cache_manager = await get_cache_manager()

        # 3. 初始化Agent核心
        logger.info("初始化Agent核心...")
        agent_core = await get_agent_core()

        logger.info("系统初始化完成")

        # 打印系统状态
        await print_system_status()

        return True

    except Exception as e:
        logger.error(f"系统初始化失败: {e}")
        return False


async def print_system_status():
    """打印系统状态"""
    logger.info("=== 系统状态 ===")
    logger.info(f"  - 数据目录: {settings.data_dir}")
    logger.info(f"  - 日志目录: {settings.logs_dir}")
    logger.info(
        f"  - 数据库: {settings.database.postgres_host}:{settings.database.postgres_port}")
    logger.info(
        f"  - Redis: {settings.database.redis_host}:{settings.database.redis_port}")
    logger.info(
        f"  - Milvus: {settings.database.milvus_host}:{settings.database.milvus_port}")

    # 获取组件状态
    if agent_core:
        health_status = await agent_core.health_check()
        logger.info(
            f"  - Agent核心: {health_status.get('agent_core', 'unknown')}")

    if model_pool:
        model_info = await model_pool.get_model_info()
        loaded_models = len(
            [info for info in model_info.values() if info["is_loaded"]])
        logger.info(f"  - 已加载模型: {loaded_models}")

    if cache_manager:
        cache_stats = await cache_manager.get_stats()
        logger.info(f"  - 缓存状态: {cache_stats}")

    logger.info("===============")


async def cleanup_system():
    """清理系统资源"""
    global agent_core, model_pool, cache_manager

    logger.info("开始清理系统资源...")

    try:
        # 清理Agent核心
        if agent_core:
            await agent_core.cleanup()

        # 清理模型池
        if model_pool:
            await model_pool.cleanup()

        # 清理缓存管理器
        if cache_manager:
            await cache_manager.close()

        logger.info("系统资源清理完成")

    except Exception as e:
        logger.error(f"系统资源清理失败: {e}")


def signal_handler(signum, frame):
    """信号处理器"""
    logger.info(f"接收到信号 {signum}，开始优雅关闭...")
    asyncio.create_task(cleanup_system())
    sys.exit(0)


async def main():
    """主程序入口"""
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # 初始化系统
        success = await initialize_system()
        if not success:
            sys.exit(1)

        logger.info("系统准备就绪！")
        logger.info("可以通过以下方式使用系统:")
        logger.info(
            "  1. 启动API服务: uvicorn src.api.app:app --host 0.0.0.0 --port 8000")
        logger.info(
            "  2. 启动前端界面: streamlit run src/frontend/app.py --server.port 8501")
        logger.info("  3. 运行示例: python examples/basic/simple_chat.py")

        # 演示基本功能
        if settings.debug:
            await demo_basic_functionality()

        # 保持运行
        logger.info("系统运行中，按 Ctrl+C 退出...")
        try:
            while True:
                await asyncio.sleep(60)  # 每分钟检查一次
                if settings.debug:
                    await print_system_status()
        except KeyboardInterrupt:
            logger.info("接收到退出信号")

    except Exception as e:
        logger.error(f"系统运行异常: {e}")
        sys.exit(1)
    finally:
        await cleanup_system()


async def demo_basic_functionality():
    """演示基本功能"""
    logger.info("=== 演示基本功能 ===")

    try:
        if agent_core:
            # 测试对话功能
            response = await agent_core.chat("你好，请介绍一下你的功能")
            logger.info(
                f"对话测试结果: {response.get('response', 'No response')[:100]}...")

            # 获取统计信息
            stats = await agent_core.get_stats()
            logger.info(f"系统统计: 总请求数={stats.get('total_requests', 0)}")

    except Exception as e:
        logger.error(f"演示功能异常: {e}")

    logger.info("==================")


if __name__ == "__main__":
    asyncio.run(main())
