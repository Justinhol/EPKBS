"""
数据库连接和会话管理
"""
import asyncio
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager

from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
import redis.asyncio as redis

from src.api.models import Base
from src.utils.logger import get_logger
from config.settings import settings

logger = get_logger("api.database")


class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self):
        self.engine = None
        self.async_engine = None
        self.SessionLocal = None
        self.AsyncSessionLocal = None
        self.redis_client = None
        
        logger.info("数据库管理器初始化完成")
    
    def initialize_sync_db(self):
        """初始化同步数据库连接"""
        try:
            # 创建同步引擎
            self.engine = create_engine(
                settings.DATABASE_URL,
                pool_pre_ping=True,
                pool_recycle=300,
                echo=settings.DEBUG
            )
            
            # 创建会话工厂
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            logger.info("同步数据库连接初始化完成")
            
        except Exception as e:
            logger.error(f"同步数据库连接初始化失败: {e}")
            raise
    
    def initialize_async_db(self):
        """初始化异步数据库连接"""
        try:
            # 将PostgreSQL URL转换为异步版本
            async_database_url = settings.DATABASE_URL.replace(
                "postgresql://", "postgresql+asyncpg://"
            )
            
            # 创建异步引擎
            self.async_engine = create_async_engine(
                async_database_url,
                pool_pre_ping=True,
                pool_recycle=300,
                echo=settings.DEBUG
            )
            
            # 创建异步会话工厂
            self.AsyncSessionLocal = async_sessionmaker(
                self.async_engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            logger.info("异步数据库连接初始化完成")
            
        except Exception as e:
            logger.error(f"异步数据库连接初始化失败: {e}")
            # 降级到同步连接
            logger.warning("降级使用同步数据库连接")
            self.initialize_sync_db()
    
    async def initialize_redis(self):
        """初始化Redis连接"""
        try:
            self.redis_client = redis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # 测试连接
            await self.redis_client.ping()
            logger.info("Redis连接初始化完成")
            
        except Exception as e:
            logger.error(f"Redis连接初始化失败: {e}")
            self.redis_client = None
    
    async def create_tables(self):
        """创建数据库表"""
        try:
            if self.async_engine:
                # 异步创建表
                async with self.async_engine.begin() as conn:
                    await conn.run_sync(Base.metadata.create_all)
            else:
                # 同步创建表
                Base.metadata.create_all(bind=self.engine)
            
            logger.info("数据库表创建完成")
            
        except Exception as e:
            logger.error(f"数据库表创建失败: {e}")
            raise
    
    async def drop_tables(self):
        """删除数据库表"""
        try:
            if self.async_engine:
                async with self.async_engine.begin() as conn:
                    await conn.run_sync(Base.metadata.drop_all)
            else:
                Base.metadata.drop_all(bind=self.engine)
            
            logger.info("数据库表删除完成")
            
        except Exception as e:
            logger.error(f"数据库表删除失败: {e}")
            raise
    
    def get_sync_session(self) -> Session:
        """获取同步数据库会话"""
        if not self.SessionLocal:
            self.initialize_sync_db()
        return self.SessionLocal()
    
    async def get_async_session(self) -> AsyncSession:
        """获取异步数据库会话"""
        if not self.AsyncSessionLocal:
            self.initialize_async_db()
        return self.AsyncSessionLocal()
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """获取数据库会话上下文管理器"""
        if self.AsyncSessionLocal:
            async with self.AsyncSessionLocal() as session:
                try:
                    yield session
                    await session.commit()
                except Exception:
                    await session.rollback()
                    raise
                finally:
                    await session.close()
        else:
            # 降级到同步会话
            session = self.get_sync_session()
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()
    
    async def get_redis(self) -> Optional[redis.Redis]:
        """获取Redis客户端"""
        if not self.redis_client:
            await self.initialize_redis()
        return self.redis_client
    
    async def close(self):
        """关闭所有连接"""
        try:
            if self.async_engine:
                await self.async_engine.dispose()
            
            if self.engine:
                self.engine.dispose()
            
            if self.redis_client:
                await self.redis_client.close()
            
            logger.info("数据库连接已关闭")
            
        except Exception as e:
            logger.warning(f"关闭数据库连接时出现警告: {e}")


# 全局数据库管理器实例
db_manager = DatabaseManager()


# 依赖注入函数
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI依赖注入：获取数据库会话"""
    async with db_manager.get_session() as session:
        yield session


async def get_redis() -> Optional[redis.Redis]:
    """FastAPI依赖注入：获取Redis客户端"""
    return await db_manager.get_redis()


# 缓存管理器
class CacheManager:
    """缓存管理器"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.default_ttl = settings.CACHE_TTL
        
        logger.info("缓存管理器初始化完成")
    
    async def get(self, key: str) -> Optional[str]:
        """获取缓存值"""
        if not self.redis_client:
            return None
        
        try:
            value = await self.redis_client.get(key)
            if value:
                logger.debug(f"缓存命中: {key}")
            return value
        except Exception as e:
            logger.warning(f"获取缓存失败: {e}")
            return None
    
    async def set(
        self,
        key: str,
        value: str,
        ttl: Optional[int] = None
    ) -> bool:
        """设置缓存值"""
        if not self.redis_client:
            return False
        
        try:
            ttl = ttl or self.default_ttl
            await self.redis_client.setex(key, ttl, value)
            logger.debug(f"缓存设置: {key}, TTL: {ttl}")
            return True
        except Exception as e:
            logger.warning(f"设置缓存失败: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """删除缓存"""
        if not self.redis_client:
            return False
        
        try:
            result = await self.redis_client.delete(key)
            logger.debug(f"缓存删除: {key}")
            return result > 0
        except Exception as e:
            logger.warning(f"删除缓存失败: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """检查缓存是否存在"""
        if not self.redis_client:
            return False
        
        try:
            result = await self.redis_client.exists(key)
            return result > 0
        except Exception as e:
            logger.warning(f"检查缓存失败: {e}")
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        """清除匹配模式的缓存"""
        if not self.redis_client:
            return 0
        
        try:
            keys = await self.redis_client.keys(pattern)
            if keys:
                result = await self.redis_client.delete(*keys)
                logger.info(f"清除缓存: {len(keys)} 个键")
                return result
            return 0
        except Exception as e:
            logger.warning(f"清除缓存失败: {e}")
            return 0
    
    def generate_key(self, prefix: str, *args) -> str:
        """生成缓存键"""
        key_parts = [prefix] + [str(arg) for arg in args]
        return ":".join(key_parts)


# 全局缓存管理器
cache_manager = CacheManager()


async def get_cache() -> CacheManager:
    """FastAPI依赖注入：获取缓存管理器"""
    if not cache_manager.redis_client:
        cache_manager.redis_client = await get_redis()
    return cache_manager


# 数据库初始化函数
async def init_database():
    """初始化数据库"""
    logger.info("开始初始化数据库...")
    
    try:
        # 初始化异步数据库连接
        db_manager.initialize_async_db()
        
        # 初始化Redis连接
        await db_manager.initialize_redis()
        
        # 创建数据库表
        await db_manager.create_tables()
        
        logger.info("数据库初始化完成")
        
    except Exception as e:
        logger.error(f"数据库初始化失败: {e}")
        raise


async def close_database():
    """关闭数据库连接"""
    await db_manager.close()


# 健康检查函数
async def check_database_health() -> dict:
    """检查数据库健康状态"""
    health_status = {
        "database": "unknown",
        "redis": "unknown"
    }
    
    # 检查数据库连接
    try:
        async with db_manager.get_session() as session:
            await session.execute("SELECT 1")
        health_status["database"] = "healthy"
    except Exception as e:
        health_status["database"] = f"unhealthy: {str(e)}"
    
    # 检查Redis连接
    try:
        redis_client = await db_manager.get_redis()
        if redis_client:
            await redis_client.ping()
            health_status["redis"] = "healthy"
        else:
            health_status["redis"] = "disabled"
    except Exception as e:
        health_status["redis"] = f"unhealthy: {str(e)}"
    
    return health_status
