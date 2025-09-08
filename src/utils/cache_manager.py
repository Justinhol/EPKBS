"""
统一缓存管理系统
实现分层缓存策略，支持内存缓存和Redis缓存
"""
import asyncio
import json
import pickle
import hashlib
from typing import Any, Optional, Dict, Union, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import weakref

import redis.asyncio as redis
from src.utils.logger import get_logger
from config.settings import settings

logger = get_logger("utils.cache_manager")


class CacheLevel(Enum):
    """缓存级别"""
    MEMORY = "memory"  # 内存缓存
    REDIS = "redis"    # Redis缓存
    BOTH = "both"      # 双层缓存


class CacheStrategy(Enum):
    """缓存策略"""
    LRU = "lru"        # 最近最少使用
    LFU = "lfu"        # 最少使用频率
    TTL = "ttl"        # 基于时间过期


@dataclass
class CacheItem:
    """缓存项"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    ttl: Optional[int] = None
    
    @property
    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.ttl is None:
            return False
        return datetime.utcnow() > self.created_at + timedelta(seconds=self.ttl)


class MemoryCache:
    """内存缓存"""
    
    def __init__(self, max_size: int = 1000, strategy: CacheStrategy = CacheStrategy.LRU):
        self.max_size = max_size
        self.strategy = strategy
        self._cache: Dict[str, CacheItem] = {}
        self._access_order: List[str] = []  # LRU顺序
        
        logger.info(f"内存缓存初始化，最大大小: {max_size}, 策略: {strategy.value}")
    
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存项"""
        if key not in self._cache:
            return None
        
        item = self._cache[key]
        
        # 检查过期
        if item.is_expired:
            await self.delete(key)
            return None
        
        # 更新访问信息
        item.last_accessed = datetime.utcnow()
        item.access_count += 1
        
        # 更新LRU顺序
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
        
        return item.value
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存项"""
        try:
            # 检查是否需要清理空间
            if len(self._cache) >= self.max_size and key not in self._cache:
                await self._evict_item()
            
            # 创建缓存项
            now = datetime.utcnow()
            item = CacheItem(
                key=key,
                value=value,
                created_at=now,
                last_accessed=now,
                access_count=1,
                ttl=ttl
            )
            
            self._cache[key] = item
            
            # 更新访问顺序
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            
            return True
            
        except Exception as e:
            logger.error(f"内存缓存设置失败: {key}, 错误: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """删除缓存项"""
        if key in self._cache:
            del self._cache[key]
        
        if key in self._access_order:
            self._access_order.remove(key)
        
        return True
    
    async def _evict_item(self):
        """驱逐缓存项"""
        if not self._cache:
            return
        
        if self.strategy == CacheStrategy.LRU:
            # 驱逐最近最少使用的项
            if self._access_order:
                key_to_evict = self._access_order[0]
                await self.delete(key_to_evict)
        elif self.strategy == CacheStrategy.LFU:
            # 驱逐使用频率最低的项
            key_to_evict = min(self._cache.keys(), key=lambda k: self._cache[k].access_count)
            await self.delete(key_to_evict)
        elif self.strategy == CacheStrategy.TTL:
            # 驱逐最早过期的项
            expired_keys = [k for k, v in self._cache.items() if v.is_expired]
            if expired_keys:
                for key in expired_keys:
                    await self.delete(key)
            else:
                # 如果没有过期项，使用LRU策略
                if self._access_order:
                    key_to_evict = self._access_order[0]
                    await self.delete(key_to_evict)
    
    async def clear(self):
        """清空缓存"""
        self._cache.clear()
        self._access_order.clear()
    
    async def size(self) -> int:
        """获取缓存大小"""
        return len(self._cache)
    
    async def keys(self) -> List[str]:
        """获取所有键"""
        return list(self._cache.keys())


class RedisCache:
    """Redis缓存"""
    
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or settings.database.redis_url
        self._redis: Optional[redis.Redis] = None
        
        logger.info(f"Redis缓存初始化，URL: {self.redis_url}")
    
    async def initialize(self):
        """初始化Redis连接"""
        try:
            self._redis = redis.from_url(self.redis_url, decode_responses=False)
            await self._redis.ping()
            logger.info("Redis缓存连接成功")
        except Exception as e:
            logger.error(f"Redis缓存连接失败: {e}")
            raise
    
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存项"""
        if not self._redis:
            return None
        
        try:
            data = await self._redis.get(key)
            if data is None:
                return None
            
            # 反序列化
            return pickle.loads(data)
            
        except Exception as e:
            logger.error(f"Redis缓存获取失败: {key}, 错误: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存项"""
        if not self._redis:
            return False
        
        try:
            # 序列化
            data = pickle.dumps(value)
            
            if ttl:
                await self._redis.setex(key, ttl, data)
            else:
                await self._redis.set(key, data)
            
            return True
            
        except Exception as e:
            logger.error(f"Redis缓存设置失败: {key}, 错误: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """删除缓存项"""
        if not self._redis:
            return False
        
        try:
            await self._redis.delete(key)
            return True
        except Exception as e:
            logger.error(f"Redis缓存删除失败: {key}, 错误: {e}")
            return False
    
    async def clear(self):
        """清空缓存"""
        if self._redis:
            await self._redis.flushdb()
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """获取匹配的键"""
        if not self._redis:
            return []
        
        try:
            keys = await self._redis.keys(pattern)
            return [key.decode() if isinstance(key, bytes) else key for key in keys]
        except Exception as e:
            logger.error(f"Redis获取键失败: {e}")
            return []
    
    async def close(self):
        """关闭连接"""
        if self._redis:
            await self._redis.close()


class UnifiedCacheManager:
    """统一缓存管理器"""
    
    def __init__(
        self,
        enable_memory: bool = True,
        enable_redis: bool = True,
        memory_size: int = 1000,
        default_ttl: int = 3600
    ):
        self.enable_memory = enable_memory
        self.enable_redis = enable_redis
        self.default_ttl = default_ttl
        
        # 初始化缓存层
        self.memory_cache = MemoryCache(memory_size) if enable_memory else None
        self.redis_cache = RedisCache() if enable_redis else None
        
        logger.info(f"统一缓存管理器初始化，内存缓存: {enable_memory}, Redis缓存: {enable_redis}")
    
    async def initialize(self):
        """初始化缓存系统"""
        if self.redis_cache:
            await self.redis_cache.initialize()
    
    def _generate_key(self, namespace: str, key: str) -> str:
        """生成缓存键"""
        return f"{namespace}:{key}"
    
    def _hash_key(self, key: str) -> str:
        """对键进行哈希"""
        return hashlib.md5(key.encode()).hexdigest()
    
    async def get(
        self,
        key: str,
        namespace: str = "default",
        level: CacheLevel = CacheLevel.BOTH
    ) -> Optional[Any]:
        """获取缓存项"""
        cache_key = self._generate_key(namespace, key)
        
        # 先尝试内存缓存
        if level in [CacheLevel.MEMORY, CacheLevel.BOTH] and self.memory_cache:
            value = await self.memory_cache.get(cache_key)
            if value is not None:
                logger.debug(f"内存缓存命中: {cache_key}")
                return value
        
        # 再尝试Redis缓存
        if level in [CacheLevel.REDIS, CacheLevel.BOTH] and self.redis_cache:
            value = await self.redis_cache.get(cache_key)
            if value is not None:
                logger.debug(f"Redis缓存命中: {cache_key}")
                
                # 回写到内存缓存
                if self.memory_cache and level == CacheLevel.BOTH:
                    await self.memory_cache.set(cache_key, value, self.default_ttl)
                
                return value
        
        return None
    
    async def set(
        self,
        key: str,
        value: Any,
        namespace: str = "default",
        ttl: Optional[int] = None,
        level: CacheLevel = CacheLevel.BOTH
    ) -> bool:
        """设置缓存项"""
        cache_key = self._generate_key(namespace, key)
        ttl = ttl or self.default_ttl
        success = True
        
        # 设置内存缓存
        if level in [CacheLevel.MEMORY, CacheLevel.BOTH] and self.memory_cache:
            memory_success = await self.memory_cache.set(cache_key, value, ttl)
            success = success and memory_success
        
        # 设置Redis缓存
        if level in [CacheLevel.REDIS, CacheLevel.BOTH] and self.redis_cache:
            redis_success = await self.redis_cache.set(cache_key, value, ttl)
            success = success and redis_success
        
        return success
    
    async def delete(
        self,
        key: str,
        namespace: str = "default",
        level: CacheLevel = CacheLevel.BOTH
    ) -> bool:
        """删除缓存项"""
        cache_key = self._generate_key(namespace, key)
        success = True
        
        # 删除内存缓存
        if level in [CacheLevel.MEMORY, CacheLevel.BOTH] and self.memory_cache:
            memory_success = await self.memory_cache.delete(cache_key)
            success = success and memory_success
        
        # 删除Redis缓存
        if level in [CacheLevel.REDIS, CacheLevel.BOTH] and self.redis_cache:
            redis_success = await self.redis_cache.delete(cache_key)
            success = success and redis_success
        
        return success
    
    async def clear(self, namespace: str = None):
        """清空缓存"""
        if namespace:
            # 清空特定命名空间
            pattern = f"{namespace}:*"
            if self.redis_cache:
                keys = await self.redis_cache.keys(pattern)
                for key in keys:
                    await self.redis_cache.delete(key)
            
            if self.memory_cache:
                keys_to_delete = [k for k in await self.memory_cache.keys() if k.startswith(f"{namespace}:")]
                for key in keys_to_delete:
                    await self.memory_cache.delete(key)
        else:
            # 清空所有缓存
            if self.memory_cache:
                await self.memory_cache.clear()
            if self.redis_cache:
                await self.redis_cache.clear()
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        stats = {
            "timestamp": datetime.utcnow().isoformat(),
            "memory_cache": {},
            "redis_cache": {}
        }
        
        if self.memory_cache:
            stats["memory_cache"] = {
                "enabled": True,
                "size": await self.memory_cache.size(),
                "max_size": self.memory_cache.max_size,
                "strategy": self.memory_cache.strategy.value
            }
        else:
            stats["memory_cache"]["enabled"] = False
        
        if self.redis_cache:
            try:
                keys = await self.redis_cache.keys()
                stats["redis_cache"] = {
                    "enabled": True,
                    "size": len(keys),
                    "connected": self.redis_cache._redis is not None
                }
            except Exception as e:
                stats["redis_cache"] = {
                    "enabled": True,
                    "error": str(e),
                    "connected": False
                }
        else:
            stats["redis_cache"]["enabled"] = False
        
        return stats
    
    async def close(self):
        """关闭缓存管理器"""
        if self.redis_cache:
            await self.redis_cache.close()


# 全局缓存管理器实例
_cache_manager_instance = None


async def get_cache_manager() -> UnifiedCacheManager:
    """获取全局缓存管理器实例"""
    global _cache_manager_instance
    
    if _cache_manager_instance is None:
        _cache_manager_instance = UnifiedCacheManager(
            enable_memory=settings.cache.enable_memory_cache,
            enable_redis=settings.cache.enable_redis_cache,
            memory_size=settings.cache.memory_cache_size,
            default_ttl=settings.cache.cache_ttl
        )
        await _cache_manager_instance.initialize()
    
    return _cache_manager_instance
