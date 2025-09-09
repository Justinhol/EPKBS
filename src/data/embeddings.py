"""
嵌入模型管理器
支持多种嵌入模型和向量化策略
"""
import asyncio
from typing import List, Dict, Any, Optional, Union
import numpy as np
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
import torch

from src.utils.logger import get_logger
from src.utils.helpers import Timer, chunk_list
from config.settings import settings

logger = get_logger("data.embeddings")


class EmbeddingManager:
    """嵌入模型管理器"""

    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL_PATH
        self.device = device or (
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.embedding_dim = None

        logger.info(f"嵌入模型管理器初始化，模型: {self.model_name}, 设备: {self.device}")

    async def initialize(self):
        """初始化嵌入模型"""
        try:
            logger.info(f"正在加载嵌入模型: {self.model_name}")

            # 使用SentenceTransformer直接加载，性能更好
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
                trust_remote_code=True
            )

            # 获取嵌入维度
            self.embedding_dim = self.model.get_sentence_embedding_dimension()

            logger.info(f"嵌入模型加载完成，维度: {self.embedding_dim}")

        except Exception as e:
            logger.error(f"嵌入模型加载失败: {e}")
            # 降级到CPU或其他模型
            try:
                logger.warning("尝试使用CPU加载模型")
                self.device = 'cpu'
                self.model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                    trust_remote_code=True
                )
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                logger.info(f"CPU模式下嵌入模型加载完成，维度: {self.embedding_dim}")
            except Exception as e2:
                logger.error(f"CPU模式下嵌入模型加载也失败: {e2}")
                raise

    async def embed_documents(self, documents: List[Document]) -> List[np.ndarray]:
        """
        对文档进行向量化

        Args:
            documents: 文档列表

        Returns:
            向量列表
        """
        if not self.model:
            await self.initialize()

        texts = [doc.page_content for doc in documents]
        return await self.embed_texts(texts)

    async def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """
        对文本进行向量化

        Args:
            texts: 文本列表

        Returns:
            向量列表
        """
        if not self.model:
            await self.initialize()

        if not texts:
            return []

        logger.info(f"开始向量化 {len(texts)} 个文本")

        with Timer() as timer:
            try:
                # 批量处理以提高效率
                batch_size = 32  # 可以根据GPU内存调整
                all_embeddings = []

                for batch in chunk_list(texts, batch_size):
                    # 异步执行嵌入
                    embeddings = await asyncio.to_thread(
                        self.model.encode,
                        batch,
                        normalize_embeddings=True,
                        show_progress_bar=False
                    )
                    all_embeddings.extend(embeddings)

                logger.info(
                    f"向量化完成，耗时: {timer.elapsed:.2f}秒，平均: {timer.elapsed/len(texts):.3f}秒/文本")
                return all_embeddings

            except Exception as e:
                logger.error(f"向量化失败: {e}")
                raise

    async def embed_query(self, query: str) -> np.ndarray:
        """
        对查询进行向量化

        Args:
            query: 查询文本

        Returns:
            查询向量
        """
        if not self.model:
            await self.initialize()

        try:
            embedding = await asyncio.to_thread(
                self.model.encode,
                [query],
                normalize_embeddings=True,
                show_progress_bar=False
            )
            return embedding[0]

        except Exception as e:
            logger.error(f"查询向量化失败: {e}")
            raise

    def get_embedding_dimension(self) -> int:
        """获取嵌入维度"""
        return self.embedding_dim


class MultiModelEmbeddingManager:
    """多模型嵌入管理器"""

    def __init__(self, models: Dict[str, str] = None):
        self.models = models or {
            'default': settings.EMBEDDING_MODEL_PATH,
            'chinese': 'BAAI/bge-large-zh-v1.5',
            'english': 'sentence-transformers/all-MiniLM-L6-v2',
        }
        self.embedding_managers = {}

        logger.info(f"多模型嵌入管理器初始化，支持模型: {list(self.models.keys())}")

    async def initialize(self):
        """初始化所有嵌入模型"""
        for name, model_path in self.models.items():
            try:
                manager = EmbeddingManager(model_path)
                await manager.initialize()
                self.embedding_managers[name] = manager
                logger.info(f"模型 {name} 初始化完成")
            except Exception as e:
                logger.error(f"模型 {name} 初始化失败: {e}")

    async def embed_documents(self, documents: List[Document], model_name: str = 'default') -> List[np.ndarray]:
        """使用指定模型对文档进行向量化"""
        if model_name not in self.embedding_managers:
            raise ValueError(f"模型 {model_name} 未初始化")

        return await self.embedding_managers[model_name].embed_documents(documents)

    async def embed_texts(self, texts: List[str], model_name: str = 'default') -> List[np.ndarray]:
        """使用指定模型对文本进行向量化"""
        if model_name not in self.embedding_managers:
            raise ValueError(f"模型 {model_name} 未初始化")

        return await self.embedding_managers[model_name].embed_texts(texts)

    async def embed_query(self, query: str, model_name: str = 'default') -> np.ndarray:
        """使用指定模型对查询进行向量化"""
        if model_name not in self.embedding_managers:
            raise ValueError(f"模型 {model_name} 未初始化")

        return await self.embedding_managers[model_name].embed_query(query)

    def get_available_models(self) -> List[str]:
        """获取可用的模型列表"""
        return list(self.embedding_managers.keys())


class EmbeddingCache:
    """嵌入缓存管理器"""

    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or (settings.DATA_DIR / "embeddings_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.cache = {}

        logger.info(f"嵌入缓存管理器初始化，缓存目录: {self.cache_dir}")

    def get_cache_key(self, text: str, model_name: str) -> str:
        """生成缓存键"""
        from src.utils.helpers import generate_hash
        return f"{model_name}_{generate_hash(text, 'sha256')}"

    async def get_embedding(self, text: str, model_name: str) -> Optional[np.ndarray]:
        """从缓存获取嵌入"""
        cache_key = self.get_cache_key(text, model_name)

        if cache_key in self.cache:
            return self.cache[cache_key]

        # 尝试从磁盘加载
        cache_file = self.cache_dir / f"{cache_key}.npy"
        if cache_file.exists():
            try:
                embedding = np.load(cache_file)
                self.cache[cache_key] = embedding
                return embedding
            except Exception as e:
                logger.warning(f"加载缓存文件失败: {e}")

        return None

    async def set_embedding(self, text: str, model_name: str, embedding: np.ndarray):
        """设置嵌入缓存"""
        cache_key = self.get_cache_key(text, model_name)

        # 内存缓存
        self.cache[cache_key] = embedding

        # 磁盘缓存
        try:
            cache_file = self.cache_dir / f"{cache_key}.npy"
            np.save(cache_file, embedding)
        except Exception as e:
            logger.warning(f"保存缓存文件失败: {e}")

    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()

        # 清空磁盘缓存
        for cache_file in self.cache_dir.glob("*.npy"):
            try:
                cache_file.unlink()
            except Exception as e:
                logger.warning(f"删除缓存文件失败: {e}")

        logger.info("嵌入缓存已清空")


class CachedEmbeddingManager(EmbeddingManager):
    """带缓存的嵌入管理器 - 支持多层缓存"""

    def __init__(self, model_name: str = None, device: str = None, enable_cache: bool = True, cache_manager=None):
        super().__init__(model_name, device)
        self.enable_cache = enable_cache and settings.ENABLE_CACHE
        self.cache = EmbeddingCache() if self.enable_cache else None
        self.cache_manager = cache_manager  # 多层缓存管理器

        # 缓存统计
        self.cache_stats = {
            "total_requests": 0,
            "l1_hits": 0,  # 内存缓存命中
            "l2_hits": 0,  # Redis缓存命中
            "cache_misses": 0
        }

        if self.enable_cache:
            logger.info("多层嵌入缓存已启用")

    async def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """带多层缓存的文本向量化"""
        if not self.enable_cache:
            return await super().embed_texts(texts)

        self.cache_stats["total_requests"] += len(texts)
        embeddings = []
        uncached_texts = []
        uncached_indices = []

        # 多层缓存检查
        for i, text in enumerate(texts):
            embedding = await self._get_from_multilevel_cache(text)
            if embedding is not None:
                embeddings.append(embedding)
            else:
                embeddings.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)

        # 处理未缓存的文本
        if uncached_texts:
            self.cache_stats["cache_misses"] += len(uncached_texts)
            hit_rate = (len(texts) - len(uncached_texts)) / len(texts) * 100
            logger.info(
                f"多层缓存命中率: {hit_rate:.1f}% (L1: {self.cache_stats['l1_hits']}, L2: {self.cache_stats['l2_hits']})")

            new_embeddings = await super().embed_texts(uncached_texts)

            # 更新多层缓存和结果
            for i, (text, embedding) in enumerate(zip(uncached_texts, new_embeddings)):
                await self._set_to_multilevel_cache(text, embedding)
                embeddings[uncached_indices[i]] = embedding
        else:
            logger.info("多层缓存命中率: 100%")

        return embeddings

    async def _get_from_multilevel_cache(self, text: str) -> Optional[np.ndarray]:
        """从多层缓存获取嵌入向量"""
        import hashlib

        # 生成缓存键
        cache_key = hashlib.md5(
            f"{self.model_name}:{text}".encode()).hexdigest()

        # L1缓存 (内存缓存)
        if self.cache_manager:
            from src.utils.cache_manager import CacheLevel
            cached = await self.cache_manager.get(
                cache_key,
                namespace="embeddings",
                level=CacheLevel.MEMORY
            )
            if cached is not None:
                self.cache_stats["l1_hits"] += 1
                return np.array(cached)

            # L2缓存 (Redis缓存)
            cached = await self.cache_manager.get(
                cache_key,
                namespace="embeddings",
                level=CacheLevel.REDIS
            )
            if cached is not None:
                self.cache_stats["l2_hits"] += 1
                # 回写到L1缓存
                await self.cache_manager.set(
                    cache_key, cached,
                    namespace="embeddings",
                    level=CacheLevel.MEMORY,
                    ttl=settings.cache.l1_cache_ttl
                )
                return np.array(cached)

        # 回退到原有缓存
        if self.cache:
            return await self.cache.get_embedding(text, self.model_name)

        return None

    async def _set_to_multilevel_cache(self, text: str, embedding: np.ndarray):
        """设置到多层缓存"""
        import hashlib

        cache_key = hashlib.md5(
            f"{self.model_name}:{text}".encode()).hexdigest()
        embedding_list = embedding.tolist()

        # 设置到多层缓存
        if self.cache_manager:
            from src.utils.cache_manager import CacheLevel
            # L1缓存 (内存)
            await self.cache_manager.set(
                cache_key, embedding_list,
                namespace="embeddings",
                level=CacheLevel.MEMORY,
                ttl=settings.cache.l1_cache_ttl
            )
            # L2缓存 (Redis)
            await self.cache_manager.set(
                cache_key, embedding_list,
                namespace="embeddings",
                level=CacheLevel.REDIS,
                ttl=settings.cache.embedding_cache_ttl
            )

        # 回退到原有缓存
        if self.cache:
            await self.cache.set_embedding(text, self.model_name, embedding)

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        if self.cache_stats["total_requests"] > 0:
            total_hits = self.cache_stats["l1_hits"] + \
                self.cache_stats["l2_hits"]
            hit_rate = total_hits / self.cache_stats["total_requests"] * 100
        else:
            hit_rate = 0

        return {
            **self.cache_stats,
            "hit_rate": hit_rate,
            "l1_hit_rate": self.cache_stats["l1_hits"] / max(1, self.cache_stats["total_requests"]) * 100,
            "l2_hit_rate": self.cache_stats["l2_hits"] / max(1, self.cache_stats["total_requests"]) * 100
        }
