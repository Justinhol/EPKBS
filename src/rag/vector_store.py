"""
向量数据库管理器
支持Milvus向量数据库的连接、索引管理和向量操作
"""
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from datetime import datetime

from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
    MilvusException
)
from langchain.schema import Document

from src.utils.logger import get_logger
from src.utils.helpers import Timer, chunk_list
from config.settings import settings

logger = get_logger("rag.vector_store")


class MilvusVectorStore:
    """Milvus向量数据库管理器"""

    def __init__(
        self,
        collection_name: str = "knowledge_base",
        host: str = None,
        port: int = None,
        user: str = None,
        password: str = None,
        embedding_dim: int = 384
    ):
        self.collection_name = collection_name
        self.host = host or settings.MILVUS_HOST
        self.port = port or settings.MILVUS_PORT
        self.user = user or settings.MILVUS_USER
        self.password = password or settings.MILVUS_PASSWORD
        self.embedding_dim = embedding_dim

        self.collection = None
        self.is_connected = False

        logger.info(
            f"Milvus向量存储初始化: {self.host}:{self.port}, 集合: {self.collection_name}")

    async def connect(self):
        """连接到Milvus数据库"""
        try:
            # 建立连接
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password
            )

            self.is_connected = True
            logger.info("Milvus连接成功")

            # 创建或加载集合
            await self._setup_collection()

        except Exception as e:
            logger.error(f"Milvus连接失败: {e}")
            raise

    async def _setup_collection(self):
        """设置集合"""
        try:
            # 检查集合是否存在
            if utility.has_collection(self.collection_name):
                logger.info(f"加载现有集合: {self.collection_name}")
                self.collection = Collection(self.collection_name)
            else:
                logger.info(f"创建新集合: {self.collection_name}")
                await self._create_collection()

            # 加载集合到内存
            self.collection.load()
            logger.info(f"集合 {self.collection_name} 已加载到内存")

        except Exception as e:
            logger.error(f"集合设置失败: {e}")
            raise

    async def _create_collection(self):
        """创建新集合"""
        # 定义字段
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64,
                        is_primary=True, auto_id=True),
            FieldSchema(name="content_hash",
                        dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="content", dtype=DataType.VARCHAR,
                        max_length=65535),
            FieldSchema(name="metadata", dtype=DataType.JSON),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR,
                        dim=self.embedding_dim),
            FieldSchema(name="created_at",
                        dtype=DataType.VARCHAR, max_length=32),
        ]

        # 创建集合schema
        schema = CollectionSchema(
            fields=fields,
            description=f"Knowledge base collection for {self.collection_name}"
        )

        # 创建集合
        self.collection = Collection(
            name=self.collection_name,
            schema=schema,
            using='default'
        )

        # 创建索引
        await self._create_index()

        logger.info(f"集合 {self.collection_name} 创建成功")

    async def _create_index(self):
        """创建向量索引"""
        try:
            # HNSW索引参数 - 优化配置
            index_params = {
                "metric_type": "COSINE",  # 余弦相似度
                "index_type": "HNSW",    # HNSW索引
                "params": {
                    "M": 32,             # 连接数 (提升精度，16->32)
                    "efConstruction": 256  # 构建时的搜索深度 (提升精度，200->256)
                }
            }

            # 创建向量字段索引
            self.collection.create_index(
                field_name="embedding",
                index_params=index_params
            )

            # 创建标量字段索引
            self.collection.create_index(field_name="content_hash")

            logger.info("向量索引创建成功")

        except Exception as e:
            logger.error(f"索引创建失败: {e}")
            raise

    async def add_documents(
        self,
        documents: List[Document],
        embeddings: List[np.ndarray],
        batch_size: int = 100
    ) -> List[int]:
        """
        添加文档到向量数据库

        Args:
            documents: 文档列表
            embeddings: 对应的向量列表
            batch_size: 批处理大小

        Returns:
            插入的文档ID列表
        """
        if not self.is_connected:
            await self.connect()

        if len(documents) != len(embeddings):
            raise ValueError("文档数量与向量数量不匹配")

        logger.info(f"开始添加 {len(documents)} 个文档到向量数据库")

        all_ids = []
        current_time = datetime.utcnow().isoformat()

        # 分批处理
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]

            # 准备数据
            data = [
                # content_hash
                [doc.metadata.get(
                    'content_hash', f'hash_{i}') for doc in batch_docs],
                [doc.page_content for doc in batch_docs],  # content
                [doc.metadata for doc in batch_docs],      # metadata
                [emb.tolist() for emb in batch_embeddings],  # embedding
                [current_time] * len(batch_docs)           # created_at
            ]

            try:
                # 插入数据
                result = self.collection.insert(data)
                batch_ids = result.primary_keys
                all_ids.extend(batch_ids)

                logger.info(
                    f"批次 {i//batch_size + 1} 插入成功，ID范围: {min(batch_ids)}-{max(batch_ids)}")

            except Exception as e:
                logger.error(f"批次 {i//batch_size + 1} 插入失败: {e}")
                raise

        # 刷新数据到磁盘
        self.collection.flush()
        logger.info(f"文档添加完成，共插入 {len(all_ids)} 个文档")

        return all_ids

    async def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        search_params: Dict[str, Any] = None,
        filter_expr: str = None
    ) -> List[Dict[str, Any]]:
        """
        向量相似度搜索

        Args:
            query_embedding: 查询向量
            top_k: 返回结果数量
            search_params: 搜索参数
            filter_expr: 过滤表达式

        Returns:
            搜索结果列表
        """
        if not self.is_connected:
            await self.connect()

        # 默认搜索参数 - 优化HNSW配置
        if search_params is None:
            search_params = {
                "metric_type": "COSINE",
                "params": {"ef": 128}  # HNSW搜索参数 (提升精度，100->128)
            }

        try:
            with Timer() as timer:
                # 执行搜索
                results = self.collection.search(
                    data=[query_embedding.tolist()],
                    anns_field="embedding",
                    param=search_params,
                    limit=top_k,
                    expr=filter_expr,
                    output_fields=["content", "metadata",
                                   "content_hash", "created_at"]
                )

            # 处理结果
            search_results = []
            for hits in results:
                for hit in hits:
                    result = {
                        'id': hit.id,
                        'score': hit.score,
                        'content': hit.entity.get('content'),
                        'metadata': hit.entity.get('metadata'),
                        'content_hash': hit.entity.get('content_hash'),
                        'created_at': hit.entity.get('created_at')
                    }
                    search_results.append(result)

            logger.info(
                f"向量搜索完成，耗时: {timer.elapsed:.3f}秒，返回 {len(search_results)} 个结果")
            return search_results

        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            raise

    async def hybrid_search(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        top_k: int = 10,
        vector_weight: float = 0.7,
        text_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        混合搜索（向量 + 文本）

        Args:
            query_embedding: 查询向量
            query_text: 查询文本
            top_k: 返回结果数量
            vector_weight: 向量搜索权重
            text_weight: 文本搜索权重

        Returns:
            混合搜索结果
        """
        # 向量搜索
        vector_results = await self.search(query_embedding, top_k * 2)

        # 文本搜索（简单的关键词匹配）
        text_filter = f'content like "%{query_text}%"'
        text_results = await self.search(query_embedding, top_k * 2, filter_expr=text_filter)

        # 合并和重新排序结果
        combined_results = {}

        # 添加向量搜索结果
        for i, result in enumerate(vector_results):
            doc_id = result['id']
            score = result['score'] * vector_weight * \
                (1 - i / len(vector_results))
            combined_results[doc_id] = {
                **result,
                'hybrid_score': score,
                'vector_rank': i + 1
            }

        # 添加文本搜索结果
        for i, result in enumerate(text_results):
            doc_id = result['id']
            text_score = text_weight * (1 - i / len(text_results))

            if doc_id in combined_results:
                combined_results[doc_id]['hybrid_score'] += text_score
                combined_results[doc_id]['text_rank'] = i + 1
            else:
                combined_results[doc_id] = {
                    **result,
                    'hybrid_score': text_score,
                    'text_rank': i + 1
                }

        # 按混合分数排序
        sorted_results = sorted(
            combined_results.values(),
            key=lambda x: x['hybrid_score'],
            reverse=True
        )

        logger.info(f"混合搜索完成，返回 {len(sorted_results[:top_k])} 个结果")
        return sorted_results[:top_k]

    async def delete_by_filter(self, filter_expr: str) -> int:
        """根据过滤条件删除文档"""
        if not self.is_connected:
            await self.connect()

        try:
            result = self.collection.delete(filter_expr)
            self.collection.flush()

            delete_count = result.delete_count
            logger.info(f"删除文档完成，删除数量: {delete_count}")
            return delete_count

        except Exception as e:
            logger.error(f"删除文档失败: {e}")
            raise

    async def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        if not self.is_connected:
            await self.connect()

        try:
            stats = self.collection.num_entities

            return {
                'collection_name': self.collection_name,
                'total_documents': stats,
                'embedding_dim': self.embedding_dim,
                'index_type': 'HNSW',
                'metric_type': 'COSINE'
            }

        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}

    async def close(self):
        """关闭连接"""
        try:
            if self.collection:
                self.collection.release()
            connections.disconnect("default")
            self.is_connected = False
            logger.info("Milvus连接已关闭")

        except Exception as e:
            logger.warning(f"关闭连接时出现警告: {e}")


class VectorStoreManager:
    """向量存储管理器 - 支持多个集合"""

    def __init__(self):
        self.stores: Dict[str, MilvusVectorStore] = {}
        logger.info("向量存储管理器初始化完成")

    async def get_store(
        self,
        collection_name: str,
        embedding_dim: int = 384
    ) -> MilvusVectorStore:
        """获取或创建向量存储"""
        if collection_name not in self.stores:
            store = MilvusVectorStore(
                collection_name=collection_name,
                embedding_dim=embedding_dim
            )
            await store.connect()
            self.stores[collection_name] = store

        return self.stores[collection_name]

    async def close_all(self):
        """关闭所有连接"""
        for store in self.stores.values():
            await store.close()
        self.stores.clear()
        logger.info("所有向量存储连接已关闭")
