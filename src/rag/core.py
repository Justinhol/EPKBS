"""
RAG核心管理器
整合检索、重排序、上下文压缩等功能
"""
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from langchain.schema import Document

from src.rag.vector_store import MilvusVectorStore, VectorStoreManager
from src.rag.retrievers import (
    VectorRetriever,
    SparseRetriever,
    FullTextRetriever,
    HybridRetriever,
    MultiQueryRetriever
)
from src.rag.rerankers import (
    CrossEncoderReranker,
    MMRReranker,
    EnsembleReranker,
    RerankerFactory
)
from src.data.embeddings import CachedEmbeddingManager
from src.utils.logger import get_logger
from src.utils.helpers import Timer
from config.settings import settings

logger = get_logger("rag.core")


class RAGCore:
    """RAG核心管理器"""

    def __init__(
        self,
        collection_name: str = "knowledge_base",
        embedding_model: str = None,
        reranker_types: List[str] = None,
        enable_cache: bool = True
    ):
        self.collection_name = collection_name
        self.embedding_model = embedding_model or settings.EMBEDDING_MODEL_PATH
        self.reranker_types = reranker_types or ['cross_encoder', 'mmr']
        self.enable_cache = enable_cache

        # 核心组件
        self.vector_store_manager = VectorStoreManager()
        self.vector_store = None
        self.embedding_manager = None

        # 检索器
        self.vector_retriever = None
        self.sparse_retriever = None
        self.fulltext_retriever = None
        self.hybrid_retriever = None
        self.multi_query_retriever = None

        # 重排序器
        self.rerankers = {}
        self.ensemble_reranker = None

        # 文档存储（用于稀疏检索）
        self.documents = []

        logger.info(
            f"RAG核心管理器初始化: 集合={collection_name}, 模型={self.embedding_model}")

    async def initialize(self):
        """初始化RAG系统"""
        logger.info("正在初始化RAG核心系统...")

        try:
            # 1. 初始化嵌入管理器
            logger.info("初始化嵌入管理器...")
            self.embedding_manager = CachedEmbeddingManager(
                model_name=self.embedding_model,
                enable_cache=self.enable_cache
            )
            await self.embedding_manager.initialize()

            # 2. 初始化向量存储
            logger.info("初始化向量存储...")
            embedding_dim = self.embedding_manager.get_embedding_dimension()
            self.vector_store = await self.vector_store_manager.get_store(
                collection_name=self.collection_name,
                embedding_dim=embedding_dim
            )

            # 3. 初始化检索器
            await self._initialize_retrievers()

            # 4. 初始化重排序器
            await self._initialize_rerankers()

            logger.info("RAG核心系统初始化完成")

        except Exception as e:
            logger.error(f"RAG核心系统初始化失败: {e}")
            raise

    async def _initialize_retrievers(self):
        """初始化检索器"""
        logger.info("初始化检索器...")

        # 向量检索器
        self.vector_retriever = VectorRetriever(
            vector_store=self.vector_store,
            embedding_manager=self.embedding_manager
        )

        # 稀疏检索器（需要文档数据）
        self.sparse_retriever = SparseRetriever()

        # 全文检索器
        self.fulltext_retriever = FullTextRetriever()

        # 混合检索器
        self.hybrid_retriever = HybridRetriever(
            vector_retriever=self.vector_retriever,
            sparse_retriever=self.sparse_retriever,
            fulltext_retriever=self.fulltext_retriever
        )

        # 多查询检索器
        self.multi_query_retriever = MultiQueryRetriever(
            base_retriever=self.hybrid_retriever
        )

        logger.info("检索器初始化完成")

    async def _initialize_rerankers(self):
        """初始化重排序器"""
        logger.info("初始化重排序器...")

        # 创建各种重排序器
        for reranker_type in self.reranker_types:
            try:
                reranker = RerankerFactory.create_reranker(reranker_type)
                if hasattr(reranker, 'initialize'):
                    await reranker.initialize()
                self.rerankers[reranker_type] = reranker
                logger.info(f"重排序器 {reranker_type} 初始化完成")
            except Exception as e:
                logger.warning(f"重排序器 {reranker_type} 初始化失败: {e}")

        # 创建集成重排序器
        if len(self.rerankers) > 1:
            # 为Cross-Encoder + MMR设置权重：Cross-Encoder权重更高
            reranker_list = list(self.rerankers.values())
            weights = []
            for reranker in reranker_list:
                if isinstance(reranker, CrossEncoderReranker):
                    weights.append(0.7)  # Cross-Encoder权重更高
                elif isinstance(reranker, MMRReranker):
                    weights.append(0.3)  # MMR权重较低
                else:
                    weights.append(0.5)  # 其他重排序器默认权重

            self.ensemble_reranker = EnsembleReranker(
                rerankers=reranker_list,
                weights=weights
            )
            logger.info(
                f"集成重排序器初始化完成，权重配置: {dict(zip([r.name for r in reranker_list], weights))}")

        logger.info("重排序器初始化完成")

    async def add_documents(
        self,
        documents: List[Document],
        embeddings: List[List[float]] = None,
        batch_size: int = 100
    ) -> List[int]:
        """
        添加文档到RAG系统

        Args:
            documents: 文档列表
            embeddings: 预计算的嵌入向量（可选）
            batch_size: 批处理大小

        Returns:
            插入的文档ID列表
        """
        if not self.vector_store:
            await self.initialize()

        logger.info(f"开始添加 {len(documents)} 个文档到RAG系统")

        try:
            # 如果没有提供嵌入向量，则计算
            if embeddings is None:
                logger.info("计算文档嵌入向量...")
                embeddings = await self.embedding_manager.embed_documents(documents)

            # 添加到向量数据库
            doc_ids = await self.vector_store.add_documents(
                documents=documents,
                embeddings=embeddings,
                batch_size=batch_size
            )

            # 更新本地文档存储（用于稀疏检索）
            self.documents.extend(documents)

            # 重建稀疏检索索引
            if self.sparse_retriever:
                self.sparse_retriever.add_documents(documents)

            # 重建全文检索索引
            if self.fulltext_retriever:
                self.fulltext_retriever.add_documents(documents)

            logger.info(f"文档添加完成，插入 {len(doc_ids)} 个文档")
            return doc_ids

        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            raise

    async def search(
        self,
        query: str,
        top_k: int = None,
        retriever_type: str = "hybrid",
        reranker_type: str = "ensemble",
        enable_multi_query: bool = False,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        执行RAG搜索

        Args:
            query: 查询文本
            top_k: 返回结果数量
            retriever_type: 检索器类型 (vector/sparse/fulltext/hybrid)
            reranker_type: 重排序器类型 (colbert/cross_encoder/mmr/ensemble)
            enable_multi_query: 是否启用多查询检索
            **kwargs: 其他参数

        Returns:
            搜索结果列表
        """
        if not self.vector_store:
            await self.initialize()

        top_k = top_k or settings.TOP_K

        logger.info(
            f"执行RAG搜索: query='{query[:50]}...', top_k={top_k}, retriever={retriever_type}, reranker={reranker_type}")

        try:
            with Timer() as total_timer:
                # 1. 检索阶段
                retrieval_results = await self._retrieve_documents(
                    query=query,
                    top_k=top_k * 2,  # 检索更多文档用于重排序
                    retriever_type=retriever_type,
                    enable_multi_query=enable_multi_query,
                    **kwargs
                )

                if not retrieval_results:
                    logger.warning("检索阶段未找到任何文档")
                    return []

                # 2. 重排序阶段
                reranked_results = await self._rerank_documents(
                    query=query,
                    documents=retrieval_results,
                    reranker_type=reranker_type,
                    top_k=top_k
                )

                # 3. 添加搜索元数据
                for i, result in enumerate(reranked_results):
                    result.update({
                        'final_rank': i + 1,
                        'search_timestamp': datetime.utcnow().isoformat(),
                        'retriever_used': retriever_type,
                        'reranker_used': reranker_type,
                        'multi_query_enabled': enable_multi_query
                    })

            logger.info(
                f"RAG搜索完成，耗时: {total_timer.elapsed:.3f}秒，返回 {len(reranked_results)} 个结果")
            return reranked_results

        except Exception as e:
            logger.error(f"RAG搜索失败: {e}")
            return []

    async def _retrieve_documents(
        self,
        query: str,
        top_k: int,
        retriever_type: str,
        enable_multi_query: bool,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """执行文档检索"""

        # 选择检索器
        if retriever_type == "vector":
            retriever = self.vector_retriever
        elif retriever_type == "sparse":
            retriever = self.sparse_retriever
        elif retriever_type == "fulltext":
            retriever = self.fulltext_retriever
        elif retriever_type == "hybrid":
            retriever = self.hybrid_retriever
        else:
            logger.warning(f"未知的检索器类型: {retriever_type}，使用混合检索器")
            retriever = self.hybrid_retriever

        # 是否使用多查询检索
        if enable_multi_query and self.multi_query_retriever:
            retriever = self.multi_query_retriever

        # 执行检索
        results = await retriever.retrieve(query, top_k, **kwargs)

        logger.info(f"检索完成，使用 {retriever.name}，返回 {len(results)} 个结果")
        return results

    async def _rerank_documents(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        reranker_type: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """执行文档重排序"""

        if not documents:
            return []

        # 选择重排序器
        if reranker_type == "ensemble" and self.ensemble_reranker:
            reranker = self.ensemble_reranker
        elif reranker_type in self.rerankers:
            reranker = self.rerankers[reranker_type]
        else:
            logger.warning(f"未找到重排序器: {reranker_type}，跳过重排序")
            return documents[:top_k]

        # 执行重排序
        reranked_docs = await reranker.rerank(query, documents, top_k)

        logger.info(f"重排序完成，使用 {reranker.name}，返回 {len(reranked_docs)} 个结果")
        return reranked_docs

    async def get_context(
        self,
        query: str,
        top_k: int = None,
        max_context_length: int = 4000,
        **kwargs
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        获取查询的上下文

        Args:
            query: 查询文本
            top_k: 检索文档数量
            max_context_length: 最大上下文长度
            **kwargs: 其他搜索参数

        Returns:
            (上下文文本, 源文档列表)
        """
        # 执行搜索
        search_results = await self.search(query, top_k, **kwargs)

        if not search_results:
            return "", []

        # 构建上下文
        context_parts = []
        current_length = 0
        used_documents = []

        for result in search_results:
            content = result.get('content', '')

            # 检查是否超过最大长度
            if current_length + len(content) > max_context_length:
                # 截断内容
                remaining_length = max_context_length - current_length
                if remaining_length > 100:  # 至少保留100字符
                    content = content[:remaining_length] + "..."
                else:
                    break

            context_parts.append(content)
            current_length += len(content)
            used_documents.append(result)

            if current_length >= max_context_length:
                break

        context = "\n\n".join(context_parts)

        logger.info(
            f"构建上下文完成，长度: {len(context)} 字符，使用 {len(used_documents)} 个文档")
        return context, used_documents

    async def get_stats(self) -> Dict[str, Any]:
        """获取RAG系统统计信息"""
        stats = {
            'collection_name': self.collection_name,
            'embedding_model': self.embedding_model,
            'total_documents': len(self.documents),
            'retrievers': {
                'vector': self.vector_retriever is not None,
                'sparse': self.sparse_retriever is not None,
                'fulltext': self.fulltext_retriever is not None,
                'hybrid': self.hybrid_retriever is not None,
                'multi_query': self.multi_query_retriever is not None,
            },
            'rerankers': list(self.rerankers.keys()),
            'ensemble_reranker': self.ensemble_reranker is not None,
        }

        # 获取向量存储统计信息
        if self.vector_store:
            vector_stats = await self.vector_store.get_collection_stats()
            stats['vector_store'] = vector_stats

        return stats

    async def close(self):
        """关闭RAG系统"""
        try:
            if self.vector_store_manager:
                await self.vector_store_manager.close_all()
            logger.info("RAG系统已关闭")
        except Exception as e:
            logger.warning(f"关闭RAG系统时出现警告: {e}")
