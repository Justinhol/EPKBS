"""
检索器模块
实现多种检索策略：向量检索、稀疏检索、混合检索等
"""
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
from abc import ABC, abstractmethod
import numpy as np

from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from rank_bm25 import BM25Okapi

from src.rag.vector_store import MilvusVectorStore
from src.data.embeddings import CachedEmbeddingManager
from src.utils.logger import get_logger
from src.utils.helpers import Timer
from config.settings import settings

logger = get_logger("rag.retrievers")


class BaseRetriever(ABC):
    """检索器基类"""
    
    def __init__(self, name: str):
        self.name = name
        logger.info(f"{name} 检索器初始化完成")
    
    @abstractmethod
    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """检索相关文档"""
        pass


class VectorRetriever(BaseRetriever):
    """向量检索器"""
    
    def __init__(
        self,
        vector_store: MilvusVectorStore,
        embedding_manager: CachedEmbeddingManager,
        name: str = "VectorRetriever"
    ):
        super().__init__(name)
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        similarity_threshold: float = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """向量检索"""
        similarity_threshold = similarity_threshold or settings.SIMILARITY_THRESHOLD
        
        try:
            # 查询向量化
            query_embedding = await self.embedding_manager.embed_query(query)
            
            # 向量搜索
            results = await self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                **kwargs
            )
            
            # 过滤低相似度结果
            filtered_results = [
                result for result in results
                if result['score'] >= similarity_threshold
            ]
            
            logger.info(f"向量检索完成，返回 {len(filtered_results)}/{len(results)} 个结果")
            return filtered_results
            
        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            return []


class SparseRetriever(BaseRetriever):
    """稀疏检索器（BM25）"""
    
    def __init__(
        self,
        documents: List[Document] = None,
        name: str = "SparseRetriever"
    ):
        super().__init__(name)
        self.documents = documents or []
        self.bm25 = None
        self.doc_contents = []
        
        if documents:
            self._build_index()
    
    def _build_index(self):
        """构建BM25索引"""
        logger.info(f"构建BM25索引，文档数量: {len(self.documents)}")
        
        # 提取文档内容并分词
        self.doc_contents = []
        tokenized_docs = []
        
        for doc in self.documents:
            content = doc.page_content
            self.doc_contents.append(content)
            
            # 简单分词（可以根据需要使用更复杂的分词器）
            tokens = self._tokenize(content)
            tokenized_docs.append(tokens)
        
        # 构建BM25索引
        self.bm25 = BM25Okapi(tokenized_docs)
        logger.info("BM25索引构建完成")
    
    def _tokenize(self, text: str) -> List[str]:
        """文本分词"""
        # 简单的中英文分词
        import re
        
        # 分离中文字符和英文单词
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        english_words = re.findall(r'[a-zA-Z]+', text.lower())
        
        return chinese_chars + english_words
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """BM25检索"""
        if not self.bm25:
            logger.warning("BM25索引未构建，返回空结果")
            return []
        
        try:
            # 查询分词
            query_tokens = self._tokenize(query)
            
            # BM25搜索
            scores = self.bm25.get_scores(query_tokens)
            
            # 获取top_k结果
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            results = []
            for i, idx in enumerate(top_indices):
                if scores[idx] > 0:  # 只返回有分数的结果
                    result = {
                        'id': idx,
                        'score': float(scores[idx]),
                        'content': self.doc_contents[idx],
                        'metadata': self.documents[idx].metadata,
                        'rank': i + 1,
                        'retriever': self.name
                    }
                    results.append(result)
            
            logger.info(f"BM25检索完成，返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"BM25检索失败: {e}")
            return []
    
    def add_documents(self, documents: List[Document]):
        """添加文档并重建索引"""
        self.documents.extend(documents)
        self._build_index()


class FullTextRetriever(BaseRetriever):
    """全文检索器"""
    
    def __init__(
        self,
        documents: List[Document] = None,
        name: str = "FullTextRetriever"
    ):
        super().__init__(name)
        self.documents = documents or []
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        case_sensitive: bool = False,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """全文检索"""
        if not self.documents:
            return []
        
        try:
            query_lower = query if case_sensitive else query.lower()
            results = []
            
            for i, doc in enumerate(self.documents):
                content = doc.page_content
                content_search = content if case_sensitive else content.lower()
                
                # 计算匹配度
                if query_lower in content_search:
                    # 简单的匹配分数计算
                    match_count = content_search.count(query_lower)
                    score = match_count / len(content.split()) * 100
                    
                    result = {
                        'id': i,
                        'score': score,
                        'content': content,
                        'metadata': doc.metadata,
                        'match_count': match_count,
                        'retriever': self.name
                    }
                    results.append(result)
            
            # 按分数排序
            results.sort(key=lambda x: x['score'], reverse=True)
            
            logger.info(f"全文检索完成，返回 {len(results[:top_k])} 个结果")
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"全文检索失败: {e}")
            return []
    
    def add_documents(self, documents: List[Document]):
        """添加文档"""
        self.documents.extend(documents)


class HybridRetriever(BaseRetriever):
    """混合检索器"""
    
    def __init__(
        self,
        vector_retriever: VectorRetriever,
        sparse_retriever: SparseRetriever,
        fulltext_retriever: FullTextRetriever = None,
        vector_weight: float = None,
        sparse_weight: float = None,
        fulltext_weight: float = 0.1,
        name: str = "HybridRetriever"
    ):
        super().__init__(name)
        self.vector_retriever = vector_retriever
        self.sparse_retriever = sparse_retriever
        self.fulltext_retriever = fulltext_retriever
        
        # 权重配置
        self.vector_weight = vector_weight or settings.VECTOR_SEARCH_WEIGHT
        self.sparse_weight = sparse_weight or settings.SPARSE_SEARCH_WEIGHT
        self.fulltext_weight = fulltext_weight
        
        # 归一化权重
        total_weight = self.vector_weight + self.sparse_weight + self.fulltext_weight
        self.vector_weight /= total_weight
        self.sparse_weight /= total_weight
        self.fulltext_weight /= total_weight
        
        logger.info(f"混合检索器权重 - 向量: {self.vector_weight:.2f}, 稀疏: {self.sparse_weight:.2f}, 全文: {self.fulltext_weight:.2f}")
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """混合检索"""
        try:
            with Timer() as timer:
                # 并行执行多种检索
                tasks = [
                    self.vector_retriever.retrieve(query, top_k * 2, **kwargs),
                    self.sparse_retriever.retrieve(query, top_k * 2, **kwargs)
                ]
                
                if self.fulltext_retriever:
                    tasks.append(self.fulltext_retriever.retrieve(query, top_k * 2, **kwargs))
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                vector_results = results[0] if not isinstance(results[0], Exception) else []
                sparse_results = results[1] if not isinstance(results[1], Exception) else []
                fulltext_results = results[2] if len(results) > 2 and not isinstance(results[2], Exception) else []
            
            # 合并和重新评分
            combined_results = self._combine_results(
                vector_results,
                sparse_results,
                fulltext_results
            )
            
            # 按混合分数排序
            combined_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
            
            logger.info(f"混合检索完成，耗时: {timer.elapsed:.3f}秒，返回 {len(combined_results[:top_k])} 个结果")
            return combined_results[:top_k]
            
        except Exception as e:
            logger.error(f"混合检索失败: {e}")
            return []
    
    def _combine_results(
        self,
        vector_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        fulltext_results: List[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """合并检索结果"""
        combined = {}
        
        # 处理向量检索结果
        for i, result in enumerate(vector_results):
            doc_id = self._get_doc_id(result)
            normalized_score = self._normalize_score(result['score'], 'vector')
            rank_bonus = (len(vector_results) - i) / len(vector_results) * 0.1
            
            combined[doc_id] = {
                **result,
                'hybrid_score': normalized_score * self.vector_weight + rank_bonus,
                'vector_score': result['score'],
                'vector_rank': i + 1,
                'sources': ['vector']
            }
        
        # 处理稀疏检索结果
        for i, result in enumerate(sparse_results):
            doc_id = self._get_doc_id(result)
            normalized_score = self._normalize_score(result['score'], 'sparse')
            rank_bonus = (len(sparse_results) - i) / len(sparse_results) * 0.1
            
            if doc_id in combined:
                combined[doc_id]['hybrid_score'] += normalized_score * self.sparse_weight + rank_bonus
                combined[doc_id]['sparse_score'] = result['score']
                combined[doc_id]['sparse_rank'] = i + 1
                combined[doc_id]['sources'].append('sparse')
            else:
                combined[doc_id] = {
                    **result,
                    'hybrid_score': normalized_score * self.sparse_weight + rank_bonus,
                    'sparse_score': result['score'],
                    'sparse_rank': i + 1,
                    'sources': ['sparse']
                }
        
        # 处理全文检索结果
        if fulltext_results:
            for i, result in enumerate(fulltext_results):
                doc_id = self._get_doc_id(result)
                normalized_score = self._normalize_score(result['score'], 'fulltext')
                rank_bonus = (len(fulltext_results) - i) / len(fulltext_results) * 0.05
                
                if doc_id in combined:
                    combined[doc_id]['hybrid_score'] += normalized_score * self.fulltext_weight + rank_bonus
                    combined[doc_id]['fulltext_score'] = result['score']
                    combined[doc_id]['fulltext_rank'] = i + 1
                    combined[doc_id]['sources'].append('fulltext')
                else:
                    combined[doc_id] = {
                        **result,
                        'hybrid_score': normalized_score * self.fulltext_weight + rank_bonus,
                        'fulltext_score': result['score'],
                        'fulltext_rank': i + 1,
                        'sources': ['fulltext']
                    }
        
        return list(combined.values())
    
    def _get_doc_id(self, result: Dict[str, Any]) -> str:
        """获取文档ID"""
        # 优先使用content_hash，否则使用content的hash
        if 'content_hash' in result:
            return result['content_hash']
        elif 'content' in result:
            import hashlib
            return hashlib.md5(result['content'].encode()).hexdigest()
        else:
            return str(result.get('id', 'unknown'))
    
    def _normalize_score(self, score: float, retriever_type: str) -> float:
        """归一化分数"""
        if retriever_type == 'vector':
            # 向量相似度通常在0-1之间
            return min(max(score, 0), 1)
        elif retriever_type == 'sparse':
            # BM25分数需要归一化
            return min(score / 10.0, 1.0)  # 简单的归一化
        elif retriever_type == 'fulltext':
            # 全文检索分数归一化
            return min(score / 100.0, 1.0)
        else:
            return score


class MultiQueryRetriever(BaseRetriever):
    """多查询检索器 - 生成查询变体并并行检索"""
    
    def __init__(
        self,
        base_retriever: BaseRetriever,
        query_generator=None,  # 将来可以集成LLM生成查询变体
        name: str = "MultiQueryRetriever"
    ):
        super().__init__(name)
        self.base_retriever = base_retriever
        self.query_generator = query_generator
    
    def _generate_query_variants(self, query: str) -> List[str]:
        """生成查询变体（简单版本）"""
        variants = [query]
        
        # 添加一些简单的变体
        if len(query) > 10:
            # 截断查询
            variants.append(query[:len(query)//2])
            
        # 添加关键词提取的变体（简单实现）
        words = query.split()
        if len(words) > 2:
            # 只保留重要词汇
            important_words = [w for w in words if len(w) > 2]
            if important_words:
                variants.append(' '.join(important_words))
        
        return list(set(variants))  # 去重
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """多查询检索"""
        try:
            # 生成查询变体
            query_variants = self._generate_query_variants(query)
            logger.info(f"生成 {len(query_variants)} 个查询变体")
            
            # 并行检索所有变体
            tasks = [
                self.base_retriever.retrieve(variant, top_k, **kwargs)
                for variant in query_variants
            ]
            
            results_list = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 合并结果
            all_results = {}
            for i, results in enumerate(results_list):
                if isinstance(results, Exception):
                    continue
                
                for result in results:
                    doc_id = result.get('content_hash', str(result.get('id', i)))
                    if doc_id not in all_results:
                        all_results[doc_id] = result
                        all_results[doc_id]['query_variants'] = [query_variants[i]]
                    else:
                        # 合并分数
                        all_results[doc_id]['score'] = max(
                            all_results[doc_id]['score'],
                            result['score']
                        )
                        all_results[doc_id]['query_variants'].append(query_variants[i])
            
            # 按分数排序
            final_results = sorted(
                all_results.values(),
                key=lambda x: x['score'],
                reverse=True
            )
            
            logger.info(f"多查询检索完成，返回 {len(final_results[:top_k])} 个结果")
            return final_results[:top_k]
            
        except Exception as e:
            logger.error(f"多查询检索失败: {e}")
            return []
