"""
重排序器模块
实现多种重排序策略：ColBERT、Cross-Encoder、Graph-Based等
"""
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
import numpy as np

from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain.schema import Document

from src.utils.logger import get_logger
from src.utils.helpers import Timer, chunk_list
from config.settings import settings

logger = get_logger("rag.rerankers")


class BaseReranker(ABC):
    """重排序器基类"""
    
    def __init__(self, name: str):
        self.name = name
        logger.info(f"{name} 重排序器初始化完成")
    
    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """重排序文档"""
        pass


class CrossEncoderReranker(BaseReranker):
    """Cross-Encoder重排序器"""
    
    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        name: str = "CrossEncoderReranker"
    ):
        super().__init__(name)
        self.model_name = model_name or settings.RERANKER_MODEL_PATH
        self.device = device or 'cpu'
        self.model = None
        
        logger.info(f"Cross-Encoder重排序器配置: {self.model_name}")
    
    async def initialize(self):
        """初始化模型"""
        try:
            logger.info(f"正在加载Cross-Encoder模型: {self.model_name}")
            
            # 尝试加载Qwen重排序模型
            try:
                self.model = CrossEncoder(self.model_name, device=self.device)
            except Exception as e:
                logger.warning(f"加载Qwen重排序模型失败: {e}")
                # 降级到通用重排序模型
                fallback_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
                logger.info(f"降级使用模型: {fallback_model}")
                self.model = CrossEncoder(fallback_model, device=self.device)
                self.model_name = fallback_model
            
            logger.info("Cross-Encoder模型加载完成")
            
        except Exception as e:
            logger.error(f"Cross-Encoder模型加载失败: {e}")
            raise
    
    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """Cross-Encoder重排序"""
        if not self.model:
            await self.initialize()
        
        if not documents:
            return []
        
        top_k = top_k or len(documents)
        
        try:
            with Timer() as timer:
                # 准备输入对
                pairs = []
                for doc in documents:
                    content = doc.get('content', '')
                    pairs.append([query, content])
                
                # 批量预测分数
                scores = await asyncio.to_thread(self.model.predict, pairs)
                
                # 添加重排序分数
                for i, doc in enumerate(documents):
                    doc['rerank_score'] = float(scores[i])
                    doc['original_rank'] = i + 1
                
                # 按重排序分数排序
                reranked_docs = sorted(
                    documents,
                    key=lambda x: x['rerank_score'],
                    reverse=True
                )
            
            logger.info(f"Cross-Encoder重排序完成，耗时: {timer.elapsed:.3f}秒，处理 {len(documents)} 个文档")
            return reranked_docs[:top_k]
            
        except Exception as e:
            logger.error(f"Cross-Encoder重排序失败: {e}")
            # 返回原始排序
            return documents[:top_k]


class ColBERTReranker(BaseReranker):
    """ColBERT重排序器（简化版本）"""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = None,
        name: str = "ColBERTReranker"
    ):
        super().__init__(name)
        self.model_name = model_name
        self.device = device or 'cpu'
        self.model = None
        
        logger.info(f"ColBERT重排序器配置: {self.model_name}")
    
    async def initialize(self):
        """初始化模型"""
        try:
            logger.info(f"正在加载ColBERT模型: {self.model_name}")
            
            self.model = SentenceTransformer(self.model_name, device=self.device)
            
            logger.info("ColBERT模型加载完成")
            
        except Exception as e:
            logger.error(f"ColBERT模型加载失败: {e}")
            raise
    
    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """ColBERT重排序（简化实现）"""
        if not self.model:
            await self.initialize()
        
        if not documents:
            return []
        
        top_k = top_k or len(documents)
        
        try:
            with Timer() as timer:
                # 编码查询
                query_embedding = await asyncio.to_thread(
                    self.model.encode,
                    [query],
                    normalize_embeddings=True
                )
                
                # 编码文档
                doc_contents = [doc.get('content', '') for doc in documents]
                doc_embeddings = await asyncio.to_thread(
                    self.model.encode,
                    doc_contents,
                    normalize_embeddings=True
                )
                
                # 计算相似度分数
                similarities = np.dot(query_embedding, doc_embeddings.T)[0]
                
                # 添加重排序分数
                for i, doc in enumerate(documents):
                    doc['colbert_score'] = float(similarities[i])
                    doc['original_rank'] = i + 1
                
                # 按ColBERT分数排序
                reranked_docs = sorted(
                    documents,
                    key=lambda x: x['colbert_score'],
                    reverse=True
                )
            
            logger.info(f"ColBERT重排序完成，耗时: {timer.elapsed:.3f}秒，处理 {len(documents)} 个文档")
            return reranked_docs[:top_k]
            
        except Exception as e:
            logger.error(f"ColBERT重排序失败: {e}")
            return documents[:top_k]


class GraphBasedReranker(BaseReranker):
    """基于图的重排序器"""
    
    def __init__(
        self,
        similarity_threshold: float = 0.7,
        name: str = "GraphBasedReranker"
    ):
        super().__init__(name)
        self.similarity_threshold = similarity_threshold
        self.document_graph = {}  # 文档关系图
        
        logger.info(f"Graph-Based重排序器配置，相似度阈值: {self.similarity_threshold}")
    
    def _build_document_graph(self, documents: List[Dict[str, Any]]):
        """构建文档关系图"""
        self.document_graph = {}
        
        for i, doc in enumerate(documents):
            doc_id = doc.get('content_hash', str(i))
            self.document_graph[doc_id] = {
                'document': doc,
                'connections': [],
                'centrality': 0.0
            }
        
        # 计算文档间相似度并建立连接
        for i, doc1 in enumerate(documents):
            doc1_id = doc1.get('content_hash', str(i))
            
            for j, doc2 in enumerate(documents[i+1:], i+1):
                doc2_id = doc2.get('content_hash', str(j))
                
                # 简单的文本相似度计算
                similarity = self._calculate_text_similarity(
                    doc1.get('content', ''),
                    doc2.get('content', '')
                )
                
                if similarity > self.similarity_threshold:
                    self.document_graph[doc1_id]['connections'].append({
                        'target': doc2_id,
                        'weight': similarity
                    })
                    self.document_graph[doc2_id]['connections'].append({
                        'target': doc1_id,
                        'weight': similarity
                    })
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度（简单实现）"""
        if not text1 or not text2:
            return 0.0
        
        # 简单的Jaccard相似度
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_centrality(self):
        """计算节点中心性"""
        for doc_id, node in self.document_graph.items():
            # 简单的度中心性
            degree_centrality = len(node['connections'])
            
            # 加权中心性
            weighted_centrality = sum(
                conn['weight'] for conn in node['connections']
            )
            
            # 综合中心性分数
            node['centrality'] = degree_centrality * 0.3 + weighted_centrality * 0.7
    
    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """基于图的重排序"""
        if not documents:
            return []
        
        top_k = top_k or len(documents)
        
        try:
            with Timer() as timer:
                # 构建文档关系图
                self._build_document_graph(documents)
                
                # 计算中心性
                self._calculate_centrality()
                
                # 结合原始分数和图中心性
                for i, doc in enumerate(documents):
                    doc_id = doc.get('content_hash', str(i))
                    original_score = doc.get('score', 0.0)
                    centrality_score = self.document_graph[doc_id]['centrality']
                    
                    # 综合分数（原始分数权重更高）
                    doc['graph_score'] = original_score * 0.8 + centrality_score * 0.2
                    doc['centrality_score'] = centrality_score
                    doc['original_rank'] = i + 1
                
                # 按图分数排序
                reranked_docs = sorted(
                    documents,
                    key=lambda x: x['graph_score'],
                    reverse=True
                )
            
            logger.info(f"Graph-Based重排序完成，耗时: {timer.elapsed:.3f}秒，处理 {len(documents)} 个文档")
            return reranked_docs[:top_k]
            
        except Exception as e:
            logger.error(f"Graph-Based重排序失败: {e}")
            return documents[:top_k]


class MMRReranker(BaseReranker):
    """最大边际相关性(MMR)重排序器"""
    
    def __init__(
        self,
        lambda_param: float = 0.5,
        name: str = "MMRReranker"
    ):
        super().__init__(name)
        self.lambda_param = lambda_param  # 相关性vs多样性权衡参数
        
        logger.info(f"MMR重排序器配置，λ参数: {self.lambda_param}")
    
    def _calculate_similarity_matrix(self, documents: List[Dict[str, Any]]) -> np.ndarray:
        """计算文档间相似度矩阵"""
        n = len(documents)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                similarity = self._calculate_text_similarity(
                    documents[i].get('content', ''),
                    documents[j].get('content', '')
                )
                similarity_matrix[i][j] = similarity
                similarity_matrix[j][i] = similarity
        
        return similarity_matrix
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """MMR重排序"""
        if not documents:
            return []
        
        top_k = top_k or len(documents)
        
        try:
            with Timer() as timer:
                # 计算文档间相似度矩阵
                similarity_matrix = self._calculate_similarity_matrix(documents)
                
                # MMR算法
                selected = []
                remaining = list(range(len(documents)))
                
                # 选择第一个文档（相关性最高的）
                first_idx = max(remaining, key=lambda i: documents[i].get('score', 0))
                selected.append(first_idx)
                remaining.remove(first_idx)
                
                # 迭代选择剩余文档
                while remaining and len(selected) < top_k:
                    best_idx = None
                    best_score = -float('inf')
                    
                    for idx in remaining:
                        # 相关性分数
                        relevance = documents[idx].get('score', 0)
                        
                        # 多样性分数（与已选文档的最大相似度）
                        max_similarity = max(
                            similarity_matrix[idx][selected_idx]
                            for selected_idx in selected
                        ) if selected else 0
                        
                        # MMR分数
                        mmr_score = (
                            self.lambda_param * relevance -
                            (1 - self.lambda_param) * max_similarity
                        )
                        
                        if mmr_score > best_score:
                            best_score = mmr_score
                            best_idx = idx
                    
                    if best_idx is not None:
                        selected.append(best_idx)
                        remaining.remove(best_idx)
                
                # 构建重排序结果
                reranked_docs = []
                for i, idx in enumerate(selected):
                    doc = documents[idx].copy()
                    doc['mmr_rank'] = i + 1
                    doc['original_rank'] = idx + 1
                    reranked_docs.append(doc)
            
            logger.info(f"MMR重排序完成，耗时: {timer.elapsed:.3f}秒，返回 {len(reranked_docs)} 个文档")
            return reranked_docs
            
        except Exception as e:
            logger.error(f"MMR重排序失败: {e}")
            return documents[:top_k]


class EnsembleReranker(BaseReranker):
    """集成重排序器"""
    
    def __init__(
        self,
        rerankers: List[BaseReranker],
        weights: List[float] = None,
        name: str = "EnsembleReranker"
    ):
        super().__init__(name)
        self.rerankers = rerankers
        self.weights = weights or [1.0] * len(rerankers)
        
        # 归一化权重
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        logger.info(f"集成重排序器配置，包含 {len(rerankers)} 个重排序器")
    
    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """集成重排序"""
        if not documents:
            return []
        
        top_k = top_k or len(documents)
        
        try:
            with Timer() as timer:
                # 并行执行所有重排序器
                tasks = [
                    reranker.rerank(query, documents.copy(), len(documents))
                    for reranker in self.rerankers
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # 合并分数
                doc_scores = {}
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.warning(f"重排序器 {i} 执行失败: {result}")
                        continue
                    
                    weight = self.weights[i]
                    for rank, doc in enumerate(result):
                        doc_id = doc.get('content_hash', str(doc.get('id', rank)))
                        
                        if doc_id not in doc_scores:
                            doc_scores[doc_id] = {
                                'document': doc,
                                'ensemble_score': 0.0,
                                'individual_scores': {}
                            }
                        
                        # 基于排名的分数（排名越高分数越高）
                        rank_score = (len(result) - rank) / len(result)
                        doc_scores[doc_id]['ensemble_score'] += weight * rank_score
                        doc_scores[doc_id]['individual_scores'][f'reranker_{i}'] = rank_score
                
                # 按集成分数排序
                sorted_docs = sorted(
                    doc_scores.values(),
                    key=lambda x: x['ensemble_score'],
                    reverse=True
                )
                
                # 构建最终结果
                final_results = []
                for i, item in enumerate(sorted_docs[:top_k]):
                    doc = item['document'].copy()
                    doc['ensemble_score'] = item['ensemble_score']
                    doc['individual_scores'] = item['individual_scores']
                    doc['ensemble_rank'] = i + 1
                    final_results.append(doc)
            
            logger.info(f"集成重排序完成，耗时: {timer.elapsed:.3f}秒，返回 {len(final_results)} 个文档")
            return final_results
            
        except Exception as e:
            logger.error(f"集成重排序失败: {e}")
            return documents[:top_k]


class RerankerFactory:
    """重排序器工厂"""
    
    @staticmethod
    def create_reranker(reranker_type: str, **kwargs) -> BaseReranker:
        """创建重排序器"""
        rerankers = {
            'cross_encoder': CrossEncoderReranker,
            'colbert': ColBERTReranker,
            'graph_based': GraphBasedReranker,
            'mmr': MMRReranker,
        }
        
        if reranker_type not in rerankers:
            raise ValueError(f"不支持的重排序器类型: {reranker_type}")
        
        return rerankers[reranker_type](**kwargs)
    
    @staticmethod
    def create_ensemble_reranker(
        reranker_types: List[str],
        weights: List[float] = None,
        **kwargs
    ) -> EnsembleReranker:
        """创建集成重排序器"""
        rerankers = [
            RerankerFactory.create_reranker(rtype, **kwargs)
            for rtype in reranker_types
        ]
        
        return EnsembleReranker(rerankers, weights)
