"""
数据接入与处理层
提供文档加载、分割、向量化等功能
"""
from .loaders import DocumentLoader, UnstructuredDocumentLoader, DocumentMetadataExtractor
from .splitters import (
    RecursiveSplitter,
    TokenSplitter,
    SemanticSplitter,
    HybridSplitter,
    ChineseSplitter,
    SplitterFactory
)
from .embeddings import (
    EmbeddingManager,
    MultiModelEmbeddingManager,
    CachedEmbeddingManager,
    EmbeddingCache
)
from .pipeline import DocumentProcessingPipeline, BatchProcessor

__all__ = [
    # 加载器
    "DocumentLoader",
    "UnstructuredDocumentLoader",
    "DocumentMetadataExtractor",

    # 分割器
    "RecursiveSplitter",
    "TokenSplitter",
    "SemanticSplitter",
    "HybridSplitter",
    "ChineseSplitter",
    "SplitterFactory",

    # 嵌入管理器
    "EmbeddingManager",
    "MultiModelEmbeddingManager",
    "CachedEmbeddingManager",
    "EmbeddingCache",

    # 处理管道
    "DocumentProcessingPipeline",
    "BatchProcessor",
]
