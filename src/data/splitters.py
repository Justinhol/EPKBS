"""
文档分割器模块
实现多种智能分割策略
"""
import re
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

from src.utils.logger import get_logger
from config.settings import settings

logger = get_logger("data.splitters")


class BaseSplitter(ABC):
    """分割器基类"""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        logger.info(f"{self.__class__.__name__} 初始化完成，chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}")
    
    @abstractmethod
    async def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档"""
        pass


class RecursiveSplitter(BaseSplitter):
    """递归字符分割器"""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        super().__init__(chunk_size, chunk_overlap)
        
        # 中英文分隔符
        separators = [
            "\n\n",  # 段落分隔
            "\n",    # 行分隔
            "。",    # 中文句号
            "！",    # 中文感叹号
            "？",    # 中文问号
            ".",     # 英文句号
            "!",     # 英文感叹号
            "?",     # 英文问号
            ";",     # 分号
            ",",     # 逗号
            " ",     # 空格
            "",      # 字符级别
        ]
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=separators,
            length_function=len,
            is_separator_regex=False,
        )
    
    async def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档"""
        logger.info(f"开始递归分割 {len(documents)} 个文档")
        
        split_docs = self.splitter.split_documents(documents)
        
        # 添加分割相关的元数据
        for i, doc in enumerate(split_docs):
            doc.metadata.update({
                'chunk_id': i,
                'splitter_type': 'recursive',
                'chunk_size': len(doc.page_content),
            })
        
        logger.info(f"递归分割完成，生成 {len(split_docs)} 个文档块")
        return split_docs


class TokenSplitter(BaseSplitter):
    """基于Token的分割器"""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None, model_name: str = "gpt-3.5-turbo"):
        super().__init__(chunk_size, chunk_overlap)
        
        self.splitter = TokenTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            model_name=model_name,
        )
    
    async def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档"""
        logger.info(f"开始Token分割 {len(documents)} 个文档")
        
        split_docs = self.splitter.split_documents(documents)
        
        # 添加分割相关的元数据
        for i, doc in enumerate(split_docs):
            doc.metadata.update({
                'chunk_id': i,
                'splitter_type': 'token',
                'chunk_size': len(doc.page_content),
            })
        
        logger.info(f"Token分割完成，生成 {len(split_docs)} 个文档块")
        return split_docs


class SemanticSplitter(BaseSplitter):
    """语义分割器"""
    
    def __init__(self, embedding_model: str = None, chunk_size: int = None):
        # 语义分割器不使用传统的chunk_overlap
        super().__init__(chunk_size, 0)
        
        self.embedding_model_name = embedding_model or settings.EMBEDDING_MODEL_PATH
        
        try:
            # 初始化嵌入模型
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={'device': 'cpu'},  # 可以根据需要改为 'cuda'
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # 初始化语义分割器
            self.splitter = SemanticChunker(
                embeddings=self.embeddings,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=95,
            )
            
            logger.info(f"语义分割器初始化完成，使用模型: {self.embedding_model_name}")
            
        except Exception as e:
            logger.error(f"语义分割器初始化失败: {e}")
            # 降级到递归分割器
            logger.warning("降级使用递归分割器")
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
    
    async def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档"""
        logger.info(f"开始语义分割 {len(documents)} 个文档")
        
        try:
            split_docs = self.splitter.split_documents(documents)
            
            # 添加分割相关的元数据
            for i, doc in enumerate(split_docs):
                doc.metadata.update({
                    'chunk_id': i,
                    'splitter_type': 'semantic',
                    'chunk_size': len(doc.page_content),
                    'embedding_model': self.embedding_model_name,
                })
            
            logger.info(f"语义分割完成，生成 {len(split_docs)} 个文档块")
            return split_docs
            
        except Exception as e:
            logger.error(f"语义分割失败: {e}")
            # 降级到递归分割
            logger.warning("降级使用递归分割")
            recursive_splitter = RecursiveSplitter(self.chunk_size, self.chunk_overlap)
            return await recursive_splitter.split_documents(documents)


class HybridSplitter(BaseSplitter):
    """混合分割器 - 结合多种分割策略"""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        super().__init__(chunk_size, chunk_overlap)
        
        # 初始化多个分割器
        self.recursive_splitter = RecursiveSplitter(chunk_size, chunk_overlap)
        self.semantic_splitter = SemanticSplitter(chunk_size=chunk_size)
    
    async def split_documents(self, documents: List[Document]) -> List[Document]:
        """混合分割文档"""
        logger.info(f"开始混合分割 {len(documents)} 个文档")
        
        all_chunks = []
        
        for doc in documents:
            # 根据文档类型和长度选择分割策略
            content_length = len(doc.page_content)
            doc_type = doc.metadata.get('document_type', 'general')
            
            if content_length > self.chunk_size * 3 and doc_type in ['technical', 'policy']:
                # 长文档且为技术或政策类文档，使用语义分割
                logger.debug(f"使用语义分割处理长文档: {doc.metadata.get('filename', 'unknown')}")
                chunks = await self.semantic_splitter.split_documents([doc])
            else:
                # 其他情况使用递归分割
                logger.debug(f"使用递归分割处理文档: {doc.metadata.get('filename', 'unknown')}")
                chunks = await self.recursive_splitter.split_documents([doc])
            
            # 添加混合分割标识
            for chunk in chunks:
                chunk.metadata['splitter_type'] = 'hybrid'
            
            all_chunks.extend(chunks)
        
        logger.info(f"混合分割完成，生成 {len(all_chunks)} 个文档块")
        return all_chunks


class ChineseSplitter(BaseSplitter):
    """中文优化分割器"""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        super().__init__(chunk_size, chunk_overlap)
        
        # 中文特定的分隔符
        chinese_separators = [
            "\n\n",
            "\n",
            "。",
            "！",
            "？",
            "；",
            "，",
            "、",
            " ",
            "",
        ]
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=chinese_separators,
            length_function=len,
        )
    
    async def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割中文文档"""
        logger.info(f"开始中文分割 {len(documents)} 个文档")
        
        split_docs = self.splitter.split_documents(documents)
        
        # 添加分割相关的元数据
        for i, doc in enumerate(split_docs):
            doc.metadata.update({
                'chunk_id': i,
                'splitter_type': 'chinese',
                'chunk_size': len(doc.page_content),
            })
        
        logger.info(f"中文分割完成，生成 {len(split_docs)} 个文档块")
        return split_docs


class SplitterFactory:
    """分割器工厂类"""
    
    @staticmethod
    def create_splitter(splitter_type: str, **kwargs) -> BaseSplitter:
        """
        创建分割器
        
        Args:
            splitter_type: 分割器类型
            **kwargs: 分割器参数
            
        Returns:
            分割器实例
        """
        splitters = {
            'recursive': RecursiveSplitter,
            'token': TokenSplitter,
            'semantic': SemanticSplitter,
            'hybrid': HybridSplitter,
            'chinese': ChineseSplitter,
        }
        
        if splitter_type not in splitters:
            raise ValueError(f"不支持的分割器类型: {splitter_type}")
        
        return splitters[splitter_type](**kwargs)
