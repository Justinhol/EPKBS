"""
智能分块策略模块
实现三步分块策略：结构分块、语义切割、递归分割
"""
import re
import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.data.parsers.integrator import IntegratedDocument, ContextualElement
from src.utils.logger import get_logger
from config.settings import settings

logger = get_logger("data.parsers.chunker")


@dataclass
class ChunkInfo:
    """分块信息"""
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    chunk_type: str  # structural, semantic, recursive
    parent_element_id: Optional[str] = None
    chunk_index: int = 0
    total_chunks: int = 1


class SmartChunker:
    """智能分块器"""
    
    def __init__(self):
        self.name = "SmartChunker"
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
        self.min_chunk_size = settings.MIN_CHUNK_SIZE
        self.max_chunk_size = settings.MAX_CHUNK_SIZE
        
        # 初始化递归分割器
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", ";", "；", " ", ""]
        )
        
        logger.info(f"智能分块器初始化完成: chunk_size={self.chunk_size}, overlap={self.chunk_overlap}")
    
    def chunk_integrated_document(self, integrated_doc: IntegratedDocument) -> List[Document]:
        """对整合文档进行智能分块"""
        try:
            all_chunks = []
            
            for ctx_element in integrated_doc.elements:
                element_chunks = self._chunk_contextual_element(ctx_element)
                all_chunks.extend(element_chunks)
            
            # 转换为Document对象
            documents = []
            for chunk_info in all_chunks:
                doc = Document(
                    page_content=chunk_info.content,
                    metadata=chunk_info.metadata
                )
                documents.append(doc)
            
            logger.info(f"智能分块完成: {len(integrated_doc.elements)} 个元素 -> {len(documents)} 个块")
            return documents
            
        except Exception as e:
            logger.error(f"智能分块失败: {e}")
            return []
    
    def _chunk_contextual_element(self, ctx_element: ContextualElement) -> List[ChunkInfo]:
        """对单个上下文元素进行分块"""
        element = ctx_element.element
        
        # 第一步：结构分块
        if settings.ENABLE_STRUCTURAL_CHUNKING:
            structural_chunks = self._structural_chunking(ctx_element)
            if self._chunks_meet_size_requirements(structural_chunks):
                return structural_chunks
        
        # 第二步：语义分块
        if settings.ENABLE_SEMANTIC_CHUNKING:
            semantic_chunks = self._semantic_chunking(ctx_element)
            if self._chunks_meet_size_requirements(semantic_chunks):
                return semantic_chunks
        
        # 第三步：递归分块
        if settings.ENABLE_RECURSIVE_CHUNKING:
            recursive_chunks = self._recursive_chunking(ctx_element)
            return recursive_chunks
        
        # 如果所有分块策略都禁用，返回原始内容
        return [self._create_single_chunk(ctx_element)]
    
    def _structural_chunking(self, ctx_element: ContextualElement) -> List[ChunkInfo]:
        """结构分块：按文档结构（标题、段落、表格等）分块"""
        element = ctx_element.element
        chunks = []
        
        if element.element_type == 'text':
            # 按段落分块
            paragraphs = element.content.split('\n\n')
            paragraphs = [p.strip() for p in paragraphs if p.strip()]
            
            for i, paragraph in enumerate(paragraphs):
                if len(paragraph) >= self.min_chunk_size:
                    chunk = ChunkInfo(
                        chunk_id=f"{ctx_element.global_position}_struct_{i}",
                        content=paragraph,
                        metadata=self._create_chunk_metadata(element, 'structural', i, len(paragraphs)),
                        chunk_type='structural',
                        parent_element_id=f"element_{ctx_element.global_position}",
                        chunk_index=i,
                        total_chunks=len(paragraphs)
                    )
                    chunks.append(chunk)
        
        elif element.element_type == 'table':
            # 表格作为单个块
            chunk = ChunkInfo(
                chunk_id=f"{ctx_element.global_position}_struct_0",
                content=element.content,
                metadata=self._create_chunk_metadata(element, 'structural', 0, 1),
                chunk_type='structural',
                parent_element_id=f"element_{ctx_element.global_position}",
                chunk_index=0,
                total_chunks=1
            )
            chunks.append(chunk)
        
        else:
            # 其他类型作为单个块
            chunk = ChunkInfo(
                chunk_id=f"{ctx_element.global_position}_struct_0",
                content=element.content,
                metadata=self._create_chunk_metadata(element, 'structural', 0, 1),
                chunk_type='structural',
                parent_element_id=f"element_{ctx_element.global_position}",
                chunk_index=0,
                total_chunks=1
            )
            chunks.append(chunk)
        
        return chunks
    
    def _semantic_chunking(self, ctx_element: ContextualElement) -> List[ChunkInfo]:
        """语义分块：按语义单元分块"""
        element = ctx_element.element
        content = element.content
        
        # 使用句子边界进行语义分块
        sentences = self._split_into_sentences(content)
        
        chunks = []
        current_chunk = ""
        current_sentences = []
        
        for sentence in sentences:
            # 检查添加当前句子后是否超过大小限制
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk) <= self.chunk_size:
                current_chunk = potential_chunk
                current_sentences.append(sentence)
            else:
                # 保存当前块
                if current_chunk and len(current_chunk) >= self.min_chunk_size:
                    chunk = ChunkInfo(
                        chunk_id=f"{ctx_element.global_position}_sem_{len(chunks)}",
                        content=current_chunk,
                        metadata=self._create_chunk_metadata(element, 'semantic', len(chunks), -1),
                        chunk_type='semantic',
                        parent_element_id=f"element_{ctx_element.global_position}",
                        chunk_index=len(chunks)
                    )
                    chunks.append(chunk)
                
                # 开始新块
                current_chunk = sentence
                current_sentences = [sentence]
        
        # 处理最后一个块
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunk = ChunkInfo(
                chunk_id=f"{ctx_element.global_position}_sem_{len(chunks)}",
                content=current_chunk,
                metadata=self._create_chunk_metadata(element, 'semantic', len(chunks), -1),
                chunk_type='semantic',
                parent_element_id=f"element_{ctx_element.global_position}",
                chunk_index=len(chunks)
            )
            chunks.append(chunk)
        
        # 更新总块数
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return chunks
    
    def _recursive_chunking(self, ctx_element: ContextualElement) -> List[ChunkInfo]:
        """递归分块：使用递归字符分割器"""
        element = ctx_element.element
        
        # 使用langchain的递归分割器
        text_chunks = self.recursive_splitter.split_text(element.content)
        
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            if len(chunk_text) >= self.min_chunk_size:
                chunk = ChunkInfo(
                    chunk_id=f"{ctx_element.global_position}_rec_{i}",
                    content=chunk_text,
                    metadata=self._create_chunk_metadata(element, 'recursive', i, len(text_chunks)),
                    chunk_type='recursive',
                    parent_element_id=f"element_{ctx_element.global_position}",
                    chunk_index=i,
                    total_chunks=len(text_chunks)
                )
                chunks.append(chunk)
        
        return chunks
    
    def _create_single_chunk(self, ctx_element: ContextualElement) -> ChunkInfo:
        """创建单个块（当其他策略都不适用时）"""
        element = ctx_element.element
        
        return ChunkInfo(
            chunk_id=f"{ctx_element.global_position}_single_0",
            content=element.content,
            metadata=self._create_chunk_metadata(element, 'single', 0, 1),
            chunk_type='single',
            parent_element_id=f"element_{ctx_element.global_position}",
            chunk_index=0,
            total_chunks=1
        )
    
    def _chunks_meet_size_requirements(self, chunks: List[ChunkInfo]) -> bool:
        """检查块是否满足大小要求"""
        if not chunks:
            return False
        
        for chunk in chunks:
            chunk_size = len(chunk.content)
            if chunk_size < self.min_chunk_size or chunk_size > self.max_chunk_size:
                return False
        
        return True
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """将文本分割为句子"""
        # 中英文句子分割
        sentence_endings = r'[。！？.!?]+'
        sentences = re.split(sentence_endings, text)
        
        # 清理和过滤
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _create_chunk_metadata(
        self,
        element,
        chunk_type: str,
        chunk_index: int,
        total_chunks: int
    ) -> Dict[str, Any]:
        """创建块的元数据"""
        metadata = element.metadata.copy()
        
        # 添加分块相关的元数据
        metadata.update({
            'chunk_type': chunk_type,
            'chunk_index': chunk_index,
            'total_chunks': total_chunks,
            'chunk_size': len(element.content),
            'chunked_at': datetime.utcnow().isoformat(),
            'chunker_version': '1.0',
            'original_element_type': element.element_type,
            'chunk_strategy': f"{chunk_type}_chunking"
        })
        
        # 添加分块配置信息
        metadata.update({
            'target_chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'min_chunk_size': self.min_chunk_size,
            'max_chunk_size': self.max_chunk_size
        })
        
        return metadata


class AdaptiveChunker(SmartChunker):
    """自适应分块器：根据内容类型自动调整分块策略"""
    
    def __init__(self):
        super().__init__()
        self.name = "AdaptiveChunker"
        
        # 不同内容类型的分块配置
        self.type_configs = {
            'text': {
                'preferred_strategy': 'semantic',
                'chunk_size': self.chunk_size,
                'overlap': self.chunk_overlap
            },
            'table': {
                'preferred_strategy': 'structural',
                'chunk_size': self.max_chunk_size,  # 表格通常需要更大的块
                'overlap': 0  # 表格不需要重叠
            },
            'image': {
                'preferred_strategy': 'structural',
                'chunk_size': self.chunk_size,
                'overlap': 0
            },
            'formula': {
                'preferred_strategy': 'structural',
                'chunk_size': self.chunk_size,
                'overlap': 0
            }
        }
    
    def _chunk_contextual_element(self, ctx_element: ContextualElement) -> List[ChunkInfo]:
        """自适应分块"""
        element = ctx_element.element
        element_type = element.element_type
        
        # 获取该类型的配置
        config = self.type_configs.get(element_type, self.type_configs['text'])
        
        # 临时调整分块参数
        original_chunk_size = self.chunk_size
        original_overlap = self.chunk_overlap
        
        self.chunk_size = config['chunk_size']
        self.chunk_overlap = config['overlap']
        
        try:
            # 根据首选策略进行分块
            preferred_strategy = config['preferred_strategy']
            
            if preferred_strategy == 'structural':
                chunks = self._structural_chunking(ctx_element)
                if self._chunks_meet_size_requirements(chunks):
                    return chunks
            elif preferred_strategy == 'semantic':
                chunks = self._semantic_chunking(ctx_element)
                if self._chunks_meet_size_requirements(chunks):
                    return chunks
            
            # 如果首选策略不满足要求，使用递归分块
            return self._recursive_chunking(ctx_element)
            
        finally:
            # 恢复原始参数
            self.chunk_size = original_chunk_size
            self.chunk_overlap = original_overlap


# 全局分块器实例
smart_chunker = SmartChunker()
adaptive_chunker = AdaptiveChunker()
