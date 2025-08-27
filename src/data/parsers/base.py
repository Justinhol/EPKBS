"""
文档解析器基类
定义统一的解析器接口
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

from langchain.schema import Document

from src.utils.logger import get_logger
from src.data.document_validator import DocumentInfo, DocumentType

logger = get_logger("data.parsers.base")


@dataclass
class ParsedElement:
    """解析后的元素"""
    element_type: str  # text, image, table, formula, etc.
    content: str
    metadata: Dict[str, Any]
    position: Optional[Dict[str, Any]] = None  # 位置信息
    confidence: float = 1.0  # 解析置信度


@dataclass
class ParseResult:
    """解析结果"""
    success: bool
    elements: List[ParsedElement]
    metadata: Dict[str, Any]
    error_message: Optional[str] = None
    processing_time: float = 0.0


class BaseParser(ABC):
    """文档解析器基类"""
    
    def __init__(self, name: str):
        """
        初始化解析器
        
        Args:
            name: 解析器名称
        """
        self.name = name
        self.supported_types: List[DocumentType] = []
        logger.info(f"解析器 {self.name} 初始化完成")
    
    @abstractmethod
    async def parse(self, file_path: Path, doc_info: DocumentInfo) -> ParseResult:
        """
        解析文档
        
        Args:
            file_path: 文件路径
            doc_info: 文档信息
            
        Returns:
            ParseResult: 解析结果
        """
        pass
    
    def can_parse(self, doc_type: DocumentType) -> bool:
        """
        检查是否支持解析指定类型的文档
        
        Args:
            doc_type: 文档类型
            
        Returns:
            bool: 是否支持
        """
        return doc_type in self.supported_types
    
    def _create_text_element(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        position: Optional[Dict[str, Any]] = None
    ) -> ParsedElement:
        """创建文本元素"""
        return ParsedElement(
            element_type="text",
            content=content,
            metadata=metadata or {},
            position=position
        )
    
    def _create_image_element(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        position: Optional[Dict[str, Any]] = None
    ) -> ParsedElement:
        """创建图像元素"""
        return ParsedElement(
            element_type="image",
            content=content,
            metadata=metadata or {},
            position=position
        )
    
    def _create_table_element(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        position: Optional[Dict[str, Any]] = None
    ) -> ParsedElement:
        """创建表格元素"""
        return ParsedElement(
            element_type="table",
            content=content,
            metadata=metadata or {},
            position=position
        )
    
    def _create_formula_element(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        position: Optional[Dict[str, Any]] = None
    ) -> ParsedElement:
        """创建公式元素"""
        return ParsedElement(
            element_type="formula",
            content=content,
            metadata=metadata or {},
            position=position
        )
    
    def _handle_error(self, error: Exception, file_path: Path) -> ParseResult:
        """处理解析错误"""
        error_msg = f"解析失败 {file_path}: {str(error)}"
        logger.error(error_msg)
        
        return ParseResult(
            success=False,
            elements=[],
            metadata={},
            error_message=error_msg
        )


class MultiModalParser(BaseParser):
    """多模态解析器基类"""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.sub_parsers: List[BaseParser] = []
    
    def add_sub_parser(self, parser: BaseParser):
        """添加子解析器"""
        self.sub_parsers.append(parser)
        logger.info(f"为 {self.name} 添加子解析器: {parser.name}")
    
    async def parse_with_sub_parsers(
        self,
        file_path: Path,
        doc_info: DocumentInfo,
        primary_result: ParseResult
    ) -> ParseResult:
        """使用子解析器增强解析结果"""
        if not primary_result.success:
            return primary_result
        
        enhanced_elements = []
        
        for element in primary_result.elements:
            # 对每个元素尝试使用子解析器增强
            enhanced_element = await self._enhance_element(element, file_path, doc_info)
            enhanced_elements.append(enhanced_element)
        
        return ParseResult(
            success=True,
            elements=enhanced_elements,
            metadata=primary_result.metadata,
            processing_time=primary_result.processing_time
        )
    
    async def _enhance_element(
        self,
        element: ParsedElement,
        file_path: Path,
        doc_info: DocumentInfo
    ) -> ParsedElement:
        """增强单个元素"""
        # 子类可以重写此方法来实现特定的增强逻辑
        return element


class ParserRegistry:
    """解析器注册表"""
    
    def __init__(self):
        self._parsers: Dict[DocumentType, List[BaseParser]] = {}
        self._primary_parsers: Dict[DocumentType, BaseParser] = {}
        logger.info("解析器注册表初始化完成")
    
    def register_parser(self, parser: BaseParser, is_primary: bool = False):
        """
        注册解析器
        
        Args:
            parser: 解析器实例
            is_primary: 是否为主解析器
        """
        for doc_type in parser.supported_types:
            if doc_type not in self._parsers:
                self._parsers[doc_type] = []
            
            self._parsers[doc_type].append(parser)
            
            if is_primary:
                self._primary_parsers[doc_type] = parser
        
        logger.info(f"注册解析器: {parser.name}, 支持类型: {parser.supported_types}")
    
    def get_primary_parser(self, doc_type: DocumentType) -> Optional[BaseParser]:
        """获取主解析器"""
        return self._primary_parsers.get(doc_type)
    
    def get_all_parsers(self, doc_type: DocumentType) -> List[BaseParser]:
        """获取所有支持指定类型的解析器"""
        return self._parsers.get(doc_type, [])
    
    def get_supported_types(self) -> List[DocumentType]:
        """获取所有支持的文档类型"""
        return list(self._parsers.keys())


# 全局解析器注册表
parser_registry = ParserRegistry()
