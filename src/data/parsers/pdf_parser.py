"""
PDF文档解析器
支持文本版和扫描版PDF的解析
"""
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import fitz  # PyMuPDF
from paddleocr import PaddleOCR

from src.data.parsers.base import BaseParser, ParseResult, ParsedElement, MultiModalParser
from src.data.document_validator import DocumentInfo, DocumentType
from src.utils.logger import get_logger
from config.settings import settings

logger = get_logger("data.parsers.pdf")


class PDFTextParser(BaseParser):
    """PDF文本解析器（使用PyMuPDF）"""
    
    def __init__(self):
        super().__init__("PDFTextParser")
        self.supported_types = [DocumentType.PDF]
    
    async def parse(self, file_path: Path, doc_info: DocumentInfo) -> ParseResult:
        """解析PDF文档"""
        start_time = time.time()
        
        try:
            doc = fitz.open(str(file_path))
            elements = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # 提取文本
                text = page.get_text()
                if text.strip():
                    text_element = self._create_text_element(
                        content=text,
                        metadata={
                            'page_number': page_num + 1,
                            'extraction_method': 'pymupdf_text'
                        },
                        position={'page': page_num + 1}
                    )
                    elements.append(text_element)
                
                # 提取图像信息
                images = page.get_images()
                for img_index, img in enumerate(images):
                    img_element = self._create_image_element(
                        content=f"[图像 {img_index + 1} 在第 {page_num + 1} 页]",
                        metadata={
                            'page_number': page_num + 1,
                            'image_index': img_index,
                            'image_info': img,
                            'extraction_method': 'pymupdf_image'
                        },
                        position={'page': page_num + 1, 'image_index': img_index}
                    )
                    elements.append(img_element)
                
                # 提取表格（简单检测）
                tables = self._extract_tables_from_page(page, page_num + 1)
                elements.extend(tables)
            
            doc.close()
            
            processing_time = time.time() - start_time
            
            return ParseResult(
                success=True,
                elements=elements,
                metadata={
                    'total_pages': len(doc),
                    'parser': self.name,
                    'extraction_method': 'pymupdf'
                },
                processing_time=processing_time
            )
            
        except Exception as e:
            return self._handle_error(e, file_path)
    
    def _extract_tables_from_page(self, page, page_num: int) -> List[ParsedElement]:
        """从页面提取表格"""
        tables = []
        
        try:
            # 使用PyMuPDF的表格检测
            tabs = page.find_tables()
            
            for tab_index, tab in enumerate(tabs):
                table_data = tab.extract()
                
                # 转换为Markdown格式
                markdown_table = self._convert_table_to_markdown(table_data)
                
                table_element = self._create_table_element(
                    content=markdown_table,
                    metadata={
                        'page_number': page_num,
                        'table_index': tab_index,
                        'rows': len(table_data),
                        'cols': len(table_data[0]) if table_data else 0,
                        'extraction_method': 'pymupdf_table'
                    },
                    position={'page': page_num, 'table_index': tab_index}
                )
                tables.append(table_element)
                
        except Exception as e:
            logger.warning(f"表格提取失败 (页面 {page_num}): {e}")
        
        return tables
    
    def _convert_table_to_markdown(self, table_data: List[List[str]]) -> str:
        """将表格数据转换为Markdown格式"""
        if not table_data:
            return ""
        
        markdown_lines = []
        
        # 表头
        if table_data:
            header = "| " + " | ".join(str(cell or "") for cell in table_data[0]) + " |"
            markdown_lines.append(header)
            
            # 分隔线
            separator = "| " + " | ".join("---" for _ in table_data[0]) + " |"
            markdown_lines.append(separator)
            
            # 数据行
            for row in table_data[1:]:
                row_line = "| " + " | ".join(str(cell or "") for cell in row) + " |"
                markdown_lines.append(row_line)
        
        return "\n".join(markdown_lines)


class PDFOCRParser(BaseParser):
    """PDF OCR解析器（使用PaddleOCR）"""
    
    def __init__(self):
        super().__init__("PDFOCRParser")
        self.supported_types = [DocumentType.PDF_SCANNED]
        self.ocr = None
        self._init_ocr()
    
    def _init_ocr(self):
        """初始化OCR引擎"""
        try:
            if settings.ENABLE_OCR:
                self.ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang=settings.OCR_LANGUAGE,
                    show_log=False
                )
                logger.info("PaddleOCR初始化完成")
            else:
                logger.info("OCR功能已禁用")
        except Exception as e:
            logger.error(f"PaddleOCR初始化失败: {e}")
            self.ocr = None
    
    async def parse(self, file_path: Path, doc_info: DocumentInfo) -> ParseResult:
        """解析扫描版PDF"""
        if not self.ocr:
            return ParseResult(
                success=False,
                elements=[],
                metadata={},
                error_message="OCR引擎未初始化"
            )
        
        start_time = time.time()
        
        try:
            doc = fitz.open(str(file_path))
            elements = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # 将页面转换为图像
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2倍缩放提高OCR精度
                img_data = pix.tobytes("png")
                
                # 使用OCR识别文本
                ocr_result = await asyncio.to_thread(self.ocr.ocr, img_data)
                
                if ocr_result and ocr_result[0]:
                    # 提取文本
                    texts = []
                    for line in ocr_result[0]:
                        text = line[1][0]
                        confidence = line[1][1]
                        texts.append(text)
                    
                    if texts:
                        combined_text = "\n".join(texts)
                        text_element = self._create_text_element(
                            content=combined_text,
                            metadata={
                                'page_number': page_num + 1,
                                'extraction_method': 'paddleocr',
                                'ocr_confidence': sum(line[1][1] for line in ocr_result[0]) / len(ocr_result[0])
                            },
                            position={'page': page_num + 1}
                        )
                        elements.append(text_element)
            
            doc.close()
            
            processing_time = time.time() - start_time
            
            return ParseResult(
                success=True,
                elements=elements,
                metadata={
                    'total_pages': len(doc),
                    'parser': self.name,
                    'extraction_method': 'paddleocr'
                },
                processing_time=processing_time
            )
            
        except Exception as e:
            return self._handle_error(e, file_path)


class PDFMultiModalParser(MultiModalParser):
    """PDF多模态解析器"""
    
    def __init__(self):
        super().__init__("PDFMultiModalParser")
        self.supported_types = [DocumentType.PDF, DocumentType.PDF_SCANNED]
        
        # 添加子解析器
        self.text_parser = PDFTextParser()
        self.ocr_parser = PDFOCRParser()
        
        self.add_sub_parser(self.text_parser)
        self.add_sub_parser(self.ocr_parser)
    
    async def parse(self, file_path: Path, doc_info: DocumentInfo) -> ParseResult:
        """解析PDF文档"""
        start_time = time.time()
        
        try:
            # 根据文档类型选择主解析器
            if doc_info.file_type == DocumentType.PDF_SCANNED:
                primary_result = await self.ocr_parser.parse(file_path, doc_info)
            else:
                primary_result = await self.text_parser.parse(file_path, doc_info)
            
            # 如果主解析器失败，尝试备用解析器
            if not primary_result.success and doc_info.file_type == DocumentType.PDF:
                logger.info("文本解析失败，尝试OCR解析")
                primary_result = await self.ocr_parser.parse(file_path, doc_info)
            
            # 使用子解析器增强结果
            enhanced_result = await self.parse_with_sub_parsers(file_path, doc_info, primary_result)
            
            enhanced_result.processing_time = time.time() - start_time
            enhanced_result.metadata['parser'] = self.name
            
            return enhanced_result
            
        except Exception as e:
            return self._handle_error(e, file_path)
