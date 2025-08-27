"""
文本和标记语言文档解析器
支持TXT、MD、HTML、XML、CSV、JSON等格式
"""
import asyncio
import time
import json
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Optional
from io import StringIO

import pandas as pd
from bs4 import BeautifulSoup
import markdown
from markdown.extensions import codehilite, tables, toc

from src.data.parsers.base import BaseParser, ParseResult, ParsedElement
from src.data.document_validator import DocumentInfo, DocumentType
from src.utils.logger import get_logger
from config.settings import settings

logger = get_logger("data.parsers.text")


class TextParser(BaseParser):
    """纯文本解析器"""
    
    def __init__(self):
        super().__init__("TextParser")
        self.supported_types = [DocumentType.TXT]
    
    async def parse(self, file_path: Path, doc_info: DocumentInfo) -> ParseResult:
        """解析文本文件"""
        start_time = time.time()
        
        try:
            # 尝试多种编码
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
            content = None
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                raise ValueError("无法解码文件内容")
            
            # 按段落分割
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            
            elements = []
            for para_index, paragraph in enumerate(paragraphs):
                text_element = self._create_text_element(
                    content=paragraph,
                    metadata={
                        'paragraph_index': para_index,
                        'extraction_method': 'plain_text'
                    },
                    position={'paragraph': para_index}
                )
                elements.append(text_element)
            
            processing_time = time.time() - start_time
            
            return ParseResult(
                success=True,
                elements=elements,
                metadata={
                    'total_paragraphs': len(paragraphs),
                    'total_chars': len(content),
                    'parser': self.name,
                    'extraction_method': 'plain_text'
                },
                processing_time=processing_time
            )
            
        except Exception as e:
            return self._handle_error(e, file_path)


class MarkdownParser(BaseParser):
    """Markdown解析器"""
    
    def __init__(self):
        super().__init__("MarkdownParser")
        self.supported_types = [DocumentType.MD]
        
        # 配置Markdown扩展
        self.md = markdown.Markdown(
            extensions=[
                'codehilite',
                'tables',
                'toc',
                'fenced_code',
                'attr_list'
            ]
        )
    
    async def parse(self, file_path: Path, doc_info: DocumentInfo) -> ParseResult:
        """解析Markdown文件"""
        start_time = time.time()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 转换为HTML
            html_content = self.md.convert(content)
            
            # 解析HTML结构
            soup = BeautifulSoup(html_content, 'html.parser')
            elements = []
            
            # 提取标题
            for i, heading in enumerate(soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])):
                text_element = self._create_text_element(
                    content=heading.get_text(),
                    metadata={
                        'element_type': 'heading',
                        'heading_level': int(heading.name[1]),
                        'heading_index': i,
                        'extraction_method': 'markdown'
                    },
                    position={'heading': i}
                )
                elements.append(text_element)
            
            # 提取段落
            for i, paragraph in enumerate(soup.find_all('p')):
                if paragraph.get_text().strip():
                    text_element = self._create_text_element(
                        content=paragraph.get_text(),
                        metadata={
                            'element_type': 'paragraph',
                            'paragraph_index': i,
                            'extraction_method': 'markdown'
                        },
                        position={'paragraph': i}
                    )
                    elements.append(text_element)
            
            # 提取表格
            for i, table in enumerate(soup.find_all('table')):
                table_data = []
                for row in table.find_all('tr'):
                    row_data = [cell.get_text().strip() for cell in row.find_all(['td', 'th'])]
                    table_data.append(row_data)
                
                markdown_table = self._convert_table_to_markdown(table_data)
                table_element = self._create_table_element(
                    content=markdown_table,
                    metadata={
                        'table_index': i,
                        'rows': len(table_data),
                        'cols': len(table_data[0]) if table_data else 0,
                        'extraction_method': 'markdown'
                    },
                    position={'table': i}
                )
                elements.append(table_element)
            
            # 提取代码块
            for i, code in enumerate(soup.find_all('code')):
                if code.parent.name == 'pre':  # 代码块而不是行内代码
                    text_element = self._create_text_element(
                        content=code.get_text(),
                        metadata={
                            'element_type': 'code',
                            'code_index': i,
                            'extraction_method': 'markdown'
                        },
                        position={'code': i}
                    )
                    elements.append(text_element)
            
            processing_time = time.time() - start_time
            
            return ParseResult(
                success=True,
                elements=elements,
                metadata={
                    'parser': self.name,
                    'extraction_method': 'markdown'
                },
                processing_time=processing_time
            )
            
        except Exception as e:
            return self._handle_error(e, file_path)
    
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


class HTMLParser(BaseParser):
    """HTML解析器"""
    
    def __init__(self):
        super().__init__("HTMLParser")
        self.supported_types = [DocumentType.HTML]
    
    async def parse(self, file_path: Path, doc_info: DocumentInfo) -> ParseResult:
        """解析HTML文件"""
        start_time = time.time()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            soup = BeautifulSoup(content, 'html.parser')
            elements = []
            
            # 移除脚本和样式
            for script in soup(["script", "style"]):
                script.decompose()
            
            # 提取标题
            for i, heading in enumerate(soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])):
                text_element = self._create_text_element(
                    content=heading.get_text().strip(),
                    metadata={
                        'element_type': 'heading',
                        'heading_level': int(heading.name[1]),
                        'heading_index': i,
                        'extraction_method': 'beautifulsoup'
                    },
                    position={'heading': i}
                )
                elements.append(text_element)
            
            # 提取段落
            for i, paragraph in enumerate(soup.find_all('p')):
                text = paragraph.get_text().strip()
                if text:
                    text_element = self._create_text_element(
                        content=text,
                        metadata={
                            'element_type': 'paragraph',
                            'paragraph_index': i,
                            'extraction_method': 'beautifulsoup'
                        },
                        position={'paragraph': i}
                    )
                    elements.append(text_element)
            
            # 提取表格
            for i, table in enumerate(soup.find_all('table')):
                table_data = []
                for row in table.find_all('tr'):
                    row_data = [cell.get_text().strip() for cell in row.find_all(['td', 'th'])]
                    table_data.append(row_data)
                
                markdown_table = self._convert_table_to_markdown(table_data)
                table_element = self._create_table_element(
                    content=markdown_table,
                    metadata={
                        'table_index': i,
                        'rows': len(table_data),
                        'cols': len(table_data[0]) if table_data else 0,
                        'extraction_method': 'beautifulsoup'
                    },
                    position={'table': i}
                )
                elements.append(table_element)
            
            processing_time = time.time() - start_time
            
            return ParseResult(
                success=True,
                elements=elements,
                metadata={
                    'parser': self.name,
                    'extraction_method': 'beautifulsoup'
                },
                processing_time=processing_time
            )
            
        except Exception as e:
            return self._handle_error(e, file_path)
    
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


class CSVParser(BaseParser):
    """CSV解析器"""
    
    def __init__(self):
        super().__init__("CSVParser")
        self.supported_types = [DocumentType.CSV]
    
    async def parse(self, file_path: Path, doc_info: DocumentInfo) -> ParseResult:
        """解析CSV文件"""
        start_time = time.time()
        
        try:
            # 使用pandas读取CSV
            df = pd.read_csv(str(file_path))
            
            # 转换为Markdown表格
            markdown_table = df.to_markdown(index=False)
            
            table_element = self._create_table_element(
                content=markdown_table,
                metadata={
                    'rows': len(df),
                    'cols': len(df.columns),
                    'columns': df.columns.tolist(),
                    'extraction_method': 'pandas'
                }
            )
            
            processing_time = time.time() - start_time
            
            return ParseResult(
                success=True,
                elements=[table_element],
                metadata={
                    'rows': len(df),
                    'cols': len(df.columns),
                    'parser': self.name,
                    'extraction_method': 'pandas'
                },
                processing_time=processing_time
            )
            
        except Exception as e:
            return self._handle_error(e, file_path)


class JSONParser(BaseParser):
    """JSON解析器"""
    
    def __init__(self):
        super().__init__("JSONParser")
        self.supported_types = [DocumentType.JSON]
    
    async def parse(self, file_path: Path, doc_info: DocumentInfo) -> ParseResult:
        """解析JSON文件"""
        start_time = time.time()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 将JSON转换为格式化的文本
            formatted_json = json.dumps(data, ensure_ascii=False, indent=2)
            
            text_element = self._create_text_element(
                content=formatted_json,
                metadata={
                    'data_type': type(data).__name__,
                    'extraction_method': 'json'
                }
            )
            
            processing_time = time.time() - start_time
            
            return ParseResult(
                success=True,
                elements=[text_element],
                metadata={
                    'data_type': type(data).__name__,
                    'parser': self.name,
                    'extraction_method': 'json'
                },
                processing_time=processing_time
            )
            
        except Exception as e:
            return self._handle_error(e, file_path)
