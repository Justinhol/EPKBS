"""
文档加载器模块
支持多种格式文档的加载和解析
整合新的解析器架构
"""
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredExcelLoader,
)
from langchain.schema import Document
from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title

from src.utils.logger import get_logger
from src.utils.helpers import get_file_extension, format_file_size, generate_hash
from src.data.document_validator import document_validator, DocumentType
from src.data.parsers.base import parser_registry
from src.data.parsers.pdf_parser import PDFMultiModalParser
from src.data.parsers.office_parser import WordParser, PowerPointParser, ExcelParser
from src.data.parsers.text_parser import TextParser, MarkdownParser, HTMLParser, CSVParser, JSONParser
from src.data.parsers.image_parser import MultiModalImageParser
from src.data.parsers.special_parser import TableExtractionParser, FormulaExtractionParser
from src.data.parsers.integrator import document_integrator
from src.data.parsers.cleaner import document_cleaner, document_formatter
from src.data.parsers.chunker import adaptive_chunker
from config.settings import settings

logger = get_logger("data.loaders")


class AdvancedDocumentLoader:
    """高级文档加载器 - 使用新的解析器架构"""

    def __init__(self):
        self._initialize_parsers()
        logger.info(
            f"高级文档加载器初始化完成，支持 {len(parser_registry.get_supported_types())} 种文档类型")

    def _initialize_parsers(self):
        """初始化所有解析器"""
        # 注册主要解析器
        parser_registry.register_parser(PDFMultiModalParser(), is_primary=True)
        parser_registry.register_parser(WordParser(), is_primary=True)
        parser_registry.register_parser(PowerPointParser(), is_primary=True)
        parser_registry.register_parser(ExcelParser(), is_primary=True)
        parser_registry.register_parser(TextParser(), is_primary=True)
        parser_registry.register_parser(MarkdownParser(), is_primary=True)
        parser_registry.register_parser(HTMLParser(), is_primary=True)
        parser_registry.register_parser(CSVParser(), is_primary=True)
        parser_registry.register_parser(JSONParser(), is_primary=True)
        parser_registry.register_parser(
            MultiModalImageParser(), is_primary=True)

        # 注册特殊解析器
        parser_registry.register_parser(
            TableExtractionParser(), is_primary=False)
        parser_registry.register_parser(
            FormulaExtractionParser(), is_primary=False)

    async def load_and_process_document(self, file_path: Union[str, Path]) -> List[Document]:
        """
        加载并处理文档的完整流水线

        Args:
            file_path: 文件路径

        Returns:
            List[Document]: 处理后的文档块列表
        """
        file_path = Path(file_path)

        try:
            # 1. 文档验证
            doc_info = document_validator.validate_file(file_path)
            if not doc_info.is_valid:
                raise ValueError(f"文档验证失败: {doc_info.error_message}")

            logger.info(
                f"开始处理文档: {file_path.name} ({doc_info.file_type.value})")

            # 2. 选择并执行解析器
            parse_results = await self._execute_parsers(file_path, doc_info)

            # 3. 整合解析结果
            integrated_doc = await document_integrator.integrate_parse_results(parse_results, doc_info)

            # 4. 清洗和格式化
            cleaned_doc = document_cleaner.clean_integrated_document(
                integrated_doc)

            # 5. 智能分块
            final_documents = adaptive_chunker.chunk_integrated_document(
                cleaned_doc)

            # 6. 生成RAG友好的JSON（可选）
            rag_json = document_formatter.format_to_rag_json(
                cleaned_doc, file_path)

            logger.info(
                f"文档处理完成: {file_path.name}, 生成 {len(final_documents)} 个文档块")
            return final_documents

        except Exception as e:
            logger.error(f"文档处理失败: {file_path.name}, 错误: {e}")
            raise

    async def _execute_parsers(self, file_path: Path, doc_info) -> List:
        """执行解析器"""
        parse_results = []

        # 获取主解析器
        primary_parser = parser_registry.get_primary_parser(doc_info.file_type)
        if primary_parser:
            try:
                result = await primary_parser.parse(file_path, doc_info)
                parse_results.append(result)
                logger.info(f"主解析器 {primary_parser.name} 执行完成")
            except Exception as e:
                logger.error(f"主解析器 {primary_parser.name} 执行失败: {e}")

        # 执行特殊解析器（如果文档包含特殊元素）
        if doc_info.has_tables and settings.ENABLE_TABLE_EXTRACTION:
            table_parser = parser_registry.get_all_parsers(doc_info.file_type)
            for parser in table_parser:
                if parser.name == "TableExtractionParser":
                    try:
                        result = await parser.parse(file_path, doc_info)
                        parse_results.append(result)
                        logger.info(f"表格解析器执行完成")
                    except Exception as e:
                        logger.error(f"表格解析器执行失败: {e}")
                    break

        if doc_info.has_formulas and settings.ENABLE_FORMULA_EXTRACTION:
            formula_parsers = parser_registry.get_all_parsers(
                doc_info.file_type)
            for parser in formula_parsers:
                if parser.name == "FormulaExtractionParser":
                    try:
                        result = await parser.parse(file_path, doc_info)
                        parse_results.append(result)
                        logger.info(f"公式解析器执行完成")
                    except Exception as e:
                        logger.error(f"公式解析器执行失败: {e}")
                    break

        return parse_results

    async def load_documents(self, file_paths: List[Union[str, Path]]) -> List[Document]:
        """
        批量加载文档

        Args:
            file_paths: 文件路径列表

        Returns:
            List[Document]: 所有文档的处理结果
        """
        all_documents = []

        for file_path in file_paths:
            try:
                documents = await self.load_and_process_document(file_path)
                all_documents.extend(documents)
            except Exception as e:
                logger.error(f"跳过文档 {file_path}: {e}")
                continue

        logger.info(
            f"批量处理完成，共处理 {len(file_paths)} 个文件，生成 {len(all_documents)} 个文档块")
        return all_documents


class DocumentLoader:
    """统一文档加载器（保持向后兼容）"""

    def __init__(self):
        self.supported_extensions = {
            '.pdf': self._load_pdf,
            '.docx': self._load_docx,
            '.pptx': self._load_pptx,
            '.txt': self._load_text,
            '.md': self._load_markdown,
            '.xlsx': self._load_excel,
            '.xls': self._load_excel,
        }
        logger.info(
            f"文档加载器初始化完成，支持格式: {list(self.supported_extensions.keys())}")

    async def load_document(self, file_path: Union[str, Path]) -> List[Document]:
        """
        加载单个文档

        Args:
            file_path: 文件路径

        Returns:
            Document列表
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        extension = get_file_extension(file_path.name)

        if extension not in self.supported_extensions:
            raise ValueError(f"不支持的文件格式: {extension}")

        logger.info(
            f"开始加载文档: {file_path.name} ({format_file_size(file_path.stat().st_size)})")

        try:
            # 调用对应的加载方法
            loader_func = self.supported_extensions[extension]
            documents = await loader_func(file_path)

            # 添加元数据
            for doc in documents:
                doc.metadata.update({
                    'source': str(file_path),
                    'filename': file_path.name,
                    'file_extension': extension,
                    'file_size': file_path.stat().st_size,
                    'created_at': datetime.utcnow().isoformat(),
                    'content_hash': generate_hash(doc.page_content),
                })

            logger.info(f"文档加载完成: {file_path.name}, 生成 {len(documents)} 个文档块")
            return documents

        except Exception as e:
            logger.error(f"文档加载失败: {file_path.name}, 错误: {e}")
            raise

    async def load_documents(self, file_paths: List[Union[str, Path]]) -> List[Document]:
        """
        批量加载文档

        Args:
            file_paths: 文件路径列表

        Returns:
            Document列表
        """
        all_documents = []

        for file_path in file_paths:
            try:
                documents = await self.load_document(file_path)
                all_documents.extend(documents)
            except Exception as e:
                logger.error(f"跳过文档 {file_path}: {e}")
                continue

        logger.info(
            f"批量加载完成，共处理 {len(file_paths)} 个文件，生成 {len(all_documents)} 个文档块")
        return all_documents

    async def _load_pdf(self, file_path: Path) -> List[Document]:
        """加载PDF文档"""
        loader = PyPDFLoader(str(file_path))
        return await asyncio.to_thread(loader.load)

    async def _load_docx(self, file_path: Path) -> List[Document]:
        """加载Word文档"""
        loader = Docx2txtLoader(str(file_path))
        return await asyncio.to_thread(loader.load)

    async def _load_pptx(self, file_path: Path) -> List[Document]:
        """加载PowerPoint文档"""
        loader = UnstructuredPowerPointLoader(str(file_path))
        return await asyncio.to_thread(loader.load)

    async def _load_text(self, file_path: Path) -> List[Document]:
        """加载文本文档"""
        loader = TextLoader(str(file_path), encoding='utf-8')
        return await asyncio.to_thread(loader.load)

    async def _load_markdown(self, file_path: Path) -> List[Document]:
        """加载Markdown文档"""
        loader = UnstructuredMarkdownLoader(str(file_path))
        return await asyncio.to_thread(loader.load)

    async def _load_excel(self, file_path: Path) -> List[Document]:
        """加载Excel文档"""
        loader = UnstructuredExcelLoader(str(file_path))
        return await asyncio.to_thread(loader.load)


class UnstructuredDocumentLoader:
    """基于Unstructured的高级文档加载器"""

    def __init__(self):
        logger.info("Unstructured文档加载器初始化完成")

    async def load_document(self, file_path: Union[str, Path]) -> List[Document]:
        """
        使用Unstructured加载文档

        Args:
            file_path: 文件路径

        Returns:
            Document列表
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        logger.info(f"使用Unstructured加载文档: {file_path.name}")

        try:
            # 使用Unstructured自动分区
            elements = await asyncio.to_thread(
                partition,
                filename=str(file_path),
                strategy="auto",
                include_page_breaks=True,
                infer_table_structure=True,
            )

            # 按标题分块
            chunks = await asyncio.to_thread(
                chunk_by_title,
                elements,
                max_characters=settings.CHUNK_SIZE * 2,  # 稍大一些，后续会进一步分割
                combine_text_under_n_chars=100,
            )

            # 转换为Document对象
            documents = []
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=str(chunk),
                    metadata={
                        'source': str(file_path),
                        'filename': file_path.name,
                        'chunk_index': i,
                        'chunk_type': 'unstructured',
                        'file_size': file_path.stat().st_size,
                        'created_at': datetime.utcnow().isoformat(),
                        'content_hash': generate_hash(str(chunk)),
                    }
                )
                documents.append(doc)

            logger.info(
                f"Unstructured文档加载完成: {file_path.name}, 生成 {len(documents)} 个文档块")
            return documents

        except Exception as e:
            logger.error(f"Unstructured文档加载失败: {file_path.name}, 错误: {e}")
            raise


class DocumentMetadataExtractor:
    """文档元数据提取器"""

    def __init__(self):
        logger.info("文档元数据提取器初始化完成")

    def extract_metadata(self, document: Document) -> Dict[str, Any]:
        """
        提取文档元数据

        Args:
            document: 文档对象

        Returns:
            元数据字典
        """
        metadata = document.metadata.copy()

        # 提取文本统计信息
        content = document.page_content
        metadata.update({
            'char_count': len(content),
            'word_count': len(content.split()),
            'line_count': len(content.split('\n')),
            'paragraph_count': len([p for p in content.split('\n\n') if p.strip()]),
        })

        # 提取语言信息（简单检测）
        if self._contains_chinese(content):
            metadata['language'] = 'zh'
        else:
            metadata['language'] = 'en'

        # 提取文档类型
        metadata['document_type'] = self._infer_document_type(content)

        return metadata

    def _contains_chinese(self, text: str) -> bool:
        """检测文本是否包含中文"""
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                return True
        return False

    def _infer_document_type(self, content: str) -> str:
        """推断文档类型"""
        content_lower = content.lower()

        if any(keyword in content_lower for keyword in ['api', 'function', 'class', 'import']):
            return 'technical'
        elif any(keyword in content_lower for keyword in ['policy', 'procedure', 'guideline']):
            return 'policy'
        elif any(keyword in content_lower for keyword in ['meeting', 'agenda', 'minutes']):
            return 'meeting'
        else:
            return 'general'
