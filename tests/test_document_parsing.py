"""
文档解析模块测试
"""
import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from src.data.document_validator import document_validator, DocumentType
from src.data.parsers.base import parser_registry
from src.data.parsers.text_parser import TextParser, MarkdownParser
from src.data.parsers.integrator import document_integrator
from src.data.parsers.cleaner import document_cleaner, document_formatter
from src.data.parsers.chunker import smart_chunker, adaptive_chunker
from src.data.parsers.error_handler import global_error_handler, SafeParserWrapper
from src.data.loaders import AdvancedDocumentLoader


class TestDocumentValidator:
    """文档验证器测试"""
    
    def test_validate_text_file(self):
        """测试文本文件验证"""
        # 创建临时文本文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("这是一个测试文档。\n包含多行文本。")
            temp_path = Path(f.name)
        
        try:
            doc_info = document_validator.validate_file(temp_path)
            
            assert doc_info.is_valid
            assert doc_info.file_type == DocumentType.TXT
            assert doc_info.file_size > 0
            assert not doc_info.is_scanned_pdf
            
        finally:
            temp_path.unlink()
    
    def test_validate_invalid_file(self):
        """测试无效文件验证"""
        doc_info = document_validator.validate_file("nonexistent.txt")
        
        assert not doc_info.is_valid
        assert "文件不存在" in doc_info.error_message
    
    def test_validate_unsupported_format(self):
        """测试不支持的格式"""
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            doc_info = document_validator.validate_file(temp_path)
            
            assert not doc_info.is_valid
            assert "不支持的文件类型" in doc_info.error_message
            
        finally:
            temp_path.unlink()


class TestTextParser:
    """文本解析器测试"""
    
    @pytest.mark.asyncio
    async def test_parse_text_file(self):
        """测试解析文本文件"""
        # 创建临时文本文件
        content = "第一段文本。\n\n第二段文本。\n\n第三段文本。"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(content)
            temp_path = Path(f.name)
        
        try:
            parser = TextParser()
            doc_info = document_validator.validate_file(temp_path)
            
            result = await parser.parse(temp_path, doc_info)
            
            assert result.success
            assert len(result.elements) > 0
            assert all(elem.element_type == 'text' for elem in result.elements)
            
        finally:
            temp_path.unlink()


class TestMarkdownParser:
    """Markdown解析器测试"""
    
    @pytest.mark.asyncio
    async def test_parse_markdown_file(self):
        """测试解析Markdown文件"""
        content = """# 标题1

这是第一段文本。

## 标题2

这是第二段文本。

| 列1 | 列2 |
|-----|-----|
| 值1 | 值2 |

```python
print("Hello World")
```
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write(content)
            temp_path = Path(f.name)
        
        try:
            parser = MarkdownParser()
            doc_info = document_validator.validate_file(temp_path)
            
            result = await parser.parse(temp_path, doc_info)
            
            assert result.success
            assert len(result.elements) > 0
            
            # 检查是否包含不同类型的元素
            element_types = [elem.element_type for elem in result.elements]
            assert 'text' in element_types
            
        finally:
            temp_path.unlink()


class TestDocumentIntegrator:
    """文档整合器测试"""
    
    @pytest.mark.asyncio
    async def test_integrate_parse_results(self):
        """测试整合解析结果"""
        # 创建模拟解析结果
        from src.data.parsers.base import ParseResult, ParsedElement
        
        elements = [
            ParsedElement(
                element_type='text',
                content='第一段文本',
                metadata={'paragraph_index': 0},
                position={'paragraph': 0}
            ),
            ParsedElement(
                element_type='text',
                content='第二段文本',
                metadata={'paragraph_index': 1},
                position={'paragraph': 1}
            )
        ]
        
        parse_result = ParseResult(
            success=True,
            elements=elements,
            metadata={'parser': 'test_parser'}
        )
        
        # 创建模拟文档信息
        doc_info = Mock()
        doc_info.file_type = DocumentType.TXT
        
        integrated_doc = await document_integrator.integrate_parse_results([parse_result], doc_info)
        
        assert len(integrated_doc.elements) == 2
        assert integrated_doc.metadata['total_elements'] == 2


class TestDocumentCleaner:
    """文档清洗器测试"""
    
    def test_clean_integrated_document(self):
        """测试清洗整合文档"""
        from src.data.parsers.integrator import IntegratedDocument, ContextualElement
        from src.data.parsers.base import ParsedElement
        
        # 创建测试元素
        element = ParsedElement(
            element_type='text',
            content='这是一个测试文档。   包含多余的空格。\n\n\n包含多余的换行。',
            metadata={},
            position={}
        )
        
        ctx_element = ContextualElement(
            element=element,
            context_before=[],
            context_after=[],
            related_elements=[],
            global_position=0
        )
        
        integrated_doc = IntegratedDocument(
            elements=[ctx_element],
            metadata={},
            structure_map={},
            cross_references={}
        )
        
        cleaned_doc = document_cleaner.clean_integrated_document(integrated_doc)
        
        assert len(cleaned_doc.elements) > 0
        cleaned_content = cleaned_doc.elements[0].element.content
        assert '   ' not in cleaned_content  # 多余空格被清理
        assert '\n\n\n' not in cleaned_content  # 多余换行被清理


class TestSmartChunker:
    """智能分块器测试"""
    
    def test_chunk_integrated_document(self):
        """测试智能分块"""
        from src.data.parsers.integrator import IntegratedDocument, ContextualElement
        from src.data.parsers.base import ParsedElement
        
        # 创建长文本元素
        long_content = "这是一个很长的文档。" * 100  # 创建足够长的内容
        
        element = ParsedElement(
            element_type='text',
            content=long_content,
            metadata={},
            position={}
        )
        
        ctx_element = ContextualElement(
            element=element,
            context_before=[],
            context_after=[],
            related_elements=[],
            global_position=0
        )
        
        integrated_doc = IntegratedDocument(
            elements=[ctx_element],
            metadata={},
            structure_map={},
            cross_references={}
        )
        
        documents = smart_chunker.chunk_integrated_document(integrated_doc)
        
        assert len(documents) > 1  # 应该被分成多个块
        assert all(len(doc.page_content) <= 1024 for doc in documents)  # 每个块不超过最大大小


class TestErrorHandler:
    """错误处理器测试"""
    
    def test_handle_error(self):
        """测试错误处理"""
        from src.data.parsers.error_handler import ErrorSeverity
        
        error = ValueError("测试错误")
        context = {
            'file_path': 'test.txt',
            'parser_name': 'TestParser'
        }
        
        result = global_error_handler.handle_error(error, context, ErrorSeverity.LOW)
        
        assert len(global_error_handler.error_log) > 0
        assert global_error_handler.error_log[-1].error_type == 'ValueError'
    
    def test_safe_parser_wrapper(self):
        """测试安全解析器包装器"""
        # 创建会抛出异常的模拟解析器
        mock_parser = Mock()
        mock_parser.name = "MockParser"
        mock_parser.parse = Mock(side_effect=ValueError("测试错误"))
        
        safe_parser = SafeParserWrapper(mock_parser, global_error_handler)
        
        # 测试异常被捕获
        result = asyncio.run(safe_parser.parse("test.txt", Mock()))
        
        # 验证错误被记录
        assert len(global_error_handler.error_log) > 0


class TestAdvancedDocumentLoader:
    """高级文档加载器测试"""
    
    @pytest.mark.asyncio
    async def test_load_and_process_document(self):
        """测试完整的文档处理流水线"""
        # 创建临时文本文件
        content = "这是一个测试文档。\n\n包含多个段落。\n\n用于测试完整的处理流水线。"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(content)
            temp_path = Path(f.name)
        
        try:
            loader = AdvancedDocumentLoader()
            
            # 模拟解析器注册（避免依赖外部库）
            with patch.object(parser_registry, 'get_primary_parser') as mock_get_parser:
                mock_parser = Mock()
                mock_parser.name = "MockParser"
                
                # 创建模拟解析结果
                from src.data.parsers.base import ParseResult, ParsedElement
                
                mock_elements = [
                    ParsedElement(
                        element_type='text',
                        content='这是一个测试文档。',
                        metadata={},
                        position={'paragraph': 0}
                    ),
                    ParsedElement(
                        element_type='text',
                        content='包含多个段落。',
                        metadata={},
                        position={'paragraph': 1}
                    )
                ]
                
                mock_result = ParseResult(
                    success=True,
                    elements=mock_elements,
                    metadata={'parser': 'MockParser'}
                )
                
                mock_parser.parse = Mock(return_value=mock_result)
                mock_get_parser.return_value = mock_parser
                
                documents = await loader.load_and_process_document(temp_path)
                
                assert len(documents) > 0
                assert all(hasattr(doc, 'page_content') for doc in documents)
                assert all(hasattr(doc, 'metadata') for doc in documents)
                
        finally:
            temp_path.unlink()


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])
