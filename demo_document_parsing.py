#!/usr/bin/env python3
"""
文档解析模块演示脚本
展示新的文档解析功能
"""
import asyncio
import tempfile
import json
from pathlib import Path

from src.data.document_validator import document_validator
from src.data.parsers.text_parser import TextParser, MarkdownParser
from src.data.parsers.integrator import document_integrator
from src.data.parsers.cleaner import document_cleaner, document_formatter
from src.data.parsers.chunker import adaptive_chunker
from src.data.parsers.error_handler import global_error_handler
from src.utils.logger import get_logger

logger = get_logger("demo")


async def demo_text_parsing():
    """演示文本解析功能"""
    print("\n=== 文本解析演示 ===")
    
    # 创建示例文本文件
    content = """企业私有知识库系统

这是一个基于RAG技术的企业私有知识库系统。

## 主要功能

1. 多格式文档解析
2. 智能内容提取
3. 向量化存储
4. 语义检索

## 技术特点

- 支持20+种文档格式
- 多模态内容处理
- 智能分块策略
- 强大的错误处理

系统采用分层架构，确保高性能和可扩展性。
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
        f.write(content)
        temp_path = Path(f.name)
    
    try:
        # 1. 文档验证
        print("1. 文档验证...")
        doc_info = document_validator.validate_file(temp_path)
        print(f"   文档类型: {doc_info.file_type.value}")
        print(f"   文件大小: {doc_info.file_size} 字节")
        print(f"   验证结果: {'通过' if doc_info.is_valid else '失败'}")
        
        # 2. 文档解析
        print("\n2. 文档解析...")
        parser = MarkdownParser()
        parse_result = await parser.parse(temp_path, doc_info)
        print(f"   解析结果: {'成功' if parse_result.success else '失败'}")
        print(f"   提取元素: {len(parse_result.elements)} 个")
        
        # 显示解析的元素
        for i, element in enumerate(parse_result.elements[:3]):  # 只显示前3个
            print(f"   元素 {i+1}: {element.element_type}")
            print(f"     内容预览: {element.content[:50]}...")
        
        # 3. 内容整合
        print("\n3. 内容整合...")
        integrated_doc = await document_integrator.integrate_parse_results([parse_result], doc_info)
        print(f"   整合元素: {len(integrated_doc.elements)} 个")
        print(f"   结构信息: {len(integrated_doc.structure_map)} 个部分")
        
        # 4. 内容清洗
        print("\n4. 内容清洗...")
        cleaned_doc = document_cleaner.clean_integrated_document(integrated_doc)
        print(f"   清洗后元素: {len(cleaned_doc.elements)} 个")
        
        # 5. 智能分块
        print("\n5. 智能分块...")
        documents = adaptive_chunker.chunk_integrated_document(cleaned_doc)
        print(f"   生成文档块: {len(documents)} 个")
        
        # 显示分块结果
        for i, doc in enumerate(documents[:2]):  # 只显示前2个
            print(f"   块 {i+1}: {len(doc.page_content)} 字符")
            print(f"     内容预览: {doc.page_content[:80]}...")
        
        # 6. 格式化输出
        print("\n6. 格式化输出...")
        rag_json = document_formatter.format_to_rag_json(cleaned_doc, temp_path)
        print(f"   JSON结构: {len(rag_json)} 个字段")
        print(f"   文档元数据: {len(rag_json.get('document_metadata', {}))} 个属性")
        
        return True
        
    except Exception as e:
        logger.error(f"演示失败: {e}")
        return False
        
    finally:
        temp_path.unlink()


async def demo_error_handling():
    """演示错误处理功能"""
    print("\n=== 错误处理演示 ===")
    
    # 1. 测试文件不存在错误
    print("1. 测试文件不存在错误...")
    doc_info = document_validator.validate_file("nonexistent.txt")
    print(f"   验证结果: {'通过' if doc_info.is_valid else '失败'}")
    print(f"   错误信息: {doc_info.error_message}")
    
    # 2. 测试不支持的格式
    print("\n2. 测试不支持的格式...")
    with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
        temp_path = Path(f.name)
    
    try:
        doc_info = document_validator.validate_file(temp_path)
        print(f"   验证结果: {'通过' if doc_info.is_valid else '失败'}")
        print(f"   错误信息: {doc_info.error_message}")
        
    finally:
        temp_path.unlink()
    
    # 3. 显示错误统计
    print("\n3. 错误统计...")
    error_summary = global_error_handler.get_error_summary()
    print(f"   总错误数: {error_summary['total_errors']}")
    print(f"   错误类型: {error_summary['error_types']}")
    print(f"   恢复成功率: {error_summary['recovery_success_rate']:.2%}")


def demo_supported_formats():
    """演示支持的文档格式"""
    print("\n=== 支持的文档格式 ===")
    
    from src.data.document_validator import DocumentType
    
    format_groups = {
        "Office文档": [DocumentType.DOC, DocumentType.DOCX, DocumentType.PPT, DocumentType.PPTX, DocumentType.XLS, DocumentType.XLSX],
        "PDF文档": [DocumentType.PDF, DocumentType.PDF_SCANNED],
        "网页和标记": [DocumentType.HTML, DocumentType.MD, DocumentType.XML],
        "电子书": [DocumentType.EPUB],
        "文本文档": [DocumentType.TXT],
        "邮件": [DocumentType.EML, DocumentType.MSG],
        "数据文件": [DocumentType.CSV, DocumentType.JSON],
        "图像文件": [DocumentType.JPG, DocumentType.JPEG, DocumentType.PNG, DocumentType.TIFF, DocumentType.TIF]
    }
    
    for group_name, formats in format_groups.items():
        print(f"\n{group_name}:")
        for fmt in formats:
            print(f"  - {fmt.value.upper()}: {fmt.name}")
    
    print(f"\n总计支持 {sum(len(formats) for formats in format_groups.values())} 种文档格式")


def demo_parsing_features():
    """演示解析功能特性"""
    print("\n=== 解析功能特性 ===")
    
    features = {
        "多模态解析": [
            "文本内容提取",
            "图像OCR识别", 
            "表格结构化提取",
            "数学公式识别"
        ],
        "智能整合": [
            "上下文关系保留",
            "交叉引用建立",
            "结构信息映射",
            "元素位置追踪"
        ],
        "内容清洗": [
            "噪声自动移除",
            "格式标准化",
            "OCR错误修复",
            "重复内容去除"
        ],
        "分块策略": [
            "结构化分块",
            "语义边界分割",
            "递归字符分割",
            "自适应大小调整"
        ],
        "错误处理": [
            "多级错误分类",
            "自动恢复策略",
            "详细错误日志",
            "容错机制保障"
        ]
    }
    
    for category, feature_list in features.items():
        print(f"\n{category}:")
        for feature in feature_list:
            print(f"  ✓ {feature}")


async def main():
    """主演示函数"""
    print("🚀 企业私有知识库系统 - 文档解析模块演示")
    print("=" * 60)
    
    # 1. 显示支持的格式
    demo_supported_formats()
    
    # 2. 显示解析功能特性
    demo_parsing_features()
    
    # 3. 演示文本解析
    success = await demo_text_parsing()
    
    # 4. 演示错误处理
    await demo_error_handling()
    
    # 5. 总结
    print("\n=== 演示总结 ===")
    if success:
        print("✅ 文档解析模块演示成功完成！")
        print("\n主要亮点:")
        print("  • 支持20+种文档格式")
        print("  • 多模态内容解析")
        print("  • 智能分块策略")
        print("  • 强大的错误处理")
        print("  • RAG友好的输出格式")
    else:
        print("❌ 演示过程中遇到问题，请检查日志")
    
    print("\n📚 文档解析模块已准备就绪，可以开始处理您的文档！")


if __name__ == "__main__":
    asyncio.run(main())
