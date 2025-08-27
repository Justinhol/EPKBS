"""
文档清洗与格式化模块
移除噪声、格式化内容、提取元数据
"""
import re
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

from src.data.parsers.integrator import IntegratedDocument, ContextualElement
from src.data.parsers.base import ParsedElement
from src.utils.logger import get_logger
from config.settings import settings

logger = get_logger("data.parsers.cleaner")


class DocumentCleaner:
    """文档清洗器"""
    
    def __init__(self):
        self.name = "DocumentCleaner"
        
        # 噪声模式
        self.noise_patterns = [
            # 页眉页脚
            r'^第\s*\d+\s*页.*$',
            r'^\d+\s*/\s*\d+$',
            r'^Page\s+\d+.*$',
            r'^.*页码.*$',
            
            # 水印
            r'机密|内部资料|CONFIDENTIAL|INTERNAL',
            r'仅供内部使用|For Internal Use Only',
            
            # 广告和推广
            r'点击.*了解更多|Click.*for more',
            r'扫码关注|Scan QR Code',
            r'微信公众号|WeChat Official Account',
            
            # 版权信息
            r'©.*版权所有|Copyright.*All Rights Reserved',
            r'保留所有权利|All Rights Reserved',
            
            # 重复的分隔符
            r'^[-=_*]{3,}$',
            r'^[.]{3,}$',
            
            # 空白和无意义内容
            r'^\s*$',
            r'^[\s\n\r\t]*$',
        ]
        
        self.compiled_noise_patterns = [re.compile(pattern, re.IGNORECASE | re.MULTILINE) 
                                       for pattern in self.noise_patterns]
        
        logger.info("文档清洗器初始化完成")
    
    def clean_integrated_document(self, integrated_doc: IntegratedDocument) -> IntegratedDocument:
        """清洗整合后的文档"""
        try:
            cleaned_elements = []
            
            for ctx_element in integrated_doc.elements:
                cleaned_element = self._clean_contextual_element(ctx_element)
                if cleaned_element:  # 只保留非空元素
                    cleaned_elements.append(cleaned_element)
            
            # 更新元数据
            cleaned_metadata = integrated_doc.metadata.copy()
            cleaned_metadata.update({
                'cleaned_at': datetime.utcnow().isoformat(),
                'original_elements': len(integrated_doc.elements),
                'cleaned_elements': len(cleaned_elements),
                'cleaning_method': 'noise_removal_and_formatting'
            })
            
            cleaned_doc = IntegratedDocument(
                elements=cleaned_elements,
                metadata=cleaned_metadata,
                structure_map=integrated_doc.structure_map,
                cross_references=integrated_doc.cross_references
            )
            
            logger.info(f"文档清洗完成: {len(integrated_doc.elements)} -> {len(cleaned_elements)} 个元素")
            return cleaned_doc
            
        except Exception as e:
            logger.error(f"文档清洗失败: {e}")
            return integrated_doc
    
    def _clean_contextual_element(self, ctx_element: ContextualElement) -> Optional[ContextualElement]:
        """清洗单个上下文元素"""
        # 清洗主元素
        cleaned_main = self._clean_parsed_element(ctx_element.element)
        if not cleaned_main:
            return None
        
        # 清洗上下文元素
        cleaned_before = [self._clean_parsed_element(elem) for elem in ctx_element.context_before]
        cleaned_before = [elem for elem in cleaned_before if elem]
        
        cleaned_after = [self._clean_parsed_element(elem) for elem in ctx_element.context_after]
        cleaned_after = [elem for elem in cleaned_after if elem]
        
        cleaned_related = [self._clean_parsed_element(elem) for elem in ctx_element.related_elements]
        cleaned_related = [elem for elem in cleaned_related if elem]
        
        return ContextualElement(
            element=cleaned_main,
            context_before=cleaned_before,
            context_after=cleaned_after,
            related_elements=cleaned_related,
            global_position=ctx_element.global_position
        )
    
    def _clean_parsed_element(self, element: ParsedElement) -> Optional[ParsedElement]:
        """清洗解析元素"""
        if not element or not element.content:
            return None
        
        # 根据元素类型进行不同的清洗
        if element.element_type == 'text':
            cleaned_content = self._clean_text_content(element.content)
        elif element.element_type == 'table':
            cleaned_content = self._clean_table_content(element.content)
        elif element.element_type == 'image':
            cleaned_content = self._clean_image_content(element.content)
        elif element.element_type == 'formula':
            cleaned_content = self._clean_formula_content(element.content)
        else:
            cleaned_content = self._clean_generic_content(element.content)
        
        if not cleaned_content or len(cleaned_content.strip()) < 10:
            return None
        
        # 创建清洗后的元素
        cleaned_element = ParsedElement(
            element_type=element.element_type,
            content=cleaned_content,
            metadata=element.metadata.copy(),
            position=element.position,
            confidence=element.confidence
        )
        
        # 更新元数据
        cleaned_element.metadata['cleaned_at'] = datetime.utcnow().isoformat()
        cleaned_element.metadata['original_length'] = len(element.content)
        cleaned_element.metadata['cleaned_length'] = len(cleaned_content)
        
        return cleaned_element
    
    def _clean_text_content(self, content: str) -> str:
        """清洗文本内容"""
        # 移除噪声
        cleaned = self._remove_noise(content)
        
        # 标准化空白字符
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned)
        
        # 移除多余的标点符号
        cleaned = re.sub(r'[.]{3,}', '...', cleaned)
        cleaned = re.sub(r'[-]{3,}', '---', cleaned)
        
        # 修复常见的OCR错误
        cleaned = self._fix_ocr_errors(cleaned)
        
        return cleaned.strip()
    
    def _clean_table_content(self, content: str) -> str:
        """清洗表格内容"""
        # 移除表格中的噪声
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # 跳过空行和分隔线
            if not line.strip() or re.match(r'^[\s|:-]+$', line):
                if line.strip():  # 保留有意义的分隔线
                    cleaned_lines.append(line)
                continue
            
            # 清洗表格行
            cleaned_line = self._remove_noise(line)
            if cleaned_line.strip():
                cleaned_lines.append(cleaned_line)
        
        return '\n'.join(cleaned_lines)
    
    def _clean_image_content(self, content: str) -> str:
        """清洗图像内容"""
        # 移除图像描述中的噪声
        cleaned = self._remove_noise(content)
        
        # 标准化图像描述格式
        cleaned = re.sub(r'\[图像\s*\d*\s*\]', '[图像]', cleaned)
        cleaned = re.sub(r'\[图片\s*\d*\s*\]', '[图像]', cleaned)
        
        return cleaned.strip()
    
    def _clean_formula_content(self, content: str) -> str:
        """清洗公式内容"""
        # 移除公式中的噪声（保守处理）
        cleaned = content.strip()
        
        # 标准化LaTeX格式
        cleaned = re.sub(r'\$\s+', '$', cleaned)
        cleaned = re.sub(r'\s+\$', '$', cleaned)
        
        return cleaned
    
    def _clean_generic_content(self, content: str) -> str:
        """通用内容清洗"""
        return self._clean_text_content(content)
    
    def _remove_noise(self, content: str) -> str:
        """移除噪声"""
        cleaned = content
        
        for pattern in self.compiled_noise_patterns:
            cleaned = pattern.sub('', cleaned)
        
        return cleaned
    
    def _fix_ocr_errors(self, content: str) -> str:
        """修复常见的OCR错误"""
        # 常见的OCR错误映射
        ocr_fixes = {
            r'0(?=\d)': 'O',  # 数字0误识别为字母O
            r'1(?=[a-zA-Z])': 'l',  # 数字1误识别为字母l
            r'5(?=[a-zA-Z])': 'S',  # 数字5误识别为字母S
            r'8(?=[a-zA-Z])': 'B',  # 数字8误识别为字母B
            r'(?<=[a-zA-Z])0': 'o',  # 字母O误识别为数字0
            r'(?<=[a-zA-Z])1': 'l',  # 字母l误识别为数字1
        }
        
        fixed = content
        for pattern, replacement in ocr_fixes.items():
            fixed = re.sub(pattern, replacement, fixed)
        
        return fixed


class DocumentFormatter:
    """文档格式化器"""
    
    def __init__(self):
        self.name = "DocumentFormatter"
        logger.info("文档格式化器初始化完成")
    
    def format_to_rag_json(self, integrated_doc: IntegratedDocument, file_path: Path) -> Dict[str, Any]:
        """格式化为RAG友好的JSON格式"""
        try:
            # 提取文档元数据
            doc_metadata = self._extract_document_metadata(integrated_doc, file_path)
            
            # 格式化元素
            formatted_elements = []
            for i, ctx_element in enumerate(integrated_doc.elements):
                formatted_element = self._format_element_to_json(ctx_element, i)
                formatted_elements.append(formatted_element)
            
            # 构建最终的JSON结构
            rag_json = {
                'document_metadata': doc_metadata,
                'elements': formatted_elements,
                'structure_map': integrated_doc.structure_map,
                'cross_references': integrated_doc.cross_references,
                'processing_info': {
                    'total_elements': len(formatted_elements),
                    'formatted_at': datetime.utcnow().isoformat(),
                    'format_version': '1.0'
                }
            }
            
            logger.info(f"文档格式化完成: {len(formatted_elements)} 个元素")
            return rag_json
            
        except Exception as e:
            logger.error(f"文档格式化失败: {e}")
            return {
                'error': str(e),
                'document_metadata': {},
                'elements': [],
                'structure_map': {},
                'cross_references': {}
            }
    
    def _extract_document_metadata(self, integrated_doc: IntegratedDocument, file_path: Path) -> Dict[str, Any]:
        """提取文档元数据"""
        metadata = {
            'filename': file_path.name,
            'file_path': str(file_path),
            'file_size': file_path.stat().st_size if file_path.exists() else 0,
            'file_extension': file_path.suffix.lower(),
            'processed_at': datetime.utcnow().isoformat(),
        }
        
        # 从整合文档中提取统计信息
        element_types = {}
        total_content_length = 0
        
        for ctx_element in integrated_doc.elements:
            element_type = ctx_element.element.element_type
            element_types[element_type] = element_types.get(element_type, 0) + 1
            total_content_length += len(ctx_element.element.content)
        
        metadata.update({
            'element_types': element_types,
            'total_elements': len(integrated_doc.elements),
            'total_content_length': total_content_length,
            'average_element_length': total_content_length / len(integrated_doc.elements) if integrated_doc.elements else 0
        })
        
        # 添加结构信息
        if integrated_doc.structure_map:
            metadata.update({
                'has_tables': len(integrated_doc.structure_map.get('tables', [])) > 0,
                'has_images': len(integrated_doc.structure_map.get('images', [])) > 0,
                'has_formulas': len(integrated_doc.structure_map.get('formulas', [])) > 0,
                'sections_count': len(integrated_doc.structure_map.get('sections', [])),
                'pages_count': len(integrated_doc.structure_map.get('pages', {}))
            })
        
        return metadata
    
    def _format_element_to_json(self, ctx_element: ContextualElement, index: int) -> Dict[str, Any]:
        """格式化元素为JSON"""
        element = ctx_element.element
        
        formatted = {
            'id': f"element_{index}",
            'type': element.element_type,
            'content': element.content,
            'metadata': element.metadata.copy(),
            'position': element.position,
            'confidence': element.confidence,
            'global_position': ctx_element.global_position,
            'context': {
                'before': [elem.content[:100] + '...' if len(elem.content) > 100 else elem.content 
                          for elem in ctx_element.context_before],
                'after': [elem.content[:100] + '...' if len(elem.content) > 100 else elem.content 
                         for elem in ctx_element.context_after],
                'related_count': len(ctx_element.related_elements)
            }
        }
        
        # 添加内容统计
        formatted['content_stats'] = {
            'char_count': len(element.content),
            'word_count': len(element.content.split()),
            'line_count': len(element.content.split('\n'))
        }
        
        return formatted


# 全局实例
document_cleaner = DocumentCleaner()
document_formatter = DocumentFormatter()
