"""
数据整合与上下文保留模块
将各个解析器的结果整合，保留数据块之间的上下文关系
"""
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

from src.data.parsers.base import BaseParser, ParseResult, ParsedElement
from src.data.document_validator import DocumentInfo, DocumentType
from src.utils.logger import get_logger
from config.settings import settings

logger = get_logger("data.parsers.integrator")


@dataclass
class ContextualElement:
    """带上下文的元素"""
    element: ParsedElement
    context_before: List[ParsedElement]
    context_after: List[ParsedElement]
    related_elements: List[ParsedElement]
    global_position: int


@dataclass
class IntegratedDocument:
    """整合后的文档"""
    elements: List[ContextualElement]
    metadata: Dict[str, Any]
    structure_map: Dict[str, Any]
    cross_references: Dict[str, List[str]]


class DocumentIntegrator:
    """文档整合器"""
    
    def __init__(self):
        self.name = "DocumentIntegrator"
        logger.info("文档整合器初始化完成")
    
    async def integrate_parse_results(
        self,
        parse_results: List[ParseResult],
        doc_info: DocumentInfo
    ) -> IntegratedDocument:
        """
        整合多个解析结果
        
        Args:
            parse_results: 解析结果列表
            doc_info: 文档信息
            
        Returns:
            IntegratedDocument: 整合后的文档
        """
        start_time = time.time()
        
        try:
            # 1. 收集所有元素
            all_elements = []
            for result in parse_results:
                if result.success:
                    all_elements.extend(result.elements)
            
            if not all_elements:
                return IntegratedDocument(
                    elements=[],
                    metadata={'error': '没有成功解析的元素'},
                    structure_map={},
                    cross_references={}
                )
            
            # 2. 按位置排序元素
            sorted_elements = self._sort_elements_by_position(all_elements)
            
            # 3. 构建文档结构图
            structure_map = self._build_structure_map(sorted_elements, doc_info)
            
            # 4. 建立上下文关系
            contextual_elements = self._build_contextual_elements(sorted_elements, structure_map)
            
            # 5. 建立交叉引用
            cross_references = self._build_cross_references(contextual_elements)
            
            # 6. 优化元素顺序和内容
            optimized_elements = self._optimize_element_order(contextual_elements)
            
            processing_time = time.time() - start_time
            
            integrated_doc = IntegratedDocument(
                elements=optimized_elements,
                metadata={
                    'total_elements': len(optimized_elements),
                    'processing_time': processing_time,
                    'integration_method': 'contextual_integration',
                    'source_parsers': [r.metadata.get('parser', 'unknown') for r in parse_results if r.success]
                },
                structure_map=structure_map,
                cross_references=cross_references
            )
            
            logger.info(f"文档整合完成: {len(optimized_elements)} 个元素")
            return integrated_doc
            
        except Exception as e:
            logger.error(f"文档整合失败: {e}")
            return IntegratedDocument(
                elements=[],
                metadata={'error': str(e)},
                structure_map={},
                cross_references={}
            )
    
    def _sort_elements_by_position(self, elements: List[ParsedElement]) -> List[ParsedElement]:
        """按位置排序元素"""
        def get_sort_key(element: ParsedElement) -> Tuple:
            """获取排序键"""
            pos = element.position or {}
            
            # 优先级：页面 > 段落/表格/图像 > 其他
            page = pos.get('page', 0)
            slide = pos.get('slide', 0)
            sheet = pos.get('sheet', 0)
            
            # 主要位置（页面、幻灯片、工作表）
            main_pos = max(page, slide, sheet)
            
            # 次要位置（段落、表格、图像等）
            paragraph = pos.get('paragraph', 0)
            table = pos.get('table', 0)
            image = pos.get('image', 0)
            heading = pos.get('heading', 0)
            
            secondary_pos = max(paragraph, table, image, heading)
            
            # 元素类型优先级
            type_priority = {
                'text': 1,
                'heading': 0,  # 标题优先
                'table': 2,
                'image': 3,
                'formula': 4
            }
            
            element_priority = type_priority.get(element.element_type, 5)
            
            return (main_pos, secondary_pos, element_priority)
        
        return sorted(elements, key=get_sort_key)
    
    def _build_structure_map(self, elements: List[ParsedElement], doc_info: DocumentInfo) -> Dict[str, Any]:
        """构建文档结构图"""
        structure = {
            'document_type': doc_info.file_type.value if doc_info.file_type else 'unknown',
            'sections': [],
            'pages': defaultdict(list),
            'tables': [],
            'images': [],
            'formulas': []
        }
        
        current_section = None
        
        for i, element in enumerate(elements):
            element_info = {
                'index': i,
                'type': element.element_type,
                'position': element.position,
                'content_preview': element.content[:100] + '...' if len(element.content) > 100 else element.content
            }
            
            # 按页面分组
            page = element.position.get('page', 1) if element.position else 1
            structure['pages'][page].append(element_info)
            
            # 按类型分组
            if element.element_type == 'text' and element.metadata.get('element_type') == 'heading':
                # 新的章节
                current_section = {
                    'title': element.content,
                    'start_index': i,
                    'elements': []
                }
                structure['sections'].append(current_section)
            elif current_section:
                current_section['elements'].append(element_info)
            
            # 特殊元素分类
            if element.element_type == 'table':
                structure['tables'].append(element_info)
            elif element.element_type == 'image':
                structure['images'].append(element_info)
            elif element.element_type == 'formula':
                structure['formulas'].append(element_info)
        
        return structure
    
    def _build_contextual_elements(
        self,
        elements: List[ParsedElement],
        structure_map: Dict[str, Any]
    ) -> List[ContextualElement]:
        """构建带上下文的元素"""
        contextual_elements = []
        
        for i, element in enumerate(elements):
            # 获取前后文
            context_before = elements[max(0, i-2):i]
            context_after = elements[i+1:min(len(elements), i+3)]
            
            # 查找相关元素
            related_elements = self._find_related_elements(element, elements, i)
            
            contextual_element = ContextualElement(
                element=element,
                context_before=context_before,
                context_after=context_after,
                related_elements=related_elements,
                global_position=i
            )
            
            contextual_elements.append(contextual_element)
        
        return contextual_elements
    
    def _find_related_elements(
        self,
        target_element: ParsedElement,
        all_elements: List[ParsedElement],
        target_index: int
    ) -> List[ParsedElement]:
        """查找相关元素"""
        related = []
        
        # 同页面/同位置的元素
        target_pos = target_element.position or {}
        
        for i, element in enumerate(all_elements):
            if i == target_index:
                continue
            
            element_pos = element.position or {}
            
            # 检查是否在同一页面/位置
            if self._is_same_location(target_pos, element_pos):
                related.append(element)
            
            # 检查内容相关性
            elif self._is_content_related(target_element, element):
                related.append(element)
        
        return related[:5]  # 限制相关元素数量
    
    def _is_same_location(self, pos1: Dict[str, Any], pos2: Dict[str, Any]) -> bool:
        """检查是否在同一位置"""
        # 检查页面
        if pos1.get('page') and pos2.get('page'):
            return pos1['page'] == pos2['page']
        
        # 检查幻灯片
        if pos1.get('slide') and pos2.get('slide'):
            return pos1['slide'] == pos2['slide']
        
        # 检查工作表
        if pos1.get('sheet') and pos2.get('sheet'):
            return pos1['sheet'] == pos2['sheet']
        
        return False
    
    def _is_content_related(self, element1: ParsedElement, element2: ParsedElement) -> bool:
        """检查内容是否相关"""
        # 简单的关键词匹配
        content1_words = set(element1.content.lower().split())
        content2_words = set(element2.content.lower().split())
        
        # 计算交集比例
        if len(content1_words) == 0 or len(content2_words) == 0:
            return False
        
        intersection = content1_words.intersection(content2_words)
        union = content1_words.union(content2_words)
        
        similarity = len(intersection) / len(union)
        return similarity > 0.1  # 10%的相似度阈值
    
    def _build_cross_references(self, contextual_elements: List[ContextualElement]) -> Dict[str, List[str]]:
        """建立交叉引用"""
        cross_refs = defaultdict(list)
        
        for i, ctx_element in enumerate(contextual_elements):
            element_id = f"element_{i}"
            
            # 添加相关元素的引用
            for related in ctx_element.related_elements:
                for j, other_ctx in enumerate(contextual_elements):
                    if other_ctx.element == related:
                        cross_refs[element_id].append(f"element_{j}")
                        break
        
        return dict(cross_refs)
    
    def _optimize_element_order(self, contextual_elements: List[ContextualElement]) -> List[ContextualElement]:
        """优化元素顺序"""
        # 这里可以实现更复杂的优化逻辑
        # 目前保持原有顺序
        return contextual_elements
    
    def create_integrated_content(self, integrated_doc: IntegratedDocument) -> str:
        """创建整合后的内容"""
        content_parts = []
        
        for ctx_element in integrated_doc.elements:
            element = ctx_element.element
            
            # 根据元素类型添加适当的格式
            if element.element_type == 'text':
                if element.metadata.get('element_type') == 'heading':
                    # 标题
                    level = element.metadata.get('heading_level', 1)
                    content_parts.append(f"{'#' * level} {element.content}")
                else:
                    # 普通文本
                    content_parts.append(element.content)
            
            elif element.element_type == 'table':
                # 表格
                content_parts.append(f"\n{element.content}\n")
            
            elif element.element_type == 'image':
                # 图像描述
                content_parts.append(f"\n[图像描述: {element.content}]\n")
            
            elif element.element_type == 'formula':
                # 公式
                content_parts.append(f"\n[数学公式: {element.content}]\n")
            
            else:
                # 其他类型
                content_parts.append(element.content)
            
            content_parts.append("")  # 添加空行分隔
        
        return "\n".join(content_parts)


# 全局整合器实例
document_integrator = DocumentIntegrator()
