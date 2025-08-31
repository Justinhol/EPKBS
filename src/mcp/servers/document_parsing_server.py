"""
文档解析MCP服务器
提供通用文档解析功能，基于Unstructured库
"""
import asyncio
import time
from typing import Dict, Any, List, Optional
from pathlib import Path

from ..types.base import MCPServer, MCPTool, ToolSchema, ToolParameter
from ..types.tools import (
    MCPToolResult, DocumentParsingResult, 
    COMMON_TOOL_PARAMETERS
)
from ...utils.logger import get_logger

logger = get_logger("mcp.document_parsing_server")


class UnstructuredParseTool(MCPTool):
    """Unstructured通用解析工具"""
    
    def __init__(self):
        super().__init__(
            name="parse_document",
            description="使用Unstructured库解析各种格式的文档，提供基础信息和处理建议"
        )
    
    def _build_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=[
                COMMON_TOOL_PARAMETERS["file_path"],
                COMMON_TOOL_PARAMETERS["strategy"],
                ToolParameter(
                    name="extract_images",
                    type="boolean",
                    description="是否提取图像",
                    required=False,
                    default=True
                ),
                ToolParameter(
                    name="chunking_strategy",
                    type="string",
                    description="分块策略",
                    required=False,
                    default="by_title",
                    enum=["by_title", "by_page", "basic"]
                )
            ]
        )
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """执行文档解析"""
        start_time = time.time()
        
        try:
            file_path = kwargs.get("file_path")
            strategy = kwargs.get("strategy", "auto")
            extract_images = kwargs.get("extract_images", True)
            chunking_strategy = kwargs.get("chunking_strategy", "by_title")
            
            logger.info(f"解析文档: {file_path}")
            
            # 检查文件是否存在
            if not Path(file_path).exists():
                return {
                    "success": False,
                    "error": f"文件不存在: {file_path}"
                }
            
            # 使用Unstructured解析
            elements = await self._parse_with_unstructured(
                file_path, strategy, extract_images, chunking_strategy
            )
            
            # 分析元素
            analysis = self._analyze_elements(elements)
            
            # 生成建议
            recommendations = self._generate_recommendations(analysis)
            
            execution_time = time.time() - start_time
            
            result = DocumentParsingResult(
                file_path=file_path,
                total_elements=len(elements),
                element_types=analysis["types"],
                has_tables=analysis["has_tables"],
                has_images=analysis["has_images"],
                has_formulas=analysis["has_formulas"],
                elements=[self._element_to_dict(elem) for elem in elements],
                recommendations=recommendations,
                quality_score=analysis.get("quality_score", 0.8)
            )
            
            result_dict = result.dict()
            result_dict["success"] = True
            result_dict["execution_time"] = execution_time
            
            return result_dict
            
        except Exception as e:
            logger.error(f"文档解析失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time
            }
    
    async def _parse_with_unstructured(
        self, 
        file_path: str, 
        strategy: str, 
        extract_images: bool,
        chunking_strategy: str
    ) -> List[Any]:
        """使用Unstructured解析文档"""
        try:
            from unstructured.partition.auto import partition
            
            # 根据策略设置参数
            partition_kwargs = {
                "filename": file_path,
                "strategy": strategy if strategy != "auto" else "auto",
                "extract_images": extract_images,
                "chunking_strategy": chunking_strategy
            }
            
            # 执行解析
            elements = partition(**partition_kwargs)
            
            logger.info(f"Unstructured解析完成，共 {len(elements)} 个元素")
            return elements
            
        except ImportError:
            logger.error("Unstructured库未安装")
            raise Exception("Unstructured库未安装，请安装: pip install unstructured[all-docs]")
        except Exception as e:
            logger.error(f"Unstructured解析失败: {e}")
            raise
    
    def _analyze_elements(self, elements: List[Any]) -> Dict[str, Any]:
        """分析解析出的元素"""
        analysis = {
            "types": {},
            "has_tables": False,
            "has_images": False,
            "has_formulas": False,
            "quality_score": 0.8
        }
        
        for element in elements:
            element_type = str(type(element).__name__)
            analysis["types"][element_type] = analysis["types"].get(element_type, 0) + 1
            
            # 检测特殊内容类型
            if "Table" in element_type:
                analysis["has_tables"] = True
            elif "Image" in element_type:
                analysis["has_images"] = True
            elif "Formula" in element_type or self._contains_formula(str(element)):
                analysis["has_formulas"] = True
        
        # 计算质量分数
        analysis["quality_score"] = self._calculate_quality_score(elements)
        
        return analysis
    
    def _contains_formula(self, text: str) -> bool:
        """检测文本是否包含数学公式"""
        formula_indicators = [
            "\\frac", "\\sum", "\\int", "\\sqrt", "\\alpha", "\\beta",
            "$$", "\\(", "\\)", "\\begin{equation}", "\\end{equation}"
        ]
        return any(indicator in text for indicator in formula_indicators)
    
    def _calculate_quality_score(self, elements: List[Any]) -> float:
        """计算解析质量分数"""
        if not elements:
            return 0.0
        
        # 简单的质量评估逻辑
        text_elements = sum(1 for elem in elements if "Text" in str(type(elem).__name__))
        total_elements = len(elements)
        
        # 文本元素比例作为质量指标之一
        text_ratio = text_elements / total_elements if total_elements > 0 else 0
        
        # 基础分数 + 文本比例调整
        base_score = 0.7
        quality_score = base_score + (text_ratio * 0.3)
        
        return min(quality_score, 1.0)
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """基于分析结果生成处理建议"""
        recommendations = []
        
        if analysis["has_tables"]:
            recommendations.append("检测到表格内容，建议使用专业表格提取工具进行进一步处理")
        
        if analysis["has_images"]:
            recommendations.append("检测到图像内容，建议使用图像分析工具进行OCR和描述生成")
        
        if analysis["has_formulas"]:
            recommendations.append("检测到数学公式，建议使用公式识别工具进行专业处理")
        
        if analysis["quality_score"] < 0.6:
            recommendations.append("解析质量较低，建议尝试其他解析策略或工具")
        
        # 根据元素类型给出建议
        element_types = analysis["types"]
        if element_types.get("Title", 0) > 0:
            recommendations.append("文档结构良好，可以按标题进行结构化分块")
        
        if element_types.get("ListItem", 0) > 5:
            recommendations.append("包含大量列表项，建议保持列表结构")
        
        return recommendations
    
    def _element_to_dict(self, element: Any) -> Dict[str, Any]:
        """将元素转换为字典格式"""
        try:
            return {
                "type": str(type(element).__name__),
                "content": str(element),
                "metadata": element.metadata.to_dict() if hasattr(element, 'metadata') else {},
                "category": getattr(element, 'category', 'unknown')
            }
        except Exception as e:
            logger.warning(f"元素转换失败: {e}")
            return {
                "type": "unknown",
                "content": str(element),
                "metadata": {},
                "category": "unknown"
            }


class DocumentStructureAnalysisTool(MCPTool):
    """文档结构分析工具"""
    
    def __init__(self):
        super().__init__(
            name="analyze_document_structure",
            description="分析文档的层次结构，为后续处理提供指导"
        )
    
    def _build_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=[
                COMMON_TOOL_PARAMETERS["file_path"]
            ]
        )
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """分析文档结构"""
        start_time = time.time()
        
        try:
            file_path = kwargs.get("file_path")
            
            logger.info(f"分析文档结构: {file_path}")
            
            # 简单的结构分析实现
            structure = await self._analyze_structure(file_path)
            
            execution_time = time.time() - start_time
            
            return {
                "success": True,
                "file_path": file_path,
                "structure": structure,
                "execution_time": execution_time
            }
            
        except Exception as e:
            logger.error(f"文档结构分析失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time
            }
    
    async def _analyze_structure(self, file_path: str) -> Dict[str, Any]:
        """分析文档结构的具体实现"""
        # 这里可以实现更复杂的结构分析逻辑
        # 目前返回基础结构信息
        file_ext = Path(file_path).suffix.lower()
        
        structure = {
            "file_type": file_ext,
            "estimated_pages": 1,
            "has_toc": False,
            "heading_levels": [],
            "complexity": "medium"
        }
        
        # 根据文件类型调整结构信息
        if file_ext == ".pdf":
            structure["estimated_pages"] = 10  # 简化估算
            structure["complexity"] = "high"
        elif file_ext in [".docx", ".doc"]:
            structure["has_toc"] = True
            structure["heading_levels"] = ["h1", "h2", "h3"]
        
        return structure


class DocumentParsingMCPServer(MCPServer):
    """文档解析MCP服务器"""
    
    def __init__(self):
        super().__init__(name="document-parsing-server", version="1.0.0")
        self._register_tools()
        
        logger.info("文档解析MCP服务器初始化完成")
    
    def _register_tools(self):
        """注册所有文档解析工具"""
        self.register_tool(UnstructuredParseTool())
        self.register_tool(DocumentStructureAnalysisTool())
        
        logger.info(f"已注册 {len(self.tools)} 个文档解析工具")
    
    async def start(self):
        """启动文档解析MCP服务器"""
        self._running = True
        logger.info(f"文档解析MCP服务器 '{self.name}' 已启动")
    
    async def stop(self):
        """停止文档解析MCP服务器"""
        self._running = False
        logger.info(f"文档解析MCP服务器 '{self.name}' 已停止")
