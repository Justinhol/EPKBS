"""
MCP工具相关类型定义
"""
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
from .base import MCPTool, ToolSchema, ToolParameter


class MCPToolResult(BaseModel):
    """MCP工具执行结果"""
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None


class DocumentParsingResult(BaseModel):
    """文档解析结果"""
    file_path: str
    total_elements: int
    element_types: Dict[str, int]
    has_tables: bool = False
    has_images: bool = False
    has_formulas: bool = False
    elements: List[Dict[str, Any]] = []
    recommendations: List[str] = []
    quality_score: Optional[float] = None


class TableExtractionResult(BaseModel):
    """表格提取结果"""
    method_used: str
    tables_found: int
    tables: List[Dict[str, Any]] = []
    average_accuracy: Optional[float] = None
    quality_assessment: Optional[Dict[str, Any]] = None


class ImageAnalysisResult(BaseModel):
    """图像分析结果"""
    images_found: int
    images: List[Dict[str, Any]] = []
    total_ocr_text: str = ""
    recommendation: Optional[str] = None


class QualityAssessmentResult(BaseModel):
    """质量评估结果"""
    overall_quality: float
    detailed_metrics: Dict[str, float]
    suggestions: List[str] = []
    needs_improvement: bool = False
    recommended_actions: List[str] = []


class ContentIntegrationResult(BaseModel):
    """内容整合结果"""
    integrated_content: str
    metadata: Dict[str, Any]
    element_count: int
    content_length: int
    chunk_info: Optional[Dict[str, Any]] = None


class RAGSearchResult(BaseModel):
    """RAG搜索结果"""
    query: str
    results: List[Dict[str, Any]]
    total_results: int
    retriever_used: str
    reranker_used: str
    execution_time: Optional[float] = None


class RAGContextResult(BaseModel):
    """RAG上下文结果"""
    context: str
    context_length: int
    sources: List[Dict[str, Any]] = []
    metadata: Optional[Dict[str, Any]] = None


# 工具参数模板
COMMON_TOOL_PARAMETERS = {
    "file_path": ToolParameter(
        name="file_path",
        type="string",
        description="文件路径",
        required=True
    ),
    "query": ToolParameter(
        name="query",
        type="string", 
        description="查询字符串",
        required=True
    ),
    "top_k": ToolParameter(
        name="top_k",
        type="integer",
        description="返回结果数量",
        required=False,
        default=5
    ),
    "strategy": ToolParameter(
        name="strategy",
        type="string",
        description="处理策略",
        required=False,
        default="auto",
        enum=["auto", "fast", "accurate", "balanced"]
    ),
    "language": ToolParameter(
        name="language",
        type="string",
        description="语言设置",
        required=False,
        default="zh",
        enum=["zh", "en", "auto"]
    ),
    "include_metadata": ToolParameter(
        name="include_metadata",
        type="boolean",
        description="是否包含元数据",
        required=False,
        default=True
    )
}
