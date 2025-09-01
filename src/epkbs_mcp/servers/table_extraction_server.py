"""
表格提取MCP服务器
提供专业的表格提取功能，支持多种工具
"""
import asyncio
import time
from typing import Dict, Any, List, Optional
from pathlib import Path

from ..types.base import MCPServer, MCPTool, ToolSchema, ToolParameter
from ..types.tools import (
    MCPToolResult, TableExtractionResult, 
    COMMON_TOOL_PARAMETERS
)
from ...utils.logger import get_logger

logger = get_logger("mcp.table_extraction_server")


class CamelotTableTool(MCPTool):
    """Camelot表格提取工具"""
    
    def __init__(self):
        super().__init__(
            name="extract_tables_camelot",
            description="使用Camelot提取PDF表格，适合结构化表格"
        )
    
    def _build_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=[
                COMMON_TOOL_PARAMETERS["file_path"],
                ToolParameter(
                    name="pages",
                    type="string",
                    description="页面范围，如'1-3'或'all'",
                    required=False,
                    default="all"
                ),
                ToolParameter(
                    name="flavor",
                    type="string",
                    description="表格检测算法",
                    required=False,
                    default="lattice",
                    enum=["lattice", "stream"]
                ),
                ToolParameter(
                    name="table_areas",
                    type="string",
                    description="表格区域坐标",
                    required=False
                )
            ]
        )
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """使用Camelot提取表格"""
        start_time = time.time()
        
        try:
            file_path = kwargs.get("file_path")
            pages = kwargs.get("pages", "all")
            flavor = kwargs.get("flavor", "lattice")
            table_areas = kwargs.get("table_areas")
            
            logger.info(f"使用Camelot提取表格: {file_path}")
            
            # 检查文件
            if not Path(file_path).exists():
                return {
                    "success": False,
                    "error": f"文件不存在: {file_path}"
                }
            
            if not file_path.lower().endswith('.pdf'):
                return {
                    "success": False,
                    "error": "Camelot只支持PDF文件"
                }
            
            # 使用Camelot提取表格
            tables = await self._extract_with_camelot(
                file_path, pages, flavor, table_areas
            )
            
            # 处理结果
            extracted_tables = []
            total_accuracy = 0
            
            for i, table in enumerate(tables):
                table_data = {
                    "table_id": i,
                    "page": getattr(table, 'page', 0),
                    "accuracy": getattr(table, 'accuracy', 0.0),
                    "whitespace": getattr(table, 'whitespace', 0.0),
                    "markdown": table.df.to_markdown() if hasattr(table, 'df') else "",
                    "csv": table.df.to_csv() if hasattr(table, 'df') else "",
                    "html": table.df.to_html() if hasattr(table, 'df') else "",
                    "shape": table.df.shape if hasattr(table, 'df') else (0, 0),
                    "bbox": getattr(table, '_bbox', None)
                }
                extracted_tables.append(table_data)
                total_accuracy += table_data["accuracy"]
            
            average_accuracy = total_accuracy / len(extracted_tables) if extracted_tables else 0
            
            execution_time = time.time() - start_time
            
            result = TableExtractionResult(
                method_used="camelot",
                tables_found=len(extracted_tables),
                tables=extracted_tables,
                average_accuracy=average_accuracy,
                quality_assessment=self._assess_quality(extracted_tables)
            )
            
            result_dict = result.dict()
            result_dict["success"] = True
            result_dict["execution_time"] = execution_time
            
            return result_dict
            
        except Exception as e:
            logger.error(f"Camelot表格提取失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time
            }
    
    async def _extract_with_camelot(
        self, 
        file_path: str, 
        pages: str, 
        flavor: str,
        table_areas: Optional[str]
    ) -> List[Any]:
        """使用Camelot提取表格"""
        try:
            import camelot
            
            # 构建参数
            kwargs = {
                "pages": pages,
                "flavor": flavor
            }
            
            if table_areas:
                kwargs["table_areas"] = [table_areas]
            
            # 执行提取
            tables = camelot.read_pdf(file_path, **kwargs)
            
            logger.info(f"Camelot提取完成，找到 {len(tables)} 个表格")
            return tables
            
        except ImportError:
            logger.error("Camelot库未安装")
            raise Exception("Camelot库未安装，请安装: pip install camelot-py[cv]")
        except Exception as e:
            logger.error(f"Camelot提取失败: {e}")
            raise
    
    def _assess_quality(self, tables: List[Dict[str, Any]]) -> Dict[str, Any]:
        """评估表格提取质量"""
        if not tables:
            return {"overall": 0.0, "details": "无表格"}
        
        total_accuracy = sum(table.get("accuracy", 0) for table in tables)
        average_accuracy = total_accuracy / len(tables)
        
        # 质量评估
        quality = "excellent" if average_accuracy > 0.9 else \
                 "good" if average_accuracy > 0.7 else \
                 "fair" if average_accuracy > 0.5 else "poor"
        
        return {
            "overall": average_accuracy,
            "quality_level": quality,
            "table_count": len(tables),
            "details": f"平均准确率: {average_accuracy:.2f}"
        }


class PDFPlumberTableTool(MCPTool):
    """PDFPlumber表格提取工具"""
    
    def __init__(self):
        super().__init__(
            name="extract_tables_pdfplumber",
            description="使用PDFPlumber提取PDF表格，适合简单表格"
        )
    
    def _build_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=[
                COMMON_TOOL_PARAMETERS["file_path"],
                ToolParameter(
                    name="page_numbers",
                    type="string",
                    description="页面编号，如'1,2,3'或'all'",
                    required=False,
                    default="all"
                )
            ]
        )
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """使用PDFPlumber提取表格"""
        start_time = time.time()
        
        try:
            file_path = kwargs.get("file_path")
            page_numbers = kwargs.get("page_numbers", "all")
            
            logger.info(f"使用PDFPlumber提取表格: {file_path}")
            
            # 检查文件
            if not Path(file_path).exists():
                return {
                    "success": False,
                    "error": f"文件不存在: {file_path}"
                }
            
            if not file_path.lower().endswith('.pdf'):
                return {
                    "success": False,
                    "error": "PDFPlumber只支持PDF文件"
                }
            
            # 使用PDFPlumber提取表格
            tables = await self._extract_with_pdfplumber(file_path, page_numbers)
            
            execution_time = time.time() - start_time
            
            result = TableExtractionResult(
                method_used="pdfplumber",
                tables_found=len(tables),
                tables=tables,
                average_accuracy=0.8,  # PDFPlumber没有准确率指标，给个估算值
                quality_assessment={"overall": 0.8, "method": "pdfplumber"}
            )
            
            result_dict = result.dict()
            result_dict["success"] = True
            result_dict["execution_time"] = execution_time
            
            return result_dict
            
        except Exception as e:
            logger.error(f"PDFPlumber表格提取失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time
            }
    
    async def _extract_with_pdfplumber(
        self, 
        file_path: str, 
        page_numbers: str
    ) -> List[Dict[str, Any]]:
        """使用PDFPlumber提取表格"""
        try:
            import pdfplumber
            import pandas as pd
            
            tables = []
            
            with pdfplumber.open(file_path) as pdf:
                pages_to_process = range(len(pdf.pages)) if page_numbers == "all" else \
                                 [int(p.strip()) - 1 for p in page_numbers.split(",")]
                
                for page_num in pages_to_process:
                    if page_num < len(pdf.pages):
                        page = pdf.pages[page_num]
                        page_tables = page.extract_tables()
                        
                        for i, table_data in enumerate(page_tables):
                            if table_data:
                                df = pd.DataFrame(table_data[1:], columns=table_data[0])
                                
                                table_info = {
                                    "table_id": len(tables),
                                    "page": page_num + 1,
                                    "accuracy": 0.8,  # 估算值
                                    "markdown": df.to_markdown(),
                                    "csv": df.to_csv(),
                                    "html": df.to_html(),
                                    "shape": df.shape
                                }
                                tables.append(table_info)
            
            logger.info(f"PDFPlumber提取完成，找到 {len(tables)} 个表格")
            return tables
            
        except ImportError:
            logger.error("PDFPlumber库未安装")
            raise Exception("PDFPlumber库未安装，请安装: pip install pdfplumber")
        except Exception as e:
            logger.error(f"PDFPlumber提取失败: {e}")
            raise


class TableComparisonTool(MCPTool):
    """表格提取方法比较工具"""
    
    def __init__(self):
        super().__init__(
            name="compare_table_methods",
            description="比较不同表格提取方法的效果，选择最佳方案"
        )
    
    def _build_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=[
                COMMON_TOOL_PARAMETERS["file_path"],
                ToolParameter(
                    name="methods",
                    type="string",
                    description="要比较的方法，用逗号分隔",
                    required=False,
                    default="camelot,pdfplumber"
                )
            ]
        )
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """比较不同表格提取方法"""
        start_time = time.time()
        
        try:
            file_path = kwargs.get("file_path")
            methods = kwargs.get("methods", "camelot,pdfplumber").split(",")
            
            logger.info(f"比较表格提取方法: {methods}")
            
            # 并行执行不同方法
            results = {}
            
            for method in methods:
                method = method.strip()
                if method == "camelot":
                    camelot_tool = CamelotTableTool()
                    results[method] = await camelot_tool.execute(file_path=file_path)
                elif method == "pdfplumber":
                    pdfplumber_tool = PDFPlumberTableTool()
                    results[method] = await pdfplumber_tool.execute(file_path=file_path)
            
            # 分析比较结果
            comparison = self._analyze_comparison(results)
            
            execution_time = time.time() - start_time
            
            return {
                "success": True,
                "file_path": file_path,
                "methods_compared": methods,
                "results": results,
                "comparison": comparison,
                "recommendation": comparison.get("best_method"),
                "execution_time": execution_time
            }
            
        except Exception as e:
            logger.error(f"表格方法比较失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time
            }
    
    def _analyze_comparison(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """分析比较结果"""
        comparison = {
            "best_method": None,
            "scores": {},
            "summary": {}
        }
        
        best_score = 0
        best_method = None
        
        for method, result in results.items():
            if result.get("success"):
                # 计算综合分数
                table_count = result.get("tables_found", 0)
                avg_accuracy = result.get("average_accuracy", 0)
                
                # 综合评分：表格数量 * 准确率
                score = table_count * avg_accuracy
                comparison["scores"][method] = score
                
                if score > best_score:
                    best_score = score
                    best_method = method
                
                comparison["summary"][method] = {
                    "tables_found": table_count,
                    "average_accuracy": avg_accuracy,
                    "score": score
                }
        
        comparison["best_method"] = best_method
        return comparison


class TableExtractionMCPServer(MCPServer):
    """表格提取MCP服务器"""
    
    def __init__(self):
        super().__init__(name="table-extraction-server", version="1.0.0")
        self._register_tools()
        
        logger.info("表格提取MCP服务器初始化完成")
    
    def _register_tools(self):
        """注册所有表格提取工具"""
        self.register_tool(CamelotTableTool())
        self.register_tool(PDFPlumberTableTool())
        self.register_tool(TableComparisonTool())
        
        logger.info(f"已注册 {len(self.tools)} 个表格提取工具")
    
    async def start(self):
        """启动表格提取MCP服务器"""
        self._running = True
        logger.info(f"表格提取MCP服务器 '{self.name}' 已启动")
    
    async def stop(self):
        """停止表格提取MCP服务器"""
        self._running = False
        logger.info(f"表格提取MCP服务器 '{self.name}' 已停止")
