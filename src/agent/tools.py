"""
Agent工具管理器
实现RAG工具、MCP工具调用等功能
"""
import asyncio
import json
from typing import List, Dict, Any, Optional, Callable, Union
from abc import ABC, abstractmethod
from datetime import datetime

from langchain.tools import BaseTool
from langchain.schema import BaseMessage
from pydantic import BaseModel, Field

from src.rag.core import RAGCore
from src.utils.logger import get_logger
from src.utils.helpers import Timer

logger = get_logger("agent.tools")


class ToolResult(BaseModel):
    """工具执行结果"""
    success: bool
    result: Any = None
    error: str = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseTool(ABC):
    """工具基类"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        logger.info(f"工具 {name} 初始化完成")
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """执行工具"""
        pass
    
    def get_schema(self) -> Dict[str, Any]:
        """获取工具schema"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self._get_parameters_schema()
        }
    
    @abstractmethod
    def _get_parameters_schema(self) -> Dict[str, Any]:
        """获取参数schema"""
        pass


class RAGSearchTool(BaseTool):
    """RAG搜索工具"""
    
    def __init__(self, rag_core: RAGCore):
        super().__init__(
            name="rag_search",
            description="在知识库中搜索相关信息，支持自然语言查询"
        )
        self.rag_core = rag_core
    
    async def execute(self, query: str, top_k: int = 5, **kwargs) -> ToolResult:
        """执行RAG搜索"""
        try:
            with Timer() as timer:
                # 执行搜索
                results = await self.rag_core.search(
                    query=query,
                    top_k=top_k,
                    **kwargs
                )
                
                # 构建搜索结果
                search_results = []
                for result in results:
                    search_results.append({
                        'content': result.get('content', ''),
                        'score': result.get('score', 0.0),
                        'source': result.get('metadata', {}).get('source', 'unknown'),
                        'rank': result.get('final_rank', 0)
                    })
            
            return ToolResult(
                success=True,
                result={
                    'query': query,
                    'results': search_results,
                    'total_results': len(search_results)
                },
                execution_time=timer.elapsed,
                metadata={
                    'tool': self.name,
                    'top_k': top_k,
                    'timestamp': datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"RAG搜索工具执行失败: {e}")
            return ToolResult(
                success=False,
                error=str(e),
                metadata={'tool': self.name}
            )
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索查询文本"
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回结果数量",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 20
                }
            },
            "required": ["query"]
        }


class RAGContextTool(BaseTool):
    """RAG上下文获取工具"""
    
    def __init__(self, rag_core: RAGCore):
        super().__init__(
            name="rag_context",
            description="获取查询相关的上下文信息，用于回答问题"
        )
        self.rag_core = rag_core
    
    async def execute(
        self,
        query: str,
        max_context_length: int = 4000,
        **kwargs
    ) -> ToolResult:
        """获取上下文"""
        try:
            with Timer() as timer:
                context, source_docs = await self.rag_core.get_context(
                    query=query,
                    max_context_length=max_context_length,
                    **kwargs
                )
            
            return ToolResult(
                success=True,
                result={
                    'query': query,
                    'context': context,
                    'context_length': len(context),
                    'source_count': len(source_docs),
                    'sources': [
                        {
                            'source': doc.get('metadata', {}).get('source', 'unknown'),
                            'score': doc.get('score', 0.0)
                        }
                        for doc in source_docs
                    ]
                },
                execution_time=timer.elapsed,
                metadata={
                    'tool': self.name,
                    'max_context_length': max_context_length
                }
            )
            
        except Exception as e:
            logger.error(f"RAG上下文工具执行失败: {e}")
            return ToolResult(
                success=False,
                error=str(e),
                metadata={'tool': self.name}
            )
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "查询文本"
                },
                "max_context_length": {
                    "type": "integer",
                    "description": "最大上下文长度",
                    "default": 4000,
                    "minimum": 500,
                    "maximum": 8000
                }
            },
            "required": ["query"]
        }


class TimeTool(BaseTool):
    """时间工具"""
    
    def __init__(self):
        super().__init__(
            name="get_time",
            description="获取当前时间和日期信息"
        )
    
    async def execute(self, format: str = "iso", **kwargs) -> ToolResult:
        """获取时间"""
        try:
            now = datetime.utcnow()
            
            if format == "iso":
                time_str = now.isoformat()
            elif format == "readable":
                time_str = now.strftime("%Y年%m月%d日 %H:%M:%S UTC")
            elif format == "date":
                time_str = now.strftime("%Y-%m-%d")
            elif format == "time":
                time_str = now.strftime("%H:%M:%S")
            else:
                time_str = now.isoformat()
            
            return ToolResult(
                success=True,
                result={
                    'current_time': time_str,
                    'format': format,
                    'timestamp': now.timestamp()
                },
                metadata={'tool': self.name}
            )
            
        except Exception as e:
            logger.error(f"时间工具执行失败: {e}")
            return ToolResult(
                success=False,
                error=str(e),
                metadata={'tool': self.name}
            )
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "description": "时间格式",
                    "enum": ["iso", "readable", "date", "time"],
                    "default": "iso"
                }
            }
        }


class CalculatorTool(BaseTool):
    """计算器工具"""
    
    def __init__(self):
        super().__init__(
            name="calculator",
            description="执行数学计算，支持基本的算术运算"
        )
    
    async def execute(self, expression: str, **kwargs) -> ToolResult:
        """执行计算"""
        try:
            # 安全的数学表达式评估
            allowed_names = {
                k: v for k, v in __builtins__.items()
                if k in ['abs', 'round', 'min', 'max', 'sum', 'pow']
            }
            allowed_names.update({
                'pi': 3.141592653589793,
                'e': 2.718281828459045
            })
            
            # 移除危险字符
            safe_expression = expression.replace('__', '').replace('import', '').replace('exec', '')
            
            result = eval(safe_expression, {"__builtins__": {}}, allowed_names)
            
            return ToolResult(
                success=True,
                result={
                    'expression': expression,
                    'result': result,
                    'result_type': type(result).__name__
                },
                metadata={'tool': self.name}
            )
            
        except Exception as e:
            logger.error(f"计算器工具执行失败: {e}")
            return ToolResult(
                success=False,
                error=f"计算错误: {str(e)}",
                metadata={'tool': self.name}
            )
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "数学表达式，如 '2 + 3 * 4'"
                }
            },
            "required": ["expression"]
        }


class ToolManager:
    """工具管理器"""
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        logger.info("工具管理器初始化完成")
    
    def register_tool(self, tool: BaseTool):
        """注册工具"""
        self.tools[tool.name] = tool
        logger.info(f"工具 {tool.name} 注册成功")
    
    def unregister_tool(self, tool_name: str):
        """注销工具"""
        if tool_name in self.tools:
            del self.tools[tool_name]
            logger.info(f"工具 {tool_name} 注销成功")
    
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """获取工具"""
        return self.tools.get(tool_name)
    
    def list_tools(self) -> List[str]:
        """列出所有工具"""
        return list(self.tools.keys())
    
    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """获取所有工具的schema"""
        return [tool.get_schema() for tool in self.tools.values()]
    
    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any] = None
    ) -> ToolResult:
        """执行工具"""
        tool = self.get_tool(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                error=f"工具 {tool_name} 不存在",
                metadata={'requested_tool': tool_name}
            )
        
        parameters = parameters or {}
        
        try:
            logger.info(f"执行工具: {tool_name}, 参数: {parameters}")
            result = await tool.execute(**parameters)
            logger.info(f"工具 {tool_name} 执行完成，成功: {result.success}")
            return result
            
        except Exception as e:
            logger.error(f"工具 {tool_name} 执行异常: {e}")
            return ToolResult(
                success=False,
                error=f"工具执行异常: {str(e)}",
                metadata={'tool': tool_name, 'parameters': parameters}
            )
    
    async def execute_tools_parallel(
        self,
        tool_calls: List[Dict[str, Any]]
    ) -> List[ToolResult]:
        """并行执行多个工具"""
        tasks = []
        
        for tool_call in tool_calls:
            tool_name = tool_call.get('name')
            parameters = tool_call.get('parameters', {})
            
            task = self.execute_tool(tool_name, parameters)
            tasks.append(task)
        
        logger.info(f"并行执行 {len(tasks)} 个工具")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常结果
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(ToolResult(
                    success=False,
                    error=str(result),
                    metadata={'tool_index': i}
                ))
            else:
                processed_results.append(result)
        
        return processed_results


class LangChainToolWrapper(BaseTool):
    """LangChain工具包装器"""
    
    def __init__(self, langchain_tool: BaseTool):
        super().__init__(
            name=langchain_tool.name,
            description=langchain_tool.description
        )
        self.langchain_tool = langchain_tool
    
    async def execute(self, **kwargs) -> ToolResult:
        """执行LangChain工具"""
        try:
            with Timer() as timer:
                # 构建输入
                if hasattr(self.langchain_tool, 'args_schema') and self.langchain_tool.args_schema:
                    # 使用schema验证参数
                    validated_input = self.langchain_tool.args_schema(**kwargs)
                    result = await asyncio.to_thread(self.langchain_tool.run, validated_input.dict())
                else:
                    # 直接传递参数
                    result = await asyncio.to_thread(self.langchain_tool.run, kwargs)
            
            return ToolResult(
                success=True,
                result=result,
                execution_time=timer.elapsed,
                metadata={'tool': self.name, 'type': 'langchain'}
            )
            
        except Exception as e:
            logger.error(f"LangChain工具包装器执行失败: {e}")
            return ToolResult(
                success=False,
                error=str(e),
                metadata={'tool': self.name, 'type': 'langchain'}
            )
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        """获取参数schema"""
        if hasattr(self.langchain_tool, 'args_schema') and self.langchain_tool.args_schema:
            return self.langchain_tool.args_schema.schema()
        else:
            return {
                "type": "object",
                "properties": {},
                "description": "参数schema未定义"
            }


def create_default_tool_manager(rag_core: RAGCore = None) -> ToolManager:
    """创建默认工具管理器"""
    manager = ToolManager()
    
    # 注册基础工具
    manager.register_tool(TimeTool())
    manager.register_tool(CalculatorTool())
    
    # 注册RAG工具（如果提供了RAG核心）
    if rag_core:
        manager.register_tool(RAGSearchTool(rag_core))
        manager.register_tool(RAGContextTool(rag_core))
    
    logger.info(f"默认工具管理器创建完成，包含 {len(manager.tools)} 个工具")
    return manager
