"""
RAG功能的MCP服务器
提供知识库搜索、上下文获取等RAG相关功能
"""
import asyncio
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..types.base import MCPServer, MCPTool, ToolSchema, ToolParameter
from ..types.tools import (
    MCPToolResult, RAGSearchResult, RAGContextResult, 
    COMMON_TOOL_PARAMETERS
)
from ...rag.core import RAGCore
from ...utils.logger import get_logger

logger = get_logger("mcp.rag_server")


class RAGSearchTool(MCPTool):
    """RAG搜索工具"""
    
    def __init__(self, rag_core: RAGCore):
        super().__init__(
            name="search_knowledge",
            description="在知识库中搜索相关信息，支持混合检索和智能重排序"
        )
        self.rag_core = rag_core
    
    def _build_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=[
                COMMON_TOOL_PARAMETERS["query"],
                COMMON_TOOL_PARAMETERS["top_k"],
                ToolParameter(
                    name="retriever_type",
                    type="string",
                    description="检索器类型",
                    required=False,
                    default="hybrid",
                    enum=["vector", "sparse", "fulltext", "hybrid"]
                ),
                ToolParameter(
                    name="reranker_type", 
                    type="string",
                    description="重排序器类型",
                    required=False,
                    default="ensemble",
                    enum=["colbert", "cross_encoder", "mmr", "ensemble"]
                ),
                ToolParameter(
                    name="enable_multi_query",
                    type="boolean",
                    description="是否启用多查询检索",
                    required=False,
                    default=True
                )
            ]
        )
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """执行RAG搜索"""
        start_time = time.time()
        
        try:
            query = kwargs.get("query")
            top_k = kwargs.get("top_k", 5)
            retriever_type = kwargs.get("retriever_type", "hybrid")
            reranker_type = kwargs.get("reranker_type", "ensemble")
            enable_multi_query = kwargs.get("enable_multi_query", True)
            
            logger.info(f"执行RAG搜索: query='{query}', top_k={top_k}")
            
            # 执行搜索
            results = await self.rag_core.search(
                query=query,
                top_k=top_k,
                retriever_type=retriever_type,
                reranker_type=reranker_type,
                enable_multi_query=enable_multi_query
            )
            
            execution_time = time.time() - start_time
            
            return RAGSearchResult(
                query=query,
                results=results,
                total_results=len(results),
                retriever_used=retriever_type,
                reranker_used=reranker_type,
                execution_time=execution_time
            ).dict()
            
        except Exception as e:
            logger.error(f"RAG搜索失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time
            }


class RAGContextTool(MCPTool):
    """RAG上下文获取工具"""
    
    def __init__(self, rag_core: RAGCore):
        super().__init__(
            name="get_context",
            description="获取查询相关的上下文信息，用于LLM生成"
        )
        self.rag_core = rag_core
    
    def _build_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=[
                COMMON_TOOL_PARAMETERS["query"],
                ToolParameter(
                    name="max_context_length",
                    type="integer",
                    description="最大上下文长度",
                    required=False,
                    default=4000
                ),
                COMMON_TOOL_PARAMETERS["include_metadata"]
            ]
        )
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """获取RAG上下文"""
        start_time = time.time()
        
        try:
            query = kwargs.get("query")
            max_context_length = kwargs.get("max_context_length", 4000)
            include_metadata = kwargs.get("include_metadata", True)
            
            logger.info(f"获取RAG上下文: query='{query}', max_length={max_context_length}")
            
            # 获取上下文
            context, source_docs = await self.rag_core.get_context(
                query=query,
                max_context_length=max_context_length
            )
            
            result = RAGContextResult(
                context=context,
                context_length=len(context)
            )
            
            if include_metadata:
                result.sources = [
                    {
                        "source": doc.get('metadata', {}).get('source', 'unknown'),
                        "score": doc.get('score', 0.0),
                        "chunk_id": doc.get('id', ''),
                        "content_preview": doc.get('content', '')[:100] + "..." if len(doc.get('content', '')) > 100 else doc.get('content', '')
                    }
                    for doc in source_docs
                ]
            
            execution_time = time.time() - start_time
            result_dict = result.dict()
            result_dict["execution_time"] = execution_time
            
            return result_dict
            
        except Exception as e:
            logger.error(f"获取RAG上下文失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time
            }


class RAGAnalysisTool(MCPTool):
    """RAG查询分析工具"""
    
    def __init__(self, rag_core: RAGCore):
        super().__init__(
            name="analyze_query",
            description="分析查询意图和复杂度，为检索策略提供建议"
        )
        self.rag_core = rag_core
    
    def _build_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=[
                COMMON_TOOL_PARAMETERS["query"]
            ]
        )
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """分析查询"""
        start_time = time.time()
        
        try:
            query = kwargs.get("query")
            
            logger.info(f"分析查询: '{query}'")
            
            # 查询分析逻辑
            analysis = await self._analyze_query(query)
            
            execution_time = time.time() - start_time
            
            return {
                "success": True,
                "query": query,
                "intent": analysis.get("intent"),
                "complexity": analysis.get("complexity"),
                "suggested_strategy": analysis.get("strategy"),
                "keywords": analysis.get("keywords", []),
                "execution_time": execution_time
            }
            
        except Exception as e:
            logger.error(f"查询分析失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time
            }
    
    async def _analyze_query(self, query: str) -> Dict[str, Any]:
        """分析查询的具体逻辑"""
        # 简单的查询分析实现
        query_lower = query.lower()
        
        # 意图分析
        intent = "general"
        if any(word in query_lower for word in ["什么是", "定义", "概念"]):
            intent = "definition"
        elif any(word in query_lower for word in ["如何", "怎么", "方法"]):
            intent = "how_to"
        elif any(word in query_lower for word in ["为什么", "原因", "why"]):
            intent = "explanation"
        elif any(word in query_lower for word in ["比较", "区别", "差异"]):
            intent = "comparison"
        
        # 复杂度分析
        complexity = "simple"
        if len(query) > 50:
            complexity = "complex"
        elif len(query.split()) > 10:
            complexity = "medium"
        
        # 策略建议
        strategy = "hybrid"
        if intent == "definition":
            strategy = "vector"  # 定义类查询适合向量检索
        elif intent == "how_to":
            strategy = "hybrid"  # 方法类查询适合混合检索
        
        return {
            "intent": intent,
            "complexity": complexity,
            "strategy": strategy,
            "keywords": query.split()[:5]  # 简单提取关键词
        }


class RAGStatsTool(MCPTool):
    """RAG统计信息工具"""
    
    def __init__(self, rag_core: RAGCore):
        super().__init__(
            name="get_retrieval_stats",
            description="获取检索系统的统计信息和状态"
        )
        self.rag_core = rag_core
    
    def _build_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=[]  # 无参数
        )
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """获取统计信息"""
        start_time = time.time()
        
        try:
            logger.info("获取RAG系统统计信息")
            
            # 获取统计信息
            stats = await self.rag_core.get_statistics()
            
            execution_time = time.time() - start_time
            
            return {
                "success": True,
                "total_documents": stats.get("total_documents", 0),
                "total_chunks": stats.get("total_chunks", 0),
                "vector_dimension": stats.get("vector_dimension", 0),
                "last_update": stats.get("last_update"),
                "index_status": stats.get("index_status", "unknown"),
                "execution_time": execution_time
            }
            
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time
            }


class RAGMCPServer(MCPServer):
    """RAG功能的MCP服务器"""
    
    def __init__(self, rag_core: RAGCore):
        super().__init__(name="rag-server", version="1.0.0")
        self.rag_core = rag_core
        self._register_tools()
        
        logger.info("RAG MCP服务器初始化完成")
    
    def _register_tools(self):
        """注册所有RAG工具"""
        self.register_tool(RAGSearchTool(self.rag_core))
        self.register_tool(RAGContextTool(self.rag_core))
        self.register_tool(RAGAnalysisTool(self.rag_core))
        self.register_tool(RAGStatsTool(self.rag_core))
        
        logger.info(f"已注册 {len(self.tools)} 个RAG工具")
    
    async def start(self):
        """启动RAG MCP服务器"""
        self._running = True
        logger.info(f"RAG MCP服务器 '{self.name}' 已启动")
    
    async def stop(self):
        """停止RAG MCP服务器"""
        self._running = False
        logger.info(f"RAG MCP服务器 '{self.name}' 已停止")
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 检查RAG核心状态
            rag_healthy = await self._check_rag_health()
            
            return {
                "status": "healthy" if rag_healthy else "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "checks": {
                    "rag_core": "healthy" if rag_healthy else "unhealthy",
                    "tools_count": len(self.tools)
                },
                "uptime": time.time() - getattr(self, '_start_time', time.time())
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _check_rag_health(self) -> bool:
        """检查RAG核心健康状态"""
        try:
            # 简单的健康检查 - 尝试获取统计信息
            stats = await self.rag_core.get_statistics()
            return stats is not None
        except Exception:
            return False
