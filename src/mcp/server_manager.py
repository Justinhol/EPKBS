"""
MCP服务器管理器
统一管理所有MCP服务器的启动、停止和调用
"""
import asyncio
import time
from typing import Dict, Any, List, Optional, Type
from datetime import datetime

from .types.base import MCPServer
from .servers.rag_server import RAGMCPServer
from .servers.document_parsing_server import DocumentParsingMCPServer
from .servers.table_extraction_server import TableExtractionMCPServer
from ..utils.logger import get_logger

logger = get_logger("mcp.server_manager")


class MCPServerManager:
    """MCP服务器管理器"""
    
    def __init__(self):
        self.servers: Dict[str, MCPServer] = {}
        self.server_stats: Dict[str, Dict[str, Any]] = {}
        self._start_time = time.time()
        
        logger.info("MCP服务器管理器初始化完成")
    
    async def register_server(self, server: MCPServer) -> bool:
        """注册MCP服务器"""
        try:
            if server.name in self.servers:
                logger.warning(f"服务器 {server.name} 已存在，将被替换")
            
            self.servers[server.name] = server
            self.server_stats[server.name] = {
                "registered_at": datetime.now().isoformat(),
                "status": "registered",
                "call_count": 0,
                "error_count": 0,
                "last_call": None
            }
            
            logger.info(f"服务器 {server.name} 注册成功")
            return True
            
        except Exception as e:
            logger.error(f"注册服务器 {server.name} 失败: {e}")
            return False
    
    async def start_server(self, server_name: str) -> bool:
        """启动指定的MCP服务器"""
        try:
            if server_name not in self.servers:
                logger.error(f"服务器 {server_name} 不存在")
                return False
            
            server = self.servers[server_name]
            await server.start()
            
            self.server_stats[server_name]["status"] = "running"
            self.server_stats[server_name]["started_at"] = datetime.now().isoformat()
            
            logger.info(f"服务器 {server_name} 启动成功")
            return True
            
        except Exception as e:
            logger.error(f"启动服务器 {server_name} 失败: {e}")
            self.server_stats[server_name]["status"] = "error"
            return False
    
    async def stop_server(self, server_name: str) -> bool:
        """停止指定的MCP服务器"""
        try:
            if server_name not in self.servers:
                logger.error(f"服务器 {server_name} 不存在")
                return False
            
            server = self.servers[server_name]
            await server.stop()
            
            self.server_stats[server_name]["status"] = "stopped"
            self.server_stats[server_name]["stopped_at"] = datetime.now().isoformat()
            
            logger.info(f"服务器 {server_name} 停止成功")
            return True
            
        except Exception as e:
            logger.error(f"停止服务器 {server_name} 失败: {e}")
            return False
    
    async def start_all_servers(self, rag_core=None) -> Dict[str, bool]:
        """启动所有MCP服务器"""
        logger.info("开始启动所有MCP服务器...")
        
        results = {}
        
        try:
            # 注册并启动RAG服务器
            if rag_core:
                rag_server = RAGMCPServer(rag_core)
                await self.register_server(rag_server)
                results["rag-server"] = await self.start_server("rag-server")
            
            # 注册并启动文档解析服务器
            doc_server = DocumentParsingMCPServer()
            await self.register_server(doc_server)
            results["document-parsing-server"] = await self.start_server("document-parsing-server")
            
            # 注册并启动表格提取服务器
            table_server = TableExtractionMCPServer()
            await self.register_server(table_server)
            results["table-extraction-server"] = await self.start_server("table-extraction-server")
            
            successful_starts = sum(1 for success in results.values() if success)
            total_servers = len(results)
            
            logger.info(f"MCP服务器启动完成: {successful_starts}/{total_servers} 成功")
            
            return results
            
        except Exception as e:
            logger.error(f"启动MCP服务器失败: {e}")
            return results
    
    async def stop_all_servers(self) -> Dict[str, bool]:
        """停止所有MCP服务器"""
        logger.info("开始停止所有MCP服务器...")
        
        results = {}
        
        for server_name in self.servers.keys():
            results[server_name] = await self.stop_server(server_name)
        
        successful_stops = sum(1 for success in results.values() if success)
        total_servers = len(results)
        
        logger.info(f"MCP服务器停止完成: {successful_stops}/{total_servers} 成功")
        
        return results
    
    async def call_tool(
        self, 
        server_name: str, 
        tool_name: str, 
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """调用指定服务器的工具"""
        start_time = time.time()
        
        try:
            if server_name not in self.servers:
                return {
                    "success": False,
                    "error": f"服务器 {server_name} 不存在"
                }
            
            server = self.servers[server_name]
            
            # 更新统计信息
            self.server_stats[server_name]["call_count"] += 1
            self.server_stats[server_name]["last_call"] = datetime.now().isoformat()
            
            # 调用工具
            result = await server.call_tool(tool_name, arguments)
            
            # 记录执行时间
            execution_time = time.time() - start_time
            result["execution_time"] = execution_time
            result["server_name"] = server_name
            result["tool_name"] = tool_name
            
            if not result.get("success", True):
                self.server_stats[server_name]["error_count"] += 1
            
            logger.info(f"工具调用完成: {server_name}.{tool_name} ({execution_time:.3f}s)")
            
            return result
            
        except Exception as e:
            self.server_stats[server_name]["error_count"] += 1
            logger.error(f"调用工具失败: {server_name}.{tool_name} - {e}")
            
            return {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time,
                "server_name": server_name,
                "tool_name": tool_name
            }
    
    async def list_all_tools(self) -> Dict[str, List[Dict[str, Any]]]:
        """列出所有服务器的工具"""
        all_tools = {}
        
        for server_name, server in self.servers.items():
            try:
                tools = await server.list_tools()
                all_tools[server_name] = tools
            except Exception as e:
                logger.error(f"获取服务器 {server_name} 工具列表失败: {e}")
                all_tools[server_name] = []
        
        return all_tools
    
    async def get_server_info(self, server_name: str) -> Optional[Dict[str, Any]]:
        """获取服务器信息"""
        if server_name not in self.servers:
            return None
        
        server = self.servers[server_name]
        stats = self.server_stats[server_name]
        
        try:
            tools = await server.list_tools()
            
            info = {
                "name": server.name,
                "version": server.version,
                "status": stats["status"],
                "tools_count": len(tools),
                "tools": tools,
                "statistics": stats
            }
            
            # 如果服务器支持健康检查
            if hasattr(server, 'health_check'):
                info["health"] = await server.health_check()
            
            return info
            
        except Exception as e:
            logger.error(f"获取服务器 {server_name} 信息失败: {e}")
            return {
                "name": server.name,
                "version": server.version,
                "status": "error",
                "error": str(e)
            }
    
    async def get_all_servers_info(self) -> Dict[str, Any]:
        """获取所有服务器信息"""
        servers_info = {}
        
        for server_name in self.servers.keys():
            servers_info[server_name] = await self.get_server_info(server_name)
        
        return {
            "servers": servers_info,
            "total_servers": len(self.servers),
            "manager_uptime": time.time() - self._start_time,
            "timestamp": datetime.now().isoformat()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """整体健康检查"""
        try:
            health_status = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "manager_uptime": time.time() - self._start_time,
                "servers": {}
            }
            
            unhealthy_count = 0
            
            for server_name, server in self.servers.items():
                try:
                    if hasattr(server, 'health_check'):
                        server_health = await server.health_check()
                        health_status["servers"][server_name] = server_health
                        
                        if server_health.get("status") != "healthy":
                            unhealthy_count += 1
                    else:
                        # 简单的状态检查
                        stats = self.server_stats[server_name]
                        health_status["servers"][server_name] = {
                            "status": stats["status"],
                            "call_count": stats["call_count"],
                            "error_count": stats["error_count"]
                        }
                        
                        if stats["status"] != "running":
                            unhealthy_count += 1
                            
                except Exception as e:
                    health_status["servers"][server_name] = {
                        "status": "unhealthy",
                        "error": str(e)
                    }
                    unhealthy_count += 1
            
            # 整体状态评估
            if unhealthy_count == 0:
                health_status["status"] = "healthy"
            elif unhealthy_count < len(self.servers):
                health_status["status"] = "degraded"
            else:
                health_status["status"] = "unhealthy"
            
            health_status["unhealthy_servers"] = unhealthy_count
            health_status["total_servers"] = len(self.servers)
            
            return health_status
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取管理器统计信息"""
        total_calls = sum(stats["call_count"] for stats in self.server_stats.values())
        total_errors = sum(stats["error_count"] for stats in self.server_stats.values())
        
        return {
            "total_servers": len(self.servers),
            "running_servers": sum(1 for stats in self.server_stats.values() if stats["status"] == "running"),
            "total_calls": total_calls,
            "total_errors": total_errors,
            "error_rate": total_errors / total_calls if total_calls > 0 else 0,
            "uptime": time.time() - self._start_time,
            "server_stats": self.server_stats
        }
