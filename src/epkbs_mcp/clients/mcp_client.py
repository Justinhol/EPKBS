"""
MCP客户端
用于连接和调用MCP服务器
"""
import asyncio
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..types.base import MCPClient
from ..server_manager import MCPServerManager
from ...utils.logger import get_logger

logger = get_logger("mcp.client")


class LocalMCPClient(MCPClient):
    """本地MCP客户端 - 直接调用本地MCP服务器"""
    
    def __init__(self, server_manager: MCPServerManager):
        super().__init__(name="local-mcp-client")
        self.server_manager = server_manager
        self.call_history: List[Dict[str, Any]] = []
        
        logger.info("本地MCP客户端初始化完成")
    
    async def connect_to_server(self, server_name: str, connection_info: str = "local"):
        """连接到本地MCP服务器"""
        try:
            # 检查服务器是否存在
            server_info = await self.server_manager.get_server_info(server_name)
            if server_info is None:
                raise Exception(f"服务器 {server_name} 不存在")
            
            self.connected_servers[server_name] = connection_info
            logger.info(f"已连接到服务器: {server_name}")
            
        except Exception as e:
            logger.error(f"连接服务器 {server_name} 失败: {e}")
            raise
    
    async def call_tool(
        self, 
        server_name: str, 
        tool_name: str, 
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """调用远程工具"""
        start_time = time.time()
        
        try:
            # 检查连接
            if server_name not in self.connected_servers:
                await self.connect_to_server(server_name)
            
            # 调用工具
            result = await self.server_manager.call_tool(server_name, tool_name, arguments)
            
            # 记录调用历史
            call_record = {
                "timestamp": datetime.now().isoformat(),
                "server_name": server_name,
                "tool_name": tool_name,
                "arguments": arguments,
                "success": result.get("success", True),
                "execution_time": time.time() - start_time
            }
            
            self.call_history.append(call_record)
            
            # 保持最近100次调用记录
            if len(self.call_history) > 100:
                self.call_history = self.call_history[-100:]
            
            logger.info(f"工具调用完成: {server_name}.{tool_name}")
            
            return result
            
        except Exception as e:
            logger.error(f"调用工具失败: {server_name}.{tool_name} - {e}")
            
            # 记录失败的调用
            call_record = {
                "timestamp": datetime.now().isoformat(),
                "server_name": server_name,
                "tool_name": tool_name,
                "arguments": arguments,
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time
            }
            
            self.call_history.append(call_record)
            
            return {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time
            }
    
    async def list_tools(self, server_name: str) -> List[Dict[str, Any]]:
        """列出服务器的所有工具"""
        try:
            if server_name not in self.connected_servers:
                await self.connect_to_server(server_name)
            
            server_info = await self.server_manager.get_server_info(server_name)
            if server_info:
                return server_info.get("tools", [])
            else:
                return []
                
        except Exception as e:
            logger.error(f"获取工具列表失败: {server_name} - {e}")
            return []
    
    async def list_all_tools(self) -> Dict[str, List[Dict[str, Any]]]:
        """列出所有连接服务器的工具"""
        return await self.server_manager.list_all_tools()
    
    async def discover_tools(self) -> Dict[str, Dict[str, Any]]:
        """发现所有可用工具"""
        all_tools = {}
        
        try:
            servers_info = await self.server_manager.get_all_servers_info()
            
            for server_name, server_info in servers_info.get("servers", {}).items():
                if server_info and server_info.get("status") == "running":
                    tools = server_info.get("tools", [])
                    
                    for tool in tools:
                        tool_key = f"{server_name}.{tool['name']}"
                        all_tools[tool_key] = {
                            "server": server_name,
                            "tool": tool["name"],
                            "description": tool.get("description", ""),
                            "schema": tool.get("inputSchema", {}),
                            "server_status": server_info.get("status", "unknown")
                        }
            
            logger.info(f"发现 {len(all_tools)} 个可用工具")
            return all_tools
            
        except Exception as e:
            logger.error(f"工具发现失败: {e}")
            return {}
    
    async def batch_call_tools(
        self, 
        calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """批量调用工具"""
        results = []
        
        # 并行执行所有调用
        tasks = []
        for call in calls:
            task = self.call_tool(
                call["server_name"],
                call["tool_name"],
                call.get("arguments", {})
            )
            tasks.append(task)
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理异常结果
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append({
                        "success": False,
                        "error": str(result),
                        "call_index": i
                    })
                else:
                    processed_results.append(result)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"批量调用失败: {e}")
            return [{"success": False, "error": str(e)} for _ in calls]
    
    def get_call_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取调用历史"""
        return self.call_history[-limit:] if limit > 0 else self.call_history
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取客户端统计信息"""
        total_calls = len(self.call_history)
        successful_calls = sum(1 for call in self.call_history if call.get("success", False))
        
        # 计算平均执行时间
        execution_times = [call.get("execution_time", 0) for call in self.call_history if call.get("execution_time")]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
        
        # 统计各服务器调用次数
        server_calls = {}
        for call in self.call_history:
            server_name = call.get("server_name", "unknown")
            server_calls[server_name] = server_calls.get(server_name, 0) + 1
        
        return {
            "total_calls": total_calls,
            "successful_calls": successful_calls,
            "failed_calls": total_calls - successful_calls,
            "success_rate": successful_calls / total_calls if total_calls > 0 else 0,
            "average_execution_time": avg_execution_time,
            "connected_servers": len(self.connected_servers),
            "server_call_distribution": server_calls
        }


class MCPClientManager:
    """MCP客户端管理器"""
    
    def __init__(self, server_manager: MCPServerManager):
        self.server_manager = server_manager
        self.client = LocalMCPClient(server_manager)
        
        logger.info("MCP客户端管理器初始化完成")
    
    async def initialize(self):
        """初始化客户端，连接到所有可用服务器"""
        try:
            servers_info = await self.server_manager.get_all_servers_info()
            
            for server_name, server_info in servers_info.get("servers", {}).items():
                if server_info and server_info.get("status") == "running":
                    await self.client.connect_to_server(server_name)
            
            logger.info("MCP客户端初始化完成")
            
        except Exception as e:
            logger.error(f"MCP客户端初始化失败: {e}")
    
    async def call_tool(
        self, 
        tool_key: str, 
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """调用工具 - 使用 server.tool 格式的工具键"""
        try:
            if "." not in tool_key:
                return {
                    "success": False,
                    "error": f"无效的工具键格式: {tool_key}，应为 'server.tool'"
                }
            
            server_name, tool_name = tool_key.split(".", 1)
            return await self.client.call_tool(server_name, tool_name, arguments)
            
        except Exception as e:
            logger.error(f"调用工具失败: {tool_key} - {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_available_tools(self) -> Dict[str, Dict[str, Any]]:
        """获取所有可用工具"""
        return await self.client.discover_tools()
    
    def get_client_statistics(self) -> Dict[str, Any]:
        """获取客户端统计信息"""
        return self.client.get_statistics()
