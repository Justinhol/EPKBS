"""
MCP基础类型定义
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum


class MCPMessageType(str, Enum):
    """MCP消息类型"""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"


class MCPError(BaseModel):
    """MCP错误信息"""
    code: int
    message: str
    data: Optional[Dict[str, Any]] = None


class MCPMessage(BaseModel):
    """MCP消息基类"""
    jsonrpc: str = "2.0"
    id: Optional[Union[str, int]] = None
    method: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    error: Optional[MCPError] = None


class ToolParameter(BaseModel):
    """工具参数定义"""
    name: str
    type: str
    description: str
    required: bool = True
    default: Optional[Any] = None
    enum: Optional[List[Any]] = None


class ToolSchema(BaseModel):
    """工具Schema定义"""
    name: str
    description: str
    parameters: List[ToolParameter] = []
    
    def to_json_schema(self) -> Dict[str, Any]:
        """转换为JSON Schema格式"""
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = {
                "type": param.type,
                "description": param.description
            }
            
            if param.enum:
                properties[param.name]["enum"] = param.enum
            
            if param.default is not None:
                properties[param.name]["default"] = param.default
            
            if param.required:
                required.append(param.name)
        
        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False
        }


class MCPTool(ABC):
    """MCP工具基类"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self._schema: Optional[ToolSchema] = None
    
    @property
    def schema(self) -> ToolSchema:
        """获取工具Schema"""
        if self._schema is None:
            self._schema = self._build_schema()
        return self._schema
    
    @abstractmethod
    def _build_schema(self) -> ToolSchema:
        """构建工具Schema"""
        pass
    
    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """执行工具"""
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.schema.to_json_schema()
        }


class MCPServer(ABC):
    """MCP服务器基类"""
    
    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.tools: Dict[str, MCPTool] = {}
        self._running = False
    
    def register_tool(self, tool: MCPTool):
        """注册工具"""
        self.tools[tool.name] = tool
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """列出所有工具"""
        return [tool.to_dict() for tool in self.tools.values()]
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """调用工具"""
        if name not in self.tools:
            return {
                "success": False,
                "error": f"Tool '{name}' not found"
            }
        
        try:
            tool = self.tools[name]
            result = await tool.execute(**arguments)
            return {
                "success": True,
                "result": result
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    @abstractmethod
    async def start(self):
        """启动服务器"""
        pass
    
    @abstractmethod
    async def stop(self):
        """停止服务器"""
        pass


class MCPClient(ABC):
    """MCP客户端基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.connected_servers: Dict[str, str] = {}  # server_name -> connection_info
    
    @abstractmethod
    async def connect_to_server(self, server_name: str, connection_info: str):
        """连接到MCP服务器"""
        pass
    
    @abstractmethod
    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """调用远程工具"""
        pass
    
    @abstractmethod
    async def list_tools(self, server_name: str) -> List[Dict[str, Any]]:
        """列出服务器的所有工具"""
        pass
