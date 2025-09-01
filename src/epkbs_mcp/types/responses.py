"""
MCP响应类型定义
"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class ToolResponse(BaseModel):
    """工具响应基类"""
    success: bool
    timestamp: datetime = Field(default_factory=datetime.now)
    execution_time: Optional[float] = None
    server_name: Optional[str] = None
    tool_name: Optional[str] = None


class SuccessResponse(ToolResponse):
    """成功响应"""
    success: bool = True
    result: Any
    metadata: Optional[Dict[str, Any]] = None


class ErrorResponse(ToolResponse):
    """错误响应"""
    success: bool = False
    error: str
    error_code: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None


class ServerInfoResponse(BaseModel):
    """服务器信息响应"""
    name: str
    version: str
    description: Optional[str] = None
    capabilities: List[str] = []
    tools_count: int = 0
    status: str = "running"


class ToolListResponse(BaseModel):
    """工具列表响应"""
    tools: List[Dict[str, Any]]
    total_count: int
    server_name: str


class HealthCheckResponse(BaseModel):
    """健康检查响应"""
    status: str  # "healthy", "unhealthy", "degraded"
    timestamp: datetime = Field(default_factory=datetime.now)
    checks: Dict[str, Any] = {}
    uptime: Optional[float] = None
