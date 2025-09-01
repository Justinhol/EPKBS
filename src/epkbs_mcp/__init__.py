"""
MCP (Model Context Protocol) 模块
提供基于MCP协议的服务器和客户端实现
"""

from .server_manager import MCPServerManager
from .clients.mcp_client import MCPClientManager, LocalMCPClient
from .servers.rag_server import RAGMCPServer
from .servers.document_parsing_server import DocumentParsingMCPServer
from .servers.table_extraction_server import TableExtractionMCPServer

__all__ = [
    'MCPServerManager',
    'MCPClientManager', 
    'LocalMCPClient',
    'RAGMCPServer',
    'DocumentParsingMCPServer',
    'TableExtractionMCPServer'
]

__version__ = "1.0.0"
