"""
MCP (Model Context Protocol) 配置文件
定义MCP服务器和客户端的配置参数
"""
from typing import Dict, Any, List
from pathlib import Path

# MCP服务器配置
MCP_SERVERS = {
    "rag-server": {
        "name": "rag-server",
        "version": "1.0.0",
        "description": "RAG功能服务器，提供知识库搜索和上下文获取",
        "host": "localhost",
        "port": 9001,
        "max_connections": 100,
        "timeout": 30,
        "tools": [
            "search_knowledge",
            "get_context", 
            "analyze_query",
            "get_retrieval_stats"
        ]
    },
    
    "document-parsing-server": {
        "name": "document-parsing-server",
        "version": "1.0.0", 
        "description": "文档解析服务器，支持多种文档格式解析",
        "host": "localhost",
        "port": 9002,
        "max_connections": 50,
        "timeout": 60,
        "tools": [
            "parse_document",
            "analyze_document_structure"
        ]
    },
    
    "table-extraction-server": {
        "name": "table-extraction-server",
        "version": "1.0.0",
        "description": "表格提取服务器，专业的表格识别和提取",
        "host": "localhost", 
        "port": 9003,
        "max_connections": 30,
        "timeout": 90,
        "tools": [
            "extract_tables_camelot",
            "extract_tables_pdfplumber",
            "compare_table_methods"
        ]
    }
}

# MCP客户端配置
MCP_CLIENT_CONFIG = {
    "name": "epkbs-mcp-client",
    "version": "1.0.0",
    "timeout": 30,
    "retry_attempts": 3,
    "retry_delay": 1.0,
    "batch_size": 10,
    "max_concurrent_calls": 5,
    "connection_pool_size": 20
}

# MCP Agent配置
MCP_AGENT_CONFIG = {
    "max_iterations": 10,
    "max_execution_time": 300,
    "enable_tool_discovery": True,
    "enable_batch_calls": True,
    "enable_parallel_execution": True,
    "reasoning_temperature": 0.1,
    "tool_selection_strategy": "intelligent"  # intelligent, sequential, parallel
}

# 工具调用配置
TOOL_CALL_CONFIG = {
    "default_timeout": 30,
    "max_retries": 3,
    "retry_backoff": 2.0,
    "enable_caching": True,
    "cache_ttl": 300,  # 5分钟
    "enable_metrics": True
}

# 服务器健康检查配置
HEALTH_CHECK_CONFIG = {
    "enabled": True,
    "interval": 30,  # 秒
    "timeout": 10,
    "failure_threshold": 3,
    "recovery_threshold": 2
}

# 监控和日志配置
MONITORING_CONFIG = {
    "enable_metrics": True,
    "metrics_port": 9090,
    "enable_tracing": True,
    "log_level": "INFO",
    "log_format": "json",
    "enable_performance_logging": True
}

# 安全配置
SECURITY_CONFIG = {
    "enable_auth": False,  # 本地开发环境
    "api_key": None,
    "allowed_hosts": ["localhost", "127.0.0.1"],
    "enable_cors": True,
    "cors_origins": ["*"]
}

# 开发环境配置
DEVELOPMENT_CONFIG = {
    "debug": True,
    "hot_reload": True,
    "enable_mock_servers": False,
    "mock_response_delay": 0.1
}

# 生产环境配置
PRODUCTION_CONFIG = {
    "debug": False,
    "hot_reload": False,
    "enable_load_balancing": True,
    "enable_circuit_breaker": True,
    "circuit_breaker_threshold": 5,
    "enable_rate_limiting": True,
    "rate_limit_per_minute": 1000
}

# 根据环境选择配置
def get_mcp_config(environment: str = "development") -> Dict[str, Any]:
    """获取MCP配置"""
    base_config = {
        "servers": MCP_SERVERS,
        "client": MCP_CLIENT_CONFIG,
        "agent": MCP_AGENT_CONFIG,
        "tool_calls": TOOL_CALL_CONFIG,
        "health_check": HEALTH_CHECK_CONFIG,
        "monitoring": MONITORING_CONFIG,
        "security": SECURITY_CONFIG
    }
    
    if environment == "development":
        base_config.update(DEVELOPMENT_CONFIG)
    elif environment == "production":
        base_config.update(PRODUCTION_CONFIG)
    
    return base_config

# 默认配置
DEFAULT_MCP_CONFIG = get_mcp_config("development")
