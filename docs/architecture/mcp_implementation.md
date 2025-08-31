# FastMCP封装RAG系统实现文档

## 🎯 概述

本项目使用FastMCP框架将RAG系统封装为标准化的MCP (Model Context Protocol) 服务器，实现了高度模块化、可扩展的智能文档处理和知识检索系统。

## 🏗️ 架构设计

### 核心组件

```
┌─────────────────────────────────────────────────────────────┐
│                    MCP Agent Layer                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   MCP Agent     │  │  ReAct Agent    │  │ Agent Core   │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                 MCP Client Layer                            │
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │ MCP Client Mgr  │  │ Local MCP Client│                  │
│  └─────────────────┘  └─────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                MCP Server Layer                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │  RAG Server     │  │ Document Parser │  │ Table Extract│ │
│  │                 │  │    Server       │  │   Server     │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                  Core Services                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   RAG Core      │  │   LLM Manager   │  │ Vector Store │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### MCP服务器集群

#### 1. RAG服务器 (`rag-server`)
- **search_knowledge**: 混合检索和智能重排序
- **get_context**: 获取查询相关上下文
- **analyze_query**: 查询意图和复杂度分析
- **get_retrieval_stats**: 检索系统统计信息

#### 2. 文档解析服务器 (`document-parsing-server`)
- **parse_document**: Unstructured通用文档解析
- **analyze_document_structure**: 文档结构分析

#### 3. 表格提取服务器 (`table-extraction-server`)
- **extract_tables_camelot**: Camelot PDF表格提取
- **extract_tables_pdfplumber**: PDFPlumber表格提取
- **compare_table_methods**: 表格提取方法比较

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 基础使用

```python
from src.agent.core import AgentCore

# 创建启用MCP的Agent
agent = AgentCore(enable_mcp=True)
await agent.initialize()

# 获取可用工具
tools = await agent.get_mcp_tools()
print(f"可用工具: {tools['count']} 个")

# 智能文档解析
result = await agent.parse_document_with_mcp("document.pdf")

# RAG问答
answer = await agent.chat_with_mcp("什么是人工智能？", use_rag=True)
```

### 3. 直接工具调用

```python
# 调用RAG搜索
result = await agent.call_mcp_tool(
    "rag-server.search_knowledge",
    {
        "query": "深度学习",
        "top_k": 5,
        "retriever_type": "hybrid"
    }
)

# 调用文档解析
result = await agent.call_mcp_tool(
    "document-parsing-server.parse_document",
    {
        "file_path": "document.pdf",
        "strategy": "auto"
    }
)
```

## 🔧 MCP工具详解

### RAG服务器工具

#### search_knowledge
```json
{
  "name": "search_knowledge",
  "description": "在知识库中搜索相关信息，支持混合检索和智能重排序",
  "parameters": {
    "query": "搜索查询文本",
    "top_k": "返回结果数量 (默认: 5)",
    "retriever_type": "检索器类型 (vector|sparse|fulltext|hybrid)",
    "reranker_type": "重排序器类型 (colbert|cross_encoder|mmr|ensemble)",
    "enable_multi_query": "是否启用多查询检索 (默认: true)"
  }
}
```

#### get_context
```json
{
  "name": "get_context", 
  "description": "获取查询相关的上下文信息，用于LLM生成",
  "parameters": {
    "query": "查询文本",
    "max_context_length": "最大上下文长度 (默认: 4000)",
    "include_metadata": "是否包含元数据 (默认: true)"
  }
}
```

### 文档解析服务器工具

#### parse_document
```json
{
  "name": "parse_document",
  "description": "使用Unstructured库解析各种格式的文档",
  "parameters": {
    "file_path": "文件路径",
    "strategy": "解析策略 (auto|fast|accurate|balanced)",
    "extract_images": "是否提取图像 (默认: true)",
    "chunking_strategy": "分块策略 (by_title|by_page|basic)"
  }
}
```

### 表格提取服务器工具

#### extract_tables_camelot
```json
{
  "name": "extract_tables_camelot",
  "description": "使用Camelot提取PDF表格，适合结构化表格",
  "parameters": {
    "file_path": "PDF文件路径",
    "pages": "页面范围 (如'1-3'或'all')",
    "flavor": "检测算法 (lattice|stream)",
    "table_areas": "表格区域坐标 (可选)"
  }
}
```

## 🤖 Agent工作流程

### 智能文档解析流程

```python
# Agent推理过程示例
"""
Thought: 用户要求解析PDF文档，我需要先获取文档基本信息

Action: document-parsing-server.parse_document
Action Input: {"file_path": "document.pdf", "strategy": "auto"}

Observation: 解析成功，发现文档包含表格和图像

Thought: 检测到表格内容，需要使用专业表格提取工具

Action: table-extraction-server.extract_tables_camelot  
Action Input: {"file_path": "document.pdf", "flavor": "lattice"}

Observation: 成功提取3个表格，平均准确率0.92

Thought: 表格提取质量良好，现在整合所有结果

Final Answer: 文档解析完成，包含文本、3个高质量表格...
"""
```

## 📊 性能特性

### 并发处理
- **异步架构**: 所有MCP服务器支持异步调用
- **并行工具调用**: 支持批量工具调用
- **负载均衡**: 支持多实例MCP服务器部署

### 容错机制
- **健康检查**: 实时监控服务器状态
- **自动重试**: 工具调用失败自动重试
- **降级策略**: 服务不可用时的备选方案

### 监控指标
```python
# 获取系统状态
server_status = await agent.get_mcp_server_status()
client_stats = agent.mcp_client_manager.get_client_statistics()

print(f"服务器状态: {server_status['status']}")
print(f"总调用次数: {client_stats['total_calls']}")
print(f"成功率: {client_stats['success_rate']:.2%}")
```

## 🔌 扩展开发

### 添加新的MCP服务器

```python
from src.mcp.types.base import MCPServer, MCPTool

class CustomMCPServer(MCPServer):
    def __init__(self):
        super().__init__(name="custom-server", version="1.0.0")
        self._register_tools()
    
    def _register_tools(self):
        self.register_tool(CustomTool())

class CustomTool(MCPTool):
    def __init__(self):
        super().__init__(
            name="custom_tool",
            description="自定义工具描述"
        )
    
    async def execute(self, **kwargs):
        # 工具实现逻辑
        return {"success": True, "result": "处理结果"}
```

### 注册到服务器管理器

```python
# 在server_manager.py中添加
custom_server = CustomMCPServer()
await self.register_server(custom_server)
await self.start_server("custom-server")
```

## 🧪 测试和验证

### 运行测试

```bash
# 基础功能测试
python test_mcp_implementation.py

# 使用示例演示
python examples/mcp_usage_example.py
```

### 测试覆盖

- ✅ MCP服务器启动和停止
- ✅ 工具注册和发现
- ✅ 客户端连接和调用
- ✅ Agent推理和工具使用
- ✅ 错误处理和恢复
- ✅ 性能监控和统计

## 🚀 部署方案

### Docker部署

```yaml
# docker-compose.mcp.yml
version: '3.8'
services:
  rag-mcp-server:
    build: ./src/mcp/servers/rag_server
    ports: ["9001:9001"]
    
  document-parsing-server:
    build: ./src/mcp/servers/document_parsing_server  
    ports: ["9002:9002"]
    
  table-extraction-server:
    build: ./src/mcp/servers/table_extraction_server
    ports: ["9003:9003"]
    
  epkbs-app:
    build: .
    ports: ["8000:8000"]
    environment:
      - MCP_SERVERS=rag-mcp-server:9001,document-parsing-server:9002
    depends_on:
      - rag-mcp-server
      - document-parsing-server
```

### 生产环境配置

```python
# config/mcp_settings.py
MCP_CONFIG = {
    "servers": {
        "rag-server": {
            "host": "localhost",
            "port": 9001,
            "max_connections": 100
        },
        "document-parsing-server": {
            "host": "localhost", 
            "port": 9002,
            "max_connections": 50
        }
    },
    "client": {
        "timeout": 30,
        "retry_attempts": 3,
        "batch_size": 10
    }
}
```

## 📈 优势总结

### 🎯 标准化
- **MCP协议**: 符合工业标准的工具调用协议
- **统一接口**: 所有工具都有一致的调用方式
- **Schema驱动**: 完整的参数验证和文档

### 🔧 模块化
- **服务分离**: 每个功能都是独立的MCP服务器
- **松耦合**: 服务器之间无直接依赖
- **可替换**: 可以轻松替换或升级单个服务

### 🚀 可扩展
- **水平扩展**: 支持多实例部署
- **插件化**: 新功能只需实现MCP服务器
- **语言无关**: MCP服务器可用任何语言实现

### 🤖 智能化
- **Agent驱动**: 智能选择和组合工具
- **自适应**: 根据任务复杂度调整策略
- **学习能力**: 可以学习最佳工具使用模式

这个FastMCP封装方案将RAG系统转变为一个高度智能、可扩展的企业级文档处理平台！
