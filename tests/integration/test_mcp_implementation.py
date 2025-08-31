#!/usr/bin/env python3
"""
测试MCP实现
验证MCP服务器、客户端和Agent的功能
"""
import asyncio
import json
from pathlib import Path

from src.agent.core import AgentCore
from src.rag.core import RAGCore
from src.mcp.server_manager import MCPServerManager
from src.mcp.clients.mcp_client import MCPClientManager
from src.utils.logger import get_logger

logger = get_logger("test_mcp")


async def test_mcp_servers():
    """测试MCP服务器功能"""
    logger.info("=== 测试MCP服务器功能 ===")
    
    try:
        # 创建RAG核心（用于RAG服务器）
        rag_core = RAGCore(collection_name="test_collection")
        await rag_core.initialize()
        
        # 创建MCP服务器管理器
        server_manager = MCPServerManager()
        
        # 启动所有服务器
        results = await server_manager.start_all_servers(rag_core)
        
        logger.info(f"服务器启动结果: {results}")
        
        # 获取所有工具
        all_tools = await server_manager.list_all_tools()
        
        logger.info("可用工具:")
        for server_name, tools in all_tools.items():
            logger.info(f"  {server_name}: {len(tools)} 个工具")
            for tool in tools:
                logger.info(f"    - {tool['name']}: {tool['description']}")
        
        # 健康检查
        health = await server_manager.health_check()
        logger.info(f"健康检查结果: {health}")
        
        return server_manager
        
    except Exception as e:
        logger.error(f"MCP服务器测试失败: {e}")
        raise


async def test_mcp_client(server_manager):
    """测试MCP客户端功能"""
    logger.info("=== 测试MCP客户端功能 ===")
    
    try:
        # 创建客户端管理器
        client_manager = MCPClientManager(server_manager)
        await client_manager.initialize()
        
        # 发现工具
        available_tools = await client_manager.get_available_tools()
        
        logger.info(f"客户端发现 {len(available_tools)} 个工具:")
        for tool_key, tool_info in available_tools.items():
            logger.info(f"  {tool_key}: {tool_info['description']}")
        
        # 测试工具调用
        if "rag-server.get_retrieval_stats" in available_tools:
            logger.info("测试RAG统计工具...")
            result = await client_manager.call_tool("rag-server.get_retrieval_stats", {})
            logger.info(f"RAG统计结果: {result}")
        
        if "document-parsing-server.parse_document" in available_tools:
            logger.info("测试文档解析工具...")
            # 创建一个测试文件
            test_file = Path("test_document.txt")
            test_file.write_text("这是一个测试文档\n包含一些测试内容")
            
            result = await client_manager.call_tool(
                "document-parsing-server.parse_document",
                {"file_path": str(test_file)}
            )
            logger.info(f"文档解析结果: {result.get('success', False)}")
            
            # 清理测试文件
            test_file.unlink()
        
        # 获取客户端统计
        stats = client_manager.get_client_statistics()
        logger.info(f"客户端统计: {stats}")
        
        return client_manager
        
    except Exception as e:
        logger.error(f"MCP客户端测试失败: {e}")
        raise


async def test_mcp_agent(server_manager, client_manager):
    """测试MCP Agent功能"""
    logger.info("=== 测试MCP Agent功能 ===")
    
    try:
        # 创建Agent核心（启用MCP）
        agent_core = AgentCore(enable_mcp=True)
        
        # 手动设置MCP组件（因为我们已经初始化了）
        agent_core.mcp_server_manager = server_manager
        agent_core.mcp_client_manager = client_manager
        
        # 初始化Agent（会跳过MCP初始化因为已经设置）
        await agent_core.initialize()
        
        # 测试MCP工具获取
        mcp_tools = await agent_core.get_mcp_tools()
        logger.info(f"Agent发现 {mcp_tools.get('count', 0)} 个MCP工具")
        
        # 测试MCP服务器状态
        server_status = await agent_core.get_mcp_server_status()
        logger.info(f"MCP服务器状态: {server_status.get('status', 'unknown')}")
        
        # 测试直接工具调用
        if mcp_tools.get('count', 0) > 0:
            # 调用RAG统计工具
            result = await agent_core.call_mcp_tool("rag-server.get_retrieval_stats", {})
            logger.info(f"直接工具调用结果: {result.get('success', False)}")
        
        # 测试MCP对话（如果有RAG功能）
        if hasattr(agent_core, 'mcp_agent') and agent_core.mcp_agent:
            logger.info("测试MCP Agent对话...")
            chat_result = await agent_core.chat_with_mcp("你好，请介绍一下你的功能", use_rag=False)
            logger.info(f"MCP对话结果: {chat_result.get('success', False)}")
            if chat_result.get('response'):
                logger.info(f"Agent回复: {chat_result['response'][:100]}...")
        
        return agent_core
        
    except Exception as e:
        logger.error(f"MCP Agent测试失败: {e}")
        raise


async def test_integration():
    """集成测试"""
    logger.info("=== MCP集成测试 ===")
    
    try:
        # 1. 测试服务器
        server_manager = await test_mcp_servers()
        
        # 2. 测试客户端
        client_manager = await test_mcp_client(server_manager)
        
        # 3. 测试Agent
        agent_core = await test_mcp_agent(server_manager, client_manager)
        
        logger.info("=== 所有测试完成 ===")
        
        # 清理资源
        await server_manager.stop_all_servers()
        
        return True
        
    except Exception as e:
        logger.error(f"集成测试失败: {e}")
        return False


async def main():
    """主函数"""
    logger.info("开始MCP实现测试...")
    
    try:
        success = await test_integration()
        
        if success:
            logger.info("✅ MCP实现测试成功！")
        else:
            logger.error("❌ MCP实现测试失败！")
            
    except Exception as e:
        logger.error(f"测试过程中发生异常: {e}")
        logger.error("❌ MCP实现测试失败！")


if __name__ == "__main__":
    asyncio.run(main())
