#!/usr/bin/env python3
"""
MCP使用示例
展示如何使用FastMCP封装的RAG系统
"""
import asyncio
import json
from pathlib import Path

from src.agent.core import AgentCore
from src.rag.core import RAGCore
from src.utils.logger import get_logger

logger = get_logger("mcp_example")


async def example_1_basic_mcp_usage():
    """示例1: 基础MCP使用"""
    logger.info("=== 示例1: 基础MCP使用 ===")
    
    # 创建启用MCP的Agent
    agent = AgentCore(enable_mcp=True)
    await agent.initialize()
    
    # 获取可用的MCP工具
    mcp_tools = await agent.get_mcp_tools()
    logger.info(f"可用工具数量: {mcp_tools.get('count', 0)}")
    
    # 列出所有工具
    for tool_key, tool_info in mcp_tools.get('tools', {}).items():
        logger.info(f"工具: {tool_key}")
        logger.info(f"  描述: {tool_info['description']}")
        logger.info(f"  服务器: {tool_info['server']}")
    
    # 检查MCP服务器状态
    status = await agent.get_mcp_server_status()
    logger.info(f"MCP服务器状态: {status.get('status')}")
    
    return agent


async def example_2_document_parsing():
    """示例2: 智能文档解析"""
    logger.info("=== 示例2: 智能文档解析 ===")
    
    # 创建测试文档
    test_doc = Path("test_document.txt")
    test_doc.write_text("""
# 测试文档

这是一个测试文档，包含以下内容：

## 表格数据
| 姓名 | 年龄 | 职业 |
|------|------|------|
| 张三 | 25   | 工程师 |
| 李四 | 30   | 设计师 |

## 图像说明
[这里应该有一个图像]

## 数学公式
E = mc²

## 总结
这是文档的总结部分。
""")
    
    try:
        # 创建Agent
        agent = AgentCore(enable_mcp=True)
        await agent.initialize()
        
        # 使用MCP Agent解析文档
        logger.info(f"开始解析文档: {test_doc}")
        result = await agent.parse_document_with_mcp(str(test_doc))
        
        if result.get('success'):
            logger.info("✅ 文档解析成功")
            
            # 显示解析结果
            final_answer = result.get('final_answer')
            if final_answer:
                logger.info(f"解析结果: {final_answer}")
            
            # 显示推理步骤
            steps = result.get('reasoning_steps', [])
            logger.info(f"推理步骤数: {len(steps)}")
            
            for i, step in enumerate(steps, 1):
                logger.info(f"步骤 {i}:")
                logger.info(f"  思考: {step.get('thought', '')[:100]}...")
                logger.info(f"  动作: {step.get('action', 'N/A')}")
                if step.get('observation'):
                    logger.info(f"  观察: 工具执行{'成功' if 'success' in str(step['observation']) else '失败'}")
        else:
            logger.error(f"❌ 文档解析失败: {result.get('error')}")
    
    finally:
        # 清理测试文件
        if test_doc.exists():
            test_doc.unlink()


async def example_3_rag_search():
    """示例3: RAG搜索和问答"""
    logger.info("=== 示例3: RAG搜索和问答 ===")
    
    # 创建带RAG的Agent
    agent = await AgentCore.create_rag_agent(
        collection_name="test_knowledge",
        enable_mcp=True
    )
    
    # 测试问题
    questions = [
        "什么是人工智能？",
        "机器学习的主要类型有哪些？",
        "深度学习和传统机器学习的区别是什么？"
    ]
    
    for question in questions:
        logger.info(f"\n问题: {question}")
        
        # 使用MCP Agent回答
        result = await agent.chat_with_mcp(question, use_rag=True)
        
        if result.get('success'):
            logger.info(f"✅ 回答: {result.get('response', '')[:200]}...")
            
            # 显示推理过程
            mcp_result = result.get('mcp_result', {})
            steps = mcp_result.get('reasoning_steps', [])
            logger.info(f"推理步骤: {len(steps)} 步")
            
        else:
            logger.error(f"❌ 回答失败: {result.get('error')}")


async def example_4_direct_tool_calls():
    """示例4: 直接工具调用"""
    logger.info("=== 示例4: 直接工具调用 ===")
    
    agent = AgentCore(enable_mcp=True)
    await agent.initialize()
    
    # 获取RAG系统统计
    logger.info("调用RAG统计工具...")
    result = await agent.call_mcp_tool("rag-server.get_retrieval_stats", {})
    
    if result.get('success'):
        logger.info("✅ RAG统计获取成功")
        stats = result.get('result', {})
        logger.info(f"  文档总数: {stats.get('total_documents', 0)}")
        logger.info(f"  分块总数: {stats.get('total_chunks', 0)}")
        logger.info(f"  向量维度: {stats.get('vector_dimension', 0)}")
    else:
        logger.error(f"❌ RAG统计获取失败: {result.get('error')}")
    
    # 分析查询
    logger.info("\n调用查询分析工具...")
    result = await agent.call_mcp_tool(
        "rag-server.analyze_query", 
        {"query": "什么是深度学习？"}
    )
    
    if result.get('success'):
        logger.info("✅ 查询分析成功")
        analysis = result.get('result', {})
        logger.info(f"  查询意图: {analysis.get('intent')}")
        logger.info(f"  复杂度: {analysis.get('complexity')}")
        logger.info(f"  建议策略: {analysis.get('suggested_strategy')}")
    else:
        logger.error(f"❌ 查询分析失败: {result.get('error')}")


async def example_5_batch_operations():
    """示例5: 批量操作"""
    logger.info("=== 示例5: 批量操作 ===")
    
    agent = AgentCore(enable_mcp=True)
    await agent.initialize()
    
    # 准备批量调用
    batch_calls = [
        {
            "server_name": "rag-server",
            "tool_name": "get_retrieval_stats",
            "arguments": {}
        },
        {
            "server_name": "rag-server", 
            "tool_name": "analyze_query",
            "arguments": {"query": "人工智能的发展历史"}
        },
        {
            "server_name": "rag-server",
            "tool_name": "analyze_query", 
            "arguments": {"query": "如何学习机器学习？"}
        }
    ]
    
    logger.info(f"执行 {len(batch_calls)} 个批量调用...")
    
    # 执行批量调用
    if hasattr(agent.mcp_client_manager.client, 'batch_call_tools'):
        results = await agent.mcp_client_manager.client.batch_call_tools(batch_calls)
        
        for i, result in enumerate(results):
            call = batch_calls[i]
            tool_name = f"{call['server_name']}.{call['tool_name']}"
            
            if result.get('success'):
                logger.info(f"✅ {tool_name}: 成功")
            else:
                logger.error(f"❌ {tool_name}: {result.get('error')}")
    else:
        logger.info("批量调用功能不可用，逐个执行...")
        for call in batch_calls:
            tool_key = f"{call['server_name']}.{call['tool_name']}"
            result = await agent.call_mcp_tool(tool_key, call['arguments'])
            
            if result.get('success'):
                logger.info(f"✅ {tool_key}: 成功")
            else:
                logger.error(f"❌ {tool_key}: {result.get('error')}")


async def main():
    """主函数 - 运行所有示例"""
    logger.info("🚀 开始MCP使用示例演示")
    
    examples = [
        ("基础MCP使用", example_1_basic_mcp_usage),
        ("智能文档解析", example_2_document_parsing),
        ("RAG搜索问答", example_3_rag_search),
        ("直接工具调用", example_4_direct_tool_calls),
        ("批量操作", example_5_batch_operations)
    ]
    
    for name, example_func in examples:
        try:
            logger.info(f"\n{'='*50}")
            logger.info(f"运行示例: {name}")
            logger.info(f"{'='*50}")
            
            await example_func()
            
            logger.info(f"✅ 示例 '{name}' 完成")
            
        except Exception as e:
            logger.error(f"❌ 示例 '{name}' 失败: {e}")
        
        # 等待一下再运行下一个示例
        await asyncio.sleep(1)
    
    logger.info("\n🎉 所有MCP示例演示完成！")


if __name__ == "__main__":
    asyncio.run(main())
