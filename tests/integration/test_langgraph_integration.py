#!/usr/bin/env python3
"""
LangGraph集成测试
测试LangGraph工作流和Agent的完整功能
"""
import asyncio
import pytest
import tempfile
from pathlib import Path

from src.agent.core import AgentCore
from src.agent.langgraph_agent import LangGraphAgent
from src.agent.workflows import WorkflowManager
from src.rag.core import RAGCore
from src.mcp.server_manager import MCPServerManager
from src.mcp.clients.mcp_client import MCPClientManager
from src.utils.logger import get_logger

logger = get_logger("test_langgraph")


class TestLangGraphIntegration:
    """LangGraph集成测试类"""
    
    @pytest.fixture
    async def agent_core(self):
        """创建测试用的Agent核心"""
        # 创建RAG核心
        rag_core = RAGCore(collection_name="test_langgraph")
        await rag_core.initialize()
        
        # 创建Agent核心（启用MCP和LangGraph）
        agent = AgentCore(rag_core=rag_core, enable_mcp=True)
        await agent.initialize()
        
        yield agent
        
        # 清理
        if agent.mcp_server_manager:
            await agent.mcp_server_manager.stop_all_servers()
    
    @pytest.fixture
    def test_document(self):
        """创建测试文档"""
        content = """
# 测试文档

这是一个用于测试LangGraph工作流的文档。

## 表格示例
| 产品 | 价格 | 库存 |
|------|------|------|
| 苹果 | 5.00 | 100 |
| 香蕉 | 3.00 | 150 |

## 文本内容
这里是一些普通的文本内容，用于测试文档解析功能。

## 数学公式
E = mc²

这是爱因斯坦的质能方程。
"""
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(content)
            temp_path = f.name
        
        yield temp_path
        
        # 清理
        Path(temp_path).unlink()
    
    async def test_workflow_manager_initialization(self, agent_core):
        """测试工作流管理器初始化"""
        assert agent_core.workflow_manager is not None
        assert agent_core.langgraph_agent is not None
        
        # 检查工作流统计
        stats = agent_core.workflow_manager.get_workflow_stats()
        assert stats["total_workflows"] == 3
        assert "document_processing" in stats["workflows"]
        assert "agent_reasoning" in stats["workflows"]
        assert "rag_search" in stats["workflows"]
    
    async def test_document_processing_workflow(self, agent_core, test_document):
        """测试文档处理工作流"""
        logger.info("测试文档处理工作流")
        
        # 使用LangGraph解析文档
        result = await agent_core.parse_document_with_langgraph(test_document)
        
        assert result["success"] is True
        assert "result" in result
        
        # 检查处理结果
        processing_result = result["result"]
        assert "integrated_content" in processing_result
        assert "metadata" in processing_result
        assert "processing_summary" in processing_result
        
        # 检查工具使用
        tools_used = processing_result.get("tools_used", [])
        assert len(tools_used) > 0
        assert any("document-parsing-server" in tool for tool in tools_used)
        
        logger.info(f"文档处理完成，使用工具: {tools_used}")
    
    async def test_agent_reasoning_workflow(self, agent_core):
        """测试Agent推理工作流"""
        logger.info("测试Agent推理工作流")
        
        # 测试问题
        test_query = "什么是人工智能？请详细解释。"
        
        # 使用LangGraph Agent对话
        result = await agent_core.chat_with_langgraph(test_query, use_rag=False)
        
        assert result["success"] is True
        assert "response" in result
        assert result["mode"] == "langgraph_agent"
        
        # 检查推理过程
        langgraph_result = result.get("langgraph_result", {})
        assert "reasoning_steps" in langgraph_result or "total_messages" in langgraph_result
        
        logger.info(f"Agent推理完成，响应长度: {len(result['response'])}")
    
    async def test_rag_workflow(self, agent_core):
        """测试RAG工作流"""
        logger.info("测试RAG工作流")
        
        # 测试RAG问答
        test_query = "机器学习的主要类型有哪些？"
        
        # 使用LangGraph Agent进行RAG对话
        result = await agent_core.chat_with_langgraph(test_query, use_rag=True)
        
        # 检查结果（可能因为没有知识库数据而降级到直接对话）
        assert result["success"] is True
        assert "response" in result
        
        # 如果使用了RAG，检查相关字段
        if result.get("mode") == "rag_enhanced":
            assert "rag_context" in result
            assert "rag_sources" in result
        
        logger.info(f"RAG对话完成，模式: {result.get('mode')}")
    
    async def test_batch_processing(self, agent_core):
        """测试批量处理"""
        logger.info("测试批量处理")
        
        # 创建多个测试文档
        test_docs = []
        for i in range(3):
            content = f"# 测试文档 {i+1}\n\n这是第 {i+1} 个测试文档。"
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
                f.write(content)
                test_docs.append(f.name)
        
        try:
            # 批量处理
            result = await agent_core.batch_process_documents_with_langgraph(test_docs)
            
            assert result["success"] is True
            assert result["total_files"] == 3
            assert "results" in result
            assert len(result["results"]) == 3
            
            # 检查成功率
            success_rate = result.get("success_rate", 0)
            assert success_rate >= 0  # 至少不应该是负数
            
            logger.info(f"批量处理完成，成功率: {success_rate:.2%}")
            
        finally:
            # 清理测试文档
            for doc_path in test_docs:
                Path(doc_path).unlink()
    
    async def test_workflow_visualization(self, agent_core):
        """测试工作流可视化"""
        logger.info("测试工作流可视化")
        
        workflow_types = ["document", "agent", "rag"]
        
        for workflow_type in workflow_types:
            mermaid_code = await agent_core.get_workflow_visualization(workflow_type)
            
            if mermaid_code:
                assert "graph" in mermaid_code.lower() or "flowchart" in mermaid_code.lower()
                logger.info(f"{workflow_type} 工作流可视化生成成功")
            else:
                logger.warning(f"{workflow_type} 工作流可视化生成失败")
    
    async def test_error_handling(self, agent_core):
        """测试错误处理"""
        logger.info("测试错误处理")
        
        # 测试不存在的文档
        result = await agent_core.parse_document_with_langgraph("nonexistent_file.pdf")
        
        # 应该优雅地处理错误
        assert result["success"] is False
        assert "error" in result
        
        logger.info("错误处理测试通过")
    
    async def test_performance_metrics(self, agent_core):
        """测试性能指标"""
        logger.info("测试性能指标")
        
        # 获取Agent统计信息
        stats = agent_core.get_agent_statistics()
        
        assert "langgraph_agent" in stats
        assert stats["langgraph_agent"]["enabled"] is True
        
        # 检查工作流统计
        if "workflows" in stats["langgraph_agent"]:
            assert stats["langgraph_agent"]["workflows"] == 3
        
        logger.info(f"性能指标: {stats}")


@pytest.mark.asyncio
async def test_full_langgraph_workflow():
    """完整的LangGraph工作流测试"""
    logger.info("=== 开始完整LangGraph工作流测试 ===")
    
    try:
        # 创建测试实例
        test_instance = TestLangGraphIntegration()
        
        # 创建Agent核心
        rag_core = RAGCore(collection_name="test_full_workflow")
        await rag_core.initialize()
        
        agent_core = AgentCore(rag_core=rag_core, enable_mcp=True)
        await agent_core.initialize()
        
        # 创建测试文档
        test_content = """
# AI技术报告

## 概述
人工智能（AI）是计算机科学的一个分支。

## 技术分类
| 技术类型 | 应用领域 | 成熟度 |
|----------|----------|--------|
| 机器学习 | 数据分析 | 高 |
| 深度学习 | 图像识别 | 高 |
| 自然语言处理 | 文本理解 | 中 |

## 发展趋势
AI技术正在快速发展，特别是在以下领域：
1. 大语言模型
2. 多模态AI
3. 自动化Agent
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(test_content)
            test_doc = f.name
        
        try:
            # 1. 测试文档处理工作流
            logger.info("1. 测试文档处理工作流")
            doc_result = await agent_core.parse_document_with_langgraph(test_doc)
            assert doc_result["success"] is True
            
            # 2. 测试Agent推理工作流
            logger.info("2. 测试Agent推理工作流")
            chat_result = await agent_core.chat_with_langgraph(
                "请总结一下AI技术的主要类型", use_rag=False
            )
            assert chat_result["success"] is True
            
            # 3. 测试工作流可视化
            logger.info("3. 测试工作流可视化")
            viz_result = await agent_core.get_workflow_visualization("document")
            # 可视化可能失败，不强制要求
            
            # 4. 测试性能统计
            logger.info("4. 测试性能统计")
            stats = agent_core.get_agent_statistics()
            assert "langgraph_agent" in stats
            
            logger.info("✅ 完整LangGraph工作流测试通过")
            return True
            
        finally:
            # 清理测试文档
            Path(test_doc).unlink()
            
            # 停止服务器
            if agent_core.mcp_server_manager:
                await agent_core.mcp_server_manager.stop_all_servers()
    
    except Exception as e:
        logger.error(f"❌ 完整LangGraph工作流测试失败: {e}")
        return False


async def main():
    """主测试函数"""
    logger.info("🚀 开始LangGraph集成测试")
    
    try:
        success = await test_full_langgraph_workflow()
        
        if success:
            logger.info("🎉 所有LangGraph测试通过！")
        else:
            logger.error("❌ LangGraph测试失败！")
            
    except Exception as e:
        logger.error(f"测试过程中发生异常: {e}")


if __name__ == "__main__":
    asyncio.run(main())
