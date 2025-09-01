#!/usr/bin/env python3
"""
LangGraph功能展示
演示LangGraph在EPKBS中的强大功能
"""
import asyncio
import json
import tempfile
from pathlib import Path

from src.agent.core import AgentCore
from src.rag.core import RAGCore
from src.utils.logger import get_logger

logger = get_logger("langgraph_showcase")


async def showcase_1_intelligent_document_processing():
    """展示1: 智能文档处理工作流"""
    logger.info("=== 展示1: 智能文档处理工作流 ===")
    
    # 创建复杂测试文档
    complex_doc_content = """
# 人工智能技术报告

## 执行摘要
本报告分析了当前人工智能技术的发展状况和未来趋势。

## 技术分类对比

### 机器学习技术对比表
| 技术类型 | 优势 | 劣势 | 适用场景 | 成熟度 |
|----------|------|------|----------|--------|
| 监督学习 | 准确率高 | 需要标注数据 | 分类、回归 | 高 |
| 无监督学习 | 无需标注 | 解释性差 | 聚类、降维 | 中 |
| 强化学习 | 自主决策 | 训练复杂 | 游戏、控制 | 中 |

### 深度学习架构对比
| 架构 | 参数量 | 训练时间 | 推理速度 | 应用领域 |
|------|--------|----------|----------|----------|
| CNN | 1M-100M | 小时级 | 毫秒级 | 图像处理 |
| RNN | 10M-1B | 天级 | 秒级 | 序列建模 |
| Transformer | 100M-100B | 周级 | 秒级 | 自然语言 |

## 市场数据分析

### 投资趋势图
[这里应该有一个投资趋势图表]

### 技术采用率
- 机器学习: 85%
- 深度学习: 65%
- 自然语言处理: 45%
- 计算机视觉: 55%

## 数学模型

### 损失函数
交叉熵损失: L = -∑(y_i * log(ŷ_i))

### 优化算法
梯度下降: θ = θ - α∇J(θ)

## 结论
人工智能技术正在快速发展，预计未来5年将有重大突破。
"""
    
    # 创建临时文档
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(complex_doc_content)
        doc_path = f.name
    
    try:
        # 创建Agent
        agent = AgentCore(enable_mcp=True)
        await agent.initialize()
        
        logger.info(f"开始处理复杂文档: {doc_path}")
        
        # 使用LangGraph工作流处理文档
        result = await agent.parse_document_with_langgraph(doc_path)
        
        if result["success"]:
            logger.info("✅ 文档处理成功")
            
            # 显示处理摘要
            summary = result.get("result", {}).get("processing_summary", {})
            logger.info(f"处理摘要:")
            logger.info(f"  - 使用工具: {summary.get('tools_list', [])}")
            logger.info(f"  - 质量分数: {summary.get('quality_score', 0):.2f}")
            logger.info(f"  - 内容长度: {summary.get('content_length', 0)}")
            logger.info(f"  - 包含表格: {summary.get('has_tables', False)}")
            
            # 显示元数据
            metadata = result.get("result", {}).get("metadata", {})
            logger.info(f"元数据:")
            logger.info(f"  - 文件类型: {metadata.get('file_info', {}).get('type')}")
            logger.info(f"  - 处理策略: {metadata.get('processing_info', {}).get('strategy')}")
            logger.info(f"  - 表格数量: {metadata.get('content_info', {}).get('table_count', 0)}")
            
        else:
            logger.error(f"❌ 文档处理失败: {result.get('error')}")
    
    finally:
        # 清理
        Path(doc_path).unlink()
        if 'agent' in locals():
            if agent.mcp_server_manager:
                await agent.mcp_server_manager.stop_all_servers()


async def showcase_2_adaptive_agent_reasoning():
    """展示2: 自适应Agent推理"""
    logger.info("=== 展示2: 自适应Agent推理 ===")
    
    # 创建带RAG的Agent
    agent = await AgentCore.create_rag_agent(
        collection_name="showcase_knowledge",
        enable_mcp=True
    )
    
    try:
        # 测试不同复杂度的问题
        test_queries = [
            {
                "query": "你好",
                "expected_complexity": "simple",
                "description": "简单问候"
            },
            {
                "query": "什么是机器学习？",
                "expected_complexity": "medium", 
                "description": "概念解释"
            },
            {
                "query": "请比较监督学习、无监督学习和强化学习的优缺点，并给出具体的应用场景和算法示例",
                "expected_complexity": "complex",
                "description": "复杂分析"
            }
        ]
        
        for i, test_case in enumerate(test_queries, 1):
            logger.info(f"\n--- 测试 {i}: {test_case['description']} ---")
            logger.info(f"问题: {test_case['query']}")
            
            # 使用LangGraph Agent处理
            result = await agent.chat_with_langgraph(
                test_case["query"], 
                use_rag=True
            )
            
            if result["success"]:
                logger.info(f"✅ 回答成功")
                logger.info(f"模式: {result.get('mode')}")
                logger.info(f"执行时间: {result.get('execution_time', 0):.2f}s")
                
                # 显示推理步骤
                langgraph_result = result.get("langgraph_result", {})
                reasoning_steps = langgraph_result.get("reasoning_steps", [])
                if reasoning_steps:
                    logger.info(f"推理步骤: {len(reasoning_steps)} 步")
                    for j, step in enumerate(reasoning_steps[:3], 1):  # 只显示前3步
                        logger.info(f"  步骤 {j}: {step.get('tool', 'N/A')}")
                
                # 显示回答预览
                response = result.get("response", "")
                preview = response[:100] + "..." if len(response) > 100 else response
                logger.info(f"回答预览: {preview}")
                
            else:
                logger.error(f"❌ 回答失败: {result.get('error')}")
    
    finally:
        # 清理
        if agent.mcp_server_manager:
            await agent.mcp_server_manager.stop_all_servers()


async def showcase_3_workflow_composition():
    """展示3: 工作流组合和编排"""
    logger.info("=== 展示3: 工作流组合和编排 ===")
    
    agent = AgentCore(enable_mcp=True)
    await agent.initialize()
    
    try:
        # 创建多个测试文档
        documents = []
        doc_contents = [
            "# 技术文档1\n\n这是关于机器学习的文档。",
            "# 技术文档2\n\n这是关于深度学习的文档。",
            "# 技术文档3\n\n这是关于自然语言处理的文档。"
        ]
        
        for i, content in enumerate(doc_contents):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
                f.write(content)
                documents.append(f.name)
        
        logger.info(f"创建了 {len(documents)} 个测试文档")
        
        # 1. 批量文档处理
        logger.info("1. 执行批量文档处理...")
        batch_result = await agent.batch_process_documents_with_langgraph(documents)
        
        if batch_result["success"]:
            logger.info(f"✅ 批量处理成功")
            logger.info(f"  - 总文件: {batch_result['total_files']}")
            logger.info(f"  - 成功: {batch_result['successful']}")
            logger.info(f"  - 失败: {batch_result['failed']}")
            logger.info(f"  - 成功率: {batch_result['success_rate']:.2%}")
            logger.info(f"  - 吞吐量: {batch_result['throughput']:.2f} 文件/秒")
        
        # 2. 基于处理结果的智能问答
        logger.info("\n2. 基于处理结果的智能问答...")
        qa_result = await agent.chat_with_langgraph(
            "根据刚才处理的文档，总结一下AI技术的主要分类", 
            use_rag=True
        )
        
        if qa_result["success"]:
            logger.info(f"✅ 智能问答成功")
            logger.info(f"回答: {qa_result['response'][:200]}...")
        
        # 3. 工作流可视化
        logger.info("\n3. 生成工作流可视化...")
        for workflow_type in ["document", "agent", "rag"]:
            viz = await agent.get_workflow_visualization(workflow_type)
            if viz:
                logger.info(f"✅ {workflow_type} 工作流可视化生成成功")
                # 保存到文件
                viz_file = f"workflow_{workflow_type}.mermaid"
                with open(viz_file, 'w') as f:
                    f.write(viz)
                logger.info(f"可视化已保存到: {viz_file}")
        
        # 4. 性能统计
        logger.info("\n4. 性能统计分析...")
        stats = agent.get_agent_statistics()
        logger.info(f"Agent统计: {json.dumps(stats, indent=2, ensure_ascii=False)}")
        
    finally:
        # 清理测试文档
        for doc_path in documents:
            Path(doc_path).unlink()
        
        # 停止服务器
        if agent.mcp_server_manager:
            await agent.mcp_server_manager.stop_all_servers()


async def showcase_4_real_world_scenario():
    """展示4: 真实世界场景模拟"""
    logger.info("=== 展示4: 真实世界场景模拟 ===")
    
    # 模拟企业文档处理场景
    scenario_description = """
场景: 企业需要处理一批技术文档，包括：
1. 产品规格书（包含大量表格）
2. 技术报告（包含图表和公式）
3. 用户手册（包含图片说明）

要求: 
- 自动解析所有文档
- 提取关键信息
- 生成结构化摘要
- 支持智能问答
"""
    
    logger.info(scenario_description)
    
    agent = AgentCore(enable_mcp=True)
    await agent.initialize()
    
    try:
        # 模拟企业文档
        enterprise_docs = [
            {
                "name": "产品规格书.md",
                "content": """
# 产品规格书

## 技术参数
| 参数 | 数值 | 单位 | 备注 |
|------|------|------|------|
| CPU | 8核 | 核心 | 高性能处理器 |
| 内存 | 32GB | GB | DDR4 |
| 存储 | 1TB | TB | SSD |

## 性能指标
| 指标 | 数值 | 标准 |
|------|------|------|
| 响应时间 | <100ms | 优秀 |
| 吞吐量 | 1000 QPS | 良好 |
| 可用性 | 99.9% | 企业级 |
"""
            },
            {
                "name": "技术报告.md", 
                "content": """
# AI系统技术报告

## 算法性能分析

### 准确率对比
| 模型 | 准确率 | F1分数 | 训练时间 |
|------|--------|--------|----------|
| BERT | 92.5% | 0.91 | 4小时 |
| GPT-3 | 94.2% | 0.93 | 12小时 |
| T5 | 93.1% | 0.92 | 8小时 |

## 数学模型

### 注意力机制
Attention(Q,K,V) = softmax(QK^T/√d_k)V

### 损失函数
L = -∑(y_i * log(ŷ_i))

## 结论
基于实验结果，推荐使用GPT-3模型。
"""
            }
        ]
        
        # 创建临时文档文件
        doc_paths = []
        for doc_info in enterprise_docs:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
                f.write(doc_info["content"])
                doc_paths.append(f.name)
        
        # 1. 批量智能处理
        logger.info("1. 批量智能处理企业文档...")
        batch_result = await agent.batch_process_documents_with_langgraph(doc_paths)
        
        if batch_result["success"]:
            logger.info("✅ 批量处理完成")
            logger.info(f"处理结果:")
            logger.info(f"  - 成功率: {batch_result['success_rate']:.2%}")
            logger.info(f"  - 平均处理时间: {batch_result['execution_time']/len(doc_paths):.2f}s/文档")
            
            # 显示每个文档的处理结果
            for result in batch_result["results"]:
                if result["success"]:
                    summary = result.get("processing_summary", {})
                    logger.info(f"  📄 {Path(result['file_path']).name}:")
                    logger.info(f"    - 质量分数: {summary.get('quality_score', 0):.2f}")
                    logger.info(f"    - 包含表格: {summary.get('has_tables', False)}")
                    logger.info(f"    - 使用工具: {len(summary.get('tools_list', []))}")
        
        # 2. 智能问答测试
        logger.info("\n2. 基于处理结果的智能问答...")
        
        questions = [
            "这些文档中提到了哪些AI技术？",
            "产品的技术参数是什么？",
            "不同AI模型的性能对比如何？",
            "文档中提到的数学公式有哪些？"
        ]
        
        for question in questions:
            logger.info(f"\n问题: {question}")
            
            answer_result = await agent.chat_with_langgraph(question, use_rag=True)
            
            if answer_result["success"]:
                logger.info(f"✅ 回答: {answer_result['response'][:150]}...")
                logger.info(f"模式: {answer_result.get('mode')}")
            else:
                logger.error(f"❌ 回答失败: {answer_result.get('error')}")
        
        # 3. 工作流性能分析
        logger.info("\n3. 工作流性能分析...")
        stats = agent.get_agent_statistics()
        
        logger.info("性能统计:")
        logger.info(f"  - LangGraph Agent: {'启用' if stats['langgraph_agent']['enabled'] else '禁用'}")
        logger.info(f"  - 可用工具数: {stats['langgraph_agent']['tools']}")
        logger.info(f"  - 工作流数: {stats['langgraph_agent']['workflows']}")
        logger.info(f"  - 对话历史: {stats['conversation_history']} 条")
        
    finally:
        # 清理临时文件
        for doc_path in doc_paths:
            Path(doc_path).unlink()
        
        # 停止服务器
        if agent.mcp_server_manager:
            await agent.mcp_server_manager.stop_all_servers()


async def showcase_5_workflow_visualization():
    """展示5: 工作流可视化"""
    logger.info("=== 展示5: 工作流可视化 ===")
    
    agent = AgentCore(enable_mcp=True)
    await agent.initialize()
    
    try:
        workflow_types = [
            ("document", "文档处理工作流"),
            ("agent", "Agent推理工作流"),
            ("rag", "RAG检索工作流")
        ]
        
        for workflow_type, description in workflow_types:
            logger.info(f"\n生成 {description} 可视化...")
            
            mermaid_code = await agent.get_workflow_visualization(workflow_type)
            
            if mermaid_code:
                logger.info(f"✅ {description} 可视化生成成功")
                
                # 保存Mermaid代码
                output_file = f"docs/workflows/{workflow_type}_workflow.mermaid"
                Path("docs/workflows").mkdir(exist_ok=True)
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(mermaid_code)
                
                logger.info(f"可视化已保存到: {output_file}")
                
                # 显示部分代码
                preview = mermaid_code[:200] + "..." if len(mermaid_code) > 200 else mermaid_code
                logger.info(f"预览:\n{preview}")
                
            else:
                logger.warning(f"❌ {description} 可视化生成失败")
    
    finally:
        # 停止服务器
        if agent.mcp_server_manager:
            await agent.mcp_server_manager.stop_all_servers()


async def main():
    """主函数 - 运行所有展示"""
    logger.info("🚀 开始LangGraph功能展示")
    
    showcases = [
        ("智能文档处理工作流", showcase_1_intelligent_document_processing),
        ("自适应Agent推理", showcase_2_adaptive_agent_reasoning),
        ("工作流可视化", showcase_5_workflow_visualization)
    ]
    
    for name, showcase_func in showcases:
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"🎯 {name}")
            logger.info(f"{'='*60}")
            
            await showcase_func()
            
            logger.info(f"✅ 展示 '{name}' 完成")
            
        except Exception as e:
            logger.error(f"❌ 展示 '{name}' 失败: {e}")
        
        # 等待一下再运行下一个展示
        await asyncio.sleep(2)
    
    logger.info("\n🎉 所有LangGraph功能展示完成！")
    logger.info("\n📊 总结:")
    logger.info("✅ LangGraph提供了强大的状态管理和工作流编排能力")
    logger.info("✅ 智能Agent可以根据任务复杂度自适应选择策略")
    logger.info("✅ 工作流可视化帮助理解和调试复杂流程")
    logger.info("✅ 批量处理和并行执行大大提升了效率")


if __name__ == "__main__":
    asyncio.run(main())
