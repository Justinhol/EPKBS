"""
LangGraph工作流定义
定义各种业务场景的工作流图
"""
from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, create_react_agent

from .states import (
    DocumentProcessingState,
    AgentReasoningState, 
    RAGWorkflowState,
    ConversationState
)
from .nodes import (
    detect_document_format,
    basic_document_parse,
    assess_parsing_quality,
    advanced_document_processing,
    integrate_processing_results,
    agent_think,
    agent_act,
    agent_observe,
    analyze_query_intent,
    execute_retrieval,
    generate_context,
    error_handling
)
from ..utils.logger import get_logger

logger = get_logger("agent.workflows")


class DocumentProcessingWorkflow:
    """文档处理工作流"""
    
    def __init__(self, mcp_client_manager):
        self.mcp_client = mcp_client_manager
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """构建文档处理工作流图"""
        workflow = StateGraph(DocumentProcessingState)
        
        # 添加节点
        workflow.add_node("detect_format", detect_document_format)
        workflow.add_node("basic_parse", self._wrap_node(basic_document_parse))
        workflow.add_node("quality_check", self._wrap_node(assess_parsing_quality))
        workflow.add_node("advanced_processing", self._wrap_node(advanced_document_processing))
        workflow.add_node("integrate_results", integrate_processing_results)
        workflow.add_node("error_handling", error_handling)
        
        # 设置入口点
        workflow.set_entry_point("detect_format")
        
        # 添加边
        workflow.add_edge("detect_format", "basic_parse")
        workflow.add_edge("basic_parse", "quality_check")
        
        # 条件边：根据质量评估决定下一步
        workflow.add_conditional_edges(
            "quality_check",
            self._should_use_advanced_processing,
            {
                "advanced": "advanced_processing",
                "integrate": "integrate_results",
                "error": "error_handling"
            }
        )
        
        workflow.add_edge("advanced_processing", "integrate_results")
        workflow.add_edge("integrate_results", END)
        workflow.add_edge("error_handling", END)
        
        return workflow.compile()
    
    def _wrap_node(self, node_func):
        """包装节点函数，注入MCP客户端"""
        async def wrapped_node(state):
            return await node_func(state, self.mcp_client)
        return wrapped_node
    
    def _should_use_advanced_processing(self, state: DocumentProcessingState) -> str:
        """判断是否需要高级处理"""
        try:
            if state.get("error_messages"):
                return "error"
            
            if state.get("needs_advanced_processing", False):
                return "advanced"
            else:
                return "integrate"
                
        except Exception:
            return "error"
    
    async def process_document(self, file_path: str) -> Dict[str, Any]:
        """处理文档"""
        from .states import create_document_processing_state
        
        initial_state = create_document_processing_state(file_path)
        
        logger.info(f"开始文档处理工作流: {file_path}")
        
        try:
            # 执行工作流
            final_state = await self.workflow.ainvoke(initial_state)
            
            return {
                "success": final_state.get("is_complete", False),
                "result": {
                    "integrated_content": final_state.get("integrated_content"),
                    "metadata": final_state.get("metadata"),
                    "processing_summary": final_state.get("processing_summary"),
                    "tools_used": final_state.get("tools_used", []),
                    "quality_score": final_state.get("quality_score", 0.0)
                },
                "errors": final_state.get("error_messages", []),
                "state": final_state
            }
            
        except Exception as e:
            logger.error(f"文档处理工作流失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "state": initial_state
            }


class AgentReasoningWorkflow:
    """Agent推理工作流"""
    
    def __init__(self, llm_manager, mcp_client_manager):
        self.llm_manager = llm_manager
        self.mcp_client = mcp_client_manager
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """构建Agent推理工作流图"""
        workflow = StateGraph(AgentReasoningState)
        
        # 添加节点
        workflow.add_node("think", self._wrap_think_node)
        workflow.add_node("act", self._wrap_act_node)
        workflow.add_node("observe", agent_observe)
        workflow.add_node("error_handling", error_handling)
        
        # 设置入口点
        workflow.set_entry_point("think")
        
        # 添加条件边
        workflow.add_conditional_edges(
            "think",
            self._route_after_thinking,
            {
                "act": "act",
                "done": END,
                "error": "error_handling"
            }
        )
        
        workflow.add_edge("act", "observe")
        
        workflow.add_conditional_edges(
            "observe",
            self._should_continue_reasoning,
            {
                "continue": "think",
                "done": END,
                "error": "error_handling"
            }
        )
        
        workflow.add_edge("error_handling", END)
        
        return workflow.compile()
    
    def _wrap_think_node(self, state):
        """包装思考节点"""
        return agent_think(state, self.llm_manager)
    
    def _wrap_act_node(self, state):
        """包装行动节点"""
        return agent_act(state, self.mcp_client)
    
    def _route_after_thinking(self, state: AgentReasoningState) -> str:
        """思考后的路由决策"""
        try:
            if state.get("error_messages"):
                return "error"
            
            if state.get("reasoning_complete"):
                return "done"
            
            if state.get("current_action"):
                return "act"
            
            return "error"  # 没有明确的行动
            
        except Exception:
            return "error"
    
    def _should_continue_reasoning(self, state: AgentReasoningState) -> str:
        """判断是否继续推理"""
        try:
            if state.get("error_messages"):
                return "error"
            
            if state.get("reasoning_complete"):
                return "done"
            
            if state.get("iteration_count", 0) >= state.get("max_iterations", 10):
                return "done"
            
            return "continue"
            
        except Exception:
            return "error"
    
    async def reason_and_answer(self, user_query: str, max_iterations: int = 10) -> Dict[str, Any]:
        """推理并回答"""
        from .states import create_agent_reasoning_state
        
        # 获取可用工具
        available_tools = await self.mcp_client.get_available_tools()
        
        initial_state = create_agent_reasoning_state(user_query, max_iterations=max_iterations)
        initial_state["available_tools"] = available_tools
        
        logger.info(f"开始Agent推理工作流: {user_query}")
        
        try:
            # 执行工作流
            final_state = await self.workflow.ainvoke(initial_state)
            
            return {
                "success": final_state.get("reasoning_complete", False),
                "answer": final_state.get("final_answer"),
                "reasoning_steps": final_state.get("tool_call_history", []),
                "iterations": final_state.get("iteration_count", 0),
                "execution_time": final_state.get("total_execution_time", 0.0),
                "errors": final_state.get("error_messages", []),
                "state": final_state
            }
            
        except Exception as e:
            logger.error(f"Agent推理工作流失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "answer": "抱歉，我无法处理您的请求。",
                "state": initial_state
            }


class RAGWorkflow:
    """RAG检索工作流"""
    
    def __init__(self, mcp_client_manager):
        self.mcp_client = mcp_client_manager
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """构建RAG工作流图"""
        workflow = StateGraph(RAGWorkflowState)
        
        # 添加节点
        workflow.add_node("analyze_intent", self._wrap_node(analyze_query_intent))
        workflow.add_node("execute_retrieval", self._wrap_node(execute_retrieval))
        workflow.add_node("generate_context", self._wrap_node(generate_context))
        workflow.add_node("error_handling", error_handling)
        
        # 设置入口点
        workflow.set_entry_point("analyze_intent")
        
        # 添加边
        workflow.add_edge("analyze_intent", "execute_retrieval")
        workflow.add_edge("execute_retrieval", "generate_context")
        workflow.add_edge("generate_context", END)
        workflow.add_edge("error_handling", END)
        
        return workflow.compile()
    
    def _wrap_node(self, node_func):
        """包装节点函数"""
        async def wrapped_node(state):
            return await node_func(state, self.mcp_client)
        return wrapped_node
    
    async def search_and_retrieve(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """搜索和检索"""
        from .states import create_rag_workflow_state
        
        initial_state = create_rag_workflow_state(query)
        initial_state["top_k"] = top_k
        
        logger.info(f"开始RAG工作流: {query}")
        
        try:
            # 执行工作流
            final_state = await self.workflow.ainvoke(initial_state)
            
            return {
                "success": True,
                "context": final_state.get("final_context", ""),
                "results": final_state.get("reranked_results", []),
                "metadata": {
                    "retrieval": final_state.get("retrieval_metadata", {}),
                    "generation": final_state.get("generation_metadata", {})
                },
                "errors": final_state.get("error_messages", []),
                "state": final_state
            }
            
        except Exception as e:
            logger.error(f"RAG工作流失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "context": "",
                "results": []
            }


class WorkflowManager:
    """工作流管理器"""
    
    def __init__(self, llm_manager, mcp_client_manager):
        self.llm_manager = llm_manager
        self.mcp_client = mcp_client_manager
        
        # 初始化所有工作流
        self.document_workflow = DocumentProcessingWorkflow(mcp_client_manager)
        self.agent_workflow = AgentReasoningWorkflow(llm_manager, mcp_client_manager)
        self.rag_workflow = RAGWorkflow(mcp_client_manager)
        
        logger.info("工作流管理器初始化完成")
    
    async def process_document(self, file_path: str) -> Dict[str, Any]:
        """处理文档"""
        return await self.document_workflow.process_document(file_path)
    
    async def agent_reasoning(self, query: str, max_iterations: int = 10) -> Dict[str, Any]:
        """Agent推理"""
        return await self.agent_workflow.reason_and_answer(query, max_iterations)
    
    async def rag_search(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """RAG搜索"""
        return await self.rag_workflow.search_and_retrieve(query, top_k)
    
    def visualize_workflow(self, workflow_type: str, output_path: str = None):
        """可视化工作流"""
        try:
            if workflow_type == "document":
                graph = self.document_workflow.workflow.get_graph()
            elif workflow_type == "agent":
                graph = self.agent_workflow.workflow.get_graph()
            elif workflow_type == "rag":
                graph = self.rag_workflow.workflow.get_graph()
            else:
                raise ValueError(f"未知的工作流类型: {workflow_type}")
            
            # 生成Mermaid图
            mermaid_code = graph.draw_mermaid()
            
            if output_path:
                # 保存到文件
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(mermaid_code)
                logger.info(f"工作流图已保存到: {output_path}")
            
            return mermaid_code
            
        except Exception as e:
            logger.error(f"工作流可视化失败: {e}")
            return None
    
    def get_workflow_stats(self) -> Dict[str, Any]:
        """获取工作流统计信息"""
        return {
            "workflows": {
                "document_processing": {
                    "nodes": len(self.document_workflow.workflow.nodes),
                    "edges": len(self.document_workflow.workflow.edges)
                },
                "agent_reasoning": {
                    "nodes": len(self.agent_workflow.workflow.nodes),
                    "edges": len(self.agent_workflow.workflow.edges)
                },
                "rag_search": {
                    "nodes": len(self.rag_workflow.workflow.nodes),
                    "edges": len(self.rag_workflow.workflow.edges)
                }
            },
            "total_workflows": 3
        }
