"""
LangGraph状态定义
定义各种工作流的状态结构
"""
from typing import TypedDict, List, Dict, Any, Optional, Annotated
from datetime import datetime
import operator


class BaseState(TypedDict):
    """基础状态类"""
    timestamp: str
    session_id: str
    error_messages: List[str]
    retry_count: int


class DocumentProcessingState(BaseState):
    """文档处理状态"""
    # 输入信息
    file_path: str
    file_type: str
    file_size: int
    
    # 处理状态
    current_stage: str  # "detection", "parsing", "quality_check", "advanced_processing", "integration"
    processing_strategy: str  # "standard", "table_focused", "image_focused", "formula_focused"
    
    # 解析结果
    basic_parsing_result: Optional[Dict[str, Any]]
    table_extraction_result: Optional[Dict[str, Any]]
    image_analysis_result: Optional[Dict[str, Any]]
    formula_recognition_result: Optional[Dict[str, Any]]
    
    # 质量评估
    quality_score: float
    quality_details: Dict[str, Any]
    needs_advanced_processing: bool
    
    # 最终结果
    integrated_content: Optional[str]
    metadata: Dict[str, Any]
    processing_summary: Dict[str, Any]
    
    # 工具使用记录
    tools_used: List[str]
    tool_results: Dict[str, Any]
    
    # 状态控制
    is_complete: bool
    next_action: str


class AgentReasoningState(BaseState):
    """Agent推理状态"""
    # 对话信息
    messages: Annotated[List[Dict[str, str]], operator.add]
    user_query: str
    
    # 推理过程
    current_thought: str
    planned_actions: List[str]
    current_action: Optional[str]
    action_input: Optional[Dict[str, Any]]
    observation: Optional[str]
    
    # 工具调用
    available_tools: Dict[str, Any]
    tool_call_history: List[Dict[str, Any]]
    
    # 上下文管理
    context_documents: List[Dict[str, Any]]
    context_length: int
    max_context_length: int
    
    # 推理控制
    iteration_count: int
    max_iterations: int
    reasoning_complete: bool
    final_answer: Optional[str]
    
    # 性能指标
    total_execution_time: float
    tool_execution_times: Dict[str, float]


class RAGWorkflowState(BaseState):
    """RAG工作流状态"""
    # 查询信息
    original_query: str
    processed_queries: List[str]  # 多查询扩展
    query_intent: str
    query_complexity: str
    
    # 检索配置
    retrieval_strategy: str  # "vector", "sparse", "hybrid", "adaptive"
    top_k: int
    rerank_top_k: int
    
    # 检索结果
    vector_results: List[Dict[str, Any]]
    sparse_results: List[Dict[str, Any]]
    fulltext_results: List[Dict[str, Any]]
    
    # 重排序结果
    reranked_results: List[Dict[str, Any]]
    final_context: str
    
    # 生成结果
    generated_answer: Optional[str]
    answer_quality: Optional[float]
    
    # 元数据
    retrieval_metadata: Dict[str, Any]
    generation_metadata: Dict[str, Any]


class MCPToolState(BaseState):
    """MCP工具调用状态"""
    # 工具信息
    server_name: str
    tool_name: str
    tool_arguments: Dict[str, Any]
    
    # 执行状态
    execution_status: str  # "pending", "running", "completed", "failed"
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    execution_time: Optional[float]
    
    # 结果
    tool_result: Optional[Dict[str, Any]]
    success: bool
    error_message: Optional[str]
    
    # 重试机制
    max_retries: int
    current_retry: int
    retry_strategy: str  # "exponential", "linear", "immediate"
    
    # 依赖关系
    depends_on: List[str]  # 依赖的其他工具调用
    blocks: List[str]      # 阻塞的其他工具调用


class BatchProcessingState(BaseState):
    """批量处理状态"""
    # 任务信息
    task_id: str
    task_type: str  # "document_batch", "query_batch", "analysis_batch"
    total_items: int
    
    # 处理进度
    processed_items: int
    failed_items: int
    current_item: Optional[str]
    
    # 批量结果
    batch_results: List[Dict[str, Any]]
    success_rate: float
    
    # 并行控制
    max_concurrent: int
    current_concurrent: int
    
    # 性能统计
    average_processing_time: float
    total_processing_time: float
    throughput: float  # items per second


class ConversationState(BaseState):
    """对话状态"""
    # 对话历史
    conversation_history: Annotated[List[Dict[str, str]], operator.add]
    current_message: str
    
    # 用户信息
    user_id: str
    user_preferences: Dict[str, Any]
    
    # 对话上下文
    conversation_context: str
    context_summary: str
    
    # RAG使用
    use_rag: bool
    rag_results: Optional[List[Dict[str, Any]]]
    
    # 响应生成
    response_type: str  # "direct", "rag_enhanced", "tool_assisted"
    generated_response: Optional[str]
    response_confidence: Optional[float]
    
    # 对话管理
    conversation_turn: int
    max_turns: int
    conversation_complete: bool


class SystemMonitoringState(BaseState):
    """系统监控状态"""
    # 系统健康
    system_health: str  # "healthy", "degraded", "unhealthy"
    component_status: Dict[str, str]
    
    # 性能指标
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    
    # 服务状态
    mcp_servers_status: Dict[str, str]
    database_status: str
    cache_status: str
    
    # 业务指标
    active_users: int
    requests_per_minute: float
    average_response_time: float
    error_rate: float
    
    # 告警信息
    alerts: List[Dict[str, Any]]
    alert_level: str  # "info", "warning", "critical"


# 状态工厂函数
def create_document_processing_state(
    file_path: str,
    session_id: str = None
) -> DocumentProcessingState:
    """创建文档处理状态"""
    return DocumentProcessingState(
        # 基础信息
        timestamp=datetime.now().isoformat(),
        session_id=session_id or f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        error_messages=[],
        retry_count=0,
        
        # 文档信息
        file_path=file_path,
        file_type="",
        file_size=0,
        
        # 处理状态
        current_stage="detection",
        processing_strategy="standard",
        
        # 结果初始化
        basic_parsing_result=None,
        table_extraction_result=None,
        image_analysis_result=None,
        formula_recognition_result=None,
        
        # 质量评估
        quality_score=0.0,
        quality_details={},
        needs_advanced_processing=False,
        
        # 最终结果
        integrated_content=None,
        metadata={},
        processing_summary={},
        
        # 工具记录
        tools_used=[],
        tool_results={},
        
        # 控制状态
        is_complete=False,
        next_action="detect_format"
    )


def create_agent_reasoning_state(
    user_query: str,
    session_id: str = None,
    max_iterations: int = 10
) -> AgentReasoningState:
    """创建Agent推理状态"""
    return AgentReasoningState(
        # 基础信息
        timestamp=datetime.now().isoformat(),
        session_id=session_id or f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        error_messages=[],
        retry_count=0,
        
        # 对话信息
        messages=[],
        user_query=user_query,
        
        # 推理过程
        current_thought="",
        planned_actions=[],
        current_action=None,
        action_input=None,
        observation=None,
        
        # 工具调用
        available_tools={},
        tool_call_history=[],
        
        # 上下文管理
        context_documents=[],
        context_length=0,
        max_context_length=4000,
        
        # 推理控制
        iteration_count=0,
        max_iterations=max_iterations,
        reasoning_complete=False,
        final_answer=None,
        
        # 性能指标
        total_execution_time=0.0,
        tool_execution_times={}
    )


def create_rag_workflow_state(
    query: str,
    session_id: str = None
) -> RAGWorkflowState:
    """创建RAG工作流状态"""
    return RAGWorkflowState(
        # 基础信息
        timestamp=datetime.now().isoformat(),
        session_id=session_id or f"rag_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        error_messages=[],
        retry_count=0,
        
        # 查询信息
        original_query=query,
        processed_queries=[],
        query_intent="",
        query_complexity="",
        
        # 检索配置
        retrieval_strategy="hybrid",
        top_k=10,
        rerank_top_k=5,
        
        # 检索结果
        vector_results=[],
        sparse_results=[],
        fulltext_results=[],
        
        # 重排序结果
        reranked_results=[],
        final_context="",
        
        # 生成结果
        generated_answer=None,
        answer_quality=None,
        
        # 元数据
        retrieval_metadata={},
        generation_metadata={}
    )
