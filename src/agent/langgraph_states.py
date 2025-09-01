"""
LangGraph状态定义
定义各种工作流的状态结构
"""
from typing import TypedDict, List, Dict, Any, Optional, Annotated
from datetime import datetime
from enum import Enum

from langchain.schema import BaseMessage
from langgraph.graph import add_messages


class ProcessingStatus(str, Enum):
    """处理状态枚举"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


class DocumentType(str, Enum):
    """文档类型枚举"""
    PDF = "pdf"
    WORD = "word"
    POWERPOINT = "powerpoint"
    EXCEL = "excel"
    TEXT = "text"
    HTML = "html"
    MARKDOWN = "markdown"
    UNKNOWN = "unknown"


class AgentState(TypedDict):
    """Agent推理状态"""
    # 消息历史
    messages: Annotated[List[BaseMessage], add_messages]
    
    # 任务信息
    task: str
    task_type: str
    user_id: Optional[str]
    session_id: Optional[str]
    
    # 推理过程
    current_step: int
    max_steps: int
    reasoning_history: List[Dict[str, Any]]
    
    # 工具调用
    available_tools: Dict[str, Any]
    tool_calls: List[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]
    
    # 状态管理
    status: ProcessingStatus
    error_count: int
    retry_count: int
    last_error: Optional[str]
    
    # 结果
    final_answer: Optional[str]
    confidence_score: Optional[float]
    
    # 元数据
    start_time: datetime
    execution_time: Optional[float]
    metadata: Dict[str, Any]


class DocumentProcessingState(TypedDict):
    """文档处理状态"""
    # 文件信息
    file_path: str
    file_name: str
    file_size: int
    document_type: DocumentType
    
    # 处理状态
    status: ProcessingStatus
    current_stage: str
    progress_percentage: float
    
    # 解析结果
    parsing_results: List[Dict[str, Any]]
    extracted_text: str
    extracted_tables: List[Dict[str, Any]]
    extracted_images: List[Dict[str, Any]]
    extracted_formulas: List[Dict[str, Any]]
    
    # 质量评估
    quality_score: float
    quality_metrics: Dict[str, float]
    quality_issues: List[str]
    
    # 处理策略
    parsing_strategy: str
    tools_used: List[str]
    retry_count: int
    max_retries: int
    
    # 错误处理
    errors: List[Dict[str, Any]]
    warnings: List[str]
    
    # 最终结果
    integrated_content: Optional[str]
    chunks: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    
    # 时间信息
    start_time: datetime
    end_time: Optional[datetime]
    processing_time: Optional[float]


class RAGSearchState(TypedDict):
    """RAG搜索状态"""
    # 查询信息
    original_query: str
    processed_query: str
    expanded_queries: List[str]
    query_intent: str
    query_complexity: str
    
    # 搜索配置
    top_k: int
    similarity_threshold: float
    retriever_types: List[str]
    reranker_type: str
    
    # 搜索结果
    vector_results: List[Dict[str, Any]]
    sparse_results: List[Dict[str, Any]]
    fulltext_results: List[Dict[str, Any]]
    merged_results: List[Dict[str, Any]]
    reranked_results: List[Dict[str, Any]]
    
    # 上下文生成
    context: str
    context_length: int
    source_documents: List[Dict[str, Any]]
    
    # 状态管理
    status: ProcessingStatus
    current_stage: str
    error_count: int
    
    # 性能指标
    search_time: Optional[float]
    rerank_time: Optional[float]
    total_time: Optional[float]
    
    # 元数据
    metadata: Dict[str, Any]


class BatchProcessingState(TypedDict):
    """批处理状态"""
    # 任务信息
    batch_id: str
    task_type: str
    total_items: int
    
    # 处理状态
    processed_items: int
    successful_items: int
    failed_items: int
    current_item: Optional[str]
    
    # 结果收集
    results: List[Dict[str, Any]]
    errors: List[Dict[str, Any]]
    
    # 进度跟踪
    progress_percentage: float
    estimated_remaining_time: Optional[float]
    
    # 配置
    parallel_workers: int
    batch_size: int
    enable_error_recovery: bool
    
    # 状态
    status: ProcessingStatus
    start_time: datetime
    end_time: Optional[datetime]


class ToolExecutionState(TypedDict):
    """工具执行状态"""
    # 工具信息
    tool_name: str
    server_name: str
    tool_description: str
    
    # 执行参数
    arguments: Dict[str, Any]
    execution_id: str
    
    # 执行状态
    status: ProcessingStatus
    start_time: datetime
    end_time: Optional[datetime]
    execution_time: Optional[float]
    
    # 结果
    result: Optional[Any]
    error: Optional[str]
    
    # 重试信息
    retry_count: int
    max_retries: int
    
    # 元数据
    metadata: Dict[str, Any]


class WorkflowState(TypedDict):
    """通用工作流状态"""
    # 工作流信息
    workflow_id: str
    workflow_type: WorkflowType
    workflow_name: str
    
    # 执行状态
    status: ProcessingStatus
    current_node: str
    completed_nodes: List[str]
    failed_nodes: List[str]
    
    # 数据流
    input_data: Dict[str, Any]
    intermediate_data: Dict[str, Any]
    output_data: Dict[str, Any]
    
    # 执行历史
    execution_history: List[Dict[str, Any]]
    node_results: Dict[str, Any]
    
    # 错误处理
    errors: List[Dict[str, Any]]
    retry_count: int
    
    # 性能指标
    start_time: datetime
    end_time: Optional[datetime]
    node_execution_times: Dict[str, float]
    
    # 配置
    config: Dict[str, Any]
    metadata: Dict[str, Any]


# 状态工厂函数
def create_agent_state(
    task: str,
    task_type: str = "general",
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    max_steps: int = 15
) -> AgentState:
    """创建Agent状态"""
    return AgentState(
        messages=[],
        task=task,
        task_type=task_type,
        user_id=user_id,
        session_id=session_id,
        current_step=0,
        max_steps=max_steps,
        reasoning_history=[],
        available_tools={},
        tool_calls=[],
        tool_results=[],
        status=ProcessingStatus.PENDING,
        error_count=0,
        retry_count=0,
        last_error=None,
        final_answer=None,
        confidence_score=None,
        start_time=datetime.now(),
        execution_time=None,
        metadata={}
    )


def create_document_processing_state(
    file_path: str,
    document_type: DocumentType = DocumentType.UNKNOWN,
    max_retries: int = 3
) -> DocumentProcessingState:
    """创建文档处理状态"""
    from pathlib import Path
    
    file_info = Path(file_path)
    
    return DocumentProcessingState(
        file_path=file_path,
        file_name=file_info.name,
        file_size=file_info.stat().st_size if file_info.exists() else 0,
        document_type=document_type,
        status=ProcessingStatus.PENDING,
        current_stage="initialization",
        progress_percentage=0.0,
        parsing_results=[],
        extracted_text="",
        extracted_tables=[],
        extracted_images=[],
        extracted_formulas=[],
        quality_score=0.0,
        quality_metrics={},
        quality_issues=[],
        parsing_strategy="auto",
        tools_used=[],
        retry_count=0,
        max_retries=max_retries,
        errors=[],
        warnings=[],
        integrated_content=None,
        chunks=[],
        metadata={},
        start_time=datetime.now(),
        end_time=None,
        processing_time=None
    )


def create_rag_search_state(
    query: str,
    top_k: int = 5,
    similarity_threshold: float = 0.7
) -> RAGSearchState:
    """创建RAG搜索状态"""
    return RAGSearchState(
        original_query=query,
        processed_query=query,
        expanded_queries=[],
        query_intent="unknown",
        query_complexity="medium",
        top_k=top_k,
        similarity_threshold=similarity_threshold,
        retriever_types=["vector", "sparse"],
        reranker_type="ensemble",
        vector_results=[],
        sparse_results=[],
        fulltext_results=[],
        merged_results=[],
        reranked_results=[],
        context="",
        context_length=0,
        source_documents=[],
        status=ProcessingStatus.PENDING,
        current_stage="initialization",
        error_count=0,
        search_time=None,
        rerank_time=None,
        total_time=None,
        metadata={}
    )


# 状态更新辅助函数
def update_state_status(state: Dict[str, Any], status: ProcessingStatus, stage: str = None) -> Dict[str, Any]:
    """更新状态"""
    updated_state = state.copy()
    updated_state["status"] = status
    
    if stage:
        updated_state["current_stage"] = stage
    
    return updated_state


def add_error_to_state(state: Dict[str, Any], error: str, error_type: str = "general") -> Dict[str, Any]:
    """向状态添加错误"""
    updated_state = state.copy()
    
    error_info = {
        "error": error,
        "error_type": error_type,
        "timestamp": datetime.now().isoformat(),
        "step": state.get("current_step", 0)
    }
    
    if "errors" not in updated_state:
        updated_state["errors"] = []
    
    updated_state["errors"].append(error_info)
    updated_state["error_count"] = updated_state.get("error_count", 0) + 1
    updated_state["last_error"] = error
    
    return updated_state


def calculate_progress(state: Dict[str, Any], completed_nodes: List[str], total_nodes: int) -> float:
    """计算进度百分比"""
    if total_nodes == 0:
        return 0.0
    
    return (len(completed_nodes) / total_nodes) * 100.0
