"""
LangGraph配置文件
定义LangGraph工作流和Agent的配置参数
"""
from typing import Dict, Any, List
from pydantic import BaseModel, Field


class WorkflowConfig(BaseModel):
    """工作流配置"""
    max_iterations: int = Field(default=10, description="最大迭代次数")
    timeout: int = Field(default=300, description="超时时间（秒）")
    enable_parallel: bool = Field(default=True, description="是否启用并行执行")
    enable_retry: bool = Field(default=True, description="是否启用重试机制")
    max_retries: int = Field(default=3, description="最大重试次数")
    retry_delay: float = Field(default=1.0, description="重试延迟（秒）")


class DocumentProcessingWorkflowConfig(WorkflowConfig):
    """文档处理工作流配置"""
    enable_quality_check: bool = Field(default=True, description="是否启用质量检查")
    quality_threshold: float = Field(default=0.7, description="质量阈值")
    enable_advanced_processing: bool = Field(default=True, description="是否启用高级处理")
    
    # 解析策略配置
    default_parsing_strategy: str = Field(default="auto", description="默认解析策略")
    enable_table_extraction: bool = Field(default=True, description="是否启用表格提取")
    enable_image_analysis: bool = Field(default=True, description="是否启用图像分析")
    enable_formula_recognition: bool = Field(default=True, description="是否启用公式识别")
    
    # 工具选择策略
    table_extraction_priority: List[str] = Field(
        default=["camelot", "pdfplumber"], 
        description="表格提取工具优先级"
    )
    image_analysis_priority: List[str] = Field(
        default=["paddleocr", "tesseract"],
        description="图像分析工具优先级"
    )


class AgentReasoningWorkflowConfig(WorkflowConfig):
    """Agent推理工作流配置"""
    reasoning_temperature: float = Field(default=0.1, description="推理温度")
    max_context_length: int = Field(default=4000, description="最大上下文长度")
    enable_tool_selection: bool = Field(default=True, description="是否启用智能工具选择")
    
    # ReAct配置
    react_prompt_template: str = Field(
        default="""你是一个专业的AI助手，能够使用各种工具完成复杂任务。

请按照以下格式进行推理：
Thought: [你的思考过程]
Action: [要使用的工具名称]
Action Input: [工具参数，JSON格式]
Observation: [工具执行结果]

重复上述过程直到你有足够信息回答用户问题，然后给出：
Final Answer: [最终答案]

用户问题: {query}

开始推理:""",
        description="ReAct推理模板"
    )


class RAGWorkflowConfig(WorkflowConfig):
    """RAG工作流配置"""
    default_top_k: int = Field(default=10, description="默认检索数量")
    rerank_top_k: int = Field(default=5, description="重排序后数量")
    enable_query_expansion: bool = Field(default=True, description="是否启用查询扩展")
    enable_multi_query: bool = Field(default=True, description="是否启用多查询检索")
    
    # 检索策略配置
    default_retrieval_strategy: str = Field(default="hybrid", description="默认检索策略")
    vector_weight: float = Field(default=0.6, description="向量检索权重")
    sparse_weight: float = Field(default=0.4, description="稀疏检索权重")
    
    # 重排序配置
    default_reranker: str = Field(default="ensemble", description="默认重排序器")
    rerank_threshold: float = Field(default=0.5, description="重排序阈值")


class LangGraphConfig(BaseModel):
    """LangGraph总配置"""
    
    # 全局配置
    enable_visualization: bool = Field(default=True, description="是否启用可视化")
    enable_streaming: bool = Field(default=True, description="是否启用流式执行")
    enable_checkpointing: bool = Field(default=True, description="是否启用检查点")
    
    # 性能配置
    max_concurrent_workflows: int = Field(default=10, description="最大并发工作流数")
    workflow_pool_size: int = Field(default=20, description="工作流池大小")
    enable_caching: bool = Field(default=True, description="是否启用缓存")
    cache_ttl: int = Field(default=300, description="缓存TTL（秒）")
    
    # 监控配置
    enable_metrics: bool = Field(default=True, description="是否启用指标收集")
    enable_tracing: bool = Field(default=True, description="是否启用链路追踪")
    log_level: str = Field(default="INFO", description="日志级别")
    
    # 工作流配置
    document_processing: DocumentProcessingWorkflowConfig = Field(
        default_factory=DocumentProcessingWorkflowConfig
    )
    agent_reasoning: AgentReasoningWorkflowConfig = Field(
        default_factory=AgentReasoningWorkflowConfig
    )
    rag_workflow: RAGWorkflowConfig = Field(
        default_factory=RAGWorkflowConfig
    )


# 预定义配置模板
DEVELOPMENT_CONFIG = LangGraphConfig(
    enable_visualization=True,
    enable_streaming=True,
    log_level="DEBUG",
    document_processing=DocumentProcessingWorkflowConfig(
        max_iterations=15,
        timeout=600,
        quality_threshold=0.6
    ),
    agent_reasoning=AgentReasoningWorkflowConfig(
        max_iterations=15,
        reasoning_temperature=0.0
    )
)

PRODUCTION_CONFIG = LangGraphConfig(
    enable_visualization=False,
    enable_streaming=False,
    log_level="INFO",
    max_concurrent_workflows=50,
    workflow_pool_size=100,
    document_processing=DocumentProcessingWorkflowConfig(
        max_iterations=10,
        timeout=300,
        quality_threshold=0.8
    ),
    agent_reasoning=AgentReasoningWorkflowConfig(
        max_iterations=8,
        reasoning_temperature=0.1
    )
)

TESTING_CONFIG = LangGraphConfig(
    enable_visualization=True,
    enable_streaming=False,
    log_level="DEBUG",
    enable_caching=False,
    document_processing=DocumentProcessingWorkflowConfig(
        max_iterations=5,
        timeout=60,
        quality_threshold=0.5
    ),
    agent_reasoning=AgentReasoningWorkflowConfig(
        max_iterations=5,
        reasoning_temperature=0.0
    )
)


def get_langgraph_config(environment: str = "development") -> LangGraphConfig:
    """根据环境获取LangGraph配置"""
    if environment == "production":
        return PRODUCTION_CONFIG
    elif environment == "testing":
        return TESTING_CONFIG
    else:
        return DEVELOPMENT_CONFIG


# 工作流可视化配置
VISUALIZATION_CONFIG = {
    "mermaid": {
        "theme": "default",
        "direction": "TD",  # Top-Down
        "node_style": {
            "fill": "#f9f9f9",
            "stroke": "#333",
            "stroke-width": "2px"
        },
        "edge_style": {
            "stroke": "#333",
            "stroke-width": "2px"
        }
    },
    "output_formats": ["mermaid", "png", "svg"],
    "default_format": "mermaid"
}

# 工作流模板
WORKFLOW_TEMPLATES = {
    "simple_document": {
        "name": "简单文档处理",
        "description": "适用于简单文档的快速处理流程",
        "nodes": ["detect_format", "basic_parse", "integrate_results"],
        "max_iterations": 3
    },
    
    "complex_document": {
        "name": "复杂文档处理", 
        "description": "适用于包含表格、图像等复杂内容的文档",
        "nodes": ["detect_format", "basic_parse", "quality_check", "advanced_processing", "integrate_results"],
        "max_iterations": 8
    },
    
    "batch_processing": {
        "name": "批量文档处理",
        "description": "适用于大量文档的并行处理",
        "enable_parallel": True,
        "max_concurrent": 5,
        "max_iterations": 5
    },
    
    "interactive_agent": {
        "name": "交互式Agent",
        "description": "适用于复杂查询的多轮推理",
        "nodes": ["think", "act", "observe"],
        "max_iterations": 10,
        "enable_streaming": True
    }
}

# 默认配置
DEFAULT_LANGGRAPH_CONFIG = get_langgraph_config("development")
