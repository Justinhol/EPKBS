"""
企业私有知识库系统统一配置文件
整合所有配置项，实现分层配置管理
"""
import os
from pathlib import Path
from typing import List, Optional
from pydantic import Field, BaseModel
from pydantic_settings import BaseSettings


class DatabaseConfig(BaseModel):
    """数据库配置"""
    postgres_host: str = Field(default="localhost", env="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, env="POSTGRES_PORT")
    postgres_db: str = Field(default="epkbs", env="POSTGRES_DB")
    postgres_user: str = Field(default="epkbs_user", env="POSTGRES_USER")
    postgres_password: str = Field(
        default="epkbs_pass", env="POSTGRES_PASSWORD")

    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_db: int = Field(default=0, env="REDIS_DB")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")

    milvus_host: str = Field(default="localhost", env="MILVUS_HOST")
    milvus_port: int = Field(default=19530, env="MILVUS_PORT")
    milvus_user: Optional[str] = Field(default=None, env="MILVUS_USER")
    milvus_password: Optional[str] = Field(default=None, env="MILVUS_PASSWORD")

    @property
    def database_url(self) -> str:
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    @property
    def redis_url(self) -> str:
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"


class ModelConfig(BaseModel):
    """模型配置"""
    qwen_model_path: str = Field(
        default="Qwen/Qwen3-8B", env="QWEN_MODEL_PATH")
    embedding_model_path: str = Field(
        default="Qwen/Qwen3-Embedding-4B", env="EMBEDDING_MODEL_PATH")
    reranker_model_path: str = Field(
        default="Qwen/Qwen3-Reranker-0.6B", env="RERANKER_MODEL_PATH")
    multimodal_model_path: str = Field(
        default="Qwen/Qwen3-8B", env="MULTIMODAL_MODEL_PATH")

    # 推理参数
    max_tokens: int = Field(default=2048, env="MAX_TOKENS")
    temperature: float = Field(default=0.7, env="TEMPERATURE")
    top_p: float = Field(default=0.9, env="TOP_P")
    multimodal_max_tokens: int = Field(
        default=1024, env="MULTIMODAL_MAX_TOKENS")

    # 设备配置
    embedding_device: str = Field(default="auto", env="EMBEDDING_DEVICE")
    reranker_device: str = Field(default="auto", env="RERANKER_DEVICE")
    embedding_batch_size: int = Field(default=32, env="EMBEDDING_BATCH_SIZE")
    reranker_batch_size: int = Field(default=16, env="RERANKER_BATCH_SIZE")


class RAGConfig(BaseModel):
    """RAG配置"""
    # 分块配置
    chunk_size: int = Field(default=512, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=77, env="CHUNK_OVERLAP")
    min_chunk_size: int = Field(default=100, env="MIN_CHUNK_SIZE")
    max_chunk_size: int = Field(default=1024, env="MAX_CHUNK_SIZE")

    # 分块策略
    enable_structural_chunking: bool = Field(
        default=True, env="ENABLE_STRUCTURAL_CHUNKING")
    enable_semantic_chunking: bool = Field(
        default=True, env="ENABLE_SEMANTIC_CHUNKING")
    enable_recursive_chunking: bool = Field(
        default=True, env="ENABLE_RECURSIVE_CHUNKING")

    # 检索配置
    top_k: int = Field(default=10, env="TOP_K")
    similarity_threshold: float = Field(
        default=0.7, env="SIMILARITY_THRESHOLD")
    vector_search_weight: float = Field(
        default=0.6, env="VECTOR_SEARCH_WEIGHT")
    sparse_search_weight: float = Field(
        default=0.4, env="SPARSE_SEARCH_WEIGHT")
    rerank_top_k: int = Field(default=5, env="RERANK_TOP_K")


class LangGraphConfig(BaseModel):
    """LangGraph工作流配置"""
    # 全局配置
    enable_visualization: bool = Field(
        default=True, env="LANGGRAPH_ENABLE_VISUALIZATION")
    enable_streaming: bool = Field(
        default=True, env="LANGGRAPH_ENABLE_STREAMING")
    enable_checkpointing: bool = Field(
        default=True, env="LANGGRAPH_ENABLE_CHECKPOINTING")

    # 性能配置
    max_concurrent_workflows: int = Field(
        default=10, env="LANGGRAPH_MAX_CONCURRENT_WORKFLOWS")
    workflow_pool_size: int = Field(
        default=20, env="LANGGRAPH_WORKFLOW_POOL_SIZE")

    # 工作流配置
    max_iterations: int = Field(default=10, env="LANGGRAPH_MAX_ITERATIONS")
    timeout: int = Field(default=300, env="LANGGRAPH_TIMEOUT")
    enable_parallel: bool = Field(
        default=True, env="LANGGRAPH_ENABLE_PARALLEL")
    enable_retry: bool = Field(default=True, env="LANGGRAPH_ENABLE_RETRY")
    max_retries: int = Field(default=3, env="LANGGRAPH_MAX_RETRIES")
    retry_delay: float = Field(default=1.0, env="LANGGRAPH_RETRY_DELAY")

    # Agent推理配置
    reasoning_temperature: float = Field(
        default=0.1, env="LANGGRAPH_REASONING_TEMPERATURE")
    max_context_length: int = Field(
        default=4000, env="LANGGRAPH_MAX_CONTEXT_LENGTH")
    enable_tool_selection: bool = Field(
        default=True, env="LANGGRAPH_ENABLE_TOOL_SELECTION")


class MCPConfig(BaseModel):
    """MCP配置"""
    # 客户端配置
    client_timeout: int = Field(default=30, env="MCP_CLIENT_TIMEOUT")
    retry_attempts: int = Field(default=3, env="MCP_RETRY_ATTEMPTS")
    retry_delay: float = Field(default=1.0, env="MCP_RETRY_DELAY")
    batch_size: int = Field(default=10, env="MCP_BATCH_SIZE")
    max_concurrent_calls: int = Field(
        default=5, env="MCP_MAX_CONCURRENT_CALLS")
    connection_pool_size: int = Field(
        default=20, env="MCP_CONNECTION_POOL_SIZE")

    # Agent配置
    agent_max_iterations: int = Field(
        default=10, env="MCP_AGENT_MAX_ITERATIONS")
    agent_max_execution_time: int = Field(
        default=300, env="MCP_AGENT_MAX_EXECUTION_TIME")
    enable_tool_discovery: bool = Field(
        default=True, env="MCP_ENABLE_TOOL_DISCOVERY")
    enable_batch_calls: bool = Field(
        default=True, env="MCP_ENABLE_BATCH_CALLS")
    enable_parallel_execution: bool = Field(
        default=True, env="MCP_ENABLE_PARALLEL_EXECUTION")
    tool_selection_strategy: str = Field(
        default="intelligent", env="MCP_TOOL_SELECTION_STRATEGY")

    # 工具调用配置
    tool_default_timeout: int = Field(
        default=30, env="MCP_TOOL_DEFAULT_TIMEOUT")
    tool_max_retries: int = Field(default=3, env="MCP_TOOL_MAX_RETRIES")
    tool_retry_backoff: float = Field(
        default=2.0, env="MCP_TOOL_RETRY_BACKOFF")


class CacheConfig(BaseModel):
    """缓存配置 - 优化多层缓存策略"""
    enable_cache: bool = Field(default=True, env="ENABLE_CACHE")
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")

    # 分层缓存配置 - 增强配置
    enable_memory_cache: bool = Field(default=True, env="ENABLE_MEMORY_CACHE")
    enable_redis_cache: bool = Field(default=True, env="ENABLE_REDIS_CACHE")
    memory_cache_size: int = Field(
        default=2000, env="MEMORY_CACHE_SIZE")  # 提升容量 1000->2000

    # L1缓存 (内存缓存) 配置
    l1_cache_size: int = Field(default=2000, env="L1_CACHE_SIZE")
    l1_cache_ttl: int = Field(default=1800, env="L1_CACHE_TTL")  # 30分钟

    # L2缓存 (Redis缓存) 配置
    l2_cache_ttl: int = Field(default=7200, env="L2_CACHE_TTL")  # 2小时
    l2_cache_max_memory: str = Field(
        default="512mb", env="L2_CACHE_MAX_MEMORY")

    # 特定缓存配置 - 优化TTL
    embedding_cache_ttl: int = Field(
        default=14400, env="EMBEDDING_CACHE_TTL")  # 4小时 (2->4小时)
    model_cache_ttl: int = Field(default=86400, env="MODEL_CACHE_TTL")  # 24小时
    rag_cache_ttl: int = Field(
        default=3600, env="RAG_CACHE_TTL")  # 1小时 (30分钟->1小时)

    # 查询结果缓存
    query_cache_ttl: int = Field(default=1800, env="QUERY_CACHE_TTL")  # 30分钟

    # 缓存预热配置
    enable_cache_warmup: bool = Field(default=True, env="ENABLE_CACHE_WARMUP")
    warmup_batch_size: int = Field(default=50, env="WARMUP_BATCH_SIZE")


class Settings(BaseSettings):
    """统一配置管理类"""

    # 项目基础配置
    project_name: str = Field(default="企业私有知识库系统", env="PROJECT_NAME")
    project_version: str = Field(default="2.0.0", env="PROJECT_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="development", env="ENVIRONMENT")

    # 路径配置
    base_dir: Path = Path(__file__).parent.parent
    data_dir: Path = base_dir / "data"
    logs_dir: Path = base_dir / "logs"

    # API配置
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_prefix: str = "/api/v1"

    # 安全配置
    secret_key: str = Field(default="your-secret-key-here", env="SECRET_KEY")
    access_token_expire_minutes: int = Field(
        default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    algorithm: str = "HS256"

    # 文件上传配置
    max_file_size: int = Field(
        default=100 * 1024 * 1024, env="MAX_FILE_SIZE")  # 100MB
    allowed_extensions: List[str] = [
        ".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx",
        ".html", ".md", ".xml", ".epub", ".txt", ".eml", ".msg",
        ".csv", ".json", ".jpg", ".jpeg", ".png", ".tiff", ".tif"
    ]

    # 文档解析配置
    enable_ocr: bool = Field(default=True, env="ENABLE_OCR")
    ocr_language: str = Field(default="ch", env="OCR_LANGUAGE")
    enable_table_extraction: bool = Field(
        default=True, env="ENABLE_TABLE_EXTRACTION")
    enable_image_description: bool = Field(
        default=True, env="ENABLE_IMAGE_DESCRIPTION")
    enable_formula_extraction: bool = Field(
        default=True, env="ENABLE_FORMULA_EXTRACTION")

    # 日志配置
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"

    # 监控配置
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=8001, env="METRICS_PORT")

    # 子配置
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    langgraph: LangGraphConfig = Field(default_factory=LangGraphConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)

    # 兼容性属性（保持向后兼容）
    @property
    def PROJECT_NAME(self) -> str:
        return self.project_name

    @property
    def PROJECT_VERSION(self) -> str:
        return self.project_version

    @property
    def DEBUG(self) -> bool:
        return self.debug

    @property
    def BASE_DIR(self) -> Path:
        return self.base_dir

    @property
    def DATA_DIR(self) -> Path:
        return self.data_dir

    @property
    def LOGS_DIR(self) -> Path:
        return self.logs_dir

    @property
    def API_HOST(self) -> str:
        return self.api_host

    @property
    def API_PORT(self) -> int:
        return self.api_port

    @property
    def API_PREFIX(self) -> str:
        return self.api_prefix

    @property
    def DATABASE_URL(self) -> str:
        return self.database.database_url

    @property
    def REDIS_URL(self) -> str:
        return self.database.redis_url

    @property
    def POSTGRES_HOST(self) -> str:
        return self.database.postgres_host

    @property
    def POSTGRES_PORT(self) -> int:
        return self.database.postgres_port

    @property
    def POSTGRES_DB(self) -> str:
        return self.database.postgres_db

    @property
    def POSTGRES_USER(self) -> str:
        return self.database.postgres_user

    @property
    def POSTGRES_PASSWORD(self) -> str:
        return self.database.postgres_password

    @property
    def REDIS_HOST(self) -> str:
        return self.database.redis_host

    @property
    def REDIS_PORT(self) -> int:
        return self.database.redis_port

    @property
    def REDIS_DB(self) -> int:
        return self.database.redis_db

    @property
    def REDIS_PASSWORD(self) -> Optional[str]:
        return self.database.redis_password

    @property
    def MILVUS_HOST(self) -> str:
        return self.database.milvus_host

    @property
    def MILVUS_PORT(self) -> int:
        return self.database.milvus_port

    @property
    def MILVUS_USER(self) -> Optional[str]:
        return self.database.milvus_user

    @property
    def MILVUS_PASSWORD(self) -> Optional[str]:
        return self.database.milvus_password

    @property
    def QWEN_MODEL_PATH(self) -> str:
        return self.model.qwen_model_path

    @property
    def EMBEDDING_MODEL_PATH(self) -> str:
        return self.model.embedding_model_path

    @property
    def RERANKER_MODEL_PATH(self) -> str:
        return self.model.reranker_model_path

    @property
    def MAX_TOKENS(self) -> int:
        return self.model.max_tokens

    @property
    def TEMPERATURE(self) -> float:
        return self.model.temperature

    @property
    def TOP_P(self) -> float:
        return self.model.top_p

    @property
    def CHUNK_SIZE(self) -> int:
        return self.rag.chunk_size

    @property
    def CHUNK_OVERLAP(self) -> int:
        return self.rag.chunk_overlap

    @property
    def TOP_K(self) -> int:
        return self.rag.top_k

    @property
    def SIMILARITY_THRESHOLD(self) -> float:
        return self.rag.similarity_threshold

    @property
    def VECTOR_SEARCH_WEIGHT(self) -> float:
        return self.rag.vector_search_weight

    @property
    def SPARSE_SEARCH_WEIGHT(self) -> float:
        return self.rag.sparse_search_weight

    @property
    def RERANK_TOP_K(self) -> int:
        return self.rag.rerank_top_k

    @property
    def CACHE_TTL(self) -> int:
        return self.cache.cache_ttl

    @property
    def ENABLE_CACHE(self) -> bool:
        return self.cache.enable_cache

    @property
    def SECRET_KEY(self) -> str:
        return self.secret_key

    @property
    def ACCESS_TOKEN_EXPIRE_MINUTES(self) -> int:
        return self.access_token_expire_minutes

    @property
    def ALGORITHM(self) -> str:
        return self.algorithm

    @property
    def MAX_FILE_SIZE(self) -> int:
        return self.max_file_size

    @property
    def ALLOWED_EXTENSIONS(self) -> List[str]:
        return self.allowed_extensions

    @property
    def ENABLE_OCR(self) -> bool:
        return self.enable_ocr

    @property
    def OCR_LANGUAGE(self) -> str:
        return self.ocr_language

    @property
    def ENABLE_TABLE_EXTRACTION(self) -> bool:
        return self.enable_table_extraction

    @property
    def ENABLE_IMAGE_DESCRIPTION(self) -> bool:
        return self.enable_image_description

    @property
    def ENABLE_FORMULA_EXTRACTION(self) -> bool:
        return self.enable_formula_extraction

    @property
    def LOG_LEVEL(self) -> str:
        return self.log_level

    @property
    def LOG_FORMAT(self) -> str:
        return self.log_format

    @property
    def ENABLE_METRICS(self) -> bool:
        return self.enable_metrics

    @property
    def METRICS_PORT(self) -> int:
        return self.metrics_port

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def get_settings(environment: str = None) -> Settings:
    """获取配置实例，支持环境特定配置"""
    if environment is None:
        environment = os.getenv("ENVIRONMENT", "development")

    # 根据环境调整配置
    settings_instance = Settings()

    if environment == "production":
        # 生产环境特定配置
        settings_instance.debug = False
        settings_instance.log_level = "WARNING"
        settings_instance.langgraph.enable_visualization = False
        settings_instance.langgraph.max_concurrent_workflows = 50
        settings_instance.cache.cache_ttl = 7200  # 2小时
    elif environment == "testing":
        # 测试环境特定配置
        settings_instance.debug = True
        settings_instance.log_level = "DEBUG"
        settings_instance.cache.enable_cache = False
        settings_instance.langgraph.max_iterations = 5
        settings_instance.langgraph.timeout = 60
    else:
        # 开发环境配置（默认）
        settings_instance.debug = True
        settings_instance.log_level = "DEBUG"
        settings_instance.langgraph.enable_visualization = True

    return settings_instance


# 全局配置实例
settings = get_settings()

# 确保必要的目录存在
settings.data_dir.mkdir(exist_ok=True)
settings.logs_dir.mkdir(exist_ok=True)
