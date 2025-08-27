"""
企业私有知识库系统配置文件
"""
import os
from pathlib import Path
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """系统配置类"""

    # 项目基础配置
    PROJECT_NAME: str = "企业私有知识库系统"
    PROJECT_VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False, env="DEBUG")

    # 路径配置
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    LOGS_DIR: Path = BASE_DIR / "logs"

    # API配置
    API_HOST: str = Field(default="0.0.0.0", env="API_HOST")
    API_PORT: int = Field(default=8000, env="API_PORT")
    API_PREFIX: str = "/api/v1"

    # 数据库配置
    POSTGRES_HOST: str = Field(default="localhost", env="POSTGRES_HOST")
    POSTGRES_PORT: int = Field(default=5432, env="POSTGRES_PORT")
    POSTGRES_DB: str = Field(default="epkbs", env="POSTGRES_DB")
    POSTGRES_USER: str = Field(default="epkbs_user", env="POSTGRES_USER")
    POSTGRES_PASSWORD: str = Field(
        default="epkbs_pass", env="POSTGRES_PASSWORD")

    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    # Redis配置
    REDIS_HOST: str = Field(default="localhost", env="REDIS_HOST")
    REDIS_PORT: int = Field(default=6379, env="REDIS_PORT")
    REDIS_DB: int = Field(default=0, env="REDIS_DB")
    REDIS_PASSWORD: Optional[str] = Field(default=None, env="REDIS_PASSWORD")

    @property
    def REDIS_URL(self) -> str:
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    # Milvus配置
    MILVUS_HOST: str = Field(default="localhost", env="MILVUS_HOST")
    MILVUS_PORT: int = Field(default=19530, env="MILVUS_PORT")
    MILVUS_USER: Optional[str] = Field(default=None, env="MILVUS_USER")
    MILVUS_PASSWORD: Optional[str] = Field(default=None, env="MILVUS_PASSWORD")

    # 模型配置 - 使用最新的Qwen3系列
    QWEN_MODEL_PATH: str = Field(
        default="Qwen/Qwen3-8B", env="QWEN_MODEL_PATH")
    EMBEDDING_MODEL_PATH: str = Field(
        default="Qwen/Qwen3-Embedding-8B", env="EMBEDDING_MODEL_PATH")
    RERANKER_MODEL_PATH: str = Field(
        default="Qwen/Qwen3-Reranker-8B", env="RERANKER_MODEL_PATH")

    # 模型推理配置
    MAX_TOKENS: int = Field(default=2048, env="MAX_TOKENS")
    TEMPERATURE: float = Field(default=0.7, env="TEMPERATURE")
    TOP_P: float = Field(default=0.9, env="TOP_P")

    # RAG配置
    CHUNK_SIZE: int = Field(default=512, env="CHUNK_SIZE")
    CHUNK_OVERLAP: int = Field(default=77, env="CHUNK_OVERLAP")  # 15% of 512
    TOP_K: int = Field(default=10, env="TOP_K")
    SIMILARITY_THRESHOLD: float = Field(
        default=0.7, env="SIMILARITY_THRESHOLD")

    # 分块策略配置
    ENABLE_STRUCTURAL_CHUNKING: bool = Field(
        default=True, env="ENABLE_STRUCTURAL_CHUNKING")
    ENABLE_SEMANTIC_CHUNKING: bool = Field(
        default=True, env="ENABLE_SEMANTIC_CHUNKING")
    ENABLE_RECURSIVE_CHUNKING: bool = Field(
        default=True, env="ENABLE_RECURSIVE_CHUNKING")
    MIN_CHUNK_SIZE: int = Field(default=100, env="MIN_CHUNK_SIZE")
    MAX_CHUNK_SIZE: int = Field(default=1024, env="MAX_CHUNK_SIZE")

    # 检索配置
    VECTOR_SEARCH_WEIGHT: float = Field(
        default=0.6, env="VECTOR_SEARCH_WEIGHT")
    SPARSE_SEARCH_WEIGHT: float = Field(
        default=0.4, env="SPARSE_SEARCH_WEIGHT")
    RERANK_TOP_K: int = Field(default=5, env="RERANK_TOP_K")

    # 缓存配置
    CACHE_TTL: int = Field(default=3600, env="CACHE_TTL")  # 1小时
    ENABLE_CACHE: bool = Field(default=True, env="ENABLE_CACHE")

    # 安全配置
    SECRET_KEY: str = Field(default="your-secret-key-here", env="SECRET_KEY")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    ALGORITHM: str = "HS256"

    # 文件上传配置
    MAX_FILE_SIZE: int = Field(
        default=100 * 1024 * 1024, env="MAX_FILE_SIZE")  # 100MB
    ALLOWED_EXTENSIONS: List[str] = [
        ".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx",
        ".html", ".md", ".xml", ".epub", ".txt", ".eml", ".msg",
        ".csv", ".json", ".jpg", ".jpeg", ".png", ".tiff", ".tif"
    ]

    # 文档解析配置
    ENABLE_OCR: bool = Field(default=True, env="ENABLE_OCR")
    OCR_LANGUAGE: str = Field(default="ch", env="OCR_LANGUAGE")  # PaddleOCR语言
    ENABLE_TABLE_EXTRACTION: bool = Field(
        default=True, env="ENABLE_TABLE_EXTRACTION")
    ENABLE_IMAGE_DESCRIPTION: bool = Field(
        default=True, env="ENABLE_IMAGE_DESCRIPTION")
    ENABLE_FORMULA_EXTRACTION: bool = Field(
        default=True, env="ENABLE_FORMULA_EXTRACTION")

    # 多模态模型配置
    MULTIMODAL_MODEL_PATH: str = Field(
        default="Qwen/Qwen3-8B", env="MULTIMODAL_MODEL_PATH")
    MULTIMODAL_MAX_TOKENS: int = Field(
        default=1024, env="MULTIMODAL_MAX_TOKENS")

    # 日志配置
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"

    # 监控配置
    ENABLE_METRICS: bool = Field(default=True, env="ENABLE_METRICS")
    METRICS_PORT: int = Field(default=8001, env="METRICS_PORT")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# 全局配置实例
settings = Settings()

# 确保必要的目录存在
settings.DATA_DIR.mkdir(exist_ok=True)
settings.LOGS_DIR.mkdir(exist_ok=True)
