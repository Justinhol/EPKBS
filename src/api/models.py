"""
数据库模型定义
"""
from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, JSON, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from pydantic import BaseModel, Field

Base = declarative_base()


class User(Base):
    """用户模型"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100))
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow,
                        onupdate=datetime.utcnow)

    # 关系
    conversations = relationship("Conversation", back_populates="user")
    documents = relationship("Document", back_populates="user")


class Document(Base):
    """文档模型"""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=False)
    file_type = Column(String(50), nullable=False)
    content_hash = Column(String(64), unique=True, index=True)

    # 处理状态
    # pending, processing, completed, failed
    status = Column(String(20), default="pending")
    processing_error = Column(Text)

    # 元数据
    doc_metadata = Column(JSON)
    chunk_count = Column(Integer, default=0)
    vector_count = Column(Integer, default=0)

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow,
                        onupdate=datetime.utcnow)
    processed_at = Column(DateTime)

    # 外键
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # 关系
    user = relationship("User", back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document")


class DocumentChunk(Base):
    """文档块模型"""
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text, nullable=False)
    content_hash = Column(String(64), unique=True, index=True)
    chunk_index = Column(Integer, nullable=False)

    # 元数据
    chunk_metadata = Column(JSON)
    char_count = Column(Integer)
    word_count = Column(Integer)

    # 向量信息
    vector_id = Column(String(100))  # Milvus中的向量ID
    embedding_model = Column(String(100))

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow)

    # 外键
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)

    # 关系
    document = relationship("Document", back_populates="chunks")


class Conversation(Base):
    """对话模型"""
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(200))

    # 配置
    model_name = Column(String(100))
    use_rag = Column(Boolean, default=True)
    max_tokens = Column(Integer, default=1000)
    temperature = Column(Float, default=0.7)

    # 统计
    message_count = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow,
                        onupdate=datetime.utcnow)

    # 外键
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # 关系
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation")


class Message(Base):
    """消息模型"""
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    role = Column(String(20), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)

    # Agent相关
    agent_steps = Column(JSON)  # ReAct步骤
    tools_used = Column(JSON)   # 使用的工具
    execution_time = Column(Float)

    # 统计
    token_count = Column(Integer)
    char_count = Column(Integer)

    # 评价
    rating = Column(Integer)  # 1-5星评价
    feedback = Column(Text)

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow)

    # 外键
    conversation_id = Column(Integer, ForeignKey(
        "conversations.id"), nullable=False)

    # 关系
    conversation = relationship("Conversation", back_populates="messages")


class SearchLog(Base):
    """搜索日志模型"""
    __tablename__ = "search_logs"

    id = Column(Integer, primary_key=True, index=True)
    query = Column(Text, nullable=False)
    query_hash = Column(String(64), index=True)

    # 搜索配置
    top_k = Column(Integer, default=10)
    retriever_type = Column(String(50))
    reranker_type = Column(String(50))

    # 结果统计
    result_count = Column(Integer, default=0)
    execution_time = Column(Float)

    # 性能指标
    retrieval_time = Column(Float)
    rerank_time = Column(Float)

    # 结果质量
    avg_score = Column(Float)
    max_score = Column(Float)
    min_score = Column(Float)

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow)

    # 外键
    user_id = Column(Integer, ForeignKey("users.id"))

    # 关系
    user = relationship("User")


class SystemMetrics(Base):
    """系统指标模型"""
    __tablename__ = "system_metrics"

    id = Column(Integer, primary_key=True, index=True)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(20))

    # 元数据
    metric_metadata = Column(JSON)

    # 时间戳
    timestamp = Column(DateTime, default=datetime.utcnow)


# Pydantic模型（API请求/响应）

class UserBase(BaseModel):
    """用户基础模型"""
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., pattern=r'^[^@]+@[^@]+\.[^@]+$')
    full_name: Optional[str] = Field(None, max_length=100)


class UserCreate(UserBase):
    """用户创建模型"""
    password: str = Field(..., min_length=6, max_length=100)


class UserUpdate(BaseModel):
    """用户更新模型"""
    email: Optional[str] = Field(None, pattern=r'^[^@]+@[^@]+\.[^@]+$')
    full_name: Optional[str] = Field(None, max_length=100)
    is_active: Optional[bool] = None


class UserResponse(UserBase):
    """用户响应模型"""
    id: int
    is_active: bool
    is_superuser: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class DocumentBase(BaseModel):
    """文档基础模型"""
    filename: str
    file_type: str


class DocumentCreate(DocumentBase):
    """文档创建模型"""
    pass


class DocumentResponse(DocumentBase):
    """文档响应模型"""
    id: int
    original_filename: str
    file_size: int
    content_hash: str
    status: str
    chunk_count: int
    vector_count: int
    created_at: datetime
    updated_at: datetime
    processed_at: Optional[datetime]

    class Config:
        from_attributes = True


class ConversationBase(BaseModel):
    """对话基础模型"""
    title: Optional[str] = None
    model_name: Optional[str] = None
    use_rag: bool = True
    max_tokens: int = Field(default=1000, ge=1, le=4000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class ConversationCreate(ConversationBase):
    """对话创建模型"""
    pass


class ConversationResponse(ConversationBase):
    """对话响应模型"""
    id: int
    message_count: int
    total_tokens: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class MessageBase(BaseModel):
    """消息基础模型"""
    role: str = Field(..., pattern=r'^(user|assistant|system)$')
    content: str = Field(..., min_length=1)


class MessageCreate(MessageBase):
    """消息创建模型"""
    pass


class MessageResponse(MessageBase):
    """消息响应模型"""
    id: int
    agent_steps: Optional[List[Dict[str, Any]]] = None
    tools_used: Optional[List[str]] = None
    execution_time: Optional[float] = None
    token_count: Optional[int] = None
    rating: Optional[int] = None
    feedback: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class SearchRequest(BaseModel):
    """搜索请求模型"""
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=10, ge=1, le=50)
    retriever_type: str = Field(
        default="hybrid", pattern=r'^(vector|sparse|fulltext|hybrid)$')
    reranker_type: str = Field(
        default="ensemble", pattern=r'^(colbert|cross_encoder|mmr|ensemble)$')
    enable_multi_query: bool = False


class SearchResponse(BaseModel):
    """搜索响应模型"""
    query: str
    results: List[Dict[str, Any]]
    total_results: int
    execution_time: float
    retriever_used: str
    reranker_used: str


class ChatRequest(BaseModel):
    """聊天请求模型"""
    message: str = Field(..., min_length=1, max_length=2000)
    conversation_id: Optional[int] = None
    use_rag: bool = True
    stream: bool = False
    max_tokens: int = Field(default=1000, ge=1, le=4000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class ChatResponse(BaseModel):
    """聊天响应模型"""
    success: bool
    response: str
    conversation_id: int
    message_id: int
    mode: str  # direct_llm, react_agent
    execution_time: Optional[float] = None
    agent_steps: Optional[List[Dict[str, Any]]] = None
    tools_used: Optional[List[str]] = None


class SystemStatus(BaseModel):
    """系统状态模型"""
    status: str
    version: str
    uptime: float
    total_users: int
    total_documents: int
    total_conversations: int
    total_messages: int
    rag_enabled: bool
    available_models: List[str]
    system_metrics: Dict[str, Any]
