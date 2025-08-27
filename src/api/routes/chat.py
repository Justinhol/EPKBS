"""
聊天相关API路由
"""
import json
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from src.api.models import (
    User, Conversation, Message,
    ChatRequest, ChatResponse,
    ConversationCreate, ConversationResponse,
    MessageResponse
)
from src.api.database import get_db, get_cache, CacheManager
from src.api.auth import get_current_active_user
from src.agent.core import AgentCore
from src.rag.core import RAGCore
from src.utils.logger import get_logger

logger = get_logger("api.routes.chat")

router = APIRouter(prefix="/chat", tags=["聊天"])

# 全局Agent实例（在实际应用中应该使用依赖注入）
_agent_core: Optional[AgentCore] = None
_rag_core: Optional[RAGCore] = None


async def get_agent_core() -> AgentCore:
    """获取Agent核心实例"""
    global _agent_core, _rag_core
    
    if _agent_core is None:
        # 初始化RAG核心
        if _rag_core is None:
            _rag_core = RAGCore()
            await _rag_core.initialize()
        
        # 初始化Agent核心
        _agent_core = AgentCore(rag_core=_rag_core)
        await _agent_core.initialize()
        
        logger.info("Agent核心实例初始化完成")
    
    return _agent_core


@router.post("/conversations", response_model=ConversationResponse, summary="创建对话")
async def create_conversation(
    conversation_create: ConversationCreate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    创建新的对话会话
    
    - **title**: 对话标题（可选）
    - **model_name**: 使用的模型名称（可选）
    - **use_rag**: 是否使用RAG（默认true）
    - **max_tokens**: 最大令牌数（1-4000）
    - **temperature**: 温度参数（0.0-2.0）
    """
    try:
        # 创建对话记录
        db_conversation = Conversation(
            title=conversation_create.title or "新对话",
            model_name=conversation_create.model_name,
            use_rag=conversation_create.use_rag,
            max_tokens=conversation_create.max_tokens,
            temperature=conversation_create.temperature,
            user_id=current_user.id
        )
        
        db.add(db_conversation)
        await db.commit()
        await db.refresh(db_conversation)
        
        logger.info(f"对话创建成功: {db_conversation.id}, 用户: {current_user.username}")
        
        return ConversationResponse.from_orm(db_conversation)
        
    except Exception as e:
        logger.error(f"对话创建失败: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="对话创建失败"
        )


@router.get("/conversations", response_model=List[ConversationResponse], summary="获取对话列表")
async def get_conversations(
    skip: int = 0,
    limit: int = 20,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    获取用户的对话列表
    
    - **skip**: 跳过的记录数
    - **limit**: 返回的记录数（最大100）
    """
    try:
        limit = min(limit, 100)  # 限制最大返回数量
        
        result = await db.execute(
            select(Conversation)
            .where(Conversation.user_id == current_user.id)
            .order_by(desc(Conversation.updated_at))
            .offset(skip)
            .limit(limit)
        )
        
        conversations = result.scalars().all()
        
        return [ConversationResponse.from_orm(conv) for conv in conversations]
        
    except Exception as e:
        logger.error(f"获取对话列表失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取对话列表失败"
        )


@router.get("/conversations/{conversation_id}", response_model=ConversationResponse, summary="获取对话详情")
async def get_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """获取指定对话的详情"""
    try:
        result = await db.execute(
            select(Conversation)
            .where(
                Conversation.id == conversation_id,
                Conversation.user_id == current_user.id
            )
        )
        
        conversation = result.scalar_one_or_none()
        
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="对话不存在"
            )
        
        return ConversationResponse.from_orm(conversation)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取对话详情失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取对话详情失败"
        )


@router.get("/conversations/{conversation_id}/messages", response_model=List[MessageResponse], summary="获取对话消息")
async def get_conversation_messages(
    conversation_id: int,
    skip: int = 0,
    limit: int = 50,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """获取指定对话的消息列表"""
    try:
        # 验证对话所有权
        conv_result = await db.execute(
            select(Conversation)
            .where(
                Conversation.id == conversation_id,
                Conversation.user_id == current_user.id
            )
        )
        
        if not conv_result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="对话不存在"
            )
        
        # 获取消息
        limit = min(limit, 100)
        
        result = await db.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at)
            .offset(skip)
            .limit(limit)
        )
        
        messages = result.scalars().all()
        
        return [MessageResponse.from_orm(msg) for msg in messages]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取对话消息失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取对话消息失败"
        )


@router.post("/send", response_model=ChatResponse, summary="发送消息")
async def send_message(
    chat_request: ChatRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
    cache: CacheManager = Depends(get_cache)
):
    """
    发送聊天消息
    
    - **message**: 用户消息内容
    - **conversation_id**: 对话ID（可选，不提供则创建新对话）
    - **use_rag**: 是否使用RAG
    - **stream**: 是否流式响应
    - **max_tokens**: 最大令牌数
    - **temperature**: 温度参数
    """
    try:
        # 获取或创建对话
        conversation = None
        if chat_request.conversation_id:
            result = await db.execute(
                select(Conversation)
                .where(
                    Conversation.id == chat_request.conversation_id,
                    Conversation.user_id == current_user.id
                )
            )
            conversation = result.scalar_one_or_none()
            
            if not conversation:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="对话不存在"
                )
        else:
            # 创建新对话
            conversation = Conversation(
                title=chat_request.message[:50] + "..." if len(chat_request.message) > 50 else chat_request.message,
                use_rag=chat_request.use_rag,
                max_tokens=chat_request.max_tokens,
                temperature=chat_request.temperature,
                user_id=current_user.id
            )
            db.add(conversation)
            await db.commit()
            await db.refresh(conversation)
        
        # 保存用户消息
        user_message = Message(
            role="user",
            content=chat_request.message,
            conversation_id=conversation.id,
            char_count=len(chat_request.message)
        )
        db.add(user_message)
        
        # 获取Agent核心
        agent_core = await get_agent_core()
        
        # 处理流式响应
        if chat_request.stream:
            return StreamingResponse(
                _stream_chat_response(
                    agent_core, chat_request, conversation, user_message, db
                ),
                media_type="text/plain"
            )
        
        # 普通响应
        chat_result = await agent_core.chat(
            message=chat_request.message,
            use_rag=chat_request.use_rag,
            stream=False,
            max_tokens=chat_request.max_tokens,
            temperature=chat_request.temperature
        )
        
        # 保存助手消息
        assistant_message = Message(
            role="assistant",
            content=chat_result.get("response", ""),
            conversation_id=conversation.id,
            agent_steps=chat_result.get("agent_result", {}).get("steps"),
            tools_used=_extract_tools_used(chat_result.get("agent_result", {})),
            execution_time=chat_result.get("agent_result", {}).get("execution_time"),
            char_count=len(chat_result.get("response", ""))
        )
        db.add(assistant_message)
        
        # 更新对话统计
        conversation.message_count += 2
        conversation.updated_at = user_message.created_at
        
        await db.commit()
        await db.refresh(user_message)
        await db.refresh(assistant_message)
        
        logger.info(f"消息发送成功: 对话={conversation.id}, 用户={current_user.username}")
        
        return ChatResponse(
            success=chat_result.get("success", False),
            response=chat_result.get("response", ""),
            conversation_id=conversation.id,
            message_id=assistant_message.id,
            mode=chat_result.get("mode", "unknown"),
            execution_time=chat_result.get("agent_result", {}).get("execution_time"),
            agent_steps=chat_result.get("agent_result", {}).get("steps"),
            tools_used=_extract_tools_used(chat_result.get("agent_result", {}))
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"发送消息失败: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="发送消息失败"
        )


async def _stream_chat_response(
    agent_core: AgentCore,
    chat_request: ChatRequest,
    conversation: Conversation,
    user_message: Message,
    db: AsyncSession
):
    """流式聊天响应生成器"""
    try:
        full_response = ""
        agent_steps = []
        tools_used = []
        
        async for event in agent_core.chat(
            message=chat_request.message,
            use_rag=chat_request.use_rag,
            stream=True,
            max_tokens=chat_request.max_tokens,
            temperature=chat_request.temperature
        ):
            # 发送事件到客户端
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            
            # 收集响应数据
            if event.get("type") == "chunk":
                full_response += event.get("content", "")
            elif event.get("type") == "final_answer":
                full_response = event.get("answer", "")
            elif event.get("type") == "action":
                tools_used.append(event.get("action", ""))
        
        # 保存助手消息
        assistant_message = Message(
            role="assistant",
            content=full_response,
            conversation_id=conversation.id,
            agent_steps=agent_steps,
            tools_used=tools_used,
            char_count=len(full_response)
        )
        db.add(assistant_message)
        
        # 更新对话统计
        conversation.message_count += 2
        
        await db.commit()
        
        # 发送完成事件
        yield f"data: {json.dumps({'type': 'done', 'message_id': assistant_message.id}, ensure_ascii=False)}\n\n"
        
    except Exception as e:
        logger.error(f"流式响应失败: {e}")
        yield f"data: {json.dumps({'type': 'error', 'error': str(e)}, ensure_ascii=False)}\n\n"


def _extract_tools_used(agent_result: Dict[str, Any]) -> List[str]:
    """从Agent结果中提取使用的工具"""
    tools = set()
    
    steps = agent_result.get("steps", [])
    for step in steps:
        action = step.get("action")
        if action and action != "final_answer":
            tools.add(action)
    
    return list(tools)


@router.delete("/conversations/{conversation_id}", summary="删除对话")
async def delete_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """删除指定对话及其所有消息"""
    try:
        # 验证对话所有权
        result = await db.execute(
            select(Conversation)
            .where(
                Conversation.id == conversation_id,
                Conversation.user_id == current_user.id
            )
        )
        
        conversation = result.scalar_one_or_none()
        
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="对话不存在"
            )
        
        # 删除对话（级联删除消息）
        await db.delete(conversation)
        await db.commit()
        
        logger.info(f"对话删除成功: {conversation_id}, 用户: {current_user.username}")
        
        return {"message": "对话删除成功"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除对话失败: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="删除对话失败"
        )
