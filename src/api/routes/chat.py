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

        logger.info(
            f"对话创建成功: {db_conversation.id}, 用户: {current_user.username}")

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
                title=chat_request.message[:50] + "..." if len(
                    chat_request.message) > 50 else chat_request.message,
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
            tools_used=_extract_tools_used(
                chat_result.get("agent_result", {})),
            execution_time=chat_result.get(
                "agent_result", {}).get("execution_time"),
            char_count=len(chat_result.get("response", ""))
        )
        db.add(assistant_message)

        # 更新对话统计
        conversation.message_count += 2
        conversation.updated_at = user_message.created_at

        await db.commit()
        await db.refresh(user_message)
        await db.refresh(assistant_message)

        logger.info(
            f"消息发送成功: 对话={conversation.id}, 用户={current_user.username}")

        return ChatResponse(
            success=chat_result.get("success", False),
            response=chat_result.get("response", ""),
            conversation_id=conversation.id,
            message_id=assistant_message.id,
            mode=chat_result.get("mode", "unknown"),
            execution_time=chat_result.get(
                "agent_result", {}).get("execution_time"),
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


# ==================== LangGraph相关API ====================

@router.post("/langgraph/chat", response_model=ChatResponse, summary="LangGraph智能对话")
async def langgraph_chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
    cache: CacheManager = Depends(get_cache)
):
    """使用LangGraph Agent进行智能对话"""
    try:
        agent_core = await get_agent_core()

        # 使用LangGraph Agent对话
        result = await agent_core.chat_with_langgraph(
            message=request.message,
            use_rag=request.use_rag
        )

        if not result.get("success"):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"LangGraph对话失败: {result.get('error')}"
            )

        # 保存对话记录
        conversation = await _get_or_create_conversation(
            db, current_user.id, request.conversation_id
        )

        # 保存用户消息
        await _save_message(
            db, conversation.id, "user", request.message
        )

        # 保存AI回复
        ai_message = await _save_message(
            db, conversation.id, "assistant", result["response"]
        )

        await db.commit()

        # 缓存结果
        cache_key = f"langgraph_chat:{current_user.id}:{request.message[:50]}"
        await cache.set(cache_key, result, expire=300)

        logger.info(f"LangGraph对话完成: 用户={current_user.username}")

        return ChatResponse(
            message=result["response"],
            conversation_id=conversation.id,
            message_id=ai_message.id,
            mode=result["mode"],
            execution_time=result.get("execution_time", 0.0),
            metadata={
                "langgraph_result": result.get("langgraph_result", {}),
                "reasoning_steps": result.get("langgraph_result", {}).get("reasoning_steps", []),
                "rag_sources": result.get("rag_sources", [])
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"LangGraph对话失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="LangGraph对话服务暂时不可用"
        )


@router.post("/langgraph/parse-document", summary="LangGraph文档解析")
async def langgraph_parse_document(
    file_path: str,
    current_user: User = Depends(get_current_active_user)
):
    """使用LangGraph工作流解析文档"""
    try:
        agent_core = await get_agent_core()

        # 使用LangGraph解析文档
        result = await agent_core.parse_document_with_langgraph(file_path)

        if not result.get("success"):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"LangGraph文档解析失败: {result.get('error')}"
            )

        logger.info(
            f"LangGraph文档解析完成: {file_path}, 用户: {current_user.username}")

        return {
            "success": True,
            "file_path": file_path,
            "result": result.get("result", {}),
            "processing_summary": result.get("result", {}).get("processing_summary", {}),
            "tools_used": result.get("result", {}).get("tools_used", []),
            "quality_score": result.get("result", {}).get("quality_score", 0.0)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"LangGraph文档解析失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="LangGraph文档解析服务暂时不可用"
        )


@router.post("/langgraph/batch-process", summary="LangGraph批量文档处理")
async def langgraph_batch_process(
    file_paths: List[str],
    current_user: User = Depends(get_current_active_user)
):
    """使用LangGraph批量处理文档"""
    try:
        agent_core = await get_agent_core()

        # 使用LangGraph批量处理
        result = await agent_core.batch_process_documents_with_langgraph(file_paths)

        if not result.get("success"):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"LangGraph批量处理失败: {result.get('error')}"
            )

        logger.info(
            f"LangGraph批量处理完成: {len(file_paths)} 个文件, 用户: {current_user.username}")

        return {
            "success": True,
            "total_files": result.get("total_files", 0),
            "successful": result.get("successful", 0),
            "failed": result.get("failed", 0),
            "success_rate": result.get("success_rate", 0.0),
            "execution_time": result.get("execution_time", 0.0),
            "throughput": result.get("throughput", 0.0),
            "results": result.get("results", [])
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"LangGraph批量处理失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="LangGraph批量处理服务暂时不可用"
        )


@router.get("/langgraph/workflows/{workflow_type}/visualization", summary="获取工作流可视化")
async def get_workflow_visualization(
    workflow_type: str,
    current_user: User = Depends(get_current_active_user)
):
    """获取LangGraph工作流的可视化图"""
    try:
        agent_core = await get_agent_core()

        # 验证工作流类型
        valid_types = ["document", "agent", "rag"]
        if workflow_type not in valid_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"无效的工作流类型，支持的类型: {valid_types}"
            )

        # 获取可视化
        mermaid_code = await agent_core.get_workflow_visualization(workflow_type)

        if not mermaid_code:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="无法生成工作流可视化"
            )

        return {
            "workflow_type": workflow_type,
            "mermaid_code": mermaid_code,
            "visualization_url": f"/api/v1/chat/langgraph/workflows/{workflow_type}/render"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取工作流可视化失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="工作流可视化服务暂时不可用"
        )


@router.get("/langgraph/statistics", summary="获取LangGraph统计信息")
async def get_langgraph_statistics(
    current_user: User = Depends(get_current_active_user)
):
    """获取LangGraph Agent和工作流的统计信息"""
    try:
        agent_core = await get_agent_core()

        # 获取Agent统计
        agent_stats = agent_core.get_agent_statistics()

        # 获取MCP服务器状态
        mcp_status = await agent_core.get_mcp_server_status()

        return {
            "agent_statistics": agent_stats,
            "mcp_server_status": mcp_status,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"获取LangGraph统计失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="统计信息服务暂时不可用"
        )
