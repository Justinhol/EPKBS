"""
搜索相关API路由
"""
import hashlib
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.api.models import User, SearchLog, SearchRequest, SearchResponse
from src.api.database import get_db, get_cache, CacheManager
from src.api.auth import get_current_active_user
from src.rag.core import RAGCore
from src.utils.logger import get_logger
from src.utils.helpers import Timer

logger = get_logger("api.routes.search")

router = APIRouter(prefix="/search", tags=["搜索"])

# 全局RAG实例
_rag_core: Optional[RAGCore] = None


async def get_rag_core() -> RAGCore:
    """获取RAG核心实例"""
    global _rag_core
    
    if _rag_core is None:
        _rag_core = RAGCore()
        await _rag_core.initialize()
        logger.info("RAG核心实例初始化完成")
    
    return _rag_core


@router.post("/", response_model=SearchResponse, summary="知识库搜索")
async def search_knowledge_base(
    search_request: SearchRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
    cache: CacheManager = Depends(get_cache)
):
    """
    在知识库中搜索相关信息
    
    - **query**: 搜索查询文本
    - **top_k**: 返回结果数量（1-50）
    - **retriever_type**: 检索器类型（vector/sparse/fulltext/hybrid）
    - **reranker_type**: 重排序器类型（colbert/cross_encoder/mmr/ensemble）
    - **enable_multi_query**: 是否启用多查询检索
    """
    try:
        # 生成查询哈希用于缓存和日志
        query_hash = hashlib.md5(
            f"{search_request.query}:{search_request.top_k}:{search_request.retriever_type}:{search_request.reranker_type}".encode()
        ).hexdigest()
        
        # 尝试从缓存获取结果
        cache_key = cache.generate_key("search", query_hash)
        cached_result = await cache.get(cache_key)
        
        if cached_result:
            logger.info(f"搜索缓存命中: {search_request.query[:50]}...")
            import json
            return SearchResponse(**json.loads(cached_result))
        
        # 获取RAG核心
        rag_core = await get_rag_core()
        
        # 执行搜索
        with Timer() as timer:
            search_results = await rag_core.search(
                query=search_request.query,
                top_k=search_request.top_k,
                retriever_type=search_request.retriever_type,
                reranker_type=search_request.reranker_type,
                enable_multi_query=search_request.enable_multi_query
            )
        
        # 构建响应
        response = SearchResponse(
            query=search_request.query,
            results=search_results,
            total_results=len(search_results),
            execution_time=timer.elapsed,
            retriever_used=search_request.retriever_type,
            reranker_used=search_request.reranker_type
        )
        
        # 缓存结果
        await cache.set(
            cache_key,
            response.json(ensure_ascii=False),
            ttl=300  # 5分钟缓存
        )
        
        # 记录搜索日志
        await _log_search(
            db=db,
            user_id=current_user.id,
            search_request=search_request,
            query_hash=query_hash,
            results=search_results,
            execution_time=timer.elapsed
        )
        
        logger.info(f"搜索完成: 查询='{search_request.query[:50]}...', 结果数={len(search_results)}, 耗时={timer.elapsed:.3f}秒")
        
        return response
        
    except Exception as e:
        logger.error(f"搜索失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="搜索失败"
        )


@router.get("/history", summary="获取搜索历史")
async def get_search_history(
    skip: int = 0,
    limit: int = 20,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    获取用户的搜索历史
    
    - **skip**: 跳过的记录数
    - **limit**: 返回的记录数（最大100）
    """
    try:
        limit = min(limit, 100)
        
        result = await db.execute(
            select(SearchLog)
            .where(SearchLog.user_id == current_user.id)
            .order_by(SearchLog.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        
        search_logs = result.scalars().all()
        
        history = []
        for log in search_logs:
            history.append({
                "id": log.id,
                "query": log.query,
                "top_k": log.top_k,
                "retriever_type": log.retriever_type,
                "reranker_type": log.reranker_type,
                "result_count": log.result_count,
                "execution_time": log.execution_time,
                "avg_score": log.avg_score,
                "created_at": log.created_at.isoformat()
            })
        
        return {
            "history": history,
            "total": len(history)
        }
        
    except Exception as e:
        logger.error(f"获取搜索历史失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取搜索历史失败"
        )


@router.get("/suggestions", summary="获取搜索建议")
async def get_search_suggestions(
    query: str,
    limit: int = 5,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    根据输入获取搜索建议
    
    - **query**: 部分查询文本
    - **limit**: 返回建议数量（最大20）
    """
    try:
        limit = min(limit, 20)
        
        # 从搜索历史中查找相似查询
        result = await db.execute(
            select(SearchLog.query)
            .where(
                SearchLog.user_id == current_user.id,
                SearchLog.query.ilike(f"%{query}%")
            )
            .distinct()
            .order_by(SearchLog.created_at.desc())
            .limit(limit)
        )
        
        suggestions = [row[0] for row in result.fetchall()]
        
        # 如果历史记录不足，可以添加一些通用建议
        if len(suggestions) < limit:
            common_suggestions = [
                f"{query} 是什么",
                f"{query} 的定义",
                f"{query} 的应用",
                f"{query} 的优缺点",
                f"如何使用 {query}"
            ]
            
            for suggestion in common_suggestions:
                if len(suggestions) >= limit:
                    break
                if suggestion not in suggestions:
                    suggestions.append(suggestion)
        
        return {
            "query": query,
            "suggestions": suggestions[:limit]
        }
        
    except Exception as e:
        logger.error(f"获取搜索建议失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取搜索建议失败"
        )


@router.get("/stats", summary="获取搜索统计")
async def get_search_stats(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """获取用户的搜索统计信息"""
    try:
        # 获取搜索总数
        total_result = await db.execute(
            select(SearchLog.id)
            .where(SearchLog.user_id == current_user.id)
        )
        total_searches = len(total_result.fetchall())
        
        # 获取平均执行时间
        avg_time_result = await db.execute(
            select(SearchLog.execution_time)
            .where(SearchLog.user_id == current_user.id)
        )
        execution_times = [row[0] for row in avg_time_result.fetchall() if row[0]]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
        
        # 获取最常用的检索器类型
        retriever_result = await db.execute(
            select(SearchLog.retriever_type)
            .where(SearchLog.user_id == current_user.id)
        )
        retriever_types = [row[0] for row in retriever_result.fetchall()]
        retriever_counts = {}
        for rt in retriever_types:
            retriever_counts[rt] = retriever_counts.get(rt, 0) + 1
        
        most_used_retriever = max(retriever_counts.items(), key=lambda x: x[1])[0] if retriever_counts else None
        
        # 获取最常用的重排序器类型
        reranker_result = await db.execute(
            select(SearchLog.reranker_type)
            .where(SearchLog.user_id == current_user.id)
        )
        reranker_types = [row[0] for row in reranker_result.fetchall()]
        reranker_counts = {}
        for rt in reranker_types:
            reranker_counts[rt] = reranker_counts.get(rt, 0) + 1
        
        most_used_reranker = max(reranker_counts.items(), key=lambda x: x[1])[0] if reranker_counts else None
        
        return {
            "total_searches": total_searches,
            "avg_execution_time": round(avg_execution_time, 3),
            "most_used_retriever": most_used_retriever,
            "most_used_reranker": most_used_reranker,
            "retriever_usage": retriever_counts,
            "reranker_usage": reranker_counts
        }
        
    except Exception as e:
        logger.error(f"获取搜索统计失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取搜索统计失败"
        )


async def _log_search(
    db: AsyncSession,
    user_id: int,
    search_request: SearchRequest,
    query_hash: str,
    results: List[Dict[str, Any]],
    execution_time: float
):
    """记录搜索日志"""
    try:
        # 计算结果统计
        scores = [result.get("score", 0) for result in results if result.get("score")]
        avg_score = sum(scores) / len(scores) if scores else 0
        max_score = max(scores) if scores else 0
        min_score = min(scores) if scores else 0
        
        # 创建搜索日志
        search_log = SearchLog(
            query=search_request.query,
            query_hash=query_hash,
            top_k=search_request.top_k,
            retriever_type=search_request.retriever_type,
            reranker_type=search_request.reranker_type,
            result_count=len(results),
            execution_time=execution_time,
            avg_score=avg_score,
            max_score=max_score,
            min_score=min_score,
            user_id=user_id
        )
        
        db.add(search_log)
        await db.commit()
        
    except Exception as e:
        logger.warning(f"记录搜索日志失败: {e}")
        # 不影响主要功能，只记录警告
