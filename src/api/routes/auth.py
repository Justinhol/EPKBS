"""
认证相关API路由
"""
from datetime import timedelta
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.models import UserCreate, UserResponse, User
from src.api.database import get_db
from src.api.auth import auth_manager, get_current_active_user
from src.utils.logger import get_logger

logger = get_logger("api.routes.auth")

router = APIRouter(prefix="/auth", tags=["认证"])


@router.post("/register", response_model=UserResponse, summary="用户注册")
async def register(
    user_create: UserCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    用户注册
    
    - **username**: 用户名（3-50字符）
    - **email**: 邮箱地址
    - **password**: 密码（6-100字符）
    - **full_name**: 全名（可选）
    """
    try:
        user = await auth_manager.create_user(db, user_create)
        logger.info(f"用户注册成功: {user.username}")
        
        return UserResponse.from_orm(user)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"用户注册失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="注册失败"
        )


@router.post("/login", summary="用户登录")
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """
    用户登录
    
    - **username**: 用户名
    - **password**: 密码
    
    返回访问令牌
    """
    try:
        # 认证用户
        user = await auth_manager.authenticate_user(
            db, form_data.username, form_data.password
        )
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="用户名或密码错误",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # 创建访问令牌
        access_token_expires = timedelta(minutes=auth_manager.access_token_expire_minutes)
        access_token = auth_manager.create_access_token(
            data={"sub": user.username, "user_id": user.id},
            expires_delta=access_token_expires
        )
        
        logger.info(f"用户登录成功: {user.username}")
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": auth_manager.access_token_expire_minutes * 60,
            "user": UserResponse.from_orm(user)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"用户登录失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="登录失败"
        )


@router.get("/me", response_model=UserResponse, summary="获取当前用户信息")
async def get_current_user_info(
    current_user: User = Depends(get_current_active_user)
):
    """
    获取当前用户信息
    
    需要有效的访问令牌
    """
    return UserResponse.from_orm(current_user)


@router.post("/refresh", summary="刷新令牌")
async def refresh_token(
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    刷新访问令牌
    
    使用当前有效令牌获取新的令牌
    """
    try:
        # 创建新的访问令牌
        access_token_expires = timedelta(minutes=auth_manager.access_token_expire_minutes)
        access_token = auth_manager.create_access_token(
            data={"sub": current_user.username, "user_id": current_user.id},
            expires_delta=access_token_expires
        )
        
        logger.info(f"令牌刷新成功: {current_user.username}")
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": auth_manager.access_token_expire_minutes * 60
        }
        
    except Exception as e:
        logger.error(f"令牌刷新失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="令牌刷新失败"
        )


@router.post("/logout", summary="用户登出")
async def logout(
    current_user: User = Depends(get_current_active_user)
):
    """
    用户登出
    
    注意：由于JWT是无状态的，实际的令牌失效需要客户端删除令牌
    """
    logger.info(f"用户登出: {current_user.username}")
    
    return {
        "message": "登出成功",
        "detail": "请在客户端删除访问令牌"
    }


@router.get("/verify", summary="验证令牌")
async def verify_token(
    current_user: User = Depends(get_current_active_user)
):
    """
    验证访问令牌是否有效
    """
    return {
        "valid": True,
        "user_id": current_user.id,
        "username": current_user.username,
        "is_active": current_user.is_active,
        "is_superuser": current_user.is_superuser
    }
