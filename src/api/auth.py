"""
用户认证和权限管理
"""
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from passlib.context import CryptContext
from jose import JWTError, jwt
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.api.models import User, UserCreate, UserResponse
from src.api.database import get_db, get_cache, CacheManager
from src.utils.logger import get_logger
from config.settings import settings

logger = get_logger("api.auth")

# 密码加密上下文
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT Bearer认证
security = HTTPBearer()


class AuthManager:
    """认证管理器"""
    
    def __init__(self):
        self.secret_key = settings.SECRET_KEY
        self.algorithm = settings.ALGORITHM
        self.access_token_expire_minutes = settings.ACCESS_TOKEN_EXPIRE_MINUTES
        
        logger.info("认证管理器初始化完成")
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """验证密码"""
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """获取密码哈希"""
        return pwd_context.hash(password)
    
    def create_access_token(
        self,
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """创建访问令牌"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        
        logger.debug(f"创建访问令牌: 用户={data.get('sub')}, 过期时间={expire}")
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """验证令牌"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            username: str = payload.get("sub")
            
            if username is None:
                return None
            
            return payload
            
        except JWTError as e:
            logger.warning(f"令牌验证失败: {e}")
            return None
    
    async def authenticate_user(
        self,
        db: AsyncSession,
        username: str,
        password: str
    ) -> Optional[User]:
        """认证用户"""
        try:
            # 查询用户
            result = await db.execute(
                select(User).where(User.username == username)
            )
            user = result.scalar_one_or_none()
            
            if not user:
                logger.warning(f"用户不存在: {username}")
                return None
            
            if not user.is_active:
                logger.warning(f"用户已禁用: {username}")
                return None
            
            if not self.verify_password(password, user.hashed_password):
                logger.warning(f"密码错误: {username}")
                return None
            
            logger.info(f"用户认证成功: {username}")
            return user
            
        except Exception as e:
            logger.error(f"用户认证失败: {e}")
            return None
    
    async def create_user(
        self,
        db: AsyncSession,
        user_create: UserCreate
    ) -> User:
        """创建用户"""
        try:
            # 检查用户名是否已存在
            result = await db.execute(
                select(User).where(User.username == user_create.username)
            )
            if result.scalar_one_or_none():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="用户名已存在"
                )
            
            # 检查邮箱是否已存在
            result = await db.execute(
                select(User).where(User.email == user_create.email)
            )
            if result.scalar_one_or_none():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="邮箱已存在"
                )
            
            # 创建新用户
            hashed_password = self.get_password_hash(user_create.password)
            db_user = User(
                username=user_create.username,
                email=user_create.email,
                full_name=user_create.full_name,
                hashed_password=hashed_password
            )
            
            db.add(db_user)
            await db.commit()
            await db.refresh(db_user)
            
            logger.info(f"用户创建成功: {user_create.username}")
            return db_user
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"用户创建失败: {e}")
            await db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="用户创建失败"
            )
    
    async def get_user_by_username(
        self,
        db: AsyncSession,
        username: str
    ) -> Optional[User]:
        """根据用户名获取用户"""
        try:
            result = await db.execute(
                select(User).where(User.username == username)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"获取用户失败: {e}")
            return None
    
    async def get_user_by_id(
        self,
        db: AsyncSession,
        user_id: int
    ) -> Optional[User]:
        """根据ID获取用户"""
        try:
            result = await db.execute(
                select(User).where(User.id == user_id)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"获取用户失败: {e}")
            return None


# 全局认证管理器
auth_manager = AuthManager()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
    cache: CacheManager = Depends(get_cache)
) -> User:
    """获取当前用户"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无效的认证凭据",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # 验证令牌
        payload = auth_manager.verify_token(credentials.credentials)
        if payload is None:
            raise credentials_exception
        
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        
        # 尝试从缓存获取用户信息
        cache_key = cache.generate_key("user", username)
        cached_user = await cache.get(cache_key)
        
        if cached_user:
            # 从缓存反序列化用户信息（简化实现）
            user = await auth_manager.get_user_by_username(db, username)
        else:
            # 从数据库获取用户
            user = await auth_manager.get_user_by_username(db, username)
            
            # 缓存用户信息
            if user:
                await cache.set(cache_key, f"user:{user.id}", ttl=300)  # 5分钟缓存
        
        if user is None:
            raise credentials_exception
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="用户已禁用"
            )
        
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取当前用户失败: {e}")
        raise credentials_exception


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """获取当前活跃用户"""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="用户已禁用"
        )
    return current_user


async def get_current_superuser(
    current_user: User = Depends(get_current_user)
) -> User:
    """获取当前超级用户"""
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="权限不足"
        )
    return current_user


class RateLimiter:
    """速率限制器"""
    
    def __init__(self, cache: CacheManager):
        self.cache = cache
        logger.info("速率限制器初始化完成")
    
    async def check_rate_limit(
        self,
        key: str,
        limit: int,
        window: int = 60
    ) -> bool:
        """检查速率限制"""
        try:
            current_count = await self.cache.get(key)
            
            if current_count is None:
                # 第一次请求
                await self.cache.set(key, "1", ttl=window)
                return True
            
            count = int(current_count)
            if count >= limit:
                logger.warning(f"速率限制触发: {key}, 当前: {count}, 限制: {limit}")
                return False
            
            # 增加计数
            await self.cache.set(key, str(count + 1), ttl=window)
            return True
            
        except Exception as e:
            logger.error(f"速率限制检查失败: {e}")
            return True  # 出错时允许通过
    
    async def get_remaining_requests(
        self,
        key: str,
        limit: int
    ) -> int:
        """获取剩余请求数"""
        try:
            current_count = await self.cache.get(key)
            if current_count is None:
                return limit
            
            count = int(current_count)
            return max(0, limit - count)
            
        except Exception as e:
            logger.error(f"获取剩余请求数失败: {e}")
            return limit


async def rate_limit_dependency(
    request_key: str,
    limit: int = 100,
    window: int = 60,
    cache: CacheManager = Depends(get_cache)
):
    """速率限制依赖"""
    rate_limiter = RateLimiter(cache)
    
    if not await rate_limiter.check_rate_limit(request_key, limit, window):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="请求过于频繁，请稍后再试"
        )


# 权限检查装饰器
def require_permission(permission: str):
    """权限检查装饰器"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # 这里可以实现更复杂的权限检查逻辑
            # 目前简化为检查用户是否为超级用户
            current_user = kwargs.get('current_user')
            if current_user and not current_user.is_superuser:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"缺少权限: {permission}"
                )
            return await func(*args, **kwargs)
        return wrapper
    return decorator
