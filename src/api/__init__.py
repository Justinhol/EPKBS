"""
后端API层
提供RESTful API接口、用户认证、数据库管理等功能
"""
from .models import *
from .database import db_manager, get_db, get_redis, get_cache
from .auth import auth_manager, get_current_user, get_current_active_user

__all__ = [
    # 数据库
    "db_manager",
    "get_db",
    "get_redis",
    "get_cache",

    # 认证
    "auth_manager",
    "get_current_user",
    "get_current_active_user",
]
