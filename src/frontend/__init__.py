"""
前端界面层
基于Streamlit的Web界面，提供用户交互功能
"""
from .app import EPKBSApp, main
from .utils import APIClient, SessionManager
from .components import (
    AuthManager,
    ChatInterface,
    SearchInterface,
    DocumentManager,
    Sidebar
)

__all__ = [
    # 主应用
    "EPKBSApp",
    "main",

    # 工具类
    "APIClient",
    "SessionManager",

    # 组件
    "AuthManager",
    "ChatInterface",
    "SearchInterface",
    "DocumentManager",
    "Sidebar"
]
