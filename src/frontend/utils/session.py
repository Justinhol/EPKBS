"""
会话管理器
管理用户会话状态和数据
"""
import streamlit as st
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import json

from src.utils.logger import get_logger

logger = get_logger("frontend.session")


class SessionManager:
    """会话管理器"""
    
    def __init__(self):
        self._initialize_session()
        logger.info("会话管理器初始化完成")
    
    def _initialize_session(self):
        """初始化会话状态"""
        # 用户认证状态
        if "authenticated" not in st.session_state:
            st.session_state.authenticated = False
        
        if "user_info" not in st.session_state:
            st.session_state.user_info = None
        
        if "access_token" not in st.session_state:
            st.session_state.access_token = None
        
        if "login_time" not in st.session_state:
            st.session_state.login_time = None
        
        # 页面状态
        if "current_page" not in st.session_state:
            st.session_state.current_page = "dashboard"
        
        # 聊天状态
        if "current_conversation_id" not in st.session_state:
            st.session_state.current_conversation_id = None
        
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        if "conversations" not in st.session_state:
            st.session_state.conversations = []
        
        # 搜索状态
        if "search_history" not in st.session_state:
            st.session_state.search_history = []
        
        if "last_search_results" not in st.session_state:
            st.session_state.last_search_results = []
        
        # 文档状态
        if "uploaded_documents" not in st.session_state:
            st.session_state.uploaded_documents = []
        
        # 系统状态
        if "system_status" not in st.session_state:
            st.session_state.system_status = {}
        
        # 用户偏好
        if "user_preferences" not in st.session_state:
            st.session_state.user_preferences = {
                "theme": "light",
                "language": "zh",
                "max_tokens": 1000,
                "temperature": 0.7,
                "use_rag": True,
                "retriever_type": "hybrid",
                "reranker_type": "ensemble"
            }
    
    # 认证相关方法
    def is_authenticated(self) -> bool:
        """检查用户是否已认证"""
        if not st.session_state.authenticated:
            return False
        
        # 检查登录是否过期（24小时）
        if st.session_state.login_time:
            login_time = datetime.fromisoformat(st.session_state.login_time)
            if datetime.now() - login_time > timedelta(hours=24):
                self.logout()
                return False
        
        return True
    
    def login(self, user_info: Dict[str, Any], access_token: str):
        """用户登录"""
        st.session_state.authenticated = True
        st.session_state.user_info = user_info
        st.session_state.access_token = access_token
        st.session_state.login_time = datetime.now().isoformat()
        
        logger.info(f"用户登录: {user_info.get('username', 'unknown')}")
    
    def logout(self):
        """用户登出"""
        username = st.session_state.user_info.get('username', 'unknown') if st.session_state.user_info else 'unknown'
        
        # 清除认证状态
        st.session_state.authenticated = False
        st.session_state.user_info = None
        st.session_state.access_token = None
        st.session_state.login_time = None
        
        # 清除聊天状态
        st.session_state.current_conversation_id = None
        st.session_state.chat_history = []
        st.session_state.conversations = []
        
        # 重置页面
        st.session_state.current_page = "dashboard"
        
        logger.info(f"用户登出: {username}")
    
    def get_user_info(self) -> Optional[Dict[str, Any]]:
        """获取用户信息"""
        return st.session_state.user_info
    
    def get_access_token(self) -> Optional[str]:
        """获取访问令牌"""
        return st.session_state.access_token
    
    # 页面状态管理
    def get_current_page(self) -> str:
        """获取当前页面"""
        return st.session_state.current_page
    
    def set_current_page(self, page: str):
        """设置当前页面"""
        st.session_state.current_page = page
        logger.debug(f"页面切换: {page}")
    
    # 聊天状态管理
    def get_current_conversation_id(self) -> Optional[int]:
        """获取当前对话ID"""
        return st.session_state.current_conversation_id
    
    def set_current_conversation_id(self, conversation_id: Optional[int]):
        """设置当前对话ID"""
        st.session_state.current_conversation_id = conversation_id
        if conversation_id:
            logger.debug(f"切换对话: {conversation_id}")
    
    def get_chat_history(self) -> List[Dict[str, Any]]:
        """获取聊天历史"""
        return st.session_state.chat_history
    
    def add_chat_message(self, role: str, content: str, **kwargs):
        """添加聊天消息"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        st.session_state.chat_history.append(message)
    
    def clear_chat_history(self):
        """清除聊天历史"""
        st.session_state.chat_history = []
        logger.debug("聊天历史已清除")
    
    def get_conversations(self) -> List[Dict[str, Any]]:
        """获取对话列表"""
        return st.session_state.conversations
    
    def set_conversations(self, conversations: List[Dict[str, Any]]):
        """设置对话列表"""
        st.session_state.conversations = conversations
    
    def add_conversation(self, conversation: Dict[str, Any]):
        """添加对话"""
        st.session_state.conversations.insert(0, conversation)
    
    def remove_conversation(self, conversation_id: int):
        """移除对话"""
        st.session_state.conversations = [
            conv for conv in st.session_state.conversations 
            if conv.get("id") != conversation_id
        ]
        
        # 如果删除的是当前对话，清除当前对话ID
        if st.session_state.current_conversation_id == conversation_id:
            st.session_state.current_conversation_id = None
            st.session_state.chat_history = []
    
    # 搜索状态管理
    def get_search_history(self) -> List[Dict[str, Any]]:
        """获取搜索历史"""
        return st.session_state.search_history
    
    def add_search_record(self, query: str, results: List[Dict[str, Any]], **kwargs):
        """添加搜索记录"""
        record = {
            "query": query,
            "results": results,
            "timestamp": datetime.now().isoformat(),
            "result_count": len(results),
            **kwargs
        }
        st.session_state.search_history.insert(0, record)
        
        # 保留最近50条记录
        if len(st.session_state.search_history) > 50:
            st.session_state.search_history = st.session_state.search_history[:50]
    
    def get_last_search_results(self) -> List[Dict[str, Any]]:
        """获取最后搜索结果"""
        return st.session_state.last_search_results
    
    def set_last_search_results(self, results: List[Dict[str, Any]]):
        """设置最后搜索结果"""
        st.session_state.last_search_results = results
    
    # 文档状态管理
    def get_uploaded_documents(self) -> List[Dict[str, Any]]:
        """获取已上传文档"""
        return st.session_state.uploaded_documents
    
    def add_uploaded_document(self, document: Dict[str, Any]):
        """添加已上传文档"""
        st.session_state.uploaded_documents.insert(0, document)
    
    def remove_uploaded_document(self, document_id: int):
        """移除已上传文档"""
        st.session_state.uploaded_documents = [
            doc for doc in st.session_state.uploaded_documents 
            if doc.get("id") != document_id
        ]
    
    # 系统状态管理
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return st.session_state.system_status
    
    def set_system_status(self, status: Dict[str, Any]):
        """设置系统状态"""
        st.session_state.system_status = status
    
    # 用户偏好管理
    def get_user_preferences(self) -> Dict[str, Any]:
        """获取用户偏好"""
        return st.session_state.user_preferences
    
    def update_user_preference(self, key: str, value: Any):
        """更新用户偏好"""
        st.session_state.user_preferences[key] = value
        logger.debug(f"用户偏好更新: {key} = {value}")
    
    def get_user_preference(self, key: str, default: Any = None) -> Any:
        """获取用户偏好值"""
        return st.session_state.user_preferences.get(key, default)
    
    # 缓存管理
    def clear_cache(self):
        """清除缓存数据"""
        st.session_state.search_history = []
        st.session_state.last_search_results = []
        st.session_state.system_status = {}
        logger.info("缓存数据已清除")
    
    def get_session_info(self) -> Dict[str, Any]:
        """获取会话信息"""
        return {
            "authenticated": st.session_state.authenticated,
            "user": st.session_state.user_info.get('username') if st.session_state.user_info else None,
            "login_time": st.session_state.login_time,
            "current_page": st.session_state.current_page,
            "conversation_count": len(st.session_state.conversations),
            "message_count": len(st.session_state.chat_history),
            "search_count": len(st.session_state.search_history),
            "document_count": len(st.session_state.uploaded_documents)
        }
