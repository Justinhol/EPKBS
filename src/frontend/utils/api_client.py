"""
API客户端
与后端API进行通信
"""
import asyncio
import aiohttp
import json
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

from src.utils.logger import get_logger
from config.settings import settings

logger = get_logger("frontend.api_client")


class APIClient:
    """API客户端"""
    
    def __init__(self):
        self.base_url = f"http://{settings.API_HOST}:{settings.API_PORT}{settings.API_PREFIX}"
        self.session = None
        self.access_token = None
        
        logger.info(f"API客户端初始化: {self.base_url}")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """获取HTTP会话"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    def _get_headers(self) -> Dict[str, str]:
        """获取请求头"""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        
        return headers
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """发送HTTP请求"""
        url = f"{self.base_url}{endpoint}"
        headers = self._get_headers()
        
        try:
            session = await self._get_session()
            
            async with session.request(
                method=method,
                url=url,
                headers=headers,
                json=data,
                params=params
            ) as response:
                
                response_data = await response.json()
                
                if response.status < 400:
                    return True, response_data
                else:
                    error_message = response_data.get("message", f"HTTP {response.status}")
                    logger.warning(f"API请求失败: {method} {endpoint} - {error_message}")
                    return False, {"error": error_message}
                    
        except aiohttp.ClientError as e:
            logger.error(f"网络请求失败: {e}")
            return False, {"error": f"网络连接失败: {str(e)}"}
        except json.JSONDecodeError as e:
            logger.error(f"响应解析失败: {e}")
            return False, {"error": "响应格式错误"}
        except Exception as e:
            logger.error(f"请求异常: {e}")
            return False, {"error": f"请求失败: {str(e)}"}
    
    def set_access_token(self, token: str):
        """设置访问令牌"""
        self.access_token = token
        logger.info("访问令牌已设置")
    
    def clear_access_token(self):
        """清除访问令牌"""
        self.access_token = None
        logger.info("访问令牌已清除")
    
    # 认证相关API
    async def register(
        self,
        username: str,
        email: str,
        password: str,
        full_name: Optional[str] = None
    ) -> Tuple[bool, str]:
        """用户注册"""
        data = {
            "username": username,
            "email": email,
            "password": password
        }
        if full_name:
            data["full_name"] = full_name
        
        success, response = await self._make_request("POST", "/auth/register", data)
        
        if success:
            return True, "注册成功"
        else:
            return False, response.get("error", "注册失败")
    
    async def login(self, username: str, password: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """用户登录"""
        # OAuth2密码流格式
        data = {
            "username": username,
            "password": password
        }
        
        # 使用表单数据格式
        session = await self._get_session()
        url = f"{self.base_url}/auth/login"
        
        try:
            async with session.post(url, data=data) as response:
                response_data = await response.json()
                
                if response.status == 200:
                    access_token = response_data.get("access_token")
                    user_info = response_data.get("user")
                    
                    if access_token:
                        self.set_access_token(access_token)
                        return True, "登录成功", user_info
                    else:
                        return False, "登录响应格式错误", None
                else:
                    error_message = response_data.get("detail", "登录失败")
                    return False, error_message, None
                    
        except Exception as e:
            logger.error(f"登录请求失败: {e}")
            return False, f"登录失败: {str(e)}", None
    
    async def get_current_user(self) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """获取当前用户信息"""
        success, response = await self._make_request("GET", "/auth/me")
        
        if success:
            return True, response
        else:
            return False, None
    
    async def logout(self) -> Tuple[bool, str]:
        """用户登出"""
        success, response = await self._make_request("POST", "/auth/logout")
        
        self.clear_access_token()
        
        if success:
            return True, "登出成功"
        else:
            return True, "登出成功"  # 即使API调用失败，本地也要清除令牌
    
    # 聊天相关API
    async def create_conversation(
        self,
        title: Optional[str] = None,
        use_rag: bool = True,
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """创建对话"""
        data = {
            "title": title,
            "use_rag": use_rag,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        success, response = await self._make_request("POST", "/chat/conversations", data)
        
        if success:
            return True, response
        else:
            return False, None
    
    async def get_conversations(
        self,
        skip: int = 0,
        limit: int = 20
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """获取对话列表"""
        params = {"skip": skip, "limit": limit}
        success, response = await self._make_request("GET", "/chat/conversations", params=params)
        
        if success:
            return True, response if isinstance(response, list) else []
        else:
            return False, []
    
    async def get_conversation_messages(
        self,
        conversation_id: int,
        skip: int = 0,
        limit: int = 50
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """获取对话消息"""
        params = {"skip": skip, "limit": limit}
        success, response = await self._make_request(
            "GET", f"/chat/conversations/{conversation_id}/messages", params=params
        )
        
        if success:
            return True, response if isinstance(response, list) else []
        else:
            return False, []
    
    async def send_message(
        self,
        message: str,
        conversation_id: Optional[int] = None,
        use_rag: bool = True,
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """发送消息"""
        data = {
            "message": message,
            "use_rag": use_rag,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        if conversation_id:
            data["conversation_id"] = conversation_id
        
        success, response = await self._make_request("POST", "/chat/send", data)
        
        if success:
            return True, response
        else:
            return False, None
    
    async def delete_conversation(self, conversation_id: int) -> Tuple[bool, str]:
        """删除对话"""
        success, response = await self._make_request("DELETE", f"/chat/conversations/{conversation_id}")
        
        if success:
            return True, "对话删除成功"
        else:
            return False, response.get("error", "删除失败")
    
    # 搜索相关API
    async def search_knowledge_base(
        self,
        query: str,
        top_k: int = 10,
        retriever_type: str = "hybrid",
        reranker_type: str = "ensemble"
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """搜索知识库"""
        data = {
            "query": query,
            "top_k": top_k,
            "retriever_type": retriever_type,
            "reranker_type": reranker_type
        }
        
        success, response = await self._make_request("POST", "/search/", data)
        
        if success:
            return True, response
        else:
            return False, None
    
    async def get_search_history(
        self,
        skip: int = 0,
        limit: int = 20
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """获取搜索历史"""
        params = {"skip": skip, "limit": limit}
        success, response = await self._make_request("GET", "/search/history", params=params)
        
        if success:
            history = response.get("history", []) if isinstance(response, dict) else []
            return True, history
        else:
            return False, []
    
    async def get_search_suggestions(
        self,
        query: str,
        limit: int = 5
    ) -> Tuple[bool, List[str]]:
        """获取搜索建议"""
        params = {"query": query, "limit": limit}
        success, response = await self._make_request("GET", "/search/suggestions", params=params)
        
        if success:
            suggestions = response.get("suggestions", []) if isinstance(response, dict) else []
            return True, suggestions
        else:
            return False, []
    
    # 系统相关API
    async def get_health_status(self) -> Tuple[bool, Dict[str, Any]]:
        """获取系统健康状态"""
        # 直接访问根路径的health端点
        url = f"http://{settings.API_HOST}:{settings.API_PORT}/health"
        
        try:
            session = await self._get_session()
            async with session.get(url) as response:
                response_data = await response.json()
                return response.status < 400, response_data
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return False, {"error": str(e)}
    
    async def get_system_info(self) -> Tuple[bool, Dict[str, Any]]:
        """获取系统信息"""
        url = f"http://{settings.API_HOST}:{settings.API_PORT}/info"
        
        try:
            session = await self._get_session()
            async with session.get(url) as response:
                response_data = await response.json()
                return response.status < 400, response_data
        except Exception as e:
            logger.error(f"获取系统信息失败: {e}")
            return False, {"error": str(e)}
    
    async def close(self):
        """关闭客户端"""
        if self.session and not self.session.closed:
            await self.session.close()
        logger.info("API客户端已关闭")
