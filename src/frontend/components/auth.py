"""
认证组件
处理用户登录、注册、登出等认证相关功能
"""
import streamlit as st
import asyncio
from typing import Tuple, Optional, Dict, Any

from src.frontend.utils.api_client import APIClient
from src.utils.logger import get_logger

logger = get_logger("frontend.auth")


class AuthManager:
    """认证管理器"""
    
    def __init__(self, api_client: APIClient):
        self.api_client = api_client
        logger.info("认证管理器初始化完成")
    
    async def login(self, username: str, password: str) -> Tuple[bool, str]:
        """用户登录"""
        try:
            success, message, user_info = await self.api_client.login(username, password)
            
            if success and user_info:
                # 保存用户信息到会话状态
                from src.frontend.utils.session import SessionManager
                session_manager = SessionManager()
                session_manager.login(user_info, self.api_client.access_token)
                
                logger.info(f"用户登录成功: {username}")
                return True, message
            else:
                logger.warning(f"用户登录失败: {username} - {message}")
                return False, message
                
        except Exception as e:
            logger.error(f"登录异常: {e}")
            return False, f"登录失败: {str(e)}"
    
    async def register(
        self,
        username: str,
        email: str,
        password: str,
        full_name: Optional[str] = None
    ) -> Tuple[bool, str]:
        """用户注册"""
        try:
            success, message = await self.api_client.register(username, email, password, full_name)
            
            if success:
                logger.info(f"用户注册成功: {username}")
            else:
                logger.warning(f"用户注册失败: {username} - {message}")
            
            return success, message
            
        except Exception as e:
            logger.error(f"注册异常: {e}")
            return False, f"注册失败: {str(e)}"
    
    async def logout(self) -> Tuple[bool, str]:
        """用户登出"""
        try:
            # 调用API登出
            success, message = await self.api_client.logout()
            
            # 清除会话状态
            from src.frontend.utils.session import SessionManager
            session_manager = SessionManager()
            session_manager.logout()
            
            logger.info("用户登出成功")
            return True, message
            
        except Exception as e:
            logger.error(f"登出异常: {e}")
            return False, f"登出失败: {str(e)}"
    
    async def get_current_user(self) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """获取当前用户信息"""
        try:
            success, user_info = await self.api_client.get_current_user()
            
            if success:
                return True, user_info
            else:
                return False, None
                
        except Exception as e:
            logger.error(f"获取用户信息异常: {e}")
            return False, None
    
    def render_login_form(self) -> Tuple[bool, Optional[str], Optional[str]]:
        """渲染登录表单"""
        st.markdown("### 🔐 用户登录")
        
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input(
                "用户名",
                placeholder="请输入用户名",
                help="请输入您的用户名"
            )
            
            password = st.text_input(
                "密码",
                type="password",
                placeholder="请输入密码",
                help="请输入您的密码"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                login_clicked = st.form_submit_button(
                    "登录",
                    use_container_width=True,
                    type="primary"
                )
            
            with col2:
                forgot_clicked = st.form_submit_button(
                    "忘记密码",
                    use_container_width=True
                )
        
        if login_clicked:
            if not username or not password:
                st.error("请输入用户名和密码")
                return False, None, None
            
            return True, username, password
        
        if forgot_clicked:
            st.info("请联系管理员重置密码")
        
        return False, None, None
    
    def render_register_form(self) -> Tuple[bool, Optional[Dict[str, str]]]:
        """渲染注册表单"""
        st.markdown("### 📝 用户注册")
        
        with st.form("register_form", clear_on_submit=False):
            username = st.text_input(
                "用户名",
                placeholder="3-50个字符",
                help="用户名长度为3-50个字符，只能包含字母、数字和下划线"
            )
            
            email = st.text_input(
                "邮箱地址",
                placeholder="请输入有效的邮箱地址",
                help="请输入有效的邮箱地址，用于账户验证"
            )
            
            full_name = st.text_input(
                "真实姓名",
                placeholder="请输入真实姓名（可选）",
                help="真实姓名是可选的，用于个人资料显示"
            )
            
            password = st.text_input(
                "密码",
                type="password",
                placeholder="至少6个字符",
                help="密码长度至少6个字符，建议包含字母、数字和特殊字符"
            )
            
            confirm_password = st.text_input(
                "确认密码",
                type="password",
                placeholder="请再次输入密码",
                help="请再次输入密码以确认"
            )
            
            # 用户协议
            agree_terms = st.checkbox(
                "我已阅读并同意用户协议和隐私政策",
                help="请仔细阅读用户协议和隐私政策"
            )
            
            register_clicked = st.form_submit_button(
                "注册",
                use_container_width=True,
                type="primary"
            )
        
        if register_clicked:
            # 验证输入
            errors = []
            
            if not username:
                errors.append("请输入用户名")
            elif len(username) < 3 or len(username) > 50:
                errors.append("用户名长度应为3-50个字符")
            
            if not email:
                errors.append("请输入邮箱地址")
            elif "@" not in email or "." not in email:
                errors.append("请输入有效的邮箱地址")
            
            if not password:
                errors.append("请输入密码")
            elif len(password) < 6:
                errors.append("密码长度至少6个字符")
            
            if password != confirm_password:
                errors.append("两次输入的密码不一致")
            
            if not agree_terms:
                errors.append("请同意用户协议和隐私政策")
            
            if errors:
                for error in errors:
                    st.error(error)
                return False, None
            
            # 返回注册数据
            register_data = {
                "username": username,
                "email": email,
                "password": password,
                "full_name": full_name if full_name else None
            }
            
            return True, register_data
        
        return False, None
    
    def render_user_profile(self, user_info: Dict[str, Any]):
        """渲染用户资料"""
        st.markdown("### 👤 用户资料")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # 用户头像（使用默认头像）
            st.image(
                "https://via.placeholder.com/150/1f77b4/ffffff?text=👤",
                width=150
            )
        
        with col2:
            st.markdown(f"**用户名:** {user_info.get('username', 'N/A')}")
            st.markdown(f"**邮箱:** {user_info.get('email', 'N/A')}")
            st.markdown(f"**姓名:** {user_info.get('full_name', 'N/A')}")
            st.markdown(f"**注册时间:** {user_info.get('created_at', 'N/A')[:10]}")
            st.markdown(f"**账户状态:** {'✅ 活跃' if user_info.get('is_active') else '❌ 禁用'}")
            
            if user_info.get('is_superuser'):
                st.markdown("**权限:** 🔑 管理员")
            else:
                st.markdown("**权限:** 👤 普通用户")
    
    def render_logout_button(self) -> bool:
        """渲染登出按钮"""
        return st.button(
            "🚪 登出",
            use_container_width=True,
            help="点击登出当前账户"
        )
    
    def render_auth_status(self, user_info: Optional[Dict[str, Any]] = None):
        """渲染认证状态"""
        if user_info:
            st.success(f"✅ 已登录: {user_info.get('username', 'Unknown')}")
        else:
            st.warning("⚠️ 未登录")
    
    def validate_session(self) -> bool:
        """验证会话有效性"""
        from src.frontend.utils.session import SessionManager
        session_manager = SessionManager()
        
        if not session_manager.is_authenticated():
            return False
        
        # 可以在这里添加更多的会话验证逻辑
        # 比如检查令牌是否过期、用户权限等
        
        return True
