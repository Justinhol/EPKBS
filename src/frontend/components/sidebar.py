"""
侧边栏组件
提供导航和用户信息
"""
import streamlit as st
import asyncio
from typing import Dict, Any, Optional

from src.frontend.utils.session import SessionManager
from src.frontend.components.auth import AuthManager
from src.utils.logger import get_logger

logger = get_logger("frontend.sidebar")


class Sidebar:
    """侧边栏组件"""
    
    def __init__(self, session_manager: SessionManager, auth_manager: AuthManager):
        self.session_manager = session_manager
        self.auth_manager = auth_manager
        logger.info("侧边栏组件初始化完成")
    
    def render(self):
        """渲染侧边栏"""
        with st.sidebar:
            self._render_user_info()
            st.markdown("---")
            self._render_navigation()
            st.markdown("---")
            self._render_system_status()
            st.markdown("---")
            self._render_user_actions()
    
    def _render_user_info(self):
        """渲染用户信息"""
        user_info = self.session_manager.get_user_info()
        
        if user_info:
            st.markdown("### 👤 用户信息")
            
            # 用户头像和基本信息
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(
                    "https://via.placeholder.com/80/1f77b4/ffffff?text=👤",
                    width=80
                )
            
            with col2:
                st.markdown(f"**{user_info.get('username', 'N/A')}**")
                st.caption(f"{user_info.get('email', 'N/A')}")
                
                if user_info.get('is_superuser'):
                    st.badge("管理员", type="secondary")
                else:
                    st.badge("用户", type="primary")
            
            # 登录时间
            login_time = self.session_manager.get_session_info().get('login_time')
            if login_time:
                st.caption(f"登录时间: {login_time[:19]}")
        else:
            st.warning("用户信息加载失败")
    
    def _render_navigation(self):
        """渲染导航菜单"""
        st.markdown("### 🧭 导航菜单")
        
        current_page = self.session_manager.get_current_page()
        
        # 导航按钮
        pages = [
            ("dashboard", "📊 仪表板", "系统概览和快速操作"),
            ("chat", "💬 智能对话", "与AI助手进行对话"),
            ("search", "🔍 知识搜索", "搜索知识库内容"),
            ("documents", "📄 文档管理", "上传和管理文档")
        ]
        
        for page_key, page_name, page_desc in pages:
            is_current = current_page == page_key
            
            if st.button(
                page_name,
                key=f"nav_{page_key}",
                use_container_width=True,
                type="primary" if is_current else "secondary",
                help=page_desc
            ):
                if not is_current:
                    self.session_manager.set_current_page(page_key)
                    st.rerun()
    
    def _render_system_status(self):
        """渲染系统状态"""
        st.markdown("### 🔧 系统状态")
        
        # 获取系统状态
        system_status = self.session_manager.get_system_status()
        
        if not system_status:
            # 尝试获取系统状态
            with st.spinner("检查系统状态..."):
                try:
                    success, status = asyncio.run(
                        self.auth_manager.api_client.get_health_status()
                    )
                    if success:
                        system_status = status
                        self.session_manager.set_system_status(status)
                except Exception as e:
                    logger.warning(f"获取系统状态失败: {e}")
                    system_status = {"status": "unknown"}
        
        # 显示状态
        status = system_status.get("status", "unknown")
        
        if status == "healthy":
            st.success("✅ 系统正常")
        elif status == "unhealthy":
            st.error("❌ 系统异常")
        else:
            st.warning("⚠️ 状态未知")
        
        # 详细状态信息
        with st.expander("详细状态", expanded=False):
            database_status = system_status.get("database", "unknown")
            redis_status = system_status.get("redis", "unknown")
            
            st.markdown(f"**数据库:** {database_status}")
            st.markdown(f"**缓存:** {redis_status}")
            
            if "timestamp" in system_status:
                st.caption(f"更新时间: {system_status['timestamp']}")
        
        # 刷新按钮
        if st.button("🔄 刷新状态", use_container_width=True):
            self.session_manager.set_system_status({})
            st.rerun()
    
    def _render_user_actions(self):
        """渲染用户操作"""
        st.markdown("### ⚙️ 用户操作")
        
        # 用户偏好设置
        with st.expander("🎛️ 偏好设置", expanded=False):
            preferences = self.session_manager.get_user_preferences()
            
            # 主题设置
            theme = st.selectbox(
                "界面主题",
                options=["light", "dark"],
                index=0 if preferences.get("theme") == "light" else 1,
                help="选择界面主题"
            )
            self.session_manager.update_user_preference("theme", theme)
            
            # 语言设置
            language = st.selectbox(
                "界面语言",
                options=["zh", "en"],
                index=0 if preferences.get("language") == "zh" else 1,
                help="选择界面语言"
            )
            self.session_manager.update_user_preference("language", language)
            
            # 默认RAG设置
            default_rag = st.checkbox(
                "默认启用RAG",
                value=preferences.get("use_rag", True),
                help="新对话是否默认启用RAG检索"
            )
            self.session_manager.update_user_preference("use_rag", default_rag)
        
        # 会话管理
        with st.expander("📊 会话信息", expanded=False):
            session_info = self.session_manager.get_session_info()
            
            st.markdown(f"**对话数量:** {session_info.get('conversation_count', 0)}")
            st.markdown(f"**消息数量:** {session_info.get('message_count', 0)}")
            st.markdown(f"**搜索次数:** {session_info.get('search_count', 0)}")
            st.markdown(f"**文档数量:** {session_info.get('document_count', 0)}")
            
            # 清除缓存按钮
            if st.button("🗑️ 清除缓存", use_container_width=True):
                self.session_manager.clear_cache()
                st.success("缓存已清除")
                st.rerun()
        
        # 导出数据
        with st.expander("📥 数据导出", expanded=False):
            st.markdown("导出您的数据：")
            
            # 导出会话信息
            if st.button("📊 导出会话信息", use_container_width=True):
                session_info = self.session_manager.get_session_info()
                st.download_button(
                    label="下载会话信息",
                    data=str(session_info),
                    file_name=f"session_info_{session_info.get('user', 'unknown')}.json",
                    mime="application/json"
                )
            
            # 导出搜索历史
            search_history = self.session_manager.get_search_history()
            if search_history:
                if st.button("🔍 导出搜索历史", use_container_width=True):
                    import json
                    st.download_button(
                        label="下载搜索历史",
                        data=json.dumps(search_history, ensure_ascii=False, indent=2),
                        file_name=f"search_history_{session_info.get('user', 'unknown')}.json",
                        mime="application/json"
                    )
        
        st.markdown("---")
        
        # 登出按钮
        if st.button("🚪 登出", use_container_width=True, type="primary"):
            with st.spinner("正在登出..."):
                success, message = asyncio.run(self.auth_manager.logout())
                
                if success:
                    st.success("登出成功")
                    st.rerun()
                else:
                    st.error(f"登出失败: {message}")
    
    def render_quick_stats(self):
        """渲染快速统计"""
        st.markdown("### 📈 快速统计")
        
        session_info = self.session_manager.get_session_info()
        
        # 使用指标卡片
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "对话数",
                session_info.get('conversation_count', 0),
                help="总对话数量"
            )
        
        with col2:
            st.metric(
                "消息数",
                session_info.get('message_count', 0),
                help="当前对话消息数"
            )
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.metric(
                "搜索次数",
                session_info.get('search_count', 0),
                help="总搜索次数"
            )
        
        with col4:
            st.metric(
                "文档数",
                session_info.get('document_count', 0),
                help="已上传文档数"
            )
    
    def render_notifications(self):
        """渲染通知"""
        st.markdown("### 🔔 通知")
        
        # 这里可以添加系统通知、更新提醒等
        notifications = [
            {
                "type": "info",
                "title": "系统提示",
                "message": "欢迎使用企业私有知识库系统！",
                "timestamp": "2024-01-01 10:00:00"
            }
        ]
        
        for notification in notifications:
            with st.container():
                if notification["type"] == "info":
                    st.info(f"**{notification['title']}**: {notification['message']}")
                elif notification["type"] == "warning":
                    st.warning(f"**{notification['title']}**: {notification['message']}")
                elif notification["type"] == "error":
                    st.error(f"**{notification['title']}**: {notification['message']}")
                
                st.caption(f"时间: {notification['timestamp']}")
    
    def render_help_section(self):
        """渲染帮助部分"""
        st.markdown("### ❓ 帮助")
        
        with st.expander("使用指南", expanded=False):
            st.markdown("""
            **智能对话:**
            - 点击"智能对话"开始与AI助手交流
            - 支持多轮对话和上下文理解
            - 可以启用RAG检索获得更准确的答案
            
            **知识搜索:**
            - 在知识库中搜索相关信息
            - 支持多种检索和重排序方式
            - 可以基于搜索结果开始对话
            
            **文档管理:**
            - 上传PDF、Word、TXT等格式文档
            - 自动处理和向量化文档内容
            - 支持文档预览和管理
            """)
        
        with st.expander("快捷键", expanded=False):
            st.markdown("""
            - `Ctrl + Enter`: 发送消息
            - `Ctrl + /`: 打开搜索
            - `Ctrl + N`: 新建对话
            - `Ctrl + S`: 保存当前对话
            """)
        
        with st.expander("联系支持", expanded=False):
            st.markdown("""
            如果您遇到问题或需要帮助：
            - 📧 邮箱: support@example.com
            - 📞 电话: 400-123-4567
            - 💬 在线客服: 工作日 9:00-18:00
            """)
