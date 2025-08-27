"""
Streamlit前端主应用
"""
import streamlit as st
import asyncio
from typing import Dict, Any, Optional
import time
from datetime import datetime

from src.frontend.components.auth import AuthManager
from src.frontend.components.chat import ChatInterface
from src.frontend.components.search import SearchInterface
from src.frontend.components.documents import DocumentManager
from src.frontend.components.sidebar import Sidebar
from src.frontend.utils.api_client import APIClient
from src.frontend.utils.session import SessionManager
from src.utils.logger import get_logger
from config.settings import settings

logger = get_logger("frontend.app")

# 页面配置
st.set_page_config(
    page_title=settings.PROJECT_NAME,
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/EPKBS',
        'Report a bug': 'https://github.com/your-repo/EPKBS/issues',
        'About': f"{settings.PROJECT_NAME} v{settings.PROJECT_VERSION}"
    }
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .chat-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .search-container {
        background-color: #fff;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .document-card {
        background-color: #ffffff;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


class EPKBSApp:
    """EPKBS前端应用主类"""

    def __init__(self):
        self.api_client = APIClient()
        self.session_manager = SessionManager()
        self.auth_manager = AuthManager(self.api_client)

        # 初始化组件
        self.chat_interface = None
        self.search_interface = None
        self.document_manager = None
        self.sidebar = None

        logger.info("EPKBS前端应用初始化完成")

    def initialize_components(self):
        """初始化界面组件"""
        if not self.chat_interface:
            self.chat_interface = ChatInterface(self.api_client)
        if not self.search_interface:
            self.search_interface = SearchInterface(self.api_client)
        if not self.document_manager:
            self.document_manager = DocumentManager(self.api_client)
        if not self.sidebar:
            self.sidebar = Sidebar(self.session_manager, self.auth_manager)

    def run(self):
        """运行主应用"""
        try:
            # 显示主标题
            st.markdown('<h1 class="main-header">🧠 企业私有知识库系统</h1>',
                        unsafe_allow_html=True)

            # 检查用户认证状态
            if not self.session_manager.is_authenticated():
                self._show_login_page()
                return

            # 初始化组件
            self.initialize_components()

            # 显示侧边栏
            self.sidebar.render()

            # 获取当前页面
            current_page = self.session_manager.get_current_page()

            # 根据页面显示不同内容
            if current_page == "chat":
                self._show_chat_page()
            elif current_page == "search":
                self._show_search_page()
            elif current_page == "documents":
                self._show_documents_page()
            elif current_page == "dashboard":
                self._show_dashboard_page()
            else:
                self._show_dashboard_page()

        except Exception as e:
            logger.error(f"应用运行错误: {e}")
            st.error(f"应用运行出错: {str(e)}")

    def _show_login_page(self):
        """显示登录页面"""
        st.markdown("## 🔐 用户登录")

        # 创建登录表单
        with st.form("login_form"):
            col1, col2, col3 = st.columns([1, 2, 1])

            with col2:
                st.markdown("### 请登录您的账户")

                username = st.text_input("用户名", placeholder="请输入用户名")
                password = st.text_input(
                    "密码", type="password", placeholder="请输入密码")

                col_login, col_register = st.columns(2)

                with col_login:
                    login_clicked = st.form_submit_button(
                        "登录", use_container_width=True)

                with col_register:
                    register_clicked = st.form_submit_button(
                        "注册", use_container_width=True)

        # 处理登录
        if login_clicked and username and password:
            with st.spinner("正在登录..."):
                success, message = asyncio.run(
                    self.auth_manager.login(username, password)
                )

                if success:
                    st.success("登录成功！")
                    st.rerun()
                else:
                    st.error(f"登录失败: {message}")

        # 处理注册
        if register_clicked:
            self._show_register_dialog()

    def _show_register_dialog(self):
        """显示注册对话框"""
        with st.expander("用户注册", expanded=True):
            with st.form("register_form"):
                username = st.text_input("用户名", placeholder="3-50个字符")
                email = st.text_input("邮箱", placeholder="请输入有效邮箱地址")
                password = st.text_input(
                    "密码", type="password", placeholder="至少6个字符")
                confirm_password = st.text_input(
                    "确认密码", type="password", placeholder="请再次输入密码")
                full_name = st.text_input("姓名", placeholder="请输入真实姓名（可选）")

                register_submit = st.form_submit_button(
                    "注册", use_container_width=True)

                if register_submit:
                    # 验证输入
                    if not all([username, email, password]):
                        st.error("请填写所有必填字段")
                        return

                    if password != confirm_password:
                        st.error("两次输入的密码不一致")
                        return

                    if len(password) < 6:
                        st.error("密码长度至少6个字符")
                        return

                    # 执行注册
                    with st.spinner("正在注册..."):
                        success, message = asyncio.run(
                            self.auth_manager.register(
                                username, email, password, full_name)
                        )

                        if success:
                            st.success("注册成功！请使用新账户登录。")
                        else:
                            st.error(f"注册失败: {message}")

    def _show_dashboard_page(self):
        """显示仪表板页面"""
        st.markdown("## 📊 系统仪表板")

        # 系统状态
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>🤖 Agent状态</h3>
                <p>运行正常</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>🔍 RAG系统</h3>
                <p>运行正常</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>📚 知识库</h3>
                <p>已就绪</p>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown("""
            <div class="metric-card">
                <h3>👥 在线用户</h3>
                <p>1</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # 快速操作
        st.markdown("### 🚀 快速操作")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("💬 开始聊天", use_container_width=True):
                self.session_manager.set_current_page("chat")
                st.rerun()

        with col2:
            if st.button("🔍 搜索知识库", use_container_width=True):
                self.session_manager.set_current_page("search")
                st.rerun()

        with col3:
            if st.button("📄 管理文档", use_container_width=True):
                self.session_manager.set_current_page("documents")
                st.rerun()

        # 最近活动
        st.markdown("### 📈 最近活动")

        # 模拟活动数据
        activities = [
            {"time": "2分钟前", "action": "用户提问", "content": "什么是人工智能？"},
            {"time": "5分钟前", "action": "文档上传", "content": "AI技术报告.pdf"},
            {"time": "10分钟前", "action": "知识库搜索", "content": "机器学习算法"},
        ]

        for activity in activities:
            with st.container():
                col1, col2, col3 = st.columns([2, 2, 6])
                with col1:
                    st.text(activity["time"])
                with col2:
                    st.text(activity["action"])
                with col3:
                    st.text(activity["content"])

    def _show_chat_page(self):
        """显示聊天页面"""
        st.markdown("## 💬 智能对话")
        self.chat_interface.render()

    def _show_search_page(self):
        """显示搜索页面"""
        st.markdown("## 🔍 知识库搜索")
        self.search_interface.render()

    def _show_documents_page(self):
        """显示文档管理页面"""
        st.markdown("## 📄 文档管理")
        self.document_manager.render()


def main():
    """主函数"""
    try:
        app = EPKBSApp()
        app.run()

    except Exception as e:
        logger.error(f"应用启动失败: {e}")
        st.error("应用启动失败，请检查系统配置")
        st.exception(e)


if __name__ == "__main__":
    main()
