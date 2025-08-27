"""
聊天界面组件
提供智能对话功能
"""
import streamlit as st
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import time

from src.frontend.utils.api_client import APIClient
from src.frontend.utils.session import SessionManager
from src.utils.logger import get_logger

logger = get_logger("frontend.chat")


class ChatInterface:
    """聊天界面组件"""

    def __init__(self, api_client: APIClient):
        self.api_client = api_client
        self.session_manager = SessionManager()
        logger.info("聊天界面组件初始化完成")

    def render(self):
        """渲染聊天界面"""
        # 创建两列布局
        col1, col2 = st.columns([1, 3])

        with col1:
            self._render_conversation_sidebar()

        with col2:
            self._render_chat_area()

    def _render_conversation_sidebar(self):
        """渲染对话侧边栏"""
        st.markdown("### 💬 对话列表")

        # 新建对话按钮
        if st.button("➕ 新建对话", use_container_width=True):
            self._create_new_conversation()

        st.markdown("---")

        # 加载对话列表
        conversations = self.session_manager.get_conversations()

        if not conversations:
            # 如果没有对话，尝试从API加载
            with st.spinner("加载对话列表..."):
                success, api_conversations = asyncio.run(
                    self.api_client.get_conversations()
                )
                if success:
                    conversations = api_conversations
                    self.session_manager.set_conversations(conversations)

        # 显示对话列表
        current_conversation_id = self.session_manager.get_current_conversation_id()

        for conversation in conversations:
            conversation_id = conversation.get("id")
            title = conversation.get("title", f"对话 {conversation_id}")
            created_at = conversation.get("created_at", "")

            # 截断标题
            if len(title) > 20:
                display_title = title[:20] + "..."
            else:
                display_title = title

            # 对话按钮
            button_key = f"conv_{conversation_id}"
            is_current = conversation_id == current_conversation_id

            if st.button(
                f"{'🔵' if is_current else '⚪'} {display_title}",
                key=button_key,
                use_container_width=True,
                help=f"创建时间: {created_at[:16] if created_at else 'N/A'}"
            ):
                self._switch_conversation(conversation_id)

            # 删除按钮
            if st.button(
                "🗑️",
                key=f"del_{conversation_id}",
                help="删除对话"
            ):
                self._delete_conversation(conversation_id)

        if not conversations:
            st.info("暂无对话记录")

    def _render_chat_area(self):
        """渲染聊天区域"""
        # 聊天设置
        with st.expander("⚙️ 聊天设置", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                use_rag = st.checkbox(
                    "启用RAG检索",
                    value=self.session_manager.get_user_preference(
                        "use_rag", True),
                    help="启用后将从知识库检索相关信息"
                )
                self.session_manager.update_user_preference("use_rag", use_rag)

                max_tokens = st.slider(
                    "最大令牌数",
                    min_value=100,
                    max_value=4000,
                    value=self.session_manager.get_user_preference(
                        "max_tokens", 1000),
                    step=100,
                    help="控制回复的最大长度"
                )
                self.session_manager.update_user_preference(
                    "max_tokens", max_tokens)

            with col2:
                temperature = st.slider(
                    "创造性",
                    min_value=0.0,
                    max_value=2.0,
                    value=self.session_manager.get_user_preference(
                        "temperature", 0.7),
                    step=0.1,
                    help="控制回复的创造性，值越高越有创意"
                )
                self.session_manager.update_user_preference(
                    "temperature", temperature)

        # 聊天历史显示区域
        chat_container = st.container()

        with chat_container:
            self._render_chat_history()

        # 消息输入区域
        st.markdown("---")
        self._render_message_input()

    def _render_chat_history(self):
        """渲染聊天历史"""
        chat_history = self.session_manager.get_chat_history()

        if not chat_history:
            st.info("👋 欢迎使用智能助手！请输入您的问题开始对话。")
            return

        for message in chat_history:
            role = message.get("role", "user")
            content = message.get("content", "")
            timestamp = message.get("timestamp", "")

            if role == "user":
                self._render_user_message(content, timestamp)
            elif role == "assistant":
                self._render_assistant_message(message)
            elif role == "system":
                self._render_system_message(content, timestamp)

    def _render_user_message(self, content: str, timestamp: str):
        """渲染用户消息"""
        with st.chat_message("user", avatar="👤"):
            st.markdown(content)
            if timestamp:
                st.caption(f"发送时间: {timestamp[:19]}")

    def _render_assistant_message(self, message: Dict[str, Any]):
        """渲染助手消息"""
        content = message.get("content", "")
        timestamp = message.get("timestamp", "")
        agent_steps = message.get("agent_steps", [])
        tools_used = message.get("tools_used", [])
        execution_time = message.get("execution_time")

        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(content)

            # 显示执行信息
            if agent_steps or tools_used or execution_time:
                with st.expander("🔍 执行详情", expanded=False):
                    if execution_time:
                        st.metric("执行时间", f"{execution_time:.2f}秒")

                    if tools_used:
                        st.markdown("**使用的工具:**")
                        for tool in tools_used:
                            st.markdown(f"- {tool}")

                    if agent_steps:
                        st.markdown("**推理步骤:**")
                        for i, step in enumerate(agent_steps, 1):
                            thought = step.get("thought", "")
                            action = step.get("action", "")
                            if thought:
                                st.markdown(f"{i}. **思考:** {thought}")
                            if action and action != "final_answer":
                                st.markdown(f"   **行动:** {action}")

            if timestamp:
                st.caption(f"回复时间: {timestamp[:19]}")

    def _render_system_message(self, content: str, timestamp: str):
        """渲染系统消息"""
        with st.chat_message("assistant", avatar="ℹ️"):
            st.info(content)
            if timestamp:
                st.caption(f"时间: {timestamp[:19]}")

    def _render_message_input(self):
        """渲染消息输入区域"""
        # 消息输入
        message = st.chat_input("请输入您的问题...")

        if message:
            self._send_message(message)

    def _send_message(self, message: str):
        """发送消息"""
        # 添加用户消息到历史
        self.session_manager.add_chat_message("user", message)

        # 重新渲染以显示用户消息
        st.rerun()

        # 发送到API并获取回复
        with st.spinner("🤖 正在思考..."):
            success, response = asyncio.run(self._send_message_to_api(message))

            if success and response:
                # 添加助手回复到历史
                self.session_manager.add_chat_message(
                    "assistant",
                    response.get("response", ""),
                    agent_steps=response.get("agent_steps"),
                    tools_used=response.get("tools_used"),
                    execution_time=response.get("execution_time")
                )

                # 更新当前对话ID
                conversation_id = response.get("conversation_id")
                if conversation_id:
                    self.session_manager.set_current_conversation_id(
                        conversation_id)

                # 重新渲染以显示助手回复
                st.rerun()
            else:
                st.error("发送消息失败，请重试")

    async def _send_message_to_api(self, message: str) -> tuple[bool, Optional[Dict[str, Any]]]:
        """发送消息到API"""
        try:
            conversation_id = self.session_manager.get_current_conversation_id()
            use_rag = self.session_manager.get_user_preference("use_rag", True)
            max_tokens = self.session_manager.get_user_preference(
                "max_tokens", 1000)
            temperature = self.session_manager.get_user_preference(
                "temperature", 0.7)

            success, response = await self.api_client.send_message(
                message=message,
                conversation_id=conversation_id,
                use_rag=use_rag,
                max_tokens=max_tokens,
                temperature=temperature
            )

            return success, response

        except Exception as e:
            logger.error(f"发送消息异常: {e}")
            return False, None

    def _create_new_conversation(self):
        """创建新对话"""
        # 清除当前对话状态
        self.session_manager.set_current_conversation_id(None)
        self.session_manager.clear_chat_history()

        # 添加系统欢迎消息
        self.session_manager.add_chat_message(
            "system",
            "🎉 新对话已创建！请输入您的问题开始对话。"
        )

        st.rerun()

    def _switch_conversation(self, conversation_id: int):
        """切换对话"""
        # 设置当前对话ID
        self.session_manager.set_current_conversation_id(conversation_id)

        # 清除当前聊天历史
        self.session_manager.clear_chat_history()

        # 加载对话消息
        with st.spinner("加载对话消息..."):
            success, messages = asyncio.run(
                self.api_client.get_conversation_messages(conversation_id)
            )

            if success:
                # 将API消息转换为会话格式
                for message in messages:
                    self.session_manager.add_chat_message(
                        message.get("role", "user"),
                        message.get("content", ""),
                        agent_steps=message.get("agent_steps"),
                        tools_used=message.get("tools_used"),
                        execution_time=message.get("execution_time")
                    )

        st.rerun()

    def _delete_conversation(self, conversation_id: int):
        """删除对话"""
        # 确认删除
        if st.session_state.get(f"confirm_delete_{conversation_id}"):
            with st.spinner("删除对话..."):
                success, message = asyncio.run(
                    self.api_client.delete_conversation(conversation_id)
                )

                if success:
                    # 从会话中移除对话
                    self.session_manager.remove_conversation(conversation_id)
                    st.success("对话删除成功")
                    st.rerun()
                else:
                    st.error(f"删除失败: {message}")

            # 重置确认状态
            st.session_state[f"confirm_delete_{conversation_id}"] = False
        else:
            # 设置确认状态
            st.session_state[f"confirm_delete_{conversation_id}"] = True
            st.warning("再次点击确认删除")

    def render_chat_statistics(self):
        """渲染聊天统计信息"""
        st.markdown("### 📊 聊天统计")

        conversations = self.session_manager.get_conversations()
        chat_history = self.session_manager.get_chat_history()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("对话数量", len(conversations))

        with col2:
            st.metric("当前消息数", len(chat_history))

        with col3:
            user_messages = [
                msg for msg in chat_history if msg.get("role") == "user"]
            st.metric("用户消息数", len(user_messages))

    def export_conversation(self, conversation_id: Optional[int] = None):
        """导出对话"""
        if conversation_id is None:
            conversation_id = self.session_manager.get_current_conversation_id()

        if not conversation_id:
            st.warning("请先选择一个对话")
            return

        # 获取对话消息
        success, messages = asyncio.run(
            self.api_client.get_conversation_messages(conversation_id)
        )

        if not success:
            st.error("获取对话消息失败")
            return

        # 生成导出内容
        export_content = f"# 对话导出 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            timestamp = message.get("created_at", "")

            if role == "user":
                export_content += f"**用户** ({timestamp[:19]}):\n{content}\n\n"
            elif role == "assistant":
                export_content += f"**助手** ({timestamp[:19]}):\n{content}\n\n"

        # 提供下载
        st.download_button(
            label="📥 下载对话记录",
            data=export_content,
            file_name=f"conversation_{conversation_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )
