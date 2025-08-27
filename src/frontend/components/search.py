"""
搜索界面组件
提供知识库搜索功能
"""
import streamlit as st
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

from src.frontend.utils.api_client import APIClient
from src.frontend.utils.session import SessionManager
from src.utils.logger import get_logger

logger = get_logger("frontend.search")


class SearchInterface:
    """搜索界面组件"""
    
    def __init__(self, api_client: APIClient):
        self.api_client = api_client
        self.session_manager = SessionManager()
        logger.info("搜索界面组件初始化完成")
    
    def render(self):
        """渲染搜索界面"""
        # 搜索输入区域
        self._render_search_input()
        
        st.markdown("---")
        
        # 创建两列布局
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self._render_search_results()
        
        with col2:
            self._render_search_sidebar()
    
    def _render_search_input(self):
        """渲染搜索输入区域"""
        st.markdown("### 🔍 知识库搜索")
        
        # 搜索设置
        with st.expander("⚙️ 搜索设置", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                top_k = st.slider(
                    "返回结果数量",
                    min_value=1,
                    max_value=50,
                    value=10,
                    help="控制返回的搜索结果数量"
                )
                
                retriever_type = st.selectbox(
                    "检索器类型",
                    options=["hybrid", "vector", "sparse", "fulltext"],
                    index=0,
                    help="选择检索器类型：混合检索效果最好"
                )
            
            with col2:
                reranker_type = st.selectbox(
                    "重排序器类型",
                    options=["ensemble", "colbert", "cross_encoder", "mmr"],
                    index=0,
                    help="选择重排序器类型：集成重排序效果最好"
                )
                
                enable_multi_query = st.checkbox(
                    "启用多查询检索",
                    value=False,
                    help="启用后会生成多个相关查询进行检索"
                )
        
        # 搜索输入框
        col1, col2 = st.columns([4, 1])
        
        with col1:
            query = st.text_input(
                "搜索查询",
                placeholder="请输入您要搜索的内容...",
                label_visibility="collapsed"
            )
        
        with col2:
            search_clicked = st.button("🔍 搜索", use_container_width=True)
        
        # 搜索建议
        if query and len(query) > 2:
            with st.spinner("获取搜索建议..."):
                success, suggestions = asyncio.run(
                    self.api_client.get_search_suggestions(query, limit=5)
                )
                
                if success and suggestions:
                    st.markdown("**搜索建议:**")
                    suggestion_cols = st.columns(len(suggestions))
                    
                    for i, suggestion in enumerate(suggestions):
                        with suggestion_cols[i]:
                            if st.button(
                                f"💡 {suggestion[:20]}...",
                                key=f"suggestion_{i}",
                                help=suggestion
                            ):
                                self._perform_search(suggestion, top_k, retriever_type, reranker_type)
        
        # 执行搜索
        if search_clicked and query:
            self._perform_search(query, top_k, retriever_type, reranker_type)
    
    def _perform_search(
        self,
        query: str,
        top_k: int = 10,
        retriever_type: str = "hybrid",
        reranker_type: str = "ensemble"
    ):
        """执行搜索"""
        with st.spinner("🔍 搜索中..."):
            success, response = asyncio.run(
                self.api_client.search_knowledge_base(
                    query=query,
                    top_k=top_k,
                    retriever_type=retriever_type,
                    reranker_type=reranker_type
                )
            )
            
            if success and response:
                results = response.get("results", [])
                execution_time = response.get("execution_time", 0)
                
                # 保存搜索结果到会话
                self.session_manager.set_last_search_results(results)
                self.session_manager.add_search_record(
                    query=query,
                    results=results,
                    execution_time=execution_time,
                    retriever_type=retriever_type,
                    reranker_type=reranker_type
                )
                
                st.success(f"✅ 搜索完成！找到 {len(results)} 个结果，耗时 {execution_time:.2f} 秒")
                st.rerun()
            else:
                st.error("搜索失败，请重试")
    
    def _render_search_results(self):
        """渲染搜索结果"""
        results = self.session_manager.get_last_search_results()
        
        if not results:
            st.info("🔍 请输入搜索查询以开始搜索知识库")
            return
        
        st.markdown(f"### 📋 搜索结果 ({len(results)} 个)")
        
        for i, result in enumerate(results, 1):
            with st.container():
                # 结果卡片
                with st.expander(f"📄 结果 {i}: {result.get('title', '无标题')[:50]}...", expanded=i <= 3):
                    # 基本信息
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.markdown(f"**文档:** {result.get('document_name', 'N/A')}")
                    
                    with col2:
                        score = result.get('score', 0)
                        st.metric("相关度", f"{score:.3f}")
                    
                    with col3:
                        chunk_id = result.get('chunk_id', 'N/A')
                        st.markdown(f"**块ID:** {chunk_id}")
                    
                    # 内容预览
                    content = result.get('content', '')
                    if content:
                        st.markdown("**内容预览:**")
                        # 高亮搜索词（简单实现）
                        st.markdown(f"```\n{content[:500]}{'...' if len(content) > 500 else ''}\n```")
                    
                    # 元数据
                    metadata = result.get('metadata', {})
                    if metadata:
                        st.markdown("**元数据:**")
                        st.json(metadata)
                    
                    # 操作按钮
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button(f"💬 基于此内容对话", key=f"chat_{i}"):
                            self._start_chat_with_context(result)
                    
                    with col2:
                        if st.button(f"📋 复制内容", key=f"copy_{i}"):
                            st.code(content)
                    
                    with col3:
                        if st.button(f"🔗 查看原文档", key=f"doc_{i}"):
                            st.info("文档查看功能开发中...")
    
    def _render_search_sidebar(self):
        """渲染搜索侧边栏"""
        st.markdown("### 📊 搜索统计")
        
        # 搜索历史统计
        search_history = self.session_manager.get_search_history()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("搜索次数", len(search_history))
        
        with col2:
            if search_history:
                avg_results = sum(record.get("result_count", 0) for record in search_history) / len(search_history)
                st.metric("平均结果数", f"{avg_results:.1f}")
        
        st.markdown("---")
        
        # 最近搜索
        st.markdown("### 🕒 最近搜索")
        
        if search_history:
            for i, record in enumerate(search_history[:5]):
                query = record.get("query", "")
                timestamp = record.get("timestamp", "")
                result_count = record.get("result_count", 0)
                
                with st.container():
                    st.markdown(f"**{query[:30]}{'...' if len(query) > 30 else ''}**")
                    st.caption(f"{timestamp[:16]} | {result_count} 个结果")
                    
                    if st.button(f"🔄 重新搜索", key=f"repeat_{i}"):
                        self._perform_search(
                            query,
                            record.get("top_k", 10),
                            record.get("retriever_type", "hybrid"),
                            record.get("reranker_type", "ensemble")
                        )
                    
                    st.markdown("---")
        else:
            st.info("暂无搜索历史")
        
        # 清除历史按钮
        if st.button("🗑️ 清除搜索历史", use_container_width=True):
            self.session_manager.search_history = []
            st.success("搜索历史已清除")
            st.rerun()
    
    def _start_chat_with_context(self, result: Dict[str, Any]):
        """基于搜索结果开始对话"""
        content = result.get('content', '')
        title = result.get('title', '搜索结果')
        
        # 构建上下文消息
        context_message = f"基于以下内容回答问题：\n\n**{title}**\n\n{content}\n\n请问："
        
        # 切换到聊天页面
        self.session_manager.set_current_page("chat")
        
        # 添加上下文消息到聊天历史
        self.session_manager.add_chat_message("system", f"已加载搜索结果作为上下文：{title}")
        
        st.success("已切换到聊天页面，并加载搜索结果作为上下文")
        st.rerun()
    
    def export_search_results(self):
        """导出搜索结果"""
        results = self.session_manager.get_last_search_results()
        
        if not results:
            st.warning("没有搜索结果可导出")
            return
        
        # 生成导出内容
        export_content = f"# 搜索结果导出 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        export_content += f"共找到 {len(results)} 个结果\n\n"
        
        for i, result in enumerate(results, 1):
            title = result.get('title', '无标题')
            content = result.get('content', '')
            score = result.get('score', 0)
            document_name = result.get('document_name', 'N/A')
            
            export_content += f"## 结果 {i}: {title}\n\n"
            export_content += f"**文档:** {document_name}\n"
            export_content += f"**相关度:** {score:.3f}\n\n"
            export_content += f"**内容:**\n{content}\n\n"
            export_content += "---\n\n"
        
        # 提供下载
        st.download_button(
            label="📥 下载搜索结果",
            data=export_content,
            file_name=f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )
    
    def render_search_analytics(self):
        """渲染搜索分析"""
        st.markdown("### 📈 搜索分析")
        
        search_history = self.session_manager.get_search_history()
        
        if not search_history:
            st.info("暂无搜索数据")
            return
        
        # 搜索趋势
        col1, col2 = st.columns(2)
        
        with col1:
            # 检索器使用统计
            retriever_counts = {}
            for record in search_history:
                retriever = record.get("retriever_type", "unknown")
                retriever_counts[retriever] = retriever_counts.get(retriever, 0) + 1
            
            st.markdown("**检索器使用统计:**")
            for retriever, count in retriever_counts.items():
                st.markdown(f"- {retriever}: {count} 次")
        
        with col2:
            # 重排序器使用统计
            reranker_counts = {}
            for record in search_history:
                reranker = record.get("reranker_type", "unknown")
                reranker_counts[reranker] = reranker_counts.get(reranker, 0) + 1
            
            st.markdown("**重排序器使用统计:**")
            for reranker, count in reranker_counts.items():
                st.markdown(f"- {reranker}: {count} 次")
        
        # 平均执行时间
        execution_times = [
            record.get("execution_time", 0) 
            for record in search_history 
            if record.get("execution_time")
        ]
        
        if execution_times:
            avg_time = sum(execution_times) / len(execution_times)
            st.metric("平均搜索时间", f"{avg_time:.3f} 秒")
