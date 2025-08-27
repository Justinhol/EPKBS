"""
文档管理组件
提供文档上传、管理和预览功能
"""
import streamlit as st
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import os
import hashlib

from src.frontend.utils.api_client import APIClient
from src.frontend.utils.session import SessionManager
from src.utils.logger import get_logger

logger = get_logger("frontend.documents")


class DocumentManager:
    """文档管理组件"""

    def __init__(self, api_client: APIClient):
        self.api_client = api_client
        self.session_manager = SessionManager()

        # 支持的文件类型
        self.supported_types = {
            "pdf": "📄 PDF文档",
            "doc": "📝 Word文档(旧版)",
            "docx": "📝 Word文档",
            "ppt": "📊 PowerPoint(旧版)",
            "pptx": "📊 PowerPoint文档",
            "xls": "📈 Excel(旧版)",
            "xlsx": "📈 Excel文档",
            "html": "🌐 HTML文档",
            "md": "📋 Markdown文件",
            "xml": "🔧 XML文件",
            "epub": "📚 电子书",
            "txt": "📃 文本文件",
            "eml": "📧 邮件文件",
            "msg": "📧 Outlook邮件",
            "csv": "📊 CSV文件",
            "json": "🔧 JSON文件",
            "jpg": "🖼️ JPEG图像",
            "jpeg": "🖼️ JPEG图像",
            "png": "🖼️ PNG图像",
            "tiff": "🖼️ TIFF图像",
            "tif": "🖼️ TIFF图像"
        }

        logger.info("文档管理组件初始化完成")

    def render(self):
        """渲染文档管理界面"""
        # 创建标签页
        tab1, tab2, tab3 = st.tabs(["📤 文档上传", "📚 文档列表", "📊 统计信息"])

        with tab1:
            self._render_upload_section()

        with tab2:
            self._render_document_list()

        with tab3:
            self._render_statistics()

    def _render_upload_section(self):
        """渲染文档上传部分"""
        st.markdown("### 📤 上传文档")

        # 上传说明
        with st.expander("📋 上传说明", expanded=False):
            st.markdown("""
            **支持的文件格式:**
            - PDF文档 (.pdf) - 支持文本版和扫描版
            - Word文档 (.doc, .docx) - 支持表格、图片提取
            - PowerPoint文档 (.ppt, .pptx) - 支持幻灯片内容提取
            - Excel文档 (.xls, .xlsx) - 支持表格和公式提取
            - 网页文档 (.html) - 支持结构化内容提取
            - 标记语言 (.md, .xml) - 支持格式化内容
            - 电子书 (.epub) - 支持章节内容提取
            - 文本文件 (.txt) - 基础文本处理
            - 邮件文件 (.eml, .msg) - 支持邮件内容提取
            - 数据文件 (.csv, .json) - 支持结构化数据
            - 图像文件 (.jpg, .png, .tiff) - 支持OCR文字识别
            
            **文件要求:**
            - 单个文件大小不超过 50MB
            - 文件名不能包含特殊字符
            - 建议使用有意义的文件名
            
            **智能处理流程:**
            1. 文档类型检查和验证
            2. 多模态内容解析（文本、图片、表格、公式）
            3. 内容整合和上下文保留
            4. 智能清洗和格式化
            5. 三步分块策略（结构→语义→递归）
            6. 向量化和索引构建
            """)

        # 文件上传器
        uploaded_files = st.file_uploader(
            "选择要上传的文件",
            type=list(self.supported_types.keys()),
            accept_multiple_files=True,
            help="可以同时选择多个文件进行批量上传"
        )

        if uploaded_files:
            st.markdown(f"**已选择 {len(uploaded_files)} 个文件:**")

            # 显示文件信息
            total_size = 0
            for file in uploaded_files:
                file_size = len(file.getvalue())
                total_size += file_size

                col1, col2, col3 = st.columns([3, 1, 1])

                with col1:
                    file_type = file.name.split('.')[-1].lower()
                    type_icon = self.supported_types.get(file_type, "📄")
                    st.markdown(f"{type_icon} {file.name}")

                with col2:
                    st.markdown(f"{self._format_file_size(file_size)}")

                with col3:
                    st.markdown(f".{file_type}")

            st.markdown(f"**总大小:** {self._format_file_size(total_size)}")

            # 上传设置
            with st.expander("⚙️ 上传设置", expanded=False):
                col1, col2 = st.columns(2)

                with col1:
                    auto_process = st.checkbox(
                        "自动处理",
                        value=True,
                        help="上传后自动进行文本提取和向量化"
                    )

                    overwrite_existing = st.checkbox(
                        "覆盖同名文件",
                        value=False,
                        help="如果存在同名文件，是否覆盖"
                    )

                with col2:
                    chunk_size = st.slider(
                        "分块大小",
                        min_value=100,
                        max_value=2000,
                        value=500,
                        step=100,
                        help="文档分块的字符数"
                    )

                    chunk_overlap = st.slider(
                        "分块重叠",
                        min_value=0,
                        max_value=200,
                        value=50,
                        step=10,
                        help="相邻分块的重叠字符数"
                    )

            # 上传按钮
            col1, col2, col3 = st.columns([1, 2, 1])

            with col2:
                if st.button("🚀 开始上传", use_container_width=True, type="primary"):
                    self._upload_files(
                        uploaded_files,
                        auto_process=auto_process,
                        overwrite_existing=overwrite_existing,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )

    def _render_document_list(self):
        """渲染文档列表"""
        st.markdown("### 📚 文档列表")

        # 搜索和过滤
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            search_query = st.text_input(
                "搜索文档",
                placeholder="输入文档名称或关键词...",
                label_visibility="collapsed"
            )

        with col2:
            status_filter = st.selectbox(
                "状态筛选",
                options=["全部", "处理中", "已完成", "失败"],
                index=0
            )

        with col3:
            sort_by = st.selectbox(
                "排序方式",
                options=["上传时间", "文件名", "文件大小"],
                index=0
            )

        # 获取文档列表
        documents = self.session_manager.get_uploaded_documents()

        # 模拟文档数据（实际应该从API获取）
        if not documents:
            documents = self._get_mock_documents()
            self.session_manager.uploaded_documents = documents

        # 应用过滤和搜索
        filtered_docs = self._filter_documents(
            documents, search_query, status_filter)

        if not filtered_docs:
            st.info("📭 暂无文档，请先上传文档")
            return

        # 显示文档列表
        for doc in filtered_docs:
            with st.container():
                self._render_document_card(doc)
                st.markdown("---")

    def _render_document_card(self, document: Dict[str, Any]):
        """渲染文档卡片"""
        doc_id = document.get("id", 0)
        filename = document.get("filename", "未知文件")
        file_size = document.get("file_size", 0)
        status = document.get("status", "unknown")
        created_at = document.get("created_at", "")
        chunk_count = document.get("chunk_count", 0)
        vector_count = document.get("vector_count", 0)

        # 文档基本信息
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

        with col1:
            file_type = filename.split('.')[-1].lower()
            type_icon = self.supported_types.get(file_type, "📄")
            st.markdown(f"### {type_icon} {filename}")
            st.caption(f"上传时间: {created_at[:19] if created_at else 'N/A'}")

        with col2:
            st.metric("文件大小", self._format_file_size(file_size))

        with col3:
            st.metric("分块数", chunk_count)

        with col4:
            st.metric("向量数", vector_count)

        # 状态显示
        if status == "completed":
            st.success("✅ 处理完成")
        elif status == "processing":
            st.info("🔄 处理中...")
        elif status == "failed":
            st.error("❌ 处理失败")
        else:
            st.warning("⏳ 等待处理")

        # 操作按钮
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            if st.button("👁️ 预览", key=f"preview_{doc_id}"):
                self._preview_document(document)

        with col2:
            if st.button("📊 详情", key=f"details_{doc_id}"):
                self._show_document_details(document)

        with col3:
            if st.button("🔄 重新处理", key=f"reprocess_{doc_id}"):
                self._reprocess_document(document)

        with col4:
            if st.button("📥 下载", key=f"download_{doc_id}"):
                self._download_document(document)

        with col5:
            if st.button("🗑️ 删除", key=f"delete_{doc_id}"):
                self._delete_document(document)

    def _render_statistics(self):
        """渲染统计信息"""
        st.markdown("### 📊 文档统计")

        documents = self.session_manager.get_uploaded_documents()

        if not documents:
            st.info("暂无统计数据")
            return

        # 基本统计
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("文档总数", len(documents))

        with col2:
            total_size = sum(doc.get("file_size", 0) for doc in documents)
            st.metric("总大小", self._format_file_size(total_size))

        with col3:
            total_chunks = sum(doc.get("chunk_count", 0) for doc in documents)
            st.metric("总分块数", total_chunks)

        with col4:
            total_vectors = sum(doc.get("vector_count", 0)
                                for doc in documents)
            st.metric("总向量数", total_vectors)

        st.markdown("---")

        # 状态统计
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📈 处理状态分布")
            status_counts = {}
            for doc in documents:
                status = doc.get("status", "unknown")
                status_counts[status] = status_counts.get(status, 0) + 1

            for status, count in status_counts.items():
                percentage = (count / len(documents)) * 100
                st.markdown(f"- {status}: {count} ({percentage:.1f}%)")

        with col2:
            st.markdown("#### 📁 文件类型分布")
            type_counts = {}
            for doc in documents:
                filename = doc.get("filename", "")
                file_type = filename.split(
                    '.')[-1].lower() if '.' in filename else "unknown"
                type_counts[file_type] = type_counts.get(file_type, 0) + 1

            for file_type, count in type_counts.items():
                percentage = (count / len(documents)) * 100
                type_name = self.supported_types.get(
                    file_type, f".{file_type}")
                st.markdown(f"- {type_name}: {count} ({percentage:.1f}%)")

    def _upload_files(
        self,
        files: List,
        auto_process: bool = True,
        overwrite_existing: bool = False,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """上传文件"""
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, file in enumerate(files):
            try:
                status_text.text(f"正在上传: {file.name}")

                # 模拟上传过程
                import time
                time.sleep(1)  # 模拟上传时间

                # 创建文档记录
                document = {
                    "id": len(self.session_manager.get_uploaded_documents()) + i + 1,
                    "filename": file.name,
                    "original_filename": file.name,
                    "file_size": len(file.getvalue()),
                    "file_type": file.name.split('.')[-1].lower(),
                    "status": "processing" if auto_process else "pending",
                    "created_at": datetime.now().isoformat(),
                    "chunk_count": 0,
                    "vector_count": 0
                }

                # 添加到会话
                self.session_manager.add_uploaded_document(document)

                # 更新进度
                progress = (i + 1) / len(files)
                progress_bar.progress(progress)

            except Exception as e:
                st.error(f"上传 {file.name} 失败: {str(e)}")

        status_text.text("上传完成！")
        st.success(f"成功上传 {len(files)} 个文件")

        # 清理进度显示
        progress_bar.empty()
        status_text.empty()

        st.rerun()

    def _filter_documents(
        self,
        documents: List[Dict[str, Any]],
        search_query: str,
        status_filter: str
    ) -> List[Dict[str, Any]]:
        """过滤文档"""
        filtered = documents

        # 搜索过滤
        if search_query:
            filtered = [
                doc for doc in filtered
                if search_query.lower() in doc.get("filename", "").lower()
            ]

        # 状态过滤
        if status_filter != "全部":
            status_map = {
                "处理中": "processing",
                "已完成": "completed",
                "失败": "failed"
            }
            target_status = status_map.get(status_filter)
            if target_status:
                filtered = [
                    doc for doc in filtered
                    if doc.get("status") == target_status
                ]

        return filtered

    def _get_mock_documents(self) -> List[Dict[str, Any]]:
        """获取模拟文档数据"""
        return [
            {
                "id": 1,
                "filename": "AI技术报告.pdf",
                "original_filename": "AI技术报告.pdf",
                "file_size": 2048576,
                "file_type": "pdf",
                "status": "completed",
                "created_at": "2024-01-01T10:00:00",
                "chunk_count": 45,
                "vector_count": 45
            },
            {
                "id": 2,
                "filename": "产品需求文档.docx",
                "original_filename": "产品需求文档.docx",
                "file_size": 1024000,
                "file_type": "docx",
                "status": "processing",
                "created_at": "2024-01-01T11:00:00",
                "chunk_count": 0,
                "vector_count": 0
            }
        ]

    def _format_file_size(self, size_bytes: int) -> str:
        """格式化文件大小"""
        if size_bytes == 0:
            return "0 B"

        size_names = ["B", "KB", "MB", "GB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1

        return f"{size_bytes:.1f} {size_names[i]}"

    def _preview_document(self, document: Dict[str, Any]):
        """预览文档"""
        st.info("文档预览功能开发中...")

    def _show_document_details(self, document: Dict[str, Any]):
        """显示文档详情"""
        st.info("文档详情功能开发中...")

    def _reprocess_document(self, document: Dict[str, Any]):
        """重新处理文档"""
        st.info("重新处理功能开发中...")

    def _download_document(self, document: Dict[str, Any]):
        """下载文档"""
        st.info("文档下载功能开发中...")

    def _delete_document(self, document: Dict[str, Any]):
        """删除文档"""
        doc_id = document.get("id")
        if st.session_state.get(f"confirm_delete_doc_{doc_id}"):
            self.session_manager.remove_uploaded_document(doc_id)
            st.success("文档删除成功")
            st.rerun()
        else:
            st.session_state[f"confirm_delete_doc_{doc_id}"] = True
            st.warning("再次点击确认删除")
