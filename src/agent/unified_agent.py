"""
简化的统一Agent架构
移除策略模式，提供智能的单一执行接口
"""
import asyncio
import re
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum

from src.agent.llm import LLMManager
from src.agent.langgraph_agent import LangGraphAgent
from src.epkbs_mcp.clients.mcp_client import MCPClientManager
from src.rag.core import RAGCore
from src.utils.logger import get_logger
from config.settings import settings

logger = get_logger("agent.simplified")


class AgentMode(Enum):
    """Agent模式"""
    STANDARD = "standard"        # 标准模式（LangGraph + RAG）
    TOOL_ENHANCED = "tool"       # 工具增强模式（+ MCP工具）


class TaskType(Enum):
    """任务类型"""
    DOCUMENT_PROCESSING = "document_processing"
    QUESTION_ANSWERING = "question_answering"
    TOOL_CALLING = "tool_calling"
    WORKFLOW_EXECUTION = "workflow_execution"
    BATCH_PROCESSING = "batch_processing"


class SimplifiedAgent:
    """简化的统一Agent

    智能Agent，自动选择最佳执行方式：
    - 基于LangGraph的工作流引擎
    - 智能MCP工具调用
    - RAG增强的问答能力
    """

    def __init__(
        self,
        llm_manager: LLMManager = None,
        mcp_client_manager: MCPClientManager = None,
        rag_core: RAGCore = None,
        mode: AgentMode = AgentMode.TOOL_ENHANCED
    ):
        self.llm_manager = llm_manager
        self.mcp_client_manager = mcp_client_manager
        self.rag_core = rag_core
        self.mode = mode

        # 核心Agent实例
        self.langgraph_agent = None

        # 工具相关
        self.available_tools = {}
        self.tools_initialized = False

        # 任务分析模式
        self.task_patterns = {
            "document": [r"解析", r"文档", r"parse", r"document", r"pdf", r"docx"],
            "search": [r"搜索", r"查找", r"search", r"find", r"什么是", r"介绍"],
            "tool": [r"调用", r"使用工具", r"tool", r"执行", r"运行"],
            "batch": [r"批量", r"多个", r"batch", r"批处理"]
        }

        # 性能统计
        self.stats = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "average_execution_time": 0.0,
            "tool_calls": 0,
            "rag_queries": 0,
            "document_processed": 0
        }

        logger.info(f"简化Agent初始化，模式: {mode.value}")

    async def initialize(self):
        """初始化Agent系统"""
        logger.info("正在初始化简化Agent系统...")

        try:
            # 初始化LLM管理器
            if not self.llm_manager:
                self.llm_manager = LLMManager()
                await self.llm_manager.initialize()

            # 初始化MCP客户端管理器（如果启用工具模式）
            if self.mode == AgentMode.TOOL_ENHANCED and not self.mcp_client_manager:
                self.mcp_client_manager = MCPClientManager()
                await self.mcp_client_manager.initialize()

            # 初始化RAG核心
            if not self.rag_core:
                self.rag_core = RAGCore()
                await self.rag_core.initialize()

            # 初始化LangGraph Agent
            self.langgraph_agent = LangGraphAgent(
                llm_manager=self.llm_manager,
                mcp_client=self.mcp_client_manager,
                rag_core=self.rag_core
            )
            await self.langgraph_agent.initialize()

            # 初始化工具（如果启用）
            if self.mode == AgentMode.TOOL_ENHANCED:
                await self._initialize_tools()

            logger.info("简化Agent系统初始化完成")

        except Exception as e:
            logger.error(f"简化Agent系统初始化失败: {e}")
            raise

    async def _initialize_tools(self):
        """初始化MCP工具"""
        if self.mcp_client_manager and not self.tools_initialized:
            try:
                self.available_tools = await self.mcp_client_manager.get_available_tools()
                self.tools_initialized = True
                logger.info(f"初始化了 {len(self.available_tools)} 个MCP工具")
            except Exception as e:
                logger.warning(f"MCP工具初始化失败: {e}")
                self.tools_initialized = False

    def _analyze_task(self, task: str) -> TaskType:
        """智能分析任务类型"""
        task_lower = task.lower()

        # 检查文档处理
        if any(re.search(pattern, task_lower) for pattern in self.task_patterns["document"]):
            return TaskType.DOCUMENT_PROCESSING

        # 检查批量处理
        if any(re.search(pattern, task_lower) for pattern in self.task_patterns["batch"]):
            return TaskType.BATCH_PROCESSING

        # 检查工具调用
        if any(re.search(pattern, task_lower) for pattern in self.task_patterns["tool"]):
            return TaskType.TOOL_CALLING

        # 检查搜索查询
        if any(re.search(pattern, task_lower) for pattern in self.task_patterns["search"]):
            return TaskType.QUESTION_ANSWERING

        # 默认为问答
        return TaskType.QUESTION_ANSWERING

    def _needs_tools(self, task: str, task_type: TaskType) -> bool:
        """判断是否需要工具调用"""
        if self.mode != AgentMode.TOOL_ENHANCED:
            return False

        if not self.tools_initialized:
            return False

        # 文档处理和工具调用明确需要工具
        if task_type in [TaskType.DOCUMENT_PROCESSING, TaskType.TOOL_CALLING]:
            return True

        # 检查是否提到特定工具
        task_lower = task.lower()
        tool_keywords = ["搜索", "search", "解析", "parse", "分析", "analyze"]
        return any(keyword in task_lower for keyword in tool_keywords)

    async def execute_task(
        self,
        task: str,
        context: Dict[str, Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """执行任务 - 智能选择执行方式"""

        start_time = datetime.utcnow()
        self.stats["total_tasks"] += 1

        logger.info(f"执行任务: {task[:50]}...")

        try:
            # 分析任务类型
            task_type = self._analyze_task(task)
            needs_tools = self._needs_tools(task, task_type)

            logger.debug(f"任务类型: {task_type.value}, 需要工具: {needs_tools}")

            # 准备执行上下文
            execution_context = {
                "task": task,
                "task_type": task_type.value,
                "needs_tools": needs_tools,
                "timestamp": start_time.isoformat(),
                "context": context or {},
                **kwargs
            }

            # 执行任务
            result = await self._execute_with_langgraph(execution_context)

            # 更新统计
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            self._update_stats(execution_time, True, task_type, needs_tools)

            return result

        except Exception as e:
            logger.error(f"任务执行失败: {e}")
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            self._update_stats(execution_time, False,
                               TaskType.QUESTION_ANSWERING, False)

            return {
                "success": False,
                "error": str(e),
                "task_type": task_type.value if 'task_type' in locals() else "unknown",
                "timestamp": datetime.utcnow().isoformat()
            }

    async def _execute_with_langgraph(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """使用LangGraph执行任务"""
        task = context["task"]
        task_type = context["task_type"]

        # 根据任务类型选择LangGraph工作流
        if task_type == TaskType.DOCUMENT_PROCESSING.value:
            if "file_path" in context:
                return await self.langgraph_agent.parse_document_with_langgraph(
                    context["file_path"]
                )
            else:
                return await self.langgraph_agent.chat_with_langgraph(
                    task, use_rag=True
                )

        elif task_type == TaskType.BATCH_PROCESSING.value:
            if "file_paths" in context:
                return await self.langgraph_agent.batch_process_documents_with_langgraph(
                    context["file_paths"]
                )

        elif task_type == TaskType.WORKFLOW_EXECUTION.value:
            return await self.langgraph_agent.execute_workflow(
                task, context.get("workflow_type", "agent_reasoning")
            )

        # 默认使用对话模式
        use_rag = context.get("use_rag", True)
        return await self.langgraph_agent.chat_with_langgraph(task, use_rag=use_rag)

    def _update_stats(self, execution_time: float, success: bool, task_type: TaskType, used_tools: bool):
        """更新统计信息"""
        if success:
            self.stats["successful_tasks"] += 1
        else:
            self.stats["failed_tasks"] += 1

        # 更新平均执行时间
        current_avg = self.stats["average_execution_time"]
        total_tasks = self.stats["total_tasks"]

        if total_tasks > 1:
            self.stats["average_execution_time"] = (
                (current_avg * (total_tasks - 1) + execution_time) / total_tasks
            )
        else:
            self.stats["average_execution_time"] = execution_time

        # 更新特定统计
        if used_tools:
            self.stats["tool_calls"] += 1

        if task_type == TaskType.QUESTION_ANSWERING:
            self.stats["rag_queries"] += 1
        elif task_type == TaskType.DOCUMENT_PROCESSING:
            self.stats["document_processed"] += 1

    async def get_capabilities(self) -> List[str]:
        """获取Agent能力列表"""
        capabilities = [
            "question_answering",
            "document_processing",
            "workflow_execution"
        ]

        if self.mode == AgentMode.TOOL_ENHANCED and self.tools_initialized:
            capabilities.extend(["tool_calling", "batch_processing"])

        return capabilities

    async def get_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        stats = self.stats.copy()
        stats["timestamp"] = datetime.utcnow().isoformat()
        stats["mode"] = self.mode.value
        stats["tools_available"] = len(self.available_tools)

        return stats

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        health_status = {
            "agent": "healthy" if self.langgraph_agent else "not_initialized",
            "mode": self.mode.value,
            "tools_initialized": self.tools_initialized,
            "timestamp": datetime.utcnow().isoformat()
        }

        return health_status


# 为了向后兼容，创建别名
UnifiedAgent = SimplifiedAgent

# 保持旧的枚举名称以兼容现有代码
AgentStrategy = AgentMode  # 策略模式 -> 模式
AgentCapability = TaskType  # 能力 -> 任务类型
