"""
基于LangGraph的新Agent实现
使用状态图和工作流进行智能推理和任务执行
"""
import asyncio
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent, ToolExecutor
from langchain.tools import BaseTool
from langchain.schema import HumanMessage, AIMessage

from .workflows import WorkflowManager
from .states import (
    create_document_processing_state,
    create_agent_reasoning_state,
    create_rag_workflow_state
)
from ..epkbs_mcp.clients.mcp_client import MCPClientManager
from ..agent.llm import LLMManager
from ..utils.logger import get_logger

logger = get_logger("agent.langgraph_agent")


class MCPToolWrapper(BaseTool):
    """MCP工具的LangChain包装器"""

    def __init__(self, tool_key: str, tool_info: Dict[str, Any], mcp_client: MCPClientManager):
        self.tool_key = tool_key
        self.tool_info = tool_info
        self.mcp_client = mcp_client

        super().__init__(
            name=tool_key.replace(".", "_"),  # LangChain工具名不能包含点
            description=tool_info.get("description", ""),
        )

    def _run(self, **kwargs) -> str:
        """同步运行（LangChain要求）"""
        # 这里需要在异步环境中运行
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 如果已经在异步环境中，创建新的任务
            task = asyncio.create_task(self._arun(**kwargs))
            return str(task.result())
        else:
            return asyncio.run(self._arun(**kwargs))

    async def _arun(self, **kwargs) -> str:
        """异步运行"""
        try:
            result = await self.mcp_client.call_tool(self.tool_key, kwargs)

            if result.get("success"):
                return str(result.get("result", ""))
            else:
                return f"工具执行失败: {result.get('error', '未知错误')}"

        except Exception as e:
            return f"工具执行异常: {str(e)}"


class LangGraphAgent:
    """基于LangGraph的智能Agent"""

    def __init__(self, llm_manager: LLMManager, mcp_client_manager: MCPClientManager):
        self.llm_manager = llm_manager
        self.mcp_client = mcp_client_manager
        self.workflow_manager = WorkflowManager(
            llm_manager, mcp_client_manager)

        # LangGraph Agent
        self.react_agent = None
        self.available_tools = {}
        self.langchain_tools = []

        logger.info("LangGraph Agent初始化完成")

    async def initialize(self):
        """初始化Agent"""
        try:
            # 发现MCP工具
            self.available_tools = await self.mcp_client.get_available_tools()

            # 将MCP工具包装为LangChain工具
            self.langchain_tools = []
            for tool_key, tool_info in self.available_tools.items():
                wrapped_tool = MCPToolWrapper(
                    tool_key, tool_info, self.mcp_client)
                self.langchain_tools.append(wrapped_tool)

            # 创建LangGraph ReAct Agent
            llm = await self.llm_manager.get_llm()
            self.react_agent = create_react_agent(
                llm,
                self.langchain_tools,
                state_modifier="你是一个专业的AI助手，能够使用各种工具完成复杂任务。请按照ReAct格式进行推理：Thought -> Action -> Observation -> ..."
            )

            logger.info(
                f"LangGraph Agent初始化完成，可用工具: {len(self.langchain_tools)} 个")

        except Exception as e:
            logger.error(f"LangGraph Agent初始化失败: {e}")
            raise

    async def chat(self, message: str, use_rag: bool = True) -> Dict[str, Any]:
        """智能对话"""
        start_time = time.time()

        try:
            logger.info(f"LangGraph Agent对话: {message}")

            if use_rag:
                # 使用RAG工作流
                rag_result = await self.workflow_manager.rag_search(message)

                if rag_result.get("success"):
                    # 基于RAG结果生成回答
                    context = rag_result.get("context", "")
                    enhanced_message = f"""
基于以下上下文信息回答用户问题：

上下文：
{context}

用户问题：{message}

请基于上下文信息给出准确、详细的回答。
"""

                    # 使用LangGraph Agent处理
                    agent_result = await self._run_react_agent(enhanced_message)

                    return {
                        "success": True,
                        "response": agent_result.get("response", ""),
                        "mode": "rag_enhanced",
                        "rag_context": context,
                        "rag_sources": rag_result.get("results", []),
                        "reasoning_steps": agent_result.get("reasoning_steps", []),
                        "execution_time": time.time() - start_time
                    }
                else:
                    # RAG失败，降级到直接对话
                    logger.warning("RAG检索失败，降级到直接对话")
                    agent_result = await self._run_react_agent(message)

                    return {
                        "success": True,
                        "response": agent_result.get("response", ""),
                        "mode": "direct",
                        "rag_error": rag_result.get("error"),
                        "reasoning_steps": agent_result.get("reasoning_steps", []),
                        "execution_time": time.time() - start_time
                    }
            else:
                # 直接对话模式
                agent_result = await self._run_react_agent(message)

                return {
                    "success": True,
                    "response": agent_result.get("response", ""),
                    "mode": "direct",
                    "reasoning_steps": agent_result.get("reasoning_steps", []),
                    "execution_time": time.time() - start_time
                }

        except Exception as e:
            logger.error(f"LangGraph Agent对话失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": "抱歉，我遇到了一些问题，无法回答您的问题。",
                "execution_time": time.time() - start_time
            }

    async def _run_react_agent(self, message: str) -> Dict[str, Any]:
        """运行ReAct Agent"""
        try:
            # 准备输入
            initial_state = {
                "messages": [HumanMessage(content=message)]
            }

            # 执行Agent
            result = await self.react_agent.ainvoke(initial_state)

            # 提取响应
            messages = result.get("messages", [])
            if messages:
                last_message = messages[-1]
                if isinstance(last_message, AIMessage):
                    response = last_message.content
                else:
                    response = str(last_message)
            else:
                response = "无法生成回答"

            # 提取推理步骤（从中间消息中）
            reasoning_steps = []
            for msg in messages[1:-1]:  # 跳过第一个用户消息和最后一个AI回答
                if hasattr(msg, 'additional_kwargs'):
                    tool_calls = msg.additional_kwargs.get('tool_calls', [])
                    for tool_call in tool_calls:
                        reasoning_steps.append({
                            "tool": tool_call.get('function', {}).get('name'),
                            "arguments": tool_call.get('function', {}).get('arguments'),
                            "result": "执行完成"  # 简化处理
                        })

            return {
                "response": response,
                "reasoning_steps": reasoning_steps,
                "total_messages": len(messages)
            }

        except Exception as e:
            logger.error(f"ReAct Agent执行失败: {e}")
            return {
                "response": f"执行失败: {str(e)}",
                "reasoning_steps": [],
                "total_messages": 0
            }

    async def parse_document(self, file_path: str) -> Dict[str, Any]:
        """智能文档解析"""
        logger.info(f"LangGraph文档解析: {file_path}")

        try:
            # 使用文档处理工作流
            result = await self.workflow_manager.process_document(file_path)

            return result

        except Exception as e:
            logger.error(f"LangGraph文档解析失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def batch_process_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """批量处理文档"""
        logger.info(f"批量处理 {len(file_paths)} 个文档")

        start_time = time.time()
        results = []

        try:
            # 并行处理文档
            tasks = []
            for file_path in file_paths:
                task = self.parse_document(file_path)
                tasks.append(task)

            # 等待所有任务完成
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # 处理结果
            successful = 0
            failed = 0

            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    results.append({
                        "file_path": file_paths[i],
                        "success": False,
                        "error": str(result)
                    })
                    failed += 1
                else:
                    results.append({
                        "file_path": file_paths[i],
                        **result
                    })
                    if result.get("success"):
                        successful += 1
                    else:
                        failed += 1

            execution_time = time.time() - start_time

            return {
                "success": True,
                "total_files": len(file_paths),
                "successful": successful,
                "failed": failed,
                "success_rate": successful / len(file_paths) if file_paths else 0,
                "results": results,
                "execution_time": execution_time,
                "throughput": len(file_paths) / execution_time if execution_time > 0 else 0
            }

        except Exception as e:
            logger.error(f"批量文档处理失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "results": results
            }

    def get_workflow_visualization(self, workflow_type: str) -> Optional[str]:
        """获取工作流可视化"""
        return self.workflow_manager.visualize_workflow(workflow_type)

    def get_statistics(self) -> Dict[str, Any]:
        """获取Agent统计信息"""
        return {
            "available_tools": len(self.available_tools),
            "langchain_tools": len(self.langchain_tools),
            "workflow_stats": self.workflow_manager.get_workflow_stats(),
            "agent_type": "LangGraph"
        }
