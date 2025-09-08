"""
Agent核心管理器 - 重构为统一架构
整合统一Agent、模型管理和缓存系统
"""
import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime

from src.agent.unified_agent import UnifiedAgent, AgentMode, TaskType
from src.agent.llm import LLMManager
from src.rag.core import RAGCore
from src.epkbs_mcp.clients.mcp_client import MCPClientManager
from src.utils.model_manager import get_model_pool, ModelType
from src.utils.cache_manager import get_cache_manager
from src.utils.logger import get_logger
from config.settings import settings

logger = get_logger("agent.core")


class AgentCore:
    """重构的Agent核心管理器 - 统一架构"""

    def __init__(
        self,
        model_name: str = None,
        rag_core: RAGCore = None,
        max_steps: int = 10,
        max_execution_time: int = 300,
        enable_mcp: bool = True,
        agent_mode: AgentMode = AgentMode.TOOL_ENHANCED
    ):
        self.model_name = model_name or settings.model.qwen_model_path
        self.rag_core = rag_core
        self.enable_mcp = enable_mcp
        self.max_steps = max_steps
        self.max_execution_time = max_execution_time
        self.agent_mode = agent_mode

        # 统一组件
        self.unified_agent = None
        self.llm_manager = None
        self.mcp_client_manager = None
        self.model_pool = None
        self.cache_manager = None

        # 性能统计
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "model_loads": 0,
            "average_response_time": 0.0
        }

        # 状态
        self.is_initialized = False
        self.conversation_history = []

        logger.info(
            f"统一Agent核心管理器初始化: 模型={self.model_name}, 策略={default_strategy.value}")

    async def initialize(self):
        """初始化统一Agent系统"""
        if self.is_initialized:
            logger.info("Agent系统已初始化")
            return

        logger.info("正在初始化统一Agent核心系统...")

        try:
            # 1. 初始化模型池
            logger.info("初始化模型池...")
            self.model_pool = get_model_pool()
            await self.model_pool.start_cleanup_task()

            # 2. 初始化缓存管理器
            logger.info("初始化缓存管理器...")
            self.cache_manager = await get_cache_manager()

            # 3. 初始化LLM管理器
            logger.info("初始化LLM管理器...")
            self.llm_manager = LLMManager(
                model_pool=self.model_pool,
                cache_manager=self.cache_manager
            )
            await self.llm_manager.initialize()

            # 4. 初始化MCP客户端管理器（如果启用）
            if self.enable_mcp:
                logger.info("初始化MCP客户端管理器...")
                self.mcp_client_manager = MCPClientManager()
                await self.mcp_client_manager.initialize()

            # 5. 初始化RAG核心（如果未提供）
            if not self.rag_core:
                logger.info("初始化RAG核心...")
                self.rag_core = RAGCore()
                await self.rag_core.initialize()

            # 6. 初始化统一Agent
            logger.info("初始化统一Agent...")
            self.unified_agent = UnifiedAgent(
                llm_manager=self.llm_manager,
                mcp_client_manager=self.mcp_client_manager,
                rag_core=self.rag_core,
                mode=self.agent_mode
            )
            await self.unified_agent.initialize()

            self.is_initialized = True
            logger.info("统一Agent核心系统初始化完成")

        except Exception as e:
            logger.error(f"统一Agent核心系统初始化失败: {e}")
            raise

    async def chat(
        self,
        message: str,
        use_rag: bool = True,
        stream: bool = False,
        task_type: TaskType = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        与统一Agent对话

        Args:
            message: 用户消息
            use_rag: 是否使用RAG
            stream: 是否流式响应
            task_type: 任务类型
            **kwargs: 其他参数

        Returns:
            对话结果
        """
        if not self.is_initialized:
            await self.initialize()

        start_time = datetime.utcnow()
        self.stats["total_requests"] += 1

        logger.info(f"统一Agent对话开始: {message[:100]}...")

        try:
            # 准备上下文
            context = {
                "use_rag": use_rag,
                "stream": stream,
                # 最近10轮对话
                "conversation_history": self.conversation_history[-10:],
                **kwargs
            }

            # 使用简化的统一Agent执行任务
            result = await self.unified_agent.execute_task(
                task=message,
                context=context
            )

            # 更新对话历史
            self.conversation_history.append({
                'role': 'user',
                'content': message,
                'timestamp': start_time.isoformat()
            })

            if result.get("success", True):
                self.conversation_history.append({
                    'role': 'assistant',
                    'content': result.get("response", result.get("result", "")),
                    'timestamp': datetime.utcnow().isoformat(),
                    'task_type': result.get("task_type", "unknown")
                })
                self.stats["successful_requests"] += 1
            else:
                self.stats["failed_requests"] += 1

            # 更新响应时间统计
            response_time = (datetime.utcnow() - start_time).total_seconds()
            self._update_response_time_stats(response_time)

            return result

        except Exception as e:
            logger.error(f"统一Agent对话失败: {e}")
            self.stats["failed_requests"] += 1
            return {
                'success': False,
                'response': f"对话过程中发生错误: {str(e)}",
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }

    async def parse_document(
        self,
        file_path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """解析文档"""
        if not self.is_initialized:
            await self.initialize()

        logger.info(f"开始解析文档: {file_path}")

        try:
            context = {
                "file_path": file_path,
                **kwargs
            }

            result = await self.unified_agent.execute_task(
                task=f"解析文档: {file_path}",
                context=context
            )

            return result

        except Exception as e:
            logger.error(f"文档解析失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }

    async def batch_process_documents(
        self,
        file_paths: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """批量处理文档"""
        if not self.is_initialized:
            await self.initialize()

        logger.info(f"开始批量处理 {len(file_paths)} 个文档")

        try:
            context = {
                "file_paths": file_paths,
                **kwargs
            }

            result = await self.unified_agent.execute_task(
                task=f"批量处理 {len(file_paths)} 个文档",
                context=context
            )

            return result

        except Exception as e:
            logger.error(f"批量文档处理失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }

    def _update_response_time_stats(self, response_time: float):
        """更新响应时间统计"""
        current_avg = self.stats["average_response_time"]
        total_requests = self.stats["total_requests"]

        # 计算新的平均响应时间
        if total_requests > 1:
            self.stats["average_response_time"] = (
                (current_avg * (total_requests - 1) + response_time) / total_requests
            )
        else:
            self.stats["average_response_time"] = response_time

    async def get_capabilities(self) -> List[str]:
        """获取Agent能力列表"""
        if self.unified_agent:
            return await self.unified_agent.get_capabilities()
        return []

    async def get_modes(self) -> List[str]:
        """获取可用模式列表"""
        return [mode.value for mode in AgentMode]

    async def get_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        stats = self.stats.copy()
        stats["timestamp"] = datetime.utcnow().isoformat()

        # 添加模型池统计
        if self.model_pool:
            stats["model_pool"] = await self.model_pool.get_model_info()

        # 添加缓存统计
        if self.cache_manager:
            stats["cache"] = await self.cache_manager.get_stats()

        # 添加Agent健康状态
        if self.unified_agent:
            stats["agent_health"] = await self.unified_agent.health_check()

        return stats

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        health_status = {
            "agent_core": "healthy" if self.is_initialized else "not_initialized",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {}
        }

        try:
            # 检查统一Agent
            if self.unified_agent:
                agent_health = await self.unified_agent.health_check()
                health_status["components"]["unified_agent"] = agent_health

            # 检查模型池
            if self.model_pool:
                model_info = await self.model_pool.get_model_info()
                health_status["components"]["model_pool"] = {
                    "status": "healthy",
                    "loaded_models": len([info for info in model_info.values() if info["is_loaded"]])
                }

            # 检查缓存管理器
            if self.cache_manager:
                cache_stats = await self.cache_manager.get_stats()
                health_status["components"]["cache_manager"] = {
                    "status": "healthy",
                    "stats": cache_stats
                }

        except Exception as e:
            health_status["agent_core"] = "unhealthy"
            health_status["error"] = str(e)

        return health_status

    async def cleanup(self):
        """清理资源"""
        logger.info("开始清理Agent核心资源...")

        try:
            # 清理模型池
            if self.model_pool:
                await self.model_pool.cleanup()

            # 清理缓存管理器
            if self.cache_manager:
                await self.cache_manager.close()

            # 清理MCP客户端
            if self.mcp_client_manager:
                await self.mcp_client_manager.close()

            self.is_initialized = False
            logger.info("Agent核心资源清理完成")

        except Exception as e:
            logger.error(f"Agent核心资源清理失败: {e}")


# 全局Agent核心实例
_agent_core_instance = None


async def get_agent_core() -> AgentCore:
    """获取全局Agent核心实例"""
    global _agent_core_instance

    if _agent_core_instance is None:
        _agent_core_instance = AgentCore()
        await _agent_core_instance.initialize()

    return _agent_core_instance

    async def _normal_chat(
        self,
        message: str,
        use_rag: bool,
        **kwargs
    ) -> Dict[str, Any]:
        """普通对话"""

        # 如果不使用RAG，直接调用LLM
        if not use_rag or not self.rag_core:
            logger.info("使用直接LLM对话模式")

            # 构建对话历史
            conversation_prompt = self._build_conversation_prompt(message)

            response = await self.llm_manager.generate(
                conversation_prompt,
                max_tokens=kwargs.get('max_tokens', 1000),
                temperature=kwargs.get('temperature', 0.7)
            )

            # 更新对话历史
            self.conversation_history.append({
                'role': 'user',
                'content': message,
                'timestamp': datetime.utcnow().isoformat()
            })
            self.conversation_history.append({
                'role': 'assistant',
                'content': response,
                'timestamp': datetime.utcnow().isoformat()
            })

            return {
                'success': True,
                'response': response,
                'mode': 'direct_llm',
                'timestamp': datetime.utcnow().isoformat()
            }

        # 使用ReAct Agent模式
        logger.info("使用ReAct Agent模式")

        result = await self.react_agent.run(message)

        # 更新对话历史
        self.conversation_history.append({
            'role': 'user',
            'content': message,
            'timestamp': datetime.utcnow().isoformat()
        })
        self.conversation_history.append({
            'role': 'assistant',
            'content': result.get('final_answer', ''),
            'timestamp': datetime.utcnow().isoformat(),
            'agent_steps': result.get('steps', []),
            'execution_time': result.get('execution_time', 0)
        })

        return {
            'success': result.get('success', False),
            'response': result.get('final_answer', ''),
            'mode': 'react_agent',
            'agent_result': result,
            'timestamp': datetime.utcnow().isoformat()
        }

    async def _stream_chat(
        self,
        message: str,
        use_rag: bool,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """流式对话"""

        if not use_rag or not self.rag_core:
            # 直接LLM流式生成
            logger.info("使用直接LLM流式对话模式")

            conversation_prompt = self._build_conversation_prompt(message)

            yield {
                'type': 'start',
                'mode': 'direct_llm',
                'timestamp': datetime.utcnow().isoformat()
            }

            response_parts = []
            async for chunk in self.llm_manager.stream_generate(
                conversation_prompt,
                max_tokens=kwargs.get('max_tokens', 1000),
                temperature=kwargs.get('temperature', 0.7)
            ):
                response_parts.append(chunk)
                yield {
                    'type': 'chunk',
                    'content': chunk,
                    'timestamp': datetime.utcnow().isoformat()
                }

            full_response = ''.join(response_parts)

            # 更新对话历史
            self.conversation_history.append({
                'role': 'user',
                'content': message,
                'timestamp': datetime.utcnow().isoformat()
            })
            self.conversation_history.append({
                'role': 'assistant',
                'content': full_response,
                'timestamp': datetime.utcnow().isoformat()
            })

            yield {
                'type': 'complete',
                'full_response': full_response,
                'timestamp': datetime.utcnow().isoformat()
            }

        else:
            # ReAct Agent流式执行
            logger.info("使用ReAct Agent流式模式")

            async for event in self.react_agent.stream_run(message):
                yield event

    def _build_conversation_prompt(self, current_message: str) -> str:
        """构建对话prompt"""
        prompt_parts = [
            "你是一个智能助手，请根据用户的问题提供有用、准确的回答。",
            ""
        ]

        # 添加对话历史（最近5轮）
        recent_history = self.conversation_history[-10:]  # 最近5轮对话

        for entry in recent_history:
            if entry['role'] == 'user':
                prompt_parts.append(f"用户: {entry['content']}")
            elif entry['role'] == 'assistant':
                prompt_parts.append(f"助手: {entry['content']}")

        # 添加当前消息
        prompt_parts.append(f"用户: {current_message}")
        prompt_parts.append("助手:")

        return "\n".join(prompt_parts)

    async def clear_conversation(self):
        """清空对话历史"""
        self.conversation_history = []
        logger.info("对话历史已清空")

    async def get_conversation_history(self) -> List[Dict[str, Any]]:
        """获取对话历史"""
        return self.conversation_history.copy()

    async def add_tool(self, tool):
        """添加工具"""
        if not self.is_initialized:
            await self.initialize()

        self.tool_manager.register_tool(tool)
        logger.info(f"工具 {tool.name} 已添加")

    async def remove_tool(self, tool_name: str):
        """移除工具"""
        if not self.is_initialized:
            await self.initialize()

        self.tool_manager.unregister_tool(tool_name)
        logger.info(f"工具 {tool_name} 已移除")

    async def list_tools(self) -> List[str]:
        """列出所有工具"""
        if not self.is_initialized:
            await self.initialize()

        return self.tool_manager.list_tools()

    async def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        status = {
            'initialized': self.is_initialized,
            'model_name': self.model_name,
            'conversation_length': len(self.conversation_history),
            'timestamp': datetime.utcnow().isoformat()
        }

        if self.is_initialized:
            status.update({
                'llm_info': self.llm_manager.get_model_info() if self.llm_manager else None,
                'available_tools': await self.list_tools(),
                'rag_enabled': self.rag_core is not None,
            })

            if self.react_agent:
                status['agent_summary'] = self.react_agent.get_execution_summary()

        return status

    async def close(self):
        """关闭Agent系统"""
        try:
            if self.rag_core:
                await self.rag_core.close()

            # 清理其他资源
            self.conversation_history = []
            self.is_initialized = False

            logger.info("Agent系统已关闭")

        except Exception as e:
            logger.warning(f"关闭Agent系统时出现警告: {e}")


class AgentFactory:
    """Agent工厂类"""

    @staticmethod
    async def create_agent(
        model_name: str = None,
        rag_core: RAGCore = None,
        **kwargs
    ) -> AgentCore:
        """创建Agent实例"""
        agent = AgentCore(
            model_name=model_name,
            rag_core=rag_core,
            **kwargs
        )

        await agent.initialize()
        return agent

    @staticmethod
    async def create_rag_agent(
        collection_name: str = "knowledge_base",
        model_name: str = None,
        **kwargs
    ) -> AgentCore:
        """创建带RAG功能的Agent"""

        # 创建RAG核心
        rag_core = RAGCore(collection_name=collection_name)
        await rag_core.initialize()

        # 创建Agent
        agent = AgentCore(
            model_name=model_name,
            rag_core=rag_core,
            **kwargs
        )

        await agent.initialize()
        return agent

    # MCP相关方法
    async def parse_document_with_mcp(self, file_path: str) -> Dict[str, Any]:
        """使用MCP Agent解析文档"""
        if not self.enable_mcp or not self.mcp_agent:
            return {
                "success": False,
                "error": "MCP功能未启用或未初始化"
            }

        logger.info(f"使用MCP Agent解析文档: {file_path}")

        try:
            result = await self.mcp_agent.parse_document_intelligently(file_path)

            # 记录到对话历史
            self.conversation_history.append({
                'role': 'system',
                'content': f'文档解析: {file_path}',
                'timestamp': datetime.utcnow().isoformat(),
                'mcp_result': result
            })

            return result

        except Exception as e:
            logger.error(f"MCP文档解析失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def chat_with_mcp(self, message: str, use_rag: bool = True) -> Dict[str, Any]:
        """使用MCP Agent进行对话"""
        if not self.enable_mcp or not self.mcp_agent:
            return {
                "success": False,
                "error": "MCP功能未启用或未初始化"
            }

        logger.info(f"使用MCP Agent对话: {message}")

        try:
            result = await self.mcp_agent.search_and_answer(message, use_rag)

            # 记录到对话历史
            self.conversation_history.append({
                'role': 'user',
                'content': message,
                'timestamp': datetime.utcnow().isoformat()
            })
            self.conversation_history.append({
                'role': 'assistant',
                'content': result.get('final_answer', ''),
                'timestamp': datetime.utcnow().isoformat(),
                'mcp_result': result
            })

            return {
                'success': result.get('success', False),
                'response': result.get('final_answer', ''),
                'mode': 'mcp_agent',
                'mcp_result': result,
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"MCP对话失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def get_mcp_tools(self) -> Dict[str, Any]:
        """获取所有可用的MCP工具"""
        if not self.enable_mcp or not self.mcp_client_manager:
            return {"tools": {}, "error": "MCP功能未启用"}

        try:
            tools = await self.mcp_client_manager.get_available_tools()
            return {"tools": tools, "count": len(tools)}
        except Exception as e:
            logger.error(f"获取MCP工具失败: {e}")
            return {"tools": {}, "error": str(e)}

    async def get_mcp_server_status(self) -> Dict[str, Any]:
        """获取MCP服务器状态"""
        if not self.enable_mcp or not self.mcp_server_manager:
            return {"status": "disabled", "error": "MCP功能未启用"}

        try:
            return await self.mcp_server_manager.health_check()
        except Exception as e:
            logger.error(f"获取MCP服务器状态失败: {e}")
            return {"status": "error", "error": str(e)}

    async def call_mcp_tool(self, tool_key: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """直接调用MCP工具"""
        if not self.enable_mcp or not self.mcp_client_manager:
            return {"success": False, "error": "MCP功能未启用"}

        try:
            result = await self.mcp_client_manager.call_tool(tool_key, arguments)

            # 记录工具调用
            self.conversation_history.append({
                'role': 'system',
                'content': f'工具调用: {tool_key}',
                'timestamp': datetime.utcnow().isoformat(),
                'tool_call': {
                    'tool': tool_key,
                    'arguments': arguments,
                    'result': result
                }
            })

            return result

        except Exception as e:
            logger.error(f"MCP工具调用失败: {e}")
            return {"success": False, "error": str(e)}

    # LangGraph相关方法
    async def chat_with_langgraph(self, message: str, use_rag: bool = True) -> Dict[str, Any]:
        """使用LangGraph Agent进行对话"""
        if not self.enable_mcp or not self.langgraph_agent:
            return {
                "success": False,
                "error": "LangGraph功能未启用或未初始化"
            }

        logger.info(f"使用LangGraph Agent对话: {message}")

        try:
            result = await self.langgraph_agent.chat(message, use_rag)

            # 记录到对话历史
            self.conversation_history.append({
                'role': 'user',
                'content': message,
                'timestamp': datetime.utcnow().isoformat()
            })
            self.conversation_history.append({
                'role': 'assistant',
                'content': result.get('response', ''),
                'timestamp': datetime.utcnow().isoformat(),
                'langgraph_result': result
            })

            return {
                'success': result.get('success', False),
                'response': result.get('response', ''),
                'mode': 'langgraph_agent',
                'langgraph_result': result,
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"LangGraph对话失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def parse_document_with_langgraph(self, file_path: str) -> Dict[str, Any]:
        """使用LangGraph工作流解析文档"""
        if not self.enable_mcp or not self.langgraph_agent:
            return {
                "success": False,
                "error": "LangGraph功能未启用或未初始化"
            }

        logger.info(f"使用LangGraph工作流解析文档: {file_path}")

        try:
            result = await self.langgraph_agent.parse_document(file_path)

            # 记录到对话历史
            self.conversation_history.append({
                'role': 'system',
                'content': f'LangGraph文档解析: {file_path}',
                'timestamp': datetime.utcnow().isoformat(),
                'langgraph_result': result
            })

            return result

        except Exception as e:
            logger.error(f"LangGraph文档解析失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def batch_process_documents_with_langgraph(self, file_paths: List[str]) -> Dict[str, Any]:
        """使用LangGraph批量处理文档"""
        if not self.enable_mcp or not self.langgraph_agent:
            return {
                "success": False,
                "error": "LangGraph功能未启用或未初始化"
            }

        logger.info(f"使用LangGraph批量处理 {len(file_paths)} 个文档")

        try:
            result = await self.langgraph_agent.batch_process_documents(file_paths)

            # 记录到对话历史
            self.conversation_history.append({
                'role': 'system',
                'content': f'LangGraph批量处理: {len(file_paths)} 个文档',
                'timestamp': datetime.utcnow().isoformat(),
                'langgraph_result': result
            })

            return result

        except Exception as e:
            logger.error(f"LangGraph批量处理失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def get_workflow_visualization(self, workflow_type: str) -> Optional[str]:
        """获取工作流可视化"""
        if not self.enable_mcp or not self.langgraph_agent:
            return None

        try:
            return self.langgraph_agent.get_workflow_visualization(workflow_type)
        except Exception as e:
            logger.error(f"获取工作流可视化失败: {e}")
            return None

    def get_agent_statistics(self) -> Dict[str, Any]:
        """获取Agent统计信息"""
        stats = {
            "traditional_agent": {
                "enabled": self.react_agent is not None,
                "tools": len(self.tool_manager.tools) if self.tool_manager else 0
            },
            "mcp_agent": {
                "enabled": self.mcp_agent is not None,
                "tools": len(self.mcp_agent.available_tools) if self.mcp_agent else 0
            },
            "langgraph_agent": {
                "enabled": self.langgraph_agent is not None,
                "tools": len(self.langgraph_agent.available_tools) if self.langgraph_agent else 0,
                "workflows": 3 if self.workflow_manager else 0
            },
            "conversation_history": len(self.conversation_history)
        }

        if self.langgraph_agent:
            stats["langgraph_agent"].update(
                self.langgraph_agent.get_statistics())

        return stats
