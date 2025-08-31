"""
Agent核心管理器
整合LLM、工具、ReAct Agent等组件
"""
import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime

from src.agent.llm import QwenLLMManager, LLMFactory
from src.agent.tools import ToolManager, create_default_tool_manager
from src.agent.react_agent import ReActAgent
from src.agent.mcp_agent import MCPAgent
from src.rag.core import RAGCore
from src.mcp.server_manager import MCPServerManager
from src.mcp.clients.mcp_client import MCPClientManager
from src.utils.logger import get_logger
from config.settings import settings

logger = get_logger("agent.core")


class AgentCore:
    """Agent核心管理器"""

    def __init__(
        self,
        model_name: str = None,
        rag_core: RAGCore = None,
        max_steps: int = 10,
        max_execution_time: int = 300,
        enable_mcp: bool = True
    ):
        self.model_name = model_name or settings.QWEN_MODEL_PATH
        self.rag_core = rag_core
        self.enable_mcp = enable_mcp
        self.max_steps = max_steps
        self.max_execution_time = max_execution_time

        # 核心组件
        self.llm_manager = None
        self.tool_manager = None
        self.react_agent = None

        # MCP组件
        self.mcp_server_manager = None
        self.mcp_client_manager = None
        self.mcp_agent = None

        # 状态
        self.is_initialized = False
        self.conversation_history = []

        logger.info(f"Agent核心管理器初始化: 模型={self.model_name}")

    async def initialize(self):
        """初始化Agent系统"""
        if self.is_initialized:
            logger.info("Agent系统已初始化")
            return

        logger.info("正在初始化Agent核心系统...")

        try:
            # 1. 初始化LLM管理器
            logger.info("初始化LLM管理器...")
            self.llm_manager = QwenLLMManager(
                model_name=self.model_name,
                load_in_8bit=False,  # 可根据需要调整
                load_in_4bit=False
            )
            await self.llm_manager.initialize()

            # 2. 初始化工具管理器
            logger.info("初始化工具管理器...")
            self.tool_manager = create_default_tool_manager(self.rag_core)

            # 3. 初始化ReAct Agent
            logger.info("初始化ReAct Agent...")
            self.react_agent = ReActAgent(
                llm_manager=self.llm_manager,
                tool_manager=self.tool_manager,
                max_steps=self.max_steps,
                max_execution_time=self.max_execution_time
            )

            # 4. 初始化MCP组件（如果启用）
            if self.enable_mcp:
                logger.info("初始化MCP组件...")

                # 初始化MCP服务器管理器
                self.mcp_server_manager = MCPServerManager()
                await self.mcp_server_manager.start_all_servers(self.rag_core)

                # 初始化MCP客户端管理器
                self.mcp_client_manager = MCPClientManager(
                    self.mcp_server_manager)
                await self.mcp_client_manager.initialize()

                # 初始化MCP Agent
                self.mcp_agent = MCPAgent(
                    self.llm_manager, self.mcp_client_manager)
                await self.mcp_agent.initialize()

                logger.info("MCP组件初始化完成")

            self.is_initialized = True
            logger.info("Agent核心系统初始化完成")

        except Exception as e:
            logger.error(f"Agent核心系统初始化失败: {e}")
            raise

    async def chat(
        self,
        message: str,
        use_rag: bool = True,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        与Agent对话

        Args:
            message: 用户消息
            use_rag: 是否使用RAG
            stream: 是否流式响应
            **kwargs: 其他参数

        Returns:
            对话结果
        """
        if not self.is_initialized:
            await self.initialize()

        logger.info(f"Agent对话开始: {message[:100]}...")

        try:
            if stream:
                # 流式响应
                return await self._stream_chat(message, use_rag, **kwargs)
            else:
                # 普通响应
                return await self._normal_chat(message, use_rag, **kwargs)

        except Exception as e:
            logger.error(f"Agent对话失败: {e}")
            return {
                'success': False,
                'response': f"对话过程中发生错误: {str(e)}",
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }

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
