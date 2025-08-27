"""
Agent层
提供智能代理、工具调用、ReAct推理等功能
"""
from .llm import QwenLLMManager, LLMFactory, LangChainLLMWrapper
from .tools import (
    ToolManager,
    RAGSearchTool,
    RAGContextTool,
    TimeTool,
    CalculatorTool,
    create_default_tool_manager
)
from .react_agent import ReActAgent, ReActStep, AgentState
from .core import AgentCore, AgentFactory

__all__ = [
    # LLM管理器
    "QwenLLMManager",
    "LLMFactory",
    "LangChainLLMWrapper",

    # 工具管理器
    "ToolManager",
    "RAGSearchTool",
    "RAGContextTool",
    "TimeTool",
    "CalculatorTool",
    "create_default_tool_manager",

    # ReAct Agent
    "ReActAgent",
    "ReActStep",
    "AgentState",

    # 核心管理器
    "AgentCore",
    "AgentFactory",
]
