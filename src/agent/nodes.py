"""
LangGraph工作流节点
定义各种工作流中的处理节点
"""
import asyncio
import time
from typing import Dict, Any, List, Optional
from pathlib import Path

from .states import (
    DocumentProcessingState,
    AgentReasoningState,
    RAGWorkflowState,
    MCPToolState
)
from ..epkbs_mcp.clients.mcp_client import MCPClientManager
from ..rag.core import RAGCore
from ..utils.logger import get_logger

logger = get_logger("agent.nodes")


# ==================== 文档处理节点 ====================

async def detect_document_format(state: DocumentProcessingState) -> DocumentProcessingState:
    """检测文档格式和类型"""
    logger.info(f"检测文档格式: {state['file_path']}")

    try:
        file_path = Path(state["file_path"])

        # 基础文件信息
        file_type = file_path.suffix.lower()
        file_size = file_path.stat().st_size if file_path.exists() else 0

        # 根据文件类型确定处理策略
        if file_type == ".pdf":
            strategy = "pdf_focused"
        elif file_type in [".docx", ".doc"]:
            strategy = "office_focused"
        elif file_type in [".xlsx", ".xls"]:
            strategy = "table_focused"
        elif file_type in [".pptx", ".ppt"]:
            strategy = "presentation_focused"
        else:
            strategy = "standard"

        return {
            **state,
            "file_type": file_type,
            "file_size": file_size,
            "processing_strategy": strategy,
            "current_stage": "parsing",
            "next_action": "basic_parse"
        }

    except Exception as e:
        logger.error(f"文档格式检测失败: {e}")
        return {
            **state,
            "error_messages": state["error_messages"] + [str(e)],
            "next_action": "error_handling"
        }


async def basic_document_parse(
    state: DocumentProcessingState,
    mcp_client: MCPClientManager
) -> DocumentProcessingState:
    """基础文档解析"""
    logger.info(f"执行基础文档解析: {state['file_path']}")

    try:
        # 调用文档解析MCP工具
        result = await mcp_client.call_tool(
            "document-parsing-server.parse_document",
            {
                "file_path": state["file_path"],
                "strategy": "auto",
                "extract_images": True
            }
        )

        if result.get("success"):
            parsing_result = result.get("result", {})

            return {
                **state,
                "basic_parsing_result": parsing_result,
                "tools_used": state["tools_used"] + ["document-parsing-server.parse_document"],
                "tool_results": {
                    **state["tool_results"],
                    "basic_parsing": parsing_result
                },
                "current_stage": "quality_check",
                "next_action": "assess_quality"
            }
        else:
            return {
                **state,
                "error_messages": state["error_messages"] + [result.get("error", "解析失败")],
                "retry_count": state["retry_count"] + 1,
                "next_action": "retry_or_fallback"
            }

    except Exception as e:
        logger.error(f"基础文档解析失败: {e}")
        return {
            **state,
            "error_messages": state["error_messages"] + [str(e)],
            "next_action": "error_handling"
        }


async def assess_parsing_quality(
    state: DocumentProcessingState,
    mcp_client: MCPClientManager
) -> DocumentProcessingState:
    """评估解析质量"""
    logger.info("评估解析质量")

    try:
        basic_result = state.get("basic_parsing_result", {})

        # 计算质量分数
        quality_score = basic_result.get("quality_score", 0.0)
        has_tables = basic_result.get("has_tables", False)
        has_images = basic_result.get("has_images", False)
        has_formulas = basic_result.get("has_formulas", False)

        # 确定是否需要高级处理
        needs_advanced = (
            quality_score < 0.8 or
            has_tables or
            has_images or
            has_formulas
        )

        quality_details = {
            "basic_quality": quality_score,
            "has_tables": has_tables,
            "has_images": has_images,
            "has_formulas": has_formulas,
            "element_count": basic_result.get("total_elements", 0)
        }

        next_action = "advanced_processing" if needs_advanced else "integrate_results"

        return {
            **state,
            "quality_score": quality_score,
            "quality_details": quality_details,
            "needs_advanced_processing": needs_advanced,
            "current_stage": "advanced_processing" if needs_advanced else "integration",
            "next_action": next_action
        }

    except Exception as e:
        logger.error(f"质量评估失败: {e}")
        return {
            **state,
            "error_messages": state["error_messages"] + [str(e)],
            "next_action": "error_handling"
        }


async def advanced_document_processing(
    state: DocumentProcessingState,
    mcp_client: MCPClientManager
) -> DocumentProcessingState:
    """高级文档处理"""
    logger.info("执行高级文档处理")

    try:
        results = {}
        tools_used = list(state["tools_used"])

        # 表格提取
        if state["quality_details"].get("has_tables"):
            logger.info("执行表格提取")
            table_result = await mcp_client.call_tool(
                "table-extraction-server.extract_tables_camelot",
                {"file_path": state["file_path"]}
            )

            if table_result.get("success"):
                results["table_extraction"] = table_result.get("result")
                tools_used.append(
                    "table-extraction-server.extract_tables_camelot")

        # 图像分析（如果有图像分析服务器）
        if state["quality_details"].get("has_images"):
            logger.info("图像分析功能待实现")
            # 这里可以添加图像分析逻辑

        # 公式识别（如果有公式识别服务器）
        if state["quality_details"].get("has_formulas"):
            logger.info("公式识别功能待实现")
            # 这里可以添加公式识别逻辑

        return {
            **state,
            "table_extraction_result": results.get("table_extraction"),
            "image_analysis_result": results.get("image_analysis"),
            "formula_recognition_result": results.get("formula_recognition"),
            "tools_used": tools_used,
            "tool_results": {
                **state["tool_results"],
                **results
            },
            "current_stage": "integration",
            "next_action": "integrate_results"
        }

    except Exception as e:
        logger.error(f"高级文档处理失败: {e}")
        return {
            **state,
            "error_messages": state["error_messages"] + [str(e)],
            "next_action": "error_handling"
        }


async def integrate_processing_results(state: DocumentProcessingState) -> DocumentProcessingState:
    """整合处理结果"""
    logger.info("整合处理结果")

    try:
        # 收集所有处理结果
        basic_result = state.get("basic_parsing_result", {})
        table_result = state.get("table_extraction_result", {})
        image_result = state.get("image_analysis_result", {})
        formula_result = state.get("formula_recognition_result", {})

        # 整合内容
        integrated_content = ""

        # 添加基础文本内容
        if basic_result.get("elements"):
            for element in basic_result["elements"]:
                if element.get("type") == "Text":
                    integrated_content += element.get("content", "") + "\n"

        # 添加表格内容
        if table_result and table_result.get("tables"):
            integrated_content += "\n## 表格内容\n"
            for i, table in enumerate(table_result["tables"]):
                integrated_content += f"\n### 表格 {i+1}\n"
                integrated_content += table.get("markdown", "") + "\n"

        # 生成处理摘要
        processing_summary = {
            "total_tools_used": len(state["tools_used"]),
            "tools_list": state["tools_used"],
            "processing_time": time.time(),  # 这里应该计算实际处理时间
            "quality_score": state["quality_score"],
            "content_length": len(integrated_content),
            "has_tables": bool(table_result),
            "has_images": bool(image_result),
            "has_formulas": bool(formula_result)
        }

        # 生成元数据
        metadata = {
            "file_info": {
                "path": state["file_path"],
                "type": state["file_type"],
                "size": state["file_size"]
            },
            "processing_info": {
                "strategy": state["processing_strategy"],
                "tools_used": state["tools_used"],
                "quality_score": state["quality_score"]
            },
            "content_info": {
                "total_elements": basic_result.get("total_elements", 0),
                "table_count": len(table_result.get("tables", [])) if table_result else 0,
                "image_count": len(image_result.get("images", [])) if image_result else 0
            }
        }

        return {
            **state,
            "integrated_content": integrated_content,
            "metadata": metadata,
            "processing_summary": processing_summary,
            "current_stage": "completed",
            "is_complete": True,
            "next_action": "done"
        }

    except Exception as e:
        logger.error(f"结果整合失败: {e}")
        return {
            **state,
            "error_messages": state["error_messages"] + [str(e)],
            "next_action": "error_handling"
        }


# ==================== Agent推理节点 ====================

async def agent_think(
    state: AgentReasoningState,
    llm_manager
) -> AgentReasoningState:
    """Agent思考节点"""
    logger.info(f"Agent思考 - 第 {state['iteration_count'] + 1} 轮")

    try:
        # 构建思考prompt
        think_prompt = f"""
你是一个智能助手，正在处理用户的请求。

用户查询: {state['user_query']}

当前上下文:
{state.get('context_documents', [])}

可用工具:
{state.get('available_tools', {})}

之前的推理历史:
{state.get('tool_call_history', [])}

请思考下一步应该采取什么行动。如果你有足够信息回答用户问题，请给出最终答案。

格式:
Thought: [你的思考过程]
Action: [工具名称] 或 Final Answer
Action Input: [工具参数] 或 [最终答案]
"""

        # 生成LLM响应
        response = await llm_manager.generate_response(
            messages=[{"role": "user", "content": think_prompt}],
            max_tokens=1000,
            temperature=0.1
        )

        response_text = response.get("content", "")

        # 解析响应
        thought, action, action_input = _parse_agent_response(response_text)

        # 检查是否完成
        if action == "Final Answer":
            return {
                **state,
                "current_thought": thought,
                "final_answer": action_input,
                "reasoning_complete": True,
                "next_action": "done"
            }

        return {
            **state,
            "current_thought": thought,
            "current_action": action,
            "action_input": action_input,
            "iteration_count": state["iteration_count"] + 1,
            "next_action": "act"
        }

    except Exception as e:
        logger.error(f"Agent思考失败: {e}")
        return {
            **state,
            "error_messages": state["error_messages"] + [str(e)],
            "next_action": "error_handling"
        }


async def agent_act(
    state: AgentReasoningState,
    mcp_client: MCPClientManager
) -> AgentReasoningState:
    """Agent行动节点"""
    logger.info(f"Agent执行行动: {state.get('current_action')}")

    try:
        action = state.get("current_action")
        action_input = state.get("action_input", {})

        if not action:
            return {
                **state,
                "observation": "没有指定要执行的行动",
                "next_action": "observe"
            }

        # 执行工具调用
        start_time = time.time()
        result = await mcp_client.call_tool(action, action_input)
        execution_time = time.time() - start_time

        # 记录工具调用
        tool_call_record = {
            "tool": action,
            "input": action_input,
            "result": result,
            "execution_time": execution_time,
            "timestamp": time.time()
        }

        observation = f"工具 {action} 执行{'成功' if result.get('success') else '失败'}"
        if result.get("success"):
            observation += f": {result.get('result', {})}"
        else:
            observation += f": {result.get('error', '未知错误')}"

        return {
            **state,
            "observation": observation,
            "tool_call_history": state["tool_call_history"] + [tool_call_record],
            "tool_execution_times": {
                **state["tool_execution_times"],
                action: execution_time
            },
            "next_action": "observe"
        }

    except Exception as e:
        logger.error(f"Agent行动失败: {e}")
        return {
            **state,
            "error_messages": state["error_messages"] + [str(e)],
            "observation": f"工具执行异常: {str(e)}",
            "next_action": "observe"
        }


async def agent_observe(state: AgentReasoningState) -> AgentReasoningState:
    """Agent观察节点"""
    logger.info("Agent观察和反思")

    try:
        # 检查是否达到最大迭代次数
        if state["iteration_count"] >= state["max_iterations"]:
            return {
                **state,
                "reasoning_complete": True,
                "final_answer": "抱歉，我无法在限定步数内完成任务。",
                "next_action": "done"
            }

        # 检查是否已经完成推理
        if state.get("reasoning_complete"):
            return {
                **state,
                "next_action": "done"
            }

        # 继续推理
        return {
            **state,
            "next_action": "think"
        }

    except Exception as e:
        logger.error(f"Agent观察失败: {e}")
        return {
            **state,
            "error_messages": state["error_messages"] + [str(e)],
            "next_action": "error_handling"
        }


# ==================== RAG工作流节点 ====================

async def analyze_query_intent(
    state: RAGWorkflowState,
    mcp_client: MCPClientManager
) -> RAGWorkflowState:
    """分析查询意图"""
    logger.info(f"分析查询意图: {state['original_query']}")

    try:
        # 调用查询分析工具
        result = await mcp_client.call_tool(
            "rag-server.analyze_query",
            {"query": state["original_query"]}
        )

        if result.get("success"):
            analysis = result.get("result", {})

            return {
                **state,
                "query_intent": analysis.get("intent", "general"),
                "query_complexity": analysis.get("complexity", "medium"),
                "retrieval_strategy": analysis.get("suggested_strategy", "hybrid"),
                "next_action": "execute_retrieval"
            }
        else:
            # 使用默认策略
            return {
                **state,
                "query_intent": "general",
                "query_complexity": "medium",
                "retrieval_strategy": "hybrid",
                "next_action": "execute_retrieval"
            }

    except Exception as e:
        logger.error(f"查询意图分析失败: {e}")
        return {
            **state,
            "error_messages": state["error_messages"] + [str(e)],
            "next_action": "execute_retrieval"  # 继续执行，使用默认策略
        }


async def execute_retrieval(
    state: RAGWorkflowState,
    mcp_client: MCPClientManager
) -> RAGWorkflowState:
    """执行检索"""
    logger.info(f"执行检索: 策略={state['retrieval_strategy']}")

    try:
        # 调用RAG搜索工具
        result = await mcp_client.call_tool(
            "rag-server.search_knowledge",
            {
                "query": state["original_query"],
                "top_k": state["top_k"],
                "retriever_type": state["retrieval_strategy"],
                "reranker_type": "ensemble"
            }
        )

        if result.get("success"):
            search_result = result.get("result", {})

            return {
                **state,
                "reranked_results": search_result.get("results", []),
                "retrieval_metadata": {
                    "retriever_used": search_result.get("retriever_used"),
                    "reranker_used": search_result.get("reranker_used"),
                    "total_results": search_result.get("total_results", 0)
                },
                "next_action": "generate_context"
            }
        else:
            return {
                **state,
                "error_messages": state["error_messages"] + [result.get("error", "检索失败")],
                "next_action": "error_handling"
            }

    except Exception as e:
        logger.error(f"检索执行失败: {e}")
        return {
            **state,
            "error_messages": state["error_messages"] + [str(e)],
            "next_action": "error_handling"
        }


async def generate_context(
    state: RAGWorkflowState,
    mcp_client: MCPClientManager
) -> RAGWorkflowState:
    """生成上下文"""
    logger.info("生成上下文")

    try:
        # 调用上下文生成工具
        result = await mcp_client.call_tool(
            "rag-server.get_context",
            {
                "query": state["original_query"],
                "max_context_length": 4000,
                "include_metadata": True
            }
        )

        if result.get("success"):
            context_result = result.get("result", {})

            return {
                **state,
                "final_context": context_result.get("context", ""),
                "generation_metadata": {
                    "context_length": context_result.get("context_length", 0),
                    "source_count": len(context_result.get("sources", []))
                },
                "next_action": "done"
            }
        else:
            return {
                **state,
                "error_messages": state["error_messages"] + [result.get("error", "上下文生成失败")],
                "next_action": "error_handling"
            }

    except Exception as e:
        logger.error(f"上下文生成失败: {e}")
        return {
            **state,
            "error_messages": state["error_messages"] + [str(e)],
            "next_action": "error_handling"
        }


# ==================== 通用节点 ====================

async def error_handling(state: Dict[str, Any]) -> Dict[str, Any]:
    """通用错误处理节点"""
    logger.error(f"进入错误处理: {state.get('error_messages', [])}")

    # 检查重试次数
    if state.get("retry_count", 0) < 3:
        logger.info("尝试重试")
        return {
            **state,
            "retry_count": state.get("retry_count", 0) + 1,
            "next_action": "retry"
        }
    else:
        logger.error("达到最大重试次数，任务失败")
        return {
            **state,
            "is_complete": True,
            "next_action": "done"
        }


def _parse_agent_response(response_text: str) -> tuple:
    """解析Agent响应"""
    thought = ""
    action = ""
    action_input = {}

    lines = response_text.strip().split('\n')
    current_section = None
    content_lines = []

    for line in lines:
        line = line.strip()

        if line.startswith("Thought:"):
            if current_section == "thought":
                thought = '\n'.join(content_lines).strip()
            current_section = "thought"
            content_lines = [line[8:].strip()]
        elif line.startswith("Action:"):
            if current_section == "thought":
                thought = '\n'.join(content_lines).strip()
            current_section = "action"
            content_lines = [line[7:].strip()]
        elif line.startswith("Action Input:"):
            if current_section == "action":
                action = '\n'.join(content_lines).strip()
            current_section = "action_input"
            content_lines = [line[13:].strip()]
        else:
            if current_section:
                content_lines.append(line)

    # 处理最后一个section
    if current_section == "thought":
        thought = '\n'.join(content_lines).strip()
    elif current_section == "action":
        action = '\n'.join(content_lines).strip()
    elif current_section == "action_input":
        action_input_str = '\n'.join(content_lines).strip()
        try:
            import json
            action_input = json.loads(action_input_str)
        except:
            action_input = {"raw": action_input_str}

    return thought, action, action_input
