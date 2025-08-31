"""
基于MCP的智能Agent
使用MCP服务器提供的工具进行智能推理和任务执行
"""
import asyncio
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..mcp.clients.mcp_client import MCPClientManager
from ..agent.llm import LLMManager
from ..utils.logger import get_logger

logger = get_logger("agent.mcp_agent")


class MCPAgent:
    """基于MCP的智能Agent"""
    
    def __init__(self, llm_manager: LLMManager, mcp_client_manager: MCPClientManager):
        self.llm_manager = llm_manager
        self.mcp_client = mcp_client_manager
        self.available_tools = {}
        self.conversation_history = []
        self.max_iterations = 10
        
        logger.info("MCP Agent初始化完成")
    
    async def initialize(self):
        """初始化Agent，发现可用工具"""
        try:
            # 发现所有可用工具
            self.available_tools = await self.mcp_client.get_available_tools()
            
            logger.info(f"发现 {len(self.available_tools)} 个可用工具")
            
            # 记录工具信息
            for tool_key, tool_info in self.available_tools.items():
                logger.debug(f"工具: {tool_key} - {tool_info['description']}")
            
        except Exception as e:
            logger.error(f"Agent初始化失败: {e}")
            raise
    
    async def parse_document_intelligently(self, file_path: str) -> Dict[str, Any]:
        """智能解析文档 - Agent驱动的解析流程"""
        
        task_prompt = f"""
你是一个专业的文档解析专家Agent。你需要解析文档: {file_path}

可用的MCP工具:
{self._format_available_tools()}

解析策略:
1. 首先使用 document-parsing-server.parse_document 获取文档基本信息
2. 根据检测到的内容类型，选择专业工具进行深度解析:
   - 如果有表格: 使用 table-extraction-server 的工具
   - 如果有图像: 使用 image-analysis-server 的工具  
   - 如果有公式: 使用 formula-recognition-server 的工具
3. 评估解析质量，如果质量不达标，尝试其他方法
4. 最后整合所有结果

请按照以下格式进行推理:
Thought: [你的思考过程]
Action: [要执行的工具，格式为 server.tool]
Action Input: [工具参数，JSON格式]
Observation: [工具执行结果]

重复上述过程直到完成任务，最后用 Final Answer: [最终结果] 结束。

开始解析:
"""
        
        # 使用ReAct模式执行解析任务
        result = await self._react_reasoning(task_prompt, {"file_path": file_path})
        return result
    
    async def search_and_answer(self, query: str, use_rag: bool = True) -> Dict[str, Any]:
        """搜索并回答问题"""
        
        if use_rag:
            task_prompt = f"""
你是一个智能助手，需要回答用户的问题: {query}

可用的工具:
{self._format_available_tools()}

请按照以下步骤:
1. 使用 rag-server.search_knowledge 搜索相关信息
2. 如果需要更多上下文，使用 rag-server.get_context
3. 基于搜索结果回答用户问题

请按照ReAct格式进行推理，最后给出 Final Answer。

开始:
"""
        else:
            task_prompt = f"""
用户问题: {query}

请直接回答这个问题，不需要搜索知识库。

Final Answer: [你的回答]
"""
        
        result = await self._react_reasoning(task_prompt, {"query": query})
        return result
    
    async def _react_reasoning(self, task_prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """ReAct推理循环"""
        start_time = time.time()
        
        try:
            conversation = [
                {"role": "system", "content": "你是一个专业的AI助手，能够使用各种工具完成复杂任务。"},
                {"role": "user", "content": task_prompt}
            ]
            
            reasoning_steps = []
            
            for iteration in range(1, self.max_iterations + 1):
                logger.info(f"ReAct推理 - 第 {iteration} 轮")
                
                # 生成LLM响应
                llm_response = await self.llm_manager.generate_response(
                    messages=conversation,
                    max_tokens=2000,
                    temperature=0.1
                )
                
                response_text = llm_response.get("content", "")
                logger.debug(f"LLM响应: {response_text}")
                
                # 解析响应
                parsed_response = self._parse_llm_response(response_text)
                
                reasoning_steps.append({
                    "iteration": iteration,
                    "thought": parsed_response.get("thought"),
                    "action": parsed_response.get("action"),
                    "action_input": parsed_response.get("action_input"),
                    "llm_response": response_text
                })
                
                # 检查是否完成
                if parsed_response.get("final_answer"):
                    reasoning_steps[-1]["final_answer"] = parsed_response["final_answer"]
                    break
                
                # 执行工具调用
                if parsed_response.get("action"):
                    observation = await self._execute_tool(
                        parsed_response["action"],
                        parsed_response.get("action_input", {})
                    )
                    
                    reasoning_steps[-1]["observation"] = observation
                    
                    # 添加观察结果到对话
                    conversation.append({"role": "assistant", "content": response_text})
                    conversation.append({"role": "user", "content": f"Observation: {json.dumps(observation, ensure_ascii=False)}"})
                else:
                    # 如果没有动作，可能是推理错误
                    conversation.append({"role": "assistant", "content": response_text})
                    conversation.append({"role": "user", "content": "请继续按照 Thought -> Action -> Action Input 的格式进行推理。"})
            
            execution_time = time.time() - start_time
            
            # 提取最终答案
            final_answer = None
            for step in reversed(reasoning_steps):
                if step.get("final_answer"):
                    final_answer = step["final_answer"]
                    break
            
            result = {
                "success": True,
                "final_answer": final_answer,
                "reasoning_steps": reasoning_steps,
                "total_iterations": len(reasoning_steps),
                "execution_time": execution_time,
                "context": context
            }
            
            # 记录到对话历史
            self.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "task": task_prompt[:100] + "..." if len(task_prompt) > 100 else task_prompt,
                "result": result,
                "success": True
            })
            
            return result
            
        except Exception as e:
            logger.error(f"ReAct推理失败: {e}")
            
            error_result = {
                "success": False,
                "error": str(e),
                "reasoning_steps": reasoning_steps if 'reasoning_steps' in locals() else [],
                "execution_time": time.time() - start_time,
                "context": context
            }
            
            self.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "task": task_prompt[:100] + "..." if len(task_prompt) > 100 else task_prompt,
                "result": error_result,
                "success": False
            })
            
            return error_result
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """解析LLM响应，提取思考、动作和参数"""
        parsed = {
            "thought": None,
            "action": None,
            "action_input": None,
            "final_answer": None
        }
        
        lines = response.strip().split('\n')
        current_section = None
        content_lines = []
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("Thought:"):
                if current_section:
                    parsed[current_section] = '\n'.join(content_lines).strip()
                current_section = "thought"
                content_lines = [line[8:].strip()]
            elif line.startswith("Action:"):
                if current_section:
                    parsed[current_section] = '\n'.join(content_lines).strip()
                current_section = "action"
                content_lines = [line[7:].strip()]
            elif line.startswith("Action Input:"):
                if current_section:
                    parsed[current_section] = '\n'.join(content_lines).strip()
                current_section = "action_input"
                content_lines = [line[13:].strip()]
            elif line.startswith("Final Answer:"):
                if current_section:
                    parsed[current_section] = '\n'.join(content_lines).strip()
                current_section = "final_answer"
                content_lines = [line[13:].strip()]
            else:
                if current_section:
                    content_lines.append(line)
        
        # 处理最后一个section
        if current_section and content_lines:
            parsed[current_section] = '\n'.join(content_lines).strip()
        
        # 解析action_input为JSON
        if parsed["action_input"]:
            try:
                parsed["action_input"] = json.loads(parsed["action_input"])
            except json.JSONDecodeError:
                logger.warning(f"无法解析action_input为JSON: {parsed['action_input']}")
                parsed["action_input"] = {}
        
        return parsed
    
    async def _execute_tool(self, tool_key: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """执行工具调用"""
        try:
            logger.info(f"执行工具: {tool_key} with {arguments}")
            
            result = await self.mcp_client.call_tool(tool_key, arguments)
            
            if result.get("success"):
                logger.info(f"工具执行成功: {tool_key}")
                return result.get("result", result)
            else:
                logger.error(f"工具执行失败: {tool_key} - {result.get('error')}")
                return {"error": result.get("error", "工具执行失败")}
                
        except Exception as e:
            logger.error(f"工具执行异常: {tool_key} - {e}")
            return {"error": str(e)}
    
    def _format_available_tools(self) -> str:
        """格式化可用工具列表"""
        if not self.available_tools:
            return "暂无可用工具"
        
        tool_descriptions = []
        for tool_key, tool_info in self.available_tools.items():
            description = f"- {tool_key}: {tool_info['description']}"
            
            # 添加参数信息
            schema = tool_info.get("schema", {})
            if schema.get("properties"):
                params = list(schema["properties"].keys())
                description += f" (参数: {', '.join(params)})"
            
            tool_descriptions.append(description)
        
        return "\n".join(tool_descriptions)
    
    def get_conversation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取对话历史"""
        return self.conversation_history[-limit:] if limit > 0 else self.conversation_history
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取Agent统计信息"""
        total_conversations = len(self.conversation_history)
        successful_conversations = sum(1 for conv in self.conversation_history if conv.get("success"))
        
        return {
            "total_conversations": total_conversations,
            "successful_conversations": successful_conversations,
            "success_rate": successful_conversations / total_conversations if total_conversations > 0 else 0,
            "available_tools": len(self.available_tools),
            "max_iterations": self.max_iterations
        }
