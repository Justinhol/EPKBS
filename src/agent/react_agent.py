"""
ReAct Agent实现
基于Reasoning + Acting框架的智能代理
"""
import asyncio
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from enum import Enum

from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage

from src.agent.llm import BaseLLMManager
from src.agent.tools import ToolManager, ToolResult
from src.utils.logger import get_logger
from src.utils.helpers import Timer

logger = get_logger("agent.react")


class AgentState(Enum):
    """Agent状态"""
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    FINISHED = "finished"
    ERROR = "error"


class ReActStep:
    """ReAct步骤"""
    
    def __init__(
        self,
        step_number: int,
        thought: str = "",
        action: str = "",
        action_input: Dict[str, Any] = None,
        observation: str = "",
        state: AgentState = AgentState.THINKING
    ):
        self.step_number = step_number
        self.thought = thought
        self.action = action
        self.action_input = action_input or {}
        self.observation = observation
        self.state = state
        self.timestamp = datetime.utcnow().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'step_number': self.step_number,
            'thought': self.thought,
            'action': self.action,
            'action_input': self.action_input,
            'observation': self.observation,
            'state': self.state.value,
            'timestamp': self.timestamp
        }


class ReActAgent:
    """ReAct Agent"""
    
    def __init__(
        self,
        llm_manager: BaseLLMManager,
        tool_manager: ToolManager,
        max_steps: int = 10,
        max_execution_time: int = 300
    ):
        self.llm_manager = llm_manager
        self.tool_manager = tool_manager
        self.max_steps = max_steps
        self.max_execution_time = max_execution_time
        
        self.steps: List[ReActStep] = []
        self.current_state = AgentState.THINKING
        self.execution_start_time = None
        
        logger.info(f"ReAct Agent初始化完成，最大步数: {max_steps}")
    
    def _build_system_prompt(self) -> str:
        """构建系统提示"""
        tools_info = []
        for tool_schema in self.tool_manager.get_tools_schema():
            tools_info.append(f"- {tool_schema['name']}: {tool_schema['description']}")
        
        tools_list = "\n".join(tools_info)
        
        return f"""你是一个智能助手，能够通过推理和行动来解决问题。

可用工具:
{tools_list}

请按照以下格式进行推理和行动:

Thought: [你的思考过程]
Action: [工具名称]
Action Input: [工具参数，JSON格式]
Observation: [工具执行结果]

重复上述过程直到找到答案，然后以以下格式结束:

Thought: [最终思考]
Final Answer: [最终答案]

重要规则:
1. 每次只能执行一个Action
2. Action Input必须是有效的JSON格式
3. 仔细分析Observation的结果
4. 如果工具执行失败，尝试其他方法
5. 给出准确、有用的最终答案

现在开始解决问题。"""
    
    def _parse_llm_output(self, output: str) -> Tuple[str, str, Dict[str, Any]]:
        """解析LLM输出"""
        thought = ""
        action = ""
        action_input = {}
        
        # 提取Thought
        thought_match = re.search(r'Thought:\s*(.*?)(?=\n(?:Action|Final Answer):|$)', output, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()
        
        # 检查是否是最终答案
        final_answer_match = re.search(r'Final Answer:\s*(.*)', output, re.DOTALL)
        if final_answer_match:
            return thought, "final_answer", {"answer": final_answer_match.group(1).strip()}
        
        # 提取Action
        action_match = re.search(r'Action:\s*(.*?)(?=\n|$)', output)
        if action_match:
            action = action_match.group(1).strip()
        
        # 提取Action Input
        action_input_match = re.search(r'Action Input:\s*(.*?)(?=\n(?:Observation|Thought|Final Answer):|$)', output, re.DOTALL)
        if action_input_match:
            action_input_str = action_input_match.group(1).strip()
            try:
                action_input = json.loads(action_input_str)
            except json.JSONDecodeError:
                logger.warning(f"无法解析Action Input: {action_input_str}")
                action_input = {"raw_input": action_input_str}
        
        return thought, action, action_input
    
    def _build_conversation_history(self, query: str) -> str:
        """构建对话历史"""
        history_parts = [f"Human: {query}"]
        
        for step in self.steps:
            if step.thought:
                history_parts.append(f"Thought: {step.thought}")
            if step.action and step.action != "final_answer":
                history_parts.append(f"Action: {step.action}")
                if step.action_input:
                    history_parts.append(f"Action Input: {json.dumps(step.action_input, ensure_ascii=False)}")
            if step.observation:
                history_parts.append(f"Observation: {step.observation}")
        
        return "\n".join(history_parts)
    
    async def run(self, query: str) -> Dict[str, Any]:
        """运行Agent"""
        logger.info(f"ReAct Agent开始执行: {query[:100]}...")
        
        self.steps = []
        self.current_state = AgentState.THINKING
        self.execution_start_time = datetime.utcnow()
        
        try:
            with Timer() as total_timer:
                # 构建系统消息
                system_prompt = self._build_system_prompt()
                
                for step_num in range(1, self.max_steps + 1):
                    # 检查执行时间
                    if total_timer.elapsed > self.max_execution_time:
                        logger.warning(f"Agent执行超时: {total_timer.elapsed}秒")
                        break
                    
                    logger.info(f"执行步骤 {step_num}")
                    
                    # 构建当前对话历史
                    conversation_history = self._build_conversation_history(query)
                    
                    # 构建完整prompt
                    full_prompt = f"{system_prompt}\n\n{conversation_history}\n"
                    
                    # 生成LLM响应
                    llm_output = await self.llm_manager.generate(
                        full_prompt,
                        max_tokens=1000,
                        temperature=0.1
                    )
                    
                    if not llm_output:
                        logger.error("LLM生成失败")
                        break
                    
                    # 解析LLM输出
                    thought, action, action_input = self._parse_llm_output(llm_output)
                    
                    # 创建步骤
                    step = ReActStep(
                        step_number=step_num,
                        thought=thought,
                        action=action,
                        action_input=action_input,
                        state=AgentState.THINKING
                    )
                    
                    # 检查是否是最终答案
                    if action == "final_answer":
                        step.state = AgentState.FINISHED
                        step.observation = "任务完成"
                        self.steps.append(step)
                        
                        final_answer = action_input.get("answer", "")
                        logger.info(f"Agent完成任务，最终答案: {final_answer[:100]}...")
                        
                        return {
                            'success': True,
                            'final_answer': final_answer,
                            'steps': [s.to_dict() for s in self.steps],
                            'total_steps': len(self.steps),
                            'execution_time': total_timer.elapsed,
                            'status': 'completed'
                        }
                    
                    # 执行工具
                    if action:
                        step.state = AgentState.ACTING
                        logger.info(f"执行工具: {action}, 参数: {action_input}")
                        
                        tool_result = await self.tool_manager.execute_tool(action, action_input)
                        
                        step.state = AgentState.OBSERVING
                        
                        if tool_result.success:
                            step.observation = f"工具执行成功: {json.dumps(tool_result.result, ensure_ascii=False)}"
                        else:
                            step.observation = f"工具执行失败: {tool_result.error}"
                        
                        logger.info(f"工具执行结果: 成功={tool_result.success}")
                    else:
                        step.observation = "未识别到有效的Action"
                        logger.warning("LLM输出中未找到有效的Action")
                    
                    self.steps.append(step)
                
                # 如果循环结束但没有最终答案
                logger.warning(f"Agent达到最大步数限制: {self.max_steps}")
                return {
                    'success': False,
                    'final_answer': "抱歉，我无法在限定步数内完成任务。",
                    'steps': [s.to_dict() for s in self.steps],
                    'total_steps': len(self.steps),
                    'execution_time': total_timer.elapsed,
                    'status': 'max_steps_reached'
                }
        
        except Exception as e:
            logger.error(f"Agent执行异常: {e}")
            return {
                'success': False,
                'final_answer': f"执行过程中发生错误: {str(e)}",
                'steps': [s.to_dict() for s in self.steps],
                'total_steps': len(self.steps),
                'execution_time': 0,
                'status': 'error',
                'error': str(e)
            }
    
    async def stream_run(self, query: str):
        """流式运行Agent"""
        logger.info(f"ReAct Agent开始流式执行: {query[:100]}...")
        
        self.steps = []
        self.current_state = AgentState.THINKING
        self.execution_start_time = datetime.utcnow()
        
        try:
            # 构建系统消息
            system_prompt = self._build_system_prompt()
            
            yield {
                'type': 'start',
                'query': query,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            for step_num in range(1, self.max_steps + 1):
                yield {
                    'type': 'step_start',
                    'step_number': step_num,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                # 构建当前对话历史
                conversation_history = self._build_conversation_history(query)
                full_prompt = f"{system_prompt}\n\n{conversation_history}\n"
                
                # 生成LLM响应
                yield {
                    'type': 'thinking',
                    'step_number': step_num,
                    'message': '正在思考...'
                }
                
                llm_output = await self.llm_manager.generate(
                    full_prompt,
                    max_tokens=1000,
                    temperature=0.1
                )
                
                if not llm_output:
                    yield {
                        'type': 'error',
                        'message': 'LLM生成失败'
                    }
                    break
                
                # 解析LLM输出
                thought, action, action_input = self._parse_llm_output(llm_output)
                
                yield {
                    'type': 'thought',
                    'step_number': step_num,
                    'thought': thought
                }
                
                # 创建步骤
                step = ReActStep(
                    step_number=step_num,
                    thought=thought,
                    action=action,
                    action_input=action_input,
                    state=AgentState.THINKING
                )
                
                # 检查是否是最终答案
                if action == "final_answer":
                    final_answer = action_input.get("answer", "")
                    
                    yield {
                        'type': 'final_answer',
                        'answer': final_answer,
                        'step_number': step_num
                    }
                    
                    step.state = AgentState.FINISHED
                    step.observation = "任务完成"
                    self.steps.append(step)
                    
                    yield {
                        'type': 'completed',
                        'total_steps': len(self.steps),
                        'final_answer': final_answer
                    }
                    return
                
                # 执行工具
                if action:
                    yield {
                        'type': 'action',
                        'step_number': step_num,
                        'action': action,
                        'action_input': action_input
                    }
                    
                    step.state = AgentState.ACTING
                    tool_result = await self.tool_manager.execute_tool(action, action_input)
                    step.state = AgentState.OBSERVING
                    
                    if tool_result.success:
                        step.observation = f"工具执行成功: {json.dumps(tool_result.result, ensure_ascii=False)}"
                    else:
                        step.observation = f"工具执行失败: {tool_result.error}"
                    
                    yield {
                        'type': 'observation',
                        'step_number': step_num,
                        'observation': step.observation,
                        'success': tool_result.success
                    }
                else:
                    step.observation = "未识别到有效的Action"
                    yield {
                        'type': 'observation',
                        'step_number': step_num,
                        'observation': step.observation,
                        'success': False
                    }
                
                self.steps.append(step)
            
            # 达到最大步数
            yield {
                'type': 'max_steps_reached',
                'total_steps': len(self.steps),
                'message': '达到最大步数限制'
            }
            
        except Exception as e:
            yield {
                'type': 'error',
                'error': str(e),
                'message': f'执行过程中发生错误: {str(e)}'
            }
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """获取执行摘要"""
        if not self.steps:
            return {'status': 'not_started'}
        
        total_time = 0
        if self.execution_start_time:
            total_time = (datetime.utcnow() - datetime.fromisoformat(self.execution_start_time.replace('Z', '+00:00'))).total_seconds()
        
        return {
            'total_steps': len(self.steps),
            'current_state': self.current_state.value,
            'execution_time': total_time,
            'tools_used': list(set(step.action for step in self.steps if step.action and step.action != "final_answer")),
            'success_rate': sum(1 for step in self.steps if step.state != AgentState.ERROR) / len(self.steps) if self.steps else 0,
            'last_step': self.steps[-1].to_dict() if self.steps else None
        }
