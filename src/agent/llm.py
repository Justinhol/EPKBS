"""
大语言模型管理器
支持Qwen3系列模型和其他开源模型
"""
import asyncio
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
from abc import ABC, abstractmethod

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextStreamer,
    GenerationConfig
)
import torch
from langchain.llms.base import LLM
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage

from src.utils.logger import get_logger
from src.utils.helpers import Timer
from config.settings import settings

logger = get_logger("agent.llm")


class BaseLLMManager(ABC):
    """LLM管理器基类"""
    
    def __init__(self, model_name: str, device: str = None):
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        
        logger.info(f"LLM管理器初始化: {model_name}, 设备: {self.device}")
    
    @abstractmethod
    async def initialize(self):
        """初始化模型"""
        pass
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        max_tokens: int = None,
        temperature: float = None,
        **kwargs
    ) -> str:
        """生成文本"""
        pass
    
    @abstractmethod
    async def chat(
        self,
        messages: List[BaseMessage],
        max_tokens: int = None,
        temperature: float = None,
        **kwargs
    ) -> str:
        """对话生成"""
        pass


class QwenLLMManager(BaseLLMManager):
    """Qwen模型管理器"""
    
    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False
    ):
        model_name = model_name or settings.QWEN_MODEL_PATH
        super().__init__(model_name, device)
        
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.generation_config = None
        
        logger.info(f"Qwen模型管理器配置: 8bit={load_in_8bit}, 4bit={load_in_4bit}")
    
    async def initialize(self):
        """初始化Qwen模型"""
        try:
            logger.info(f"正在加载Qwen模型: {self.model_name}")
            
            # 加载tokenizer
            logger.info("加载tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side='left'
            )
            
            # 设置pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 加载模型
            logger.info("加载模型...")
            model_kwargs = {
                'trust_remote_code': True,
                'torch_dtype': torch.float16 if self.device == 'cuda' else torch.float32,
                'device_map': 'auto' if self.device == 'cuda' else None,
            }
            
            if self.load_in_8bit:
                model_kwargs['load_in_8bit'] = True
            elif self.load_in_4bit:
                model_kwargs['load_in_4bit'] = True
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            if self.device == 'cpu':
                self.model = self.model.to(self.device)
            
            # 设置生成配置
            self.generation_config = GenerationConfig(
                max_new_tokens=settings.MAX_TOKENS,
                temperature=settings.TEMPERATURE,
                top_p=settings.TOP_P,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
            self.is_loaded = True
            logger.info("Qwen模型加载完成")
            
        except Exception as e:
            logger.error(f"Qwen模型加载失败: {e}")
            # 尝试降级到更小的模型
            await self._try_fallback_model()
    
    async def _try_fallback_model(self):
        """尝试降级模型"""
        fallback_models = [
            "Qwen/Qwen2.5-3B-Instruct",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "microsoft/DialoGPT-medium"
        ]
        
        for fallback_model in fallback_models:
            try:
                logger.warning(f"尝试降级到模型: {fallback_model}")
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    fallback_model,
                    trust_remote_code=True
                )
                
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    fallback_model,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                    device_map=None
                ).to(self.device)
                
                self.model_name = fallback_model
                self.is_loaded = True
                logger.info(f"降级模型 {fallback_model} 加载成功")
                return
                
            except Exception as e:
                logger.warning(f"降级模型 {fallback_model} 加载失败: {e}")
                continue
        
        raise RuntimeError("所有模型加载失败")
    
    async def generate(
        self,
        prompt: str,
        max_tokens: int = None,
        temperature: float = None,
        **kwargs
    ) -> str:
        """生成文本"""
        if not self.is_loaded:
            await self.initialize()
        
        try:
            with Timer() as timer:
                # 编码输入
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048
                ).to(self.device)
                
                # 更新生成配置
                gen_config = self.generation_config
                if max_tokens:
                    gen_config.max_new_tokens = max_tokens
                if temperature:
                    gen_config.temperature = temperature
                
                # 生成文本
                with torch.no_grad():
                    outputs = await asyncio.to_thread(
                        self.model.generate,
                        **inputs,
                        generation_config=gen_config,
                        **kwargs
                    )
                
                # 解码输出
                generated_text = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
            
            logger.info(f"文本生成完成，耗时: {timer.elapsed:.2f}秒，长度: {len(generated_text)}")
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"文本生成失败: {e}")
            return ""
    
    async def chat(
        self,
        messages: List[BaseMessage],
        max_tokens: int = None,
        temperature: float = None,
        **kwargs
    ) -> str:
        """对话生成"""
        if not self.is_loaded:
            await self.initialize()
        
        try:
            # 构建对话prompt
            chat_prompt = self._build_chat_prompt(messages)
            
            # 生成回复
            response = await self.generate(
                chat_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            return response
            
        except Exception as e:
            logger.error(f"对话生成失败: {e}")
            return ""
    
    def _build_chat_prompt(self, messages: List[BaseMessage]) -> str:
        """构建对话prompt"""
        prompt_parts = []
        
        for message in messages:
            if isinstance(message, SystemMessage):
                prompt_parts.append(f"System: {message.content}")
            elif isinstance(message, HumanMessage):
                prompt_parts.append(f"Human: {message.content}")
            elif isinstance(message, AIMessage):
                prompt_parts.append(f"Assistant: {message.content}")
        
        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)
    
    async def stream_generate(
        self,
        prompt: str,
        max_tokens: int = None,
        temperature: float = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """流式生成文本"""
        if not self.is_loaded:
            await self.initialize()
        
        try:
            # 编码输入
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            # 更新生成配置
            gen_config = self.generation_config
            if max_tokens:
                gen_config.max_new_tokens = max_tokens
            if temperature:
                gen_config.temperature = temperature
            
            # 流式生成
            streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            
            with torch.no_grad():
                # 这里简化实现，实际应该使用真正的流式生成
                outputs = await asyncio.to_thread(
                    self.model.generate,
                    **inputs,
                    generation_config=gen_config,
                    streamer=streamer,
                    **kwargs
                )
                
                generated_text = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
                
                # 模拟流式输出
                words = generated_text.split()
                for word in words:
                    yield word + " "
                    await asyncio.sleep(0.01)  # 模拟延迟
            
        except Exception as e:
            logger.error(f"流式生成失败: {e}")
            yield ""
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'is_loaded': self.is_loaded,
            'load_in_8bit': self.load_in_8bit,
            'load_in_4bit': self.load_in_4bit,
            'max_tokens': settings.MAX_TOKENS,
            'temperature': settings.TEMPERATURE,
            'top_p': settings.TOP_P,
        }


class LangChainLLMWrapper(LLM):
    """LangChain LLM包装器"""
    
    def __init__(self, llm_manager: BaseLLMManager):
        super().__init__()
        self.llm_manager = llm_manager
    
    @property
    def _llm_type(self) -> str:
        return "qwen_custom"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """同步调用（LangChain要求）"""
        # 在实际应用中，这里应该使用同步版本或者事件循环
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果已经在事件循环中，创建新的任务
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.llm_manager.generate(prompt))
                    return future.result()
            else:
                return asyncio.run(self.llm_manager.generate(prompt))
        except Exception as e:
            logger.error(f"LangChain包装器调用失败: {e}")
            return ""
    
    async def acall(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """异步调用"""
        return await self.llm_manager.generate(prompt)


class LLMFactory:
    """LLM工厂类"""
    
    @staticmethod
    def create_llm(
        model_type: str = "qwen",
        model_name: str = None,
        **kwargs
    ) -> BaseLLMManager:
        """创建LLM管理器"""
        
        if model_type.lower() == "qwen":
            return QwenLLMManager(model_name=model_name, **kwargs)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
    
    @staticmethod
    def create_langchain_llm(
        model_type: str = "qwen",
        model_name: str = None,
        **kwargs
    ) -> LangChainLLMWrapper:
        """创建LangChain兼容的LLM"""
        llm_manager = LLMFactory.create_llm(model_type, model_name, **kwargs)
        return LangChainLLMWrapper(llm_manager)
