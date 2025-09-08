"""
统一大语言模型管理器
支持模型池、缓存和异步处理
"""
import asyncio
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
from abc import ABC, abstractmethod
from datetime import datetime

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
from src.utils.model_manager import ModelPool, ModelType
from src.utils.cache_manager import UnifiedCacheManager, CacheLevel
from config.settings import settings

logger = get_logger("agent.llm")


class BaseLLMManager(ABC):
    """LLM管理器基类"""

    def __init__(self, model_name: str, device: str = None):
        self.model_name = model_name
        self.device = device or (
            'cuda' if torch.cuda.is_available() else 'cpu')
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

            logger.info(
                f"文本生成完成，耗时: {timer.elapsed:.2f}秒，长度: {len(generated_text)}")
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
            streamer = TextStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True)

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
                    future = executor.submit(
                        asyncio.run, self.llm_manager.generate(prompt))
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


class UnifiedLLMManager:
    """统一LLM管理器 - 使用模型池和缓存"""

    def __init__(
        self,
        model_pool: ModelPool = None,
        cache_manager: UnifiedCacheManager = None,
        model_name: str = None,
        device: str = "auto"
    ):
        self.model_pool = model_pool
        self.cache_manager = cache_manager
        self.model_name = model_name or settings.model.qwen_model_path
        self.device = device
        self.model_key = f"llm_{self.model_name.replace('/', '_')}"

        # 性能统计
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_tokens_generated": 0,
            "average_generation_time": 0.0
        }

        logger.info(f"统一LLM管理器初始化: {self.model_name}, 设备: {device}")

    async def initialize(self):
        """初始化LLM管理器"""
        logger.info("初始化统一LLM管理器...")

        try:
            # 预加载模型（可选）
            if self.model_pool:
                await self._ensure_model_loaded()

            logger.info("统一LLM管理器初始化完成")

        except Exception as e:
            logger.error(f"统一LLM管理器初始化失败: {e}")
            raise

    async def _ensure_model_loaded(self) -> Dict[str, Any]:
        """确保模型已加载"""
        if not self.model_pool:
            raise RuntimeError("模型池未初始化")

        return await self.model_pool.get_model(
            model_key=self.model_key,
            model_type=ModelType.LLM,
            model_path=self.model_name,
            device=self.device
        )

    async def generate(
        self,
        prompt: str,
        max_tokens: int = None,
        temperature: float = None,
        use_cache: bool = True,
        **kwargs
    ) -> str:
        """生成文本"""
        start_time = datetime.utcnow()
        self.stats["total_requests"] += 1

        # 生成缓存键
        cache_key = self._generate_cache_key(
            prompt, max_tokens, temperature, **kwargs)

        # 尝试从缓存获取
        if use_cache and self.cache_manager:
            cached_result = await self.cache_manager.get(
                cache_key,
                namespace="llm_generation",
                level=CacheLevel.BOTH
            )
            if cached_result:
                self.stats["cache_hits"] += 1
                logger.debug(f"LLM缓存命中: {cache_key[:32]}...")
                return cached_result
            else:
                self.stats["cache_misses"] += 1

        try:
            # 获取模型
            model_info = await self._ensure_model_loaded()
            model = model_info.get("model")
            tokenizer = model_info.get("tokenizer")

            if not model or not tokenizer:
                raise RuntimeError("模型或分词器未正确加载")

            # 生成文本
            result = await self._generate_with_model(
                model, tokenizer, prompt, max_tokens, temperature, **kwargs
            )

            # 缓存结果
            if use_cache and self.cache_manager:
                await self.cache_manager.set(
                    cache_key,
                    result,
                    namespace="llm_generation",
                    ttl=settings.cache.model_cache_ttl,
                    level=CacheLevel.BOTH
                )

            # 更新统计
            generation_time = (datetime.utcnow() - start_time).total_seconds()
            self._update_stats(generation_time, len(result.split()))

            return result

        except Exception as e:
            logger.error(f"LLM生成失败: {e}")
            raise

    async def _generate_with_model(
        self,
        model,
        tokenizer,
        prompt: str,
        max_tokens: int = None,
        temperature: float = None,
        **kwargs
    ) -> str:
        """使用模型生成文本"""
        max_tokens = max_tokens or settings.model.max_tokens
        temperature = temperature or settings.model.temperature

        try:
            # 编码输入
            inputs = tokenizer.encode(prompt, return_tensors="pt")
            if hasattr(model, 'device'):
                inputs = inputs.to(model.device)

            # 生成配置
            generation_config = GenerationConfig(
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=settings.model.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                **kwargs
            )

            # 生成文本
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    generation_config=generation_config
                )

            # 解码输出
            generated_text = tokenizer.decode(
                outputs[0][inputs.shape[1]:],
                skip_special_tokens=True
            )

            return generated_text.strip()

        except Exception as e:
            logger.error(f"模型生成异常: {e}")
            raise

    def _generate_cache_key(
        self,
        prompt: str,
        max_tokens: int = None,
        temperature: float = None,
        **kwargs
    ) -> str:
        """生成缓存键"""
        import hashlib

        cache_data = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }

        cache_str = str(sorted(cache_data.items()))
        return hashlib.md5(cache_str.encode()).hexdigest()

    def _update_stats(self, generation_time: float, token_count: int):
        """更新统计信息"""
        # 更新平均生成时间
        current_avg = self.stats["average_generation_time"]
        total_requests = self.stats["total_requests"]

        if total_requests > 1:
            self.stats["average_generation_time"] = (
                (current_avg * (total_requests - 1) +
                 generation_time) / total_requests
            )
        else:
            self.stats["average_generation_time"] = generation_time

        # 更新token统计
        self.stats["total_tokens_generated"] += token_count

    async def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.stats.copy()
        stats["timestamp"] = datetime.utcnow().isoformat()
        stats["model_name"] = self.model_name
        stats["device"] = self.device

        return stats


# 为了向后兼容，创建别名
LLMManager = UnifiedLLMManager
