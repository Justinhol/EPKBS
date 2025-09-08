"""
统一模型管理器
实现模型池、共享机制和延迟加载
"""
import asyncio
import threading
from typing import Dict, Any, Optional, Union, Type
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import weakref
import gc

from src.utils.logger import get_logger
from config.settings import settings

logger = get_logger("utils.model_manager")


class ModelType(Enum):
    """模型类型"""
    LLM = "llm"
    EMBEDDING = "embedding"
    RERANKER = "reranker"
    MULTIMODAL = "multimodal"


@dataclass
class ModelInfo:
    """模型信息"""
    model_type: ModelType
    model_path: str
    device: str
    batch_size: int
    max_memory: Optional[int] = None
    load_time: Optional[datetime] = None
    last_used: Optional[datetime] = None
    use_count: int = 0
    is_loaded: bool = False


class ModelPool:
    """模型池管理器"""
    
    def __init__(self, max_models: int = 3, cleanup_interval: int = 300):
        self.max_models = max_models
        self.cleanup_interval = cleanup_interval  # 5分钟
        self._models: Dict[str, Any] = {}
        self._model_info: Dict[str, ModelInfo] = {}
        self._model_refs: Dict[str, weakref.ref] = {}
        self._lock = threading.RLock()
        self._cleanup_task = None
        
        logger.info(f"模型池初始化，最大模型数: {max_models}")
    
    async def start_cleanup_task(self):
        """启动清理任务"""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
    
    async def _periodic_cleanup(self):
        """定期清理未使用的模型"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_unused_models()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"模型清理任务异常: {e}")
    
    async def _cleanup_unused_models(self):
        """清理未使用的模型"""
        with self._lock:
            current_time = datetime.utcnow()
            models_to_remove = []
            
            for model_key, info in self._model_info.items():
                if info.is_loaded and info.last_used:
                    # 超过30分钟未使用的模型
                    if current_time - info.last_used > timedelta(minutes=30):
                        models_to_remove.append(model_key)
            
            for model_key in models_to_remove:
                await self._unload_model(model_key)
                logger.info(f"清理未使用模型: {model_key}")
    
    async def get_model(
        self,
        model_key: str,
        model_type: ModelType,
        model_path: str = None,
        device: str = "auto",
        batch_size: int = None,
        **kwargs
    ) -> Any:
        """获取模型实例"""
        with self._lock:
            # 检查是否已加载
            if model_key in self._models and self._model_info[model_key].is_loaded:
                model = self._models[model_key]
                self._model_info[model_key].last_used = datetime.utcnow()
                self._model_info[model_key].use_count += 1
                logger.debug(f"复用已加载模型: {model_key}")
                return model
            
            # 检查模型池是否已满
            loaded_count = sum(1 for info in self._model_info.values() if info.is_loaded)
            if loaded_count >= self.max_models:
                # 卸载最久未使用的模型
                await self._evict_least_used_model()
            
            # 加载新模型
            return await self._load_model(
                model_key, model_type, model_path, device, batch_size, **kwargs
            )
    
    async def _load_model(
        self,
        model_key: str,
        model_type: ModelType,
        model_path: str,
        device: str,
        batch_size: int,
        **kwargs
    ) -> Any:
        """加载模型"""
        logger.info(f"正在加载模型: {model_key} ({model_type.value})")
        load_start = datetime.utcnow()
        
        try:
            # 根据模型类型加载
            if model_type == ModelType.LLM:
                model = await self._load_llm_model(model_path, device, **kwargs)
            elif model_type == ModelType.EMBEDDING:
                model = await self._load_embedding_model(model_path, device, batch_size, **kwargs)
            elif model_type == ModelType.RERANKER:
                model = await self._load_reranker_model(model_path, device, batch_size, **kwargs)
            elif model_type == ModelType.MULTIMODAL:
                model = await self._load_multimodal_model(model_path, device, **kwargs)
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")
            
            # 记录模型信息
            load_time = datetime.utcnow()
            self._models[model_key] = model
            self._model_info[model_key] = ModelInfo(
                model_type=model_type,
                model_path=model_path,
                device=device,
                batch_size=batch_size or 32,
                load_time=load_time,
                last_used=load_time,
                use_count=1,
                is_loaded=True
            )
            
            # 创建弱引用
            self._model_refs[model_key] = weakref.ref(model, lambda ref: self._on_model_deleted(model_key))
            
            load_duration = (load_time - load_start).total_seconds()
            logger.info(f"模型加载完成: {model_key}, 耗时: {load_duration:.2f}s")
            
            return model
            
        except Exception as e:
            logger.error(f"模型加载失败: {model_key}, 错误: {e}")
            raise
    
    async def _load_llm_model(self, model_path: str, device: str, **kwargs) -> Any:
        """加载LLM模型"""
        # 这里应该根据实际的LLM库来实现
        # 示例使用transformers
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            # 确定设备
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map=device if device != "cpu" else None,
                **kwargs
            )
            
            return {"model": model, "tokenizer": tokenizer}
            
        except ImportError:
            logger.warning("transformers库未安装，返回模拟模型")
            return {"model": None, "tokenizer": None}
    
    async def _load_embedding_model(self, model_path: str, device: str, batch_size: int, **kwargs) -> Any:
        """加载嵌入模型"""
        try:
            from sentence_transformers import SentenceTransformer
            
            model = SentenceTransformer(model_path, device=device)
            return {"model": model, "batch_size": batch_size}
            
        except ImportError:
            logger.warning("sentence-transformers库未安装，返回模拟模型")
            return {"model": None, "batch_size": batch_size}
    
    async def _load_reranker_model(self, model_path: str, device: str, batch_size: int, **kwargs) -> Any:
        """加载重排序模型"""
        try:
            from sentence_transformers import CrossEncoder
            
            model = CrossEncoder(model_path, device=device)
            return {"model": model, "batch_size": batch_size}
            
        except ImportError:
            logger.warning("重排序模型库未安装，返回模拟模型")
            return {"model": None, "batch_size": batch_size}
    
    async def _load_multimodal_model(self, model_path: str, device: str, **kwargs) -> Any:
        """加载多模态模型"""
        # 这里应该根据实际的多模态模型库来实现
        logger.warning("多模态模型加载未实现，返回模拟模型")
        return {"model": None}
    
    async def _evict_least_used_model(self):
        """驱逐最久未使用的模型"""
        with self._lock:
            if not self._model_info:
                return
            
            # 找到最久未使用的模型
            least_used_key = min(
                (key for key, info in self._model_info.items() if info.is_loaded),
                key=lambda k: self._model_info[k].last_used,
                default=None
            )
            
            if least_used_key:
                await self._unload_model(least_used_key)
                logger.info(f"驱逐最久未使用模型: {least_used_key}")
    
    async def _unload_model(self, model_key: str):
        """卸载模型"""
        with self._lock:
            if model_key in self._models:
                del self._models[model_key]
            
            if model_key in self._model_info:
                self._model_info[model_key].is_loaded = False
            
            if model_key in self._model_refs:
                del self._model_refs[model_key]
            
            # 强制垃圾回收
            gc.collect()
    
    def _on_model_deleted(self, model_key: str):
        """模型被删除时的回调"""
        logger.debug(f"模型被垃圾回收: {model_key}")
        if model_key in self._model_info:
            self._model_info[model_key].is_loaded = False
    
    async def get_model_info(self) -> Dict[str, Dict[str, Any]]:
        """获取模型信息"""
        with self._lock:
            return {
                key: {
                    "model_type": info.model_type.value,
                    "model_path": info.model_path,
                    "device": info.device,
                    "batch_size": info.batch_size,
                    "is_loaded": info.is_loaded,
                    "load_time": info.load_time.isoformat() if info.load_time else None,
                    "last_used": info.last_used.isoformat() if info.last_used else None,
                    "use_count": info.use_count
                }
                for key, info in self._model_info.items()
            }
    
    async def cleanup(self):
        """清理资源"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        with self._lock:
            for model_key in list(self._models.keys()):
                await self._unload_model(model_key)
        
        logger.info("模型池清理完成")


# 全局模型池实例
_model_pool_instance = None


def get_model_pool() -> ModelPool:
    """获取全局模型池实例"""
    global _model_pool_instance
    
    if _model_pool_instance is None:
        _model_pool_instance = ModelPool(
            max_models=getattr(settings, 'model_pool_size', 3)
        )
    
    return _model_pool_instance
