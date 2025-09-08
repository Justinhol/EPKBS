"""
异步处理工具
统一异步架构，提供并发控制和错误处理
"""
import asyncio
import functools
from typing import Any, Callable, List, Dict, Optional, Union, Coroutine
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import weakref
import time

from src.utils.logger import get_logger

logger = get_logger("utils.async_utils")


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskResult:
    """任务结果"""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[Exception] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[float] = None


class AsyncTaskManager:
    """异步任务管理器"""
    
    def __init__(self, max_concurrent_tasks: int = 10):
        self.max_concurrent_tasks = max_concurrent_tasks
        self._semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self._tasks: Dict[str, asyncio.Task] = {}
        self._results: Dict[str, TaskResult] = {}
        self._task_counter = 0
        
        logger.info(f"异步任务管理器初始化，最大并发数: {max_concurrent_tasks}")
    
    def _generate_task_id(self) -> str:
        """生成任务ID"""
        self._task_counter += 1
        return f"task_{self._task_counter}_{int(time.time())}"
    
    async def submit_task(
        self,
        coro: Coroutine,
        task_id: str = None,
        timeout: Optional[float] = None
    ) -> str:
        """提交异步任务"""
        task_id = task_id or self._generate_task_id()
        
        # 创建任务结果
        result = TaskResult(
            task_id=task_id,
            status=TaskStatus.PENDING,
            start_time=datetime.utcnow()
        )
        self._results[task_id] = result
        
        # 创建受限制的任务
        limited_coro = self._run_with_semaphore(coro, task_id, timeout)
        task = asyncio.create_task(limited_coro)
        self._tasks[task_id] = task
        
        # 设置任务完成回调
        task.add_done_callback(lambda t: self._on_task_done(task_id, t))
        
        logger.debug(f"提交异步任务: {task_id}")
        return task_id
    
    async def _run_with_semaphore(
        self,
        coro: Coroutine,
        task_id: str,
        timeout: Optional[float]
    ):
        """在信号量控制下运行协程"""
        async with self._semaphore:
            result = self._results[task_id]
            result.status = TaskStatus.RUNNING
            
            try:
                if timeout:
                    task_result = await asyncio.wait_for(coro, timeout=timeout)
                else:
                    task_result = await coro
                
                result.result = task_result
                result.status = TaskStatus.COMPLETED
                return task_result
                
            except asyncio.TimeoutError as e:
                result.error = e
                result.status = TaskStatus.FAILED
                logger.warning(f"任务超时: {task_id}")
                raise
            except asyncio.CancelledError as e:
                result.error = e
                result.status = TaskStatus.CANCELLED
                logger.info(f"任务被取消: {task_id}")
                raise
            except Exception as e:
                result.error = e
                result.status = TaskStatus.FAILED
                logger.error(f"任务执行失败: {task_id}, 错误: {e}")
                raise
    
    def _on_task_done(self, task_id: str, task: asyncio.Task):
        """任务完成回调"""
        result = self._results.get(task_id)
        if result:
            result.end_time = datetime.utcnow()
            if result.start_time:
                result.duration = (result.end_time - result.start_time).total_seconds()
        
        # 清理任务引用
        if task_id in self._tasks:
            del self._tasks[task_id]
    
    async def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> TaskResult:
        """等待任务完成"""
        if task_id not in self._tasks:
            result = self._results.get(task_id)
            if result:
                return result
            else:
                raise ValueError(f"任务不存在: {task_id}")
        
        task = self._tasks[task_id]
        
        try:
            if timeout:
                await asyncio.wait_for(task, timeout=timeout)
            else:
                await task
        except asyncio.TimeoutError:
            logger.warning(f"等待任务超时: {task_id}")
        except Exception as e:
            logger.error(f"等待任务异常: {task_id}, 错误: {e}")
        
        return self._results[task_id]
    
    async def wait_for_all(
        self,
        task_ids: List[str],
        timeout: Optional[float] = None,
        return_when: str = "ALL_COMPLETED"
    ) -> List[TaskResult]:
        """等待多个任务完成"""
        tasks = [self._tasks[task_id] for task_id in task_ids if task_id in self._tasks]
        
        if not tasks:
            return [self._results.get(task_id) for task_id in task_ids]
        
        try:
            if return_when == "ALL_COMPLETED":
                await asyncio.gather(*tasks, return_exceptions=True)
            elif return_when == "FIRST_COMPLETED":
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED, timeout=timeout)
                # 取消未完成的任务
                for task in pending:
                    task.cancel()
            else:
                await asyncio.wait(tasks, timeout=timeout)
                
        except Exception as e:
            logger.error(f"等待多个任务异常: {e}")
        
        return [self._results[task_id] for task_id in task_ids]
    
    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        if task_id in self._tasks:
            task = self._tasks[task_id]
            if not task.done():
                task.cancel()
                logger.info(f"取消任务: {task_id}")
                return True
        return False
    
    def get_task_status(self, task_id: str) -> Optional[TaskResult]:
        """获取任务状态"""
        return self._results.get(task_id)
    
    def get_running_tasks(self) -> List[str]:
        """获取正在运行的任务"""
        return [
            task_id for task_id, result in self._results.items()
            if result.status == TaskStatus.RUNNING
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        status_counts = {}
        for status in TaskStatus:
            status_counts[status.value] = sum(
                1 for result in self._results.values()
                if result.status == status
            )
        
        return {
            "total_tasks": len(self._results),
            "active_tasks": len(self._tasks),
            "max_concurrent": self.max_concurrent_tasks,
            "status_counts": status_counts,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def cleanup_completed_tasks(self, max_age_hours: int = 24):
        """清理已完成的任务"""
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        
        tasks_to_remove = []
        for task_id, result in self._results.items():
            if (result.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED] and
                result.end_time and result.end_time < cutoff_time):
                tasks_to_remove.append(task_id)
        
        for task_id in tasks_to_remove:
            del self._results[task_id]
        
        logger.info(f"清理了 {len(tasks_to_remove)} 个已完成的任务")


class AsyncBatchProcessor:
    """异步批处理器"""
    
    def __init__(
        self,
        batch_size: int = 10,
        max_concurrent_batches: int = 3,
        delay_between_batches: float = 0.1
    ):
        self.batch_size = batch_size
        self.max_concurrent_batches = max_concurrent_batches
        self.delay_between_batches = delay_between_batches
        self.task_manager = AsyncTaskManager(max_concurrent_batches)
        
        logger.info(f"异步批处理器初始化: 批大小={batch_size}, 最大并发批数={max_concurrent_batches}")
    
    async def process_batch(
        self,
        items: List[Any],
        processor_func: Callable,
        **kwargs
    ) -> List[Any]:
        """批量处理项目"""
        if not items:
            return []
        
        # 分批
        batches = [items[i:i + self.batch_size] for i in range(0, len(items), self.batch_size)]
        logger.info(f"分成 {len(batches)} 个批次处理 {len(items)} 个项目")
        
        # 提交批处理任务
        task_ids = []
        for i, batch in enumerate(batches):
            coro = self._process_single_batch(batch, processor_func, **kwargs)
            task_id = await self.task_manager.submit_task(coro, f"batch_{i}")
            task_ids.append(task_id)
            
            # 批次间延迟
            if i < len(batches) - 1:
                await asyncio.sleep(self.delay_between_batches)
        
        # 等待所有批次完成
        results = await self.task_manager.wait_for_all(task_ids)
        
        # 合并结果
        all_results = []
        for result in results:
            if result.status == TaskStatus.COMPLETED and result.result:
                all_results.extend(result.result)
            elif result.status == TaskStatus.FAILED:
                logger.error(f"批处理失败: {result.error}")
        
        return all_results
    
    async def _process_single_batch(
        self,
        batch: List[Any],
        processor_func: Callable,
        **kwargs
    ) -> List[Any]:
        """处理单个批次"""
        results = []
        
        for item in batch:
            try:
                if asyncio.iscoroutinefunction(processor_func):
                    result = await processor_func(item, **kwargs)
                else:
                    result = processor_func(item, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"处理项目失败: {item}, 错误: {e}")
                results.append(None)
        
        return results


def async_retry(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """异步重试装饰器"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"函数 {func.__name__} 第 {attempt + 1} 次尝试失败: {e}, {current_delay}秒后重试")
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        logger.error(f"函数 {func.__name__} 重试 {max_retries} 次后仍然失败")
            
            raise last_exception
        
        return wrapper
    return decorator


def async_timeout(timeout_seconds: float):
    """异步超时装饰器"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout_seconds)
            except asyncio.TimeoutError:
                logger.error(f"函数 {func.__name__} 执行超时 ({timeout_seconds}秒)")
                raise
        
        return wrapper
    return decorator


# 全局任务管理器实例
_global_task_manager = None


def get_global_task_manager() -> AsyncTaskManager:
    """获取全局任务管理器"""
    global _global_task_manager
    
    if _global_task_manager is None:
        _global_task_manager = AsyncTaskManager()
    
    return _global_task_manager
