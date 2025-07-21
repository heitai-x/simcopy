"""请求数据模型"""

import time
import asyncio
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from vllm import SamplingParams

from .enums import RequestStatus


@dataclass
class SimpleRequest:
    """简化的请求数据模型"""
    request_id: str
    prompt: str
    sampling_params: SamplingParams
    timestamp: float = field(default_factory=time.time)
    future: Optional[asyncio.Future] = field(default=None, init=False)
    status: RequestStatus = field(default=RequestStatus.WAITING)
    hash_id: Optional[str] = field(default=None)
    similar_cache_keys: List[str] = field(default_factory=list)
    similarity_scores: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.future is None:
            self.future = asyncio.Future()
    
    def is_waiting(self) -> bool:
        """检查是否在等待状态"""
        return self.status == RequestStatus.WAITING
    
    def is_running(self) -> bool:
        """检查是否在运行状态"""
        return self.status == RequestStatus.RUNNING
    
    def is_completed(self) -> bool:
        """检查是否已完成"""
        return self.status == RequestStatus.COMPLETED
    
    def is_failed(self) -> bool:
        """检查是否失败"""
        return self.status == RequestStatus.FAILED
    
    def get_elapsed_time(self) -> float:
        """获取已用时间"""
        return time.time() - self.timestamp
    
    def set_result(self, result: str) -> None:
        """设置请求结果"""
        if not self.future.done():
            self.future.set_result(result)
    
    def set_error(self, error: Exception) -> None:
        """设置请求错误"""
        if not self.future.done():
            self.future.set_exception(error)
    
    async def get_result(self) -> str:
        """获取请求结果"""
        return await self.future