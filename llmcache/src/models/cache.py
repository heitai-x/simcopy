"""缓存数据模型"""

import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from .enums import CacheStatus


@dataclass
class CacheEntry:
    """缓存条目数据模型"""
    hash_id: str
    status: CacheStatus = field(default=CacheStatus.PROCESSING)
    result_tokens: Optional[List[int]] = field(default=None)  # 改为存储token ID列表
    result: Optional[str] = field(default=None)
    created_time: float = field(default_factory=time.time)
    last_access_time: float = field(default_factory=time.time)
    access_count: int = field(default=0)
    original_request_id: Optional[str] = field(default=None)
    waiting_requests: List[str] = field(default_factory=list)
    
    def update_access_time(self) -> None:
        """更新访问时间"""
        self.last_access_time = time.time()
        self.access_count += 1
    
    def get_age(self) -> float:
        """获取缓存年龄（秒）"""
        return time.time() - self.created_time
    
    def get_idle_time(self) -> float:
        """获取空闲时间（秒）"""
        return time.time() - self.last_access_time
    
    def is_completed(self) -> bool:
        """检查是否已完成"""
        return self.status == CacheStatus.COMPLETED
    
    def is_processing(self) -> bool:
        """检查是否正在处理"""
        return self.status == CacheStatus.PROCESSING
    
    def add_waiting_request(self, request_id: str) -> None:
        """添加等待请求"""
        if request_id not in self.waiting_requests:
            self.waiting_requests.append(request_id)
    
    def remove_waiting_request(self, request_id: str) -> None:
        """移除等待请求"""
        if request_id in self.waiting_requests:
            self.waiting_requests.remove(request_id)
    
    def get_waiting_count(self) -> int:
        """获取等待请求数量"""
        return len(self.waiting_requests)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        data = {
            'hash_id': self.hash_id,
            'status': self.status.value,
            'result': self.result,
            'created_time': self.created_time,
            'last_access_time': self.last_access_time,
            'access_count': self.access_count,
            'original_request_id': self.original_request_id,
            'waiting_requests': self.waiting_requests
        }
        # 直接存储token列表，无需转换
        if self.result_tokens is not None:
            data['result_tokens'] = self.result_tokens
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """从字典创建缓存条目"""
        entry = cls(
            hash_id=data['hash_id'],
            status=CacheStatus(data['status']),
            result=data.get('result'),
            created_time=data.get('created_time', time.time()),
            last_access_time=data.get('last_access_time', time.time()),
            access_count=data.get('access_count', 0),
            original_request_id=data.get('original_request_id'),
            waiting_requests=data.get('waiting_requests', []),
            result_tokens=data.get('result_tokens')  # 直接获取token列表
        )
        return entry
    