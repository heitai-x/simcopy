"""请求管理器"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Set
from collections import deque
import time

from ..models.request import SimpleRequest
# CacheEntry管理已移至独立的缓存组件
from ..models.enums import RequestStatus, CacheStatus

logger = logging.getLogger(__name__)


class RequestManager:
    """请求管理器
    
    负责管理请求的生命周期，包括：
    - 请求的添加、移除和状态更新
    - 请求队列的调度和管理
    - 相似性上下文的管理
    - 请求统计信息的维护
    - 等待重复请求完成的请求记录
    
    注意：重复请求检测逻辑由缓存层（CacheEntry）处理，
    RequestManager仅负责记录WAITING_FOR_DUPLICATE状态的请求
    """
    
    def __init__(self, max_concurrent_requests: int = 100):
        self.max_concurrent_requests = max_concurrent_requests
        
        # 请求存储
        self.requests: Dict[str, SimpleRequest] = {}  # request_id -> SimpleRequest
        self.all_requests = self.requests 
        
        # 状态索引和队列
        self.waiting_requests: Set[str] = set()
        self.running_requests: Set[str] = set()
        self.completed_requests: Set[str] = set()
        self.waiting_for_duplicate_requests: Set[str] = set()  # 等待重复请求完成的request_id
        
        # 调度队列
        self.waiting_queue: deque = deque()  # 等待处理的请求队列
        # 锁
        self._lock = asyncio.Lock()
    
    async def add_request(self, request: SimpleRequest) -> bool:
        """添加请求"""
        async with self._lock:
            try:
                # 检查是否已存在
                if request.request_id in self.requests:
                    logger.debug(f"请求已存在: {request.request_id}")
                    return True
                
                # 检查并发限制
                if len(self.running_requests) >= self.max_concurrent_requests:
                    logger.warning(f"达到最大并发限制: {self.max_concurrent_requests}")
                    return False
                
                # 添加请求
                self.requests[request.request_id] = request
                
                
                # 根据状态添加到相应集合和队列
                if request.status == RequestStatus.WAITING:
                    self.waiting_requests.add(request.request_id)
                    self.waiting_queue.append(request.request_id)
                elif request.status == RequestStatus.RUNNING:
                    self.running_requests.add(request.request_id)
                elif request.status == RequestStatus.COMPLETED:
                    self.completed_requests.add(request.request_id)
                elif request.status == RequestStatus.WAITING_FOR_DUPLICATE:
                    self.waiting_for_duplicate_requests.add(request.request_id)
                
                logger.debug(f"已添加请求: {request.request_id}")
                return True
                
            except Exception as e:
                logger.error(f"添加请求失败: {e}")
                return False
    
    async def get_request(self, request_id: str) -> Optional[SimpleRequest]:
        """获取请求"""
        async with self._lock:
            return self.requests.get(request_id)
    
    async def update_request_status(self, request_id: str, status: RequestStatus) -> bool:
        """更新请求状态"""
        async with self._lock:
            try:
                request = self.requests.get(request_id)
                if not request:
                    logger.warning(f"请求不存在: {request_id}")
                    return False
                
                old_status = request.status
                
                # 从旧状态集合中移除
                if old_status == RequestStatus.WAITING:
                    self.waiting_requests.discard(request_id)
                elif old_status == RequestStatus.RUNNING:
                    self.running_requests.discard(request_id)
                elif old_status == RequestStatus.COMPLETED:
                    self.completed_requests.discard(request_id)
                elif old_status == RequestStatus.WAITING_FOR_DUPLICATE:
                    self.waiting_for_duplicate_requests.discard(request_id)
                
                
                # 添加到新状态集合
                if status == RequestStatus.WAITING:
                    self.waiting_requests.add(request_id)
                elif status == RequestStatus.RUNNING:
                    self.running_requests.add(request_id)
                elif status == RequestStatus.COMPLETED:
                    self.completed_requests.add(request_id)
                    self._stats['completed_requests'] += 1
                elif status == RequestStatus.WAITING_FOR_DUPLICATE:
                    self.waiting_for_duplicate_requests.add(request_id)
                
                logger.debug(f"请求状态已更新: {request_id} {old_status} -> {status}")
                return True
                
            except Exception as e:
                logger.error(f"更新请求状态失败: {e}")
                return False
    
    async def remove_request(self, request_id: str) -> bool:
        """移除请求"""
        async with self._lock:
            try:
                request = self.requests.get(request_id)
                if not request:
                    logger.debug(f"请求不存在: {request_id}")
                    return True
                
                # 注意：hash_id索引清理已移至历史答案管理组件
                
                # 从所有集合中移除
                self.waiting_requests.discard(request_id)
                self.running_requests.discard(request_id)
                self.completed_requests.discard(request_id)
                self.waiting_for_duplicate_requests.discard(request_id)
                
                # 从等待队列中移除
                try:
                    self.waiting_queue.remove(request_id)
                except ValueError:
                    pass  # 如果不在队列中，忽略错误
                
                # 移除请求
                del self.requests[request_id]
                
                # 注意：缓存条目清理已移至独立的缓存管理器
                
                # 清理相似性映射
                if request_id in self.similar_cache_keys:
                    del self.similar_cache_keys[request_id]
                
                if request_id in self.similarity_scores:
                    del self.similarity_scores[request_id]
                
                # 清理其他请求中的相似性引用
                for other_id in list(self.similarity_scores.keys()):
                    if request_id in self.similarity_scores[other_id]:
                        del self.similarity_scores[other_id][request_id]
                
                logger.debug(f"已移除请求: {request_id}")
                return True
                
            except Exception as e:
                logger.error(f"移除请求失败: {e}")
                return False
    
    async def get_waiting_requests(self) -> List[str]:
        """获取等待中的请求"""
        async with self._lock:
            return list(self.waiting_requests)
    
    async def get_running_requests(self) -> List[str]:
        """获取运行中的请求"""
        async with self._lock:
            return list(self.running_requests)
    
    async def get_completed_requests(self) -> List[str]:
        """获取已完成的请求"""
        async with self._lock:
            return list(self.completed_requests)
    
    async def get_waiting_for_duplicate_requests(self) -> List[str]:
        """获取等待重复请求完成的请求"""
        async with self._lock:
            return list(self.waiting_for_duplicate_requests)
    
    async def get_requests_by_status(self, status: RequestStatus) -> List[SimpleRequest]:
        """根据状态获取请求"""
        async with self._lock:
            if status == RequestStatus.WAITING:
                request_ids = self.waiting_requests
            elif status == RequestStatus.RUNNING:
                request_ids = self.running_requests
            elif status == RequestStatus.COMPLETED:
                request_ids = self.completed_requests
            elif status == RequestStatus.WAITING_FOR_DUPLICATE:
                request_ids = self.waiting_for_duplicate_requests
            else:
                return []
            
            result = []
            for request_id in request_ids:
                if request_id in self.requests:
                    result.append(self.requests[request_id])
            return result
    
    async def set_request_result(self, request_id: str, result: Any) -> bool:
        """设置请求结果"""
        async with self._lock:
            try:
                request = self.requests.get(request_id)
                if not request:
                    logger.warning(f"请求不存在: {request_id}")
                    return False
                
                # 设置结果
                request.set_result(result)
                
                # 更新状态
                await self.update_request_status(request_id, RequestStatus.COMPLETED)
                
                # 更新处理时间统计
                if request.created_at:
                    processing_time = time.time() - request.created_at
                    completed_count = self._stats['completed_requests']
                    current_avg = self._stats['avg_processing_time']
                    self._stats['avg_processing_time'] = (
                        (current_avg * (completed_count - 1) + processing_time) / completed_count
                    )
                
                logger.debug(f"已设置请求结果: {request_id}")
                return True
                
            except Exception as e:
                logger.error(f"设置请求结果失败: {e}")
                return False
    
    async def set_request_error(self, request_id: str, error: str) -> bool:
        """设置请求错误"""
        async with self._lock:
            try:
                request = self.requests.get(request_id)
                if not request:
                    logger.warning(f"请求不存在: {request_id}")
                    return False
                
                # 设置错误
                request.set_error(error)
                
                # 更新状态
                await self.update_request_status(request_id, RequestStatus.COMPLETED)
                
                # 更新失败统计
                self._stats['failed_requests'] += 1
                
                logger.debug(f"已设置请求错误: {request_id}")
                return True
                
            except Exception as e:
                logger.error(f"设置请求错误失败: {e}")
                return False
    

    async def cleanup_completed_requests(self, max_age_seconds: int = 3600) -> int:
        """清理已完成的请求"""
        async with self._lock:
            try:
                current_time = time.time()
                to_remove = []
                
                for request_id in self.completed_requests:
                    request = self.requests.get(request_id)
                    if request and request.created_at:
                        age = current_time - request.created_at
                        if age > max_age_seconds:
                            to_remove.append(request_id)
                
                # 移除过期请求
                removed_count = 0
                for request_id in to_remove:
                    if await self.remove_request(request_id):
                        removed_count += 1
                
                logger.info(f"清理了 {removed_count} 个过期请求")
                return removed_count
                
            except Exception as e:
                logger.error(f"清理已完成请求失败: {e}")
                return 0
    
    
    def get_running_requests_count(self) -> int:
        """获取正在运行的请求数量"""
        return len(self.running_requests)
    
    def get_from_waiting_queue(self) -> Optional[str]:
        """从等待队列中获取一个请求ID"""
        if self.waiting_queue:
            return self.waiting_queue.popleft()
        return None
    
    def add_running_request(self, request_id: str) -> bool:
        """将请求添加到运行队列
        
        Args:
            request_id: 请求ID
            
        Returns:
            是否成功添加
        """
        if request_id in self.requests:
            self.running_requests.add(request_id)
            self.waiting_requests.discard(request_id)
            # 更新请求状态
            request = self.requests[request_id]
            request.status = RequestStatus.RUNNING
            return True
        return False
    
    def remove_completed_request(self, request_id: str) -> bool:
        """从运行队列中移除已完成的请求
        
        Args:
            request_id: 请求ID
            
        Returns:
            是否成功移除
        """
        if request_id in self.running_requests:
            self.running_requests.remove(request_id)
            self.completed_requests.add(request_id)
            # 更新请求状态
            if request_id in self.requests:
                request = self.requests[request_id]
                request.status = RequestStatus.COMPLETED
            return True
        return False
    
    
    # ==================== 注意 ====================
    # 包括以下方法：
    # - get_requests_by_hash_id: 根据hash_id获取请求
    # - find_duplicate_request: 查找重复请求
    # - set_waiting_for_duplicate: 设置等待重复请求状态
    # - notify_duplicate_requests: 通知重复请求完成
    # 
    # 这些功能现在由独立的历史答案管理组件处理，
    # 实现了请求管理与缓存/历史答案管理的完全解耦
    
    async def clear_all(self) -> None:
        """清空所有数据"""
        async with self._lock:
            try:
                self.requests.clear()
                self.waiting_requests.clear()
                self.running_requests.clear()
                self.completed_requests.clear()
                self.waiting_for_duplicate_requests.clear()
                self.waiting_queue.clear()

                logger.info("请求管理器已清空")
                
            except Exception as e:
                logger.error(f"清空请求管理器失败: {e}")
    
    def get_request_count(self) -> int:
        """获取请求总数"""
        return len(self.requests)
    
    def get_concurrent_count(self) -> int:
        """获取当前并发数"""
        return len(self.running_requests)
    
    async def is_request_exists(self, request_id: str) -> bool:
        """检查请求是否存在"""
        async with self._lock:
            return request_id in self.requests