"""多级缓存系统"""

import time
import logging
from typing import Optional, Dict, Any, List
from collections import OrderedDict

from ..models.cache import CacheEntry
from ..models.enums import CacheStatus, CacheLevel
from .redis_manager import RedisManager

logger = logging.getLogger(__name__)


class MultiLevelCache:
    """多级缓存管理器
    
    实现L1内存缓存和L2 Redis缓存的统一管理
    """
    
    def __init__(self, 
                 max_memory_entries: int = 1000,
                 memory_ttl: int = 3600,
                 redis_config: Optional[Dict[str, Any]] = None):
        self.max_memory_entries = max_memory_entries
        self.memory_ttl = memory_ttl
        
        # L1内存缓存 (LRU)
        self._memory_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        
        # L2 Redis缓存
        self._redis_manager = None
        if redis_config and redis_config.get('enable_redis', False):
            try:
                self._redis_manager = RedisManager(
                    host=redis_config.get('redis_host', 'localhost'),
                    port=redis_config.get('redis_port', 6379),
                    db=redis_config.get('redis_db', 0),
                    password=redis_config.get('redis_password'),
                    key_prefix=redis_config.get('redis_prefix', 'vllm_cache:'),
                    ttl=redis_config.get('redis_ttl', 86400)
                )
                logger.info("Redis缓存已启用")
            except Exception as e:
                logger.warning(f"Redis连接失败，仅使用内存缓存: {e}")
                self._redis_manager = None
    
    async def get(self, hash_id: str) -> Optional[CacheEntry]:
        """获取缓存条目"""
        # 1. 先查L1内存缓存
        if hash_id in self._memory_cache:
            entry = self._memory_cache[hash_id]
            # 检查TTL
            if time.time() - entry.created_time < self.memory_ttl:
                # 更新LRU顺序
                self._memory_cache.move_to_end(hash_id)
                entry.update_access_time()
                return entry
            else:
                # 过期，删除
                del self._memory_cache[hash_id]
        
        # 2. 查L2 Redis缓存
        if self._redis_manager:
            try:
                entry = await self._redis_manager.get(hash_id)
                if entry:
                    # 提升到L1缓存
                    await self._put_memory(hash_id, entry)
                    return entry
            except Exception as e:
                logger.warning(f"Redis查询失败: {e}")
        
        return None
    
    async def put(self, hash_id: str, entry: CacheEntry) -> None:
        """存储缓存条目"""
        # 存储到L1内存缓存
        await self._put_memory(hash_id, entry)
        
        # 存储到L2 Redis缓存
        if self._redis_manager:
            try:
                await self._redis_manager.put(hash_id, entry)
            except Exception as e:
                logger.warning(f"Redis存储失败: {e}")
    
    async def _put_memory(self, hash_id: str, entry: CacheEntry) -> None:
        """存储到内存缓存"""
        # LRU淘汰
        print("try_put_memory")
        if hash_id in self._memory_cache:
            self._memory_cache[hash_id] = entry
            self._memory_cache.move_to_end(hash_id)
            return
        if len(self._memory_cache) >= self.max_memory_entries:
            # 删除最旧的条目
            oldest_key = next(iter(self._memory_cache))
            del self._memory_cache[oldest_key]
        
        self._memory_cache[hash_id] = entry
        self._memory_cache.move_to_end(hash_id)
    
    async def remove(self, hash_id: str) -> bool:
        """删除缓存条目"""
        removed = False
        
        # 从L1删除
        if hash_id in self._memory_cache:
            del self._memory_cache[hash_id]
            removed = True
        
        # 从L2删除
        if self._redis_manager:
            try:
                redis_removed = await self._redis_manager.remove(hash_id)
                removed = removed or redis_removed
            except Exception as e:
                logger.warning(f"Redis删除失败: {e}")
        
        return removed
    
    async def exists(self, hash_id: str) -> bool:
        """检查缓存是否存在"""
        # 检查L1
        if hash_id in self._memory_cache:
            entry = self._memory_cache[hash_id]
            if time.time() - entry.created_time < self.memory_ttl:
                return True
            else:
                del self._memory_cache[hash_id]
        
        # 检查L2
        if self._redis_manager:
            try:
                return await self._redis_manager.exists(hash_id)
            except Exception as e:
                logger.warning(f"Redis检查失败: {e}")
        
        return False
    
    async def clear(self) -> None:
        """清空所有缓存"""
        # 清空L1
        self._memory_cache.clear()
        
        # 清空L2
        if self._redis_manager:
            try:
                await self._redis_manager.clear()
            except Exception as e:
                logger.warning(f"Redis清空失败: {e}")
    
  
    async def cleanup_expired(self) -> int:
        """清理过期条目"""
        current_time = time.time()
        expired_keys = []
        
        for hash_id, entry in self._memory_cache.items():
            if current_time - entry.created_time > self.memory_ttl:
                expired_keys.append(hash_id)
        
        for key in expired_keys:
            del self._memory_cache[key]
        
        logger.info(f"清理了 {len(expired_keys)} 个过期缓存条目")
        return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息
        
        Returns:
            包含缓存统计信息的字典
        """
        current_time = time.time()
        
        # 计算内存缓存统计
        memory_stats = {
            "total_entries": len(self._memory_cache),
            "max_entries": self.max_memory_entries,
            "usage_ratio": len(self._memory_cache) / self.max_memory_entries if self.max_memory_entries > 0 else 0,
            "ttl_seconds": self.memory_ttl
        }
        
        # 计算过期条目数量
        expired_count = 0
        for entry in self._memory_cache.values():
            if current_time - entry.created_time > self.memory_ttl:
                expired_count += 1
        
        memory_stats["expired_entries"] = expired_count
        memory_stats["active_entries"] = len(self._memory_cache) - expired_count
        
        stats = {
            "cache_type": "MultiLevelCache",
            "memory_cache": memory_stats,
            "redis_enabled": self._redis_manager is not None,
            "timestamp": current_time
        }
        
        # 如果Redis可用，添加Redis统计信息
        if self._redis_manager:
            try:
                # 这里可以添加Redis统计信息，如果RedisManager有相应方法
                stats["redis_cache"] = {
                    "status": "connected",
                    "host": getattr(self._redis_manager, 'host', 'unknown'),
                    "port": getattr(self._redis_manager, 'port', 'unknown')
                }
            except Exception as e:
                stats["redis_cache"] = {
                    "status": "error",
                    "error": str(e)
                }
        else:
            stats["redis_cache"] = {
                "status": "disabled"
            }
        
        return stats
