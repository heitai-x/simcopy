"""Redis缓存管理器"""

import json
import logging
from typing import Optional, Dict, Any

try:
    import redis.asyncio as redis
except ImportError:
    import redis

from ..models.cache import CacheEntry

logger = logging.getLogger(__name__)


class RedisManager:
    """Redis缓存管理器"""
    
    def __init__(self, 
                 host: str = "localhost",
                 port: int = 6379,
                 db: int = 0,
                 password: Optional[str] = None,
                 key_prefix: str = "vllm_cache:",
                 ttl: int = 86400):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.key_prefix = key_prefix
        self.ttl = ttl
        self._client: Optional[redis.Redis] = None
        self.is_connected = False
    
    async def connect(self) -> bool:
        """连接到Redis"""
        try:
            self._client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True
            )
            
            # 测试连接
            await self._client.ping()
            self.is_connected = True
            logger.info(f"Redis连接成功: {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Redis连接失败: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self) -> None:
        """断开Redis连接"""
        if self._client:
            await self._client.close()
            self.is_connected = False
            logger.info("Redis连接已断开")
    
    def _get_key(self, hash_id: str) -> str:
        """获取完整的Redis键名"""
        return f"{self.key_prefix}{hash_id}"
    
    async def get(self, hash_id: str) -> Optional[CacheEntry]:
        """获取缓存条目"""
        if not self.is_connected:
            await self.connect()
        
        if not self.is_connected:
            return None
        
        try:
            key = self._get_key(hash_id)
            data = await self._client.get(key)
            
            if data:
                cache_data = json.loads(data)
                return CacheEntry.from_dict(cache_data)
            
            return None
        except Exception as e:
            logger.error(f"Redis获取失败 {hash_id}: {e}")
            return None
    
    async def put(self, hash_id: str, entry: CacheEntry) -> bool:
        """存储缓存条目"""
        if not self.is_connected:
            await self.connect()
        
        if not self.is_connected:
            return False
        
        try:
            key = self._get_key(hash_id)
            data = json.dumps(entry.to_dict())
            
            await self._client.setex(key, self.ttl, data)
            return True
        except Exception as e:
            logger.error(f"Redis存储失败 {hash_id}: {e}")
            return False
    
    async def remove(self, hash_id: str) -> bool:
        """删除缓存条目"""
        if not self.is_connected:
            return False
        
        try:
            key = self._get_key(hash_id)
            result = await self._client.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Redis删除失败 {hash_id}: {e}")
            return False
    
    async def exists(self, hash_id: str) -> bool:
        """检查缓存是否存在"""
        if not self.is_connected:
            return False
        
        try:
            key = self._get_key(hash_id)
            result = await self._client.exists(key)
            return result > 0
        except Exception as e:
            logger.error(f"Redis检查失败 {hash_id}: {e}")
            return False
    
    async def clear(self) -> bool:
        """清空所有缓存"""
        if not self.is_connected:
            return False
        
        try:
            pattern = f"{self.key_prefix}*"
            keys = await self._client.keys(pattern)
            
            if keys:
                await self._client.delete(*keys)
                logger.info(f"清空了 {len(keys)} 个Redis缓存条目")
            
            return True
        except Exception as e:
            logger.error(f"Redis清空失败: {e}")
            return False
    
