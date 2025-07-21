import os
import msgpack
import pickle
import threading
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from multiprocessing import shared_memory, Lock, RLock
from threading import RLock as ThreadingRLock
from enum import Enum

class ReadWriteLock:
    """读写锁实现，允许多个读者并发访问，但写者独占访问"""
    
    def __init__(self):
        self._read_ready = threading.Condition(threading.RLock())
        self._readers = 0
    
    def acquire_read(self):
        """获取读锁"""
        self._read_ready.acquire()
        try:
            self._readers += 1
        finally:
            self._read_ready.release()
    
    def release_read(self):
        """释放读锁"""
        self._read_ready.acquire()
        try:
            self._readers -= 1
            if self._readers == 0:
                self._read_ready.notifyAll()
        finally:
            self._read_ready.release()
    
    def acquire_write(self):
        """获取写锁"""
        self._read_ready.acquire()
        while self._readers > 0:
            self._read_ready.wait()
    
    def release_write(self):
        """释放写锁"""
        self._read_ready.release()
    
    def read_lock(self):
        """读锁上下文管理器"""
        return ReadLockContext(self)
    
    def write_lock(self):
        """写锁上下文管理器"""
        return WriteLockContext(self)

class ReadLockContext:
    """读锁上下文管理器"""
    
    def __init__(self, rw_lock: ReadWriteLock):
        self.rw_lock = rw_lock
    
    def __enter__(self):
        self.rw_lock.acquire_read()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.rw_lock.release_read()

class WriteLockContext:
    """写锁上下文管理器"""
    
    def __init__(self, rw_lock: ReadWriteLock):
        self.rw_lock = rw_lock
    
    def __enter__(self):
        self.rw_lock.acquire_write()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.rw_lock.release_write()

class SerializationFormat(Enum):
    """序列化格式枚举"""
    MSGPACK = "msgpack"
    PICKLE = "pickle"

@dataclass
class SimilarRequestMemoryConfig:
    """相似请求内存配置"""
    request_mapping_memory_size: int = 8 * 1024 * 1024  # 8MB for request mappings
    token_mapping_memory_size: int = 8 * 1024 * 1024    # 8MB for token mappings
    max_entries: int = 5000
    request_mapping_shared_name: str = "vllm_request_mappings"
    token_mapping_shared_name: str = "vllm_token_mappings"
    
    serialization_format: SerializationFormat = SerializationFormat.PICKLE  # 默认使用pickle
    

class SimilarRequestMemoryManager:
    """专门管理相似请求的共享内存管理器
    
    使用分离的共享内存区域存储两种类型的数据：
    1. 请求hash_id -> 相似请求hash_id列表的映射 (独立共享内存区域)
    2. 相似请求hash_id -> token序列的映射 (独立共享内存区域)
    
    优势：
    - 独立更新：每个映射可以独立更新，无需序列化整个数据结构
    - 减少锁竞争：不同映射使用独立的锁
    - 提高并发性：读写操作可以并行进行
    - 优化的序列化：基于数量变化的简单触发机制，避免不必要的序列化开销
    """
    
    def __init__(self, config: Optional[SimilarRequestMemoryConfig] = None):
        self.config = config or SimilarRequestMemoryConfig()
        
        # 分离的读写锁和共享内存
        self._request_rw_lock = ReadWriteLock()  # 请求映射读写锁
        self._token_rw_lock = ReadWriteLock()    # token映射读写锁
        self._request_shm = None     # 请求映射共享内存
        self._token_shm = None       # token映射共享内存
        
        # 内存中的数据结构 - 使用简单字典存储
        self._request_to_similar: Dict[str, List[str]] = {}  # 请求ID -> 相似请求ID列表
        self._similar_to_tokens: Dict[str, List[int]] = {}   # 相似请求ID -> token序列
        
        
        self._initialize_shared_memory()
        self._load_data()
    
    def _initialize_shared_memory(self):
        """初始化分离的共享内存区域"""
        try:
            self._request_shm = shared_memory.SharedMemory(
                name=self.config.request_mapping_shared_name,
                create=True,
                size=self.config.request_mapping_memory_size
            )
        except FileExistsError:
            self._request_shm = shared_memory.SharedMemory(
                name=self.config.request_mapping_shared_name,
                create=False,
                size=self.config.request_mapping_memory_size
            )
        except Exception as e:
            print(f"Failed to initialize request mapping shared memory: {e}")
            self._request_shm = None
        
        # 初始化token映射共享内存
        try:
            self._token_shm = shared_memory.SharedMemory(
                name=self.config.token_mapping_shared_name,
                create=True,
                size=self.config.token_mapping_memory_size
            )
        except FileExistsError:
            self._token_shm = shared_memory.SharedMemory(
                name=self.config.token_mapping_shared_name,
                create=False,
                size=self.config.token_mapping_memory_size
            )
        except Exception as e:
            print(f"Failed to initialize token mapping shared memory: {e}")
            self._token_shm = None
    
    def _serialize_data(self, data: Any) -> bytes:
        """根据配置的格式序列化数据"""
        if self.config.serialization_format == SerializationFormat.PICKLE:
            return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        elif self.config.serialization_format == SerializationFormat.MSGPACK:
            return msgpack.packb(data, use_bin_type=True)
        else:
            # 默认使用pickle
            return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _deserialize_data(self, data_bytes: bytes) -> Any:
        """根据配置的格式反序列化数据"""
        if self.config.serialization_format == SerializationFormat.PICKLE:
            return pickle.loads(data_bytes)
        elif self.config.serialization_format == SerializationFormat.MSGPACK:
            return msgpack.unpackb(data_bytes, raw=False)
        else:
            # 默认使用pickle
            return pickle.loads(data_bytes)
    

    
    def _load_data(self):
        """从分离的共享内存加载数据"""
        # 加载请求映射数据
        if self._request_shm:
            try:
                length_bytes = self._request_shm.buf[0:4]
                data_length = int.from_bytes(length_bytes, 'little')
                if data_length > 0:
                    serialized_bytes = self._request_shm.buf[4:4 + data_length]
                    self._request_to_similar = self._deserialize_data(serialized_bytes.tobytes())
            except Exception as e:
                print(f"Failed to load request mapping data: {e}")
        
        # 加载token映射数据
        if self._token_shm:
            try:
                length_bytes = self._token_shm.buf[0:4]
                data_length = int.from_bytes(length_bytes, 'little')
                if data_length > 0:
                    serialized_bytes = self._token_shm.buf[4:4 + data_length]
                    self._similar_to_tokens = self._deserialize_data(serialized_bytes.tobytes())
            except Exception as e:
                print(f"Failed to load token mapping data: {e}")
    
    def _save_request_mapping_data(self):
        """保存请求映射数据到共享内存"""
        if not self._request_shm:
            return
            
        try:
            # 序列化数据
            serialized_bytes = self._serialize_data(self._request_to_similar)
            
            # 检查数据大小
            if len(serialized_bytes) + 4 > self.config.request_mapping_memory_size:
                self._cleanup_request_mapping_data()
                serialized_bytes = self._serialize_data(self._request_to_similar)
            
            # 写入数据长度和数据
            self._request_shm.buf[0:4] = len(serialized_bytes).to_bytes(4, 'little')
            self._request_shm.buf[4:4 + len(serialized_bytes)] = serialized_bytes
            
        except Exception as e:
            print(f"Failed to save request mapping data: {e}")
    
    def _save_token_mapping_data(self):
        """保存token映射数据到共享内存"""
        if not self._token_shm:
            return
            
        try:
            # 序列化数据
            serialized_bytes = self._serialize_data(self._similar_to_tokens)
            
            # 检查数据大小
            if len(serialized_bytes) + 4 > self.config.token_mapping_memory_size:
                self._cleanup_token_mapping_data()
                serialized_bytes = self._serialize_data(self._similar_to_tokens)
            
            # 写入数据长度和数据
            self._token_shm.buf[0:4] = len(serialized_bytes).to_bytes(4, 'little')
            self._token_shm.buf[4:4 + len(serialized_bytes)] = serialized_bytes
            
        except Exception as e:
            print(f"Failed to save token mapping data: {e}")
    
    def _cleanup_request_mapping_data(self):
        """简单清理请求映射数据以释放空间"""
        if len(self._request_to_similar) > self.config.max_entries:
            # 简单清理：删除一半数据
            keys_to_remove = list(self._request_to_similar.keys())[:len(self._request_to_similar) // 2]
            for key in keys_to_remove:
                del self._request_to_similar[key]
    
    def _cleanup_token_mapping_data(self):
        """简单清理token映射数据以释放空间"""
        if len(self._similar_to_tokens) > self.config.max_entries:
            # 简单清理：删除一半数据
            keys_to_remove = list(self._similar_to_tokens.keys())[:len(self._similar_to_tokens) // 2]
            for key in keys_to_remove:
                del self._similar_to_tokens[key]
    

    
    def store_similar_request_mapping(self, request_hash: str, similar_hashes: List[str]):
        """存储请求hash到相似请求hash列表的映射"""
        with self._request_rw_lock.write_lock():
            self._request_to_similar[request_hash] = similar_hashes
            self._save_request_mapping_data()
    
    def store_multiple_similar_request_mappings(self, request_mappings: Dict[str, List[str]]):
        """批量存储多个请求hash到相似请求hash列表的映射
        
        Args:
            request_mappings: 字典，key为request_hash，value为相似请求hash列表
        """
        if not request_mappings:
            return
            
        with self._request_rw_lock.write_lock():
            self._request_to_similar.update(request_mappings)
            self._save_request_mapping_data()
    
    def get_similar_request_mapping(self, request_hash: str) -> Optional[List[str]]:
        """获取请求hash对应的相似请求hash列表"""
        with self._request_rw_lock.read_lock():
            return self._request_to_similar.get(request_hash)
    
    def store_answer_tokens(self, similar_hash: str, tokens: Union[List[int], np.ndarray]):
        """存储相似请求hash到token序列的映射"""
        # 转换NumPy数组为列表（如果需要）
        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()
            
        with self._token_rw_lock.write_lock():
            self._similar_to_tokens[similar_hash] = tokens
            self._save_token_mapping_data()
    
    def store_multiple_answer_tokens(self, token_mappings: Dict[str, Union[List[int], np.ndarray]]):
        """批量存储多个相似请求hash到token序列的映射
        
        Args:
            token_mappings: 字典，key为similar_hash，value为token序列（支持List[int]或np.ndarray）
        """
        if not token_mappings:
            return
            
        # 转换NumPy数组为列表（如果需要）
        converted_mappings = {}
        for key, value in token_mappings.items():
            if isinstance(value, np.ndarray):
                converted_mappings[key] = value.tolist()
            else:
                converted_mappings[key] = value
            
        with self._token_rw_lock.write_lock():
            self._similar_to_tokens.update(converted_mappings)
            self._save_token_mapping_data()
    
    def get_answer_tokens(self, similar_hash: str) -> Optional[List[int]]:
        """获取相似请求hash对应的token序列"""
        with self._token_rw_lock.read_lock():
            return self._similar_to_tokens.get(similar_hash)
    
    def delete_request_mapping(self, request_hash: str) -> bool:
        """直接删除指定的请求映射"""
        with self._request_rw_lock.write_lock():
            if request_hash in self._request_to_similar:
                del self._request_to_similar[request_hash]
                self._save_request_mapping_data()
                return True
            return False
    
    def delete_multiple_request_mappings(self, request_hashes: List[str]) -> int:
        """批量删除多个请求映射
        
        Args:
            request_hashes: 要删除的请求hash列表
            
        Returns:
            实际删除的数量
        """
        if not request_hashes:
            return 0
            
        deleted_count = 0
        with self._request_rw_lock.write_lock():
            for request_hash in request_hashes:
                if request_hash in self._request_to_similar:
                    del self._request_to_similar[request_hash]
                    deleted_count += 1
            
            if deleted_count > 0:
                self._save_request_mapping_data()
                
        return deleted_count
    
    def delete_answer_tokens(self, similar_hash: str) -> bool:
        """直接删除指定的token序列"""
        with self._token_rw_lock.write_lock():
            if similar_hash in self._similar_to_tokens:
                del self._similar_to_tokens[similar_hash]
                self._save_token_mapping_data()
                return True
            return False
    
    def delete_multiple_answer_tokens(self, similar_hashes: List[str]) -> int:
        """批量删除多个token序列
        
        Args:
            similar_hashes: 要删除的相似请求hash列表
            
        Returns:
            实际删除的数量
        """
        if not similar_hashes:
            return 0
            
        deleted_count = 0
        with self._token_rw_lock.write_lock():
            for similar_hash in similar_hashes:
                if similar_hash in self._similar_to_tokens:
                    del self._similar_to_tokens[similar_hash]
                    deleted_count += 1
            
            if deleted_count > 0:
                self._save_token_mapping_data()
                
        return deleted_count
    
    def clear_request_mappings(self):
        """清空所有请求映射"""
        with self._request_rw_lock.write_lock():
            self._request_to_similar.clear()
            self._save_request_mapping_data()

    def clear_answer_tokens(self):
        """清空所有token序列"""
        with self._token_rw_lock.write_lock():
            self._similar_to_tokens.clear()
            self._save_token_mapping_data()
    
    
    def cleanup(self):
        """清理资源"""
        # 清理请求映射共享内存
        if self._request_shm:
            try:
                self._request_shm.close()
                self._request_shm.unlink()
            except Exception as e:
                print(f"Failed to cleanup request mapping shared memory: {e}")
        
        # 清理token映射共享内存
        if self._token_shm:
            try:
                self._token_shm.close()
                self._token_shm.unlink()
            except Exception as e:
                print(f"Failed to cleanup token mapping shared memory: {e}")
        
        # 清理内存数据结构
        self._request_to_similar.clear()
        self._similar_to_tokens.clear()
    
    def __del__(self):
        """析构函数，确保资源被正确清理"""
        self.cleanup()