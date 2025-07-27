# SPDX-License-Identifier: Apache-2.0
from typing import Optional, List, Union, Dict, Any

import numpy as np
from numba import jit

from vllm.config import VllmConfig

try:
    import sys
    import os
    import importlib.util
    
    # 直接指定模块文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    vllm_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    module_path = os.path.join(vllm_root, 'llmcache', 'src', 'handler', 'similar_request_memory.py')
    
    if not os.path.exists(module_path):
        raise ImportError(f"Module file not found: {module_path}")
    
    # 使用importlib动态加载模块
    spec = importlib.util.spec_from_file_location("similar_request_memory", module_path)
    similar_request_memory_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(similar_request_memory_module)
    
    SimilarRequestMemoryManager = similar_request_memory_module.SimilarRequestMemoryManager
    print(f"✓ SimilarRequestMemoryManager通过importlib导入成功")
    
except Exception as e:
    print(f"✗ SimilarRequestMemoryManager导入失败: {e}")
    SimilarRequestMemoryManager = None
# Local cache for speculative decoding optimization
class LocalSpeculativeCache:
    """本地推测解码缓存，优先访问本地缓存，缓存未命中时再访问共享内存"""
    
    def __init__(self, max_size: int = 500):
        self.cache = {}
        self.access_order = []
        self.max_size = max_size
        self.hit_count = 0
        self.miss_count = 0
    
    def get(self, request_id: str) -> Optional[List[np.ndarray]]:
        """获取缓存的相似文本tokens"""
        if request_id in self.cache:
            # 更新访问顺序
            self.access_order.remove(request_id)
            self.access_order.append(request_id)
            self.hit_count += 1
            return self.cache[request_id].copy()
        
        self.miss_count += 1
        return None
    
    def put(self, request_id: str, similar_texts: List[np.ndarray]) -> None:
        """存储相似文本tokens到本地缓存"""
        if not similar_texts:
            return
        
        # LRU淘汰
        if len(self.cache) >= self.max_size and request_id not in self.cache:
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]
        
        # 存储副本避免外部修改
        self.cache[request_id] = [text.copy() for text in similar_texts]
        
        if request_id in self.access_order:
            self.access_order.remove(request_id)
        self.access_order.append(request_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total if total > 0 else 0.0
        return {
            "cache_size": len(self.cache),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate
        }


class MYNgramProposer:
    """优化的N-gram提议器，直接访问SimilarRequestMemoryManager，去除不必要的中间层
    
    优化点：
    1. 直接访问SimilarRequestMemoryManager，避免通过RequestManager中转
    2. 简化数据传递路径：SimilarRequestMemoryManager -> MYNgramProposer
    3. 减少内存拷贝和序列化开销
    """

    def __init__(self, vllm_config: VllmConfig, **kwargs):
        # Basic N-gram configuration from vllm_config
        self.min_n = vllm_config.speculative_config.prompt_lookup_min
        self.max_n = vllm_config.speculative_config.prompt_lookup_max
        self.max_k = vllm_config.speculative_config.num_speculative_tokens
        self.max_model_len = vllm_config.model_config.max_model_len
        self.enable_direct_memory_access = getattr(vllm_config.speculative_config, 'enable_direct_memory_access', True)
        
        # 相似度阈值配置
        self.similarity_threshold_min = 0.7
        self.similarity_threshold_max = 1.0
        self.min_k_ratio = 0.5
        self.k_scale_factor = 2.0
        
        self.memory_manager = None
        self._init_shared_memory_manager()
        
        # 验证共享内存导入状态
        validation_result = self.validate_shared_memory_import()
        self._report_shared_memory_status(validation_result)
        
        # Initialize local cache for speculative decoding optimization
        cache_size = kwargs.get("local_cache_size", 500)
        self.local_cache = LocalSpeculativeCache(max_size=cache_size)
        print(f"推测解码本地缓存已初始化，缓存大小: {cache_size}")

        # Trigger Numba JIT compilation for enhanced N-gram proposer
        try:
            self.propose(np.zeros(1024, dtype=np.int32), "req_id", 1)
        except Exception as e:
            print(f"Numba JIT编译失败: {e}")

    def propose(
        self,
        context_token_ids: np.ndarray,
        request_id: str,
        k: int,
    ) -> Optional[np.ndarray]:
        k = min(k, self.max_model_len - context_token_ids.shape[0])
        if k <= 0:
            return None
        

        similar_texts = self._get_similar_texts_direct(request_id)
        # similar_texts.append(context_token_ids)
        
        result = self._sequential_match_similar_texts(context_token_ids, similar_texts, k)
        if result is not None:
            print("length:",result)
            return result
        return None
        
    def _get_similar_texts_direct(self, request_id: str, tokenizer=None) -> List[np.ndarray]:
        """直接从SimilarRequestMemoryManager获取相似文本，去除RequestManager中间层
        
        优化点：
        1. 直接访问共享内存，避免通过RequestManager中转
        2. 减少数据拷贝和序列化开销
        3. 简化调用链路
        """
        # 首先尝试从本地缓存获取
        cached_result = self.local_cache.get(request_id)
        if cached_result is not None:
            
            return cached_result
        
        if not self.memory_manager or not self.enable_direct_memory_access:
            return []
        
        try:
            # 添加调试信息
            print(f"[共享内存读取] 正在查找请求ID: '{request_id}'")
            print(f"[共享内存读取] 请求ID长度: {len(request_id)}")
            print(f"[共享内存读取] 请求ID类型: {type(request_id)}")
            
            # 检查共享内存管理器状态
            if hasattr(self.memory_manager, '_request_to_similar'):
                all_keys = list(self.memory_manager._request_to_similar.keys())
                print(f"[共享内存读取] 当前共享内存中的所有请求ID: {all_keys}")
                print(f"[共享内存读取] 共享内存中请求映射总数: {len(all_keys)}")
                
                # 检查是否有相似的key
                for key in all_keys:
                    if request_id in key or key in request_id:
                        print(f"[共享内存读取] 发现相似key: '{key}'")
            
            # 直接从SimilarRequestMemoryManager获取相似请求映射
            similar_hashes = self.memory_manager.get_similar_request_mapping(request_id)
            if not similar_hashes:
                print(f"[共享内存读取] 请求 {request_id} 未找到相似请求映射")
                
                # 尝试重新加载数据
                if hasattr(self.memory_manager, '_load_data'):
                    print(f"[共享内存读取] 尝试重新加载共享内存数据")
                    self.memory_manager._load_data()
                    similar_hashes = self.memory_manager.get_similar_request_mapping(request_id)
                    if similar_hashes:
                        print(f"[共享内存读取] 重新加载后找到映射: {similar_hashes}")
                
                return []
            print(f"[共享内存] 请求 {request_id} 找到 {len(similar_hashes)} 个相似请求: {similar_hashes}")
            
            similar_token_sequences = []
            for similar_hash in similar_hashes:
                tokens = self.memory_manager.get_answer_tokens(similar_hash)
                if tokens:
                    print(f"[共享内存] 相似请求 {similar_hash} 获取到 {len(tokens)} 个tokens: {tokens[:20]}{'...' if len(tokens) > 20 else ''}")
                    # 将token列表转换为numpy数组
                    token_array = np.array(tokens, dtype=np.int32)
                    similar_token_sequences.append(token_array)
                else:
                    print(f"[共享内存] 相似请求 {similar_hash} 未获取到tokens")
            
            # 将结果存储到本地缓存
            if similar_token_sequences:
                print(f"[共享内存] 成功获取 {len(similar_token_sequences)} 个相似文本序列，已存储到本地缓存")
                self.local_cache.put(request_id, similar_token_sequences)
            else:
                print(f"[共享内存] 请求 {request_id} 未获取到有效的token序列")
            return similar_token_sequences
            
        except Exception as e:
            return []

    def _sequential_match_similar_texts(
        self,
        context_token_ids: np.ndarray,
        similar_texts: List[np.ndarray],
        k: int
    ) -> Optional[np.ndarray]:
        """Performs sequential matching across similar text contexts."""
        if not similar_texts:
            return None
        for text in similar_texts:
            try:
                result = self._match_tokens_by_suffix(context_token_ids, text, k)
                if result is not None:
                    return result
            except Exception:
                continue
        
        return None

    def _match_tokens_by_suffix(
        self,
        context_token_ids: np.ndarray,
        response_tokens: np.ndarray,
        k: int
    ) -> Optional[np.ndarray]:
        """Matches context suffix against response tokens using N-gram patterns."""
        if len(context_token_ids) == 0 or len(response_tokens) == 0:
            return None
        
        max_n = min(self.max_n, len(context_token_ids), len(response_tokens))
        
        for n in range(max_n, self.min_n - 1, -1):
            context_suffix = context_token_ids[-n:]
            match_pos = _find_pattern_kmp(response_tokens, context_suffix)
            
            if match_pos != -1:
                start_pos = match_pos + n
                adaptive_k = self._calculate_adaptive_k(n, k)
                end_pos = min(start_pos + adaptive_k, len(response_tokens))
                
                if end_pos > start_pos:
                    return response_tokens[start_pos:end_pos]
        
        return None

    def load_model(self, *args, **kwargs):
        # No model to load for N-gram proposer.
        pass
        
    def _calculate_adaptive_k(self, n: int, base_k: int) -> int:
        """根据n的大小动态调整k值。
        
        Args:
            n: 当前匹配的n-gram长度
            base_k: 基础k值
            
        Returns:
            调整后的k值
        """
        n_ratio = n / self.max_n
        k_multiplier = self.min_k_ratio + (1 - self.min_k_ratio) * (n_ratio ** self.k_scale_factor)
        
        adjusted_k = int(base_k * k_multiplier)
        
        # 确保k值在合理范围内
        return max(1, min(adjusted_k, base_k))
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取本地缓存统计信息
        
        Returns:
            包含缓存命中率、大小等统计信息的字典
        """
        stats = self.local_cache.get_stats()
        if self.memory_manager:
            memory_stats = self._get_memory_manager_stats()
            stats.update({
                "memory_manager_stats": memory_stats,
                "direct_memory_access": self.enable_direct_memory_access
            })
        return stats
    
    def _get_memory_manager_stats(self) -> Dict[str, Any]:
        """获取共享内存管理器的统计信息
        
        Returns:
            包含共享内存状态的详细统计信息
        """
        if not self.memory_manager:
            return {"status": "not_initialized"}
        
        try:
            # 检查共享内存是否可访问
            stats = {
                "status": "active",
                "shared_memory_accessible": True,
                "request_mapping_count": 0,
                "token_mapping_count": 0,
                "memory_usage": {
                    "request_mapping_size": 0,
                    "token_mapping_size": 0
                }
            }
            
            # 尝试获取映射数量
            try:
                if hasattr(self.memory_manager, '_request_to_similar'):
                    stats["request_mapping_count"] = len(self.memory_manager._request_to_similar)
                if hasattr(self.memory_manager, '_similar_to_tokens'):
                    stats["token_mapping_count"] = len(self.memory_manager._similar_to_tokens)
            except Exception as e:
                stats["mapping_count_error"] = str(e)
            
            # 检查共享内存配置
            if hasattr(self.memory_manager, 'config'):
                config = self.memory_manager.config
                stats["config"] = {
                    "request_mapping_memory_size": config.request_mapping_memory_size,
                    "token_mapping_memory_size": config.token_mapping_memory_size,
                    "max_entries": config.max_entries,
                    "serialization_format": config.serialization_format.name if hasattr(config.serialization_format, 'name') else str(config.serialization_format)
                }
            
            # 检查共享内存对象状态
            if hasattr(self.memory_manager, '_request_shm') and self.memory_manager._request_shm:
                stats["request_shm_active"] = True
                stats["request_shm_name"] = self.memory_manager._request_shm.name
            else:
                stats["request_shm_active"] = False
                
            if hasattr(self.memory_manager, '_token_shm') and self.memory_manager._token_shm:
                stats["token_shm_active"] = True
                stats["token_shm_name"] = self.memory_manager._token_shm.name
            else:
                stats["token_shm_active"] = False
            
            return stats
            
        except Exception as e:
            return {
                "status": "error",
                "shared_memory_accessible": False,
                "error": str(e)
            }
    
    def is_shared_memory_ready(self) -> bool:
        """检查共享内存是否准备就绪
        
        Returns:
            True if shared memory is properly initialized and accessible
        """
        if not self.memory_manager or not self.enable_direct_memory_access:
            return False
        
        try:
            # 尝试执行一个简单的操作来验证共享内存是否可用
            test_result = self.memory_manager.get_similar_request_mapping("__test_key__")
            return True  # 如果没有抛出异常，说明共享内存可用
        except Exception as e:
            print(f"共享内存状态检查失败: {e}")
            return False
    
    def validate_shared_memory_import(self) -> Dict[str, Any]:
        """验证共享内存导入状态
        
        Returns:
            包含导入状态详细信息的字典
        """
        validation_result = {
            "import_successful": SimilarRequestMemoryManager is not None,
            "class_available": SimilarRequestMemoryManager is not None,
            "instance_created": self.memory_manager is not None,
            "direct_access_enabled": self.enable_direct_memory_access,
            "shared_memory_ready": False,
            "errors": []
        }
        
        # 检查导入状态
        if SimilarRequestMemoryManager is None:
            validation_result["errors"].append("SimilarRequestMemoryManager导入失败")
        
        # 检查实例创建状态
        if self.memory_manager is None:
            if self.enable_direct_memory_access:
                validation_result["errors"].append("启用了直接内存访问但未能创建SimilarRequestMemoryManager实例")
        else:
            # 检查共享内存是否准备就绪
            validation_result["shared_memory_ready"] = self.is_shared_memory_ready()
            if not validation_result["shared_memory_ready"]:
                validation_result["errors"].append("共享内存未准备就绪或不可访问")
        
        # 添加详细的内存管理器统计信息
        validation_result["memory_manager_stats"] = self._get_memory_manager_stats()
        
        return validation_result
    
    def _init_shared_memory_manager(self) -> None:
        """初始化共享内存管理器"""
        if SimilarRequestMemoryManager and self.enable_direct_memory_access:
            try:
                # 导入配置类
                spec = importlib.util.spec_from_file_location(
                    "similar_request_memory", 
                    os.path.join(vllm_root, 'llmcache', 'src', 'handler', 'similar_request_memory.py')
                )
                similar_request_memory_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(similar_request_memory_module)
                SimilarRequestMemoryConfig = similar_request_memory_module.SimilarRequestMemoryConfig
                
                # 使用与enhanced_async_llm.py完全相同的配置
                config = SimilarRequestMemoryConfig(
                    request_mapping_memory_size=8 * 1024 * 1024,  # 8MB
                    token_mapping_memory_size=8 * 1024 * 1024,    # 8MB
                    max_entries=5000,
                    request_mapping_shared_name="vllm_request_mappings",
                    token_mapping_shared_name="vllm_token_mappings"
                )
                
                self.memory_manager = SimilarRequestMemoryManager(config)
                print("✓ 直接内存访问已启用，使用统一配置")
                
                # 验证共享内存是否可访问
                if hasattr(self.memory_manager, '_request_shm') and self.memory_manager._request_shm:
                    print(f"✓ 请求共享内存连接成功: {self.memory_manager._request_shm.name}")
                else:
                    print("✗ 请求共享内存连接失败")
                    
                if hasattr(self.memory_manager, '_token_shm') and self.memory_manager._token_shm:
                    print(f"✓ Token共享内存连接成功: {self.memory_manager._token_shm.name}")
                else:
                    print("✗ Token共享内存连接失败")
                    
            except Exception as e:
                print(f"✗ 初始化SimilarRequestMemoryManager失败: {e}")
                import traceback
                print(f"详细错误: {traceback.format_exc()}")
                self.enable_direct_memory_access = False
        elif not SimilarRequestMemoryManager:
            print("✗ SimilarRequestMemoryManager导入失败，共享内存功能不可用")
            self.enable_direct_memory_access = False
        else:
            print("ⓘ 直接内存访问已禁用")
    
    def _report_shared_memory_status(self, validation_result: Dict[str, Any]) -> None:
        """报告共享内存状态
        
        Args:
            validation_result: 验证结果字典
        """
        print("\n=== 共享内存状态报告 ===")
        print(f"导入成功: {'✓' if validation_result['import_successful'] else '✗'}")
        print(f"类可用: {'✓' if validation_result['class_available'] else '✗'}")
        print(f"实例已创建: {'✓' if validation_result['instance_created'] else '✗'}")
        print(f"直接访问启用: {'✓' if validation_result['direct_access_enabled'] else '✗'}")
        print(f"共享内存就绪: {'✓' if validation_result['shared_memory_ready'] else '✗'}")
        
        if validation_result['errors']:
            print("\n错误信息:")
            for error in validation_result['errors']:
                print(f"  ✗ {error}")
        
        # 显示内存管理器统计信息
        memory_stats = validation_result.get('memory_manager_stats', {})
        if memory_stats.get('status') == 'active':
            print(f"\n内存状态: {memory_stats['status']}")
            print(f"请求映射数量: {memory_stats.get('request_mapping_count', 0)}")
            print(f"Token映射数量: {memory_stats.get('token_mapping_count', 0)}")
            print(f"请求共享内存: {'✓' if memory_stats.get('request_shm_active') else '✗'}")
            print(f"Token共享内存: {'✓' if memory_stats.get('token_shm_active') else '✗'}")
        elif memory_stats.get('status') == 'error':
            print(f"\n内存状态: 错误 - {memory_stats.get('error', '未知错误')}")
        
        print("========================\n")
    
    def clear_local_cache(self) -> None:
        """清空本地缓存"""
        self.local_cache.cache.clear()
        self.local_cache.access_order.clear()
        print("本地推测解码缓存已清空")

    def cleanup(self):
        """清理资源"""
        if self.memory_manager:
            self.memory_manager.cleanup()
            print("SimilarRequestMemoryManager资源已清理")


@jit(nopython=True)
def _find_pattern_kmp(text: np.ndarray, pattern: np.ndarray) -> int:
    """Find pattern in text using KMP algorithm.
    
    Returns:
        Index of first match, or -1 if not found.
    """
    if len(pattern) == 0:
        return 0
    if len(text) < len(pattern):
        return -1
    
    # Build LPS array
    lps = np.zeros(len(pattern), dtype=np.int32)
    prev_lps = 0
    i = 1
    
    while i < len(pattern):
        if pattern[i] == pattern[prev_lps]:
            prev_lps += 1
            lps[i] = prev_lps
            i += 1
        else:
            if prev_lps != 0:
                prev_lps = lps[prev_lps - 1]
            else:
                lps[i] = 0
                i += 1
    
    # Search pattern in text
    i = 0  # index for text
    j = 0  # index for pattern
    
    while i < len(text):
        if pattern[j] == text[i]:
            i += 1
            j += 1
        
        if j == len(pattern):
            return i - j  # Found pattern
        elif i < len(text) and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    
    return -1  # Pattern not found
__all__ = ['MYNgramProposer']
