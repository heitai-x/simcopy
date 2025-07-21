"""请求哈希工具"""

import hashlib
from typing import Optional
from vllm import SamplingParams


class RequestHasher:
    """请求哈希生成器"""
    
    @staticmethod
    def compute_hash_id(prompt: str, sampling_params: SamplingParams) -> str:
        """计算请求的哈希ID
        
        Args:
            prompt: 输入提示
            sampling_params: 采样参数
            
        Returns:
            str: 16字符的哈希ID
        """
        # 提取关键采样参数
        key_params = {
            'temperature': getattr(sampling_params, 'temperature', 1.0),
            'max_tokens': getattr(sampling_params, 'max_tokens', None),
            'top_p': getattr(sampling_params, 'top_p', 1.0),
            'top_k': getattr(sampling_params, 'top_k', -1)
        }
        
        # 创建用于哈希的字符串
        hash_string = f"{prompt}|{key_params['temperature']}|{key_params['max_tokens']}|{key_params['top_p']}|{key_params['top_k']}"
        
        # 生成SHA256哈希并取前16位
        hash_object = hashlib.sha256(hash_string.encode('utf-8'))
        return hash_object.hexdigest()[:16]
    
    @staticmethod
    def compute_simple_hash(text: str) -> str:
        """计算简单文本哈希
        
        Args:
            text: 输入文本
            
        Returns:
            str: 16字符的哈希ID
        """
        hash_object = hashlib.sha256(text.encode('utf-8'))
        return hash_object.hexdigest()[:16]
