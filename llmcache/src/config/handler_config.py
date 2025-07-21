"""Handler配置模块"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class HandlerFeatureConfig:
    """Handler功能配置"""
    
    # NLP增强功能
    enable_nlp_enhancement: bool = True
    nlp_similarity_threshold: float = 0.85
    nlp_high_similarity_threshold: float = 0.95
    
    # 相似度搜索功能
    enable_similarity_search: bool = True
    similarity_threshold: float = 0.7
    high_similarity_threshold: float = 0.9
    max_similarity_results: int = 10
    
    # 向量搜索功能
    enable_vector_search: bool = True
    vector_search_batch_size: int = 100
    
    # 缓存功能
    enable_cache_deduplication: bool = True
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    
    # 异步任务管理
    max_concurrent_tasks: int = 50
    task_timeout_seconds: int = 300
    
    # 详细日志
    enable_detailed_logging: bool = False
    
    # 错误处理
    max_retry_attempts: int = 3
    retry_delay_seconds: float = 1.0
    
    # 资源管理
    enable_memory_optimization: bool = True
    cleanup_interval_seconds: int = 60


@dataclass
class HandlerConfig:
    """Handler主配置"""
    
    # 基础配置
    max_concurrent: int = 10
    request_timeout: int = 300
    
    # 功能配置
    features: HandlerFeatureConfig = None
    
    def __post_init__(self):
        if self.features is None:
            self.features = HandlerFeatureConfig()
    
    @classmethod
    def create_default(cls) -> 'HandlerConfig':
        """创建默认配置"""
        return cls(
            max_concurrent=10,
            request_timeout=300,
            features=HandlerFeatureConfig()
        )
    
    @classmethod
    def create_performance_optimized(cls) -> 'HandlerConfig':
        """创建性能优化配置"""
        features = HandlerFeatureConfig(
            enable_nlp_enhancement=True,
            enable_vector_search=True,
            enable_cache_deduplication=True,
            max_concurrent_tasks=100,

            enable_detailed_logging=False,
            enable_memory_optimization=True
        )
        
        return cls(
            max_concurrent=20,
            request_timeout=300,
            features=features
        )
    
    @classmethod
    def create_debug(cls) -> 'HandlerConfig':
        """创建调试配置"""
        features = HandlerFeatureConfig(
            enable_nlp_enhancement=True,
            enable_vector_search=True,
            enable_cache_deduplication=True,
            max_concurrent_tasks=10,

            enable_detailed_logging=True,
            enable_memory_optimization=False
        )
        
        return cls(
            max_concurrent=5,
            request_timeout=600,
            features=features
        )