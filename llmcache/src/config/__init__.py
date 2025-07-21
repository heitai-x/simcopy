#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理模块

提供项目的各种配置管理功能。
"""

# 从现有模块导入
from .settings import (
    SharedMemoryConfig,
    VectorSearchConfig,
    CacheConfig,
    ModelConfig,
    NLPConfig,
    VLLMConfig,
    EnhancedVLLMConfig,
)
# 导入配置管理器
from .handler_config import HandlerConfig, HandlerFeatureConfig

try:
    from .nlp_config import (
        NLPEnhancedConfig,
        NLPPerformanceConfig,
        NLPConfigTemplates,
        load_nlp_config,
        save_nlp_config
    )
except ImportError:
    # NLP config module might not exist yet
    NLPEnhancedConfig = None
    NLPPerformanceConfig = None
    NLPConfigTemplates = None
    load_nlp_config = None
    save_nlp_config = None

# 工具函数
def get_default_config():
    """获取默认配置"""
    return VLLMConfig()

def setup_environment():
    """设置环境"""
    pass

def create_default_engine_args():
    """创建默认引擎参数"""
    return {}

def create_demo_engine_args():
    """创建演示引擎参数"""
    return {}

def create_benchmark_engine_args():
    """创建基准测试引擎参数"""
    return {}

def get_common_handler_config():
    """获取通用处理器配置"""
    return {}

def setup_demo_logging():
    """设置演示日志"""
    pass

def create_sample_requests():
    """创建示例请求"""
    return []

def create_nlp_sample_requests():
    """创建NLP示例请求"""
    return []

# 兼容性函数
def get_config_manager():
    """获取配置管理器（兼容性函数）"""
    return get_default_config()

def get_config():
    """获取配置（兼容性函数）"""
    return get_default_config()

__all__ = [
    # Settings模块
    'SharedMemoryConfig',
    'VectorSearchConfig', 
    'CacheConfig',
    'ModelConfig',
    'NLPConfig',
    'VLLMConfig',
    'EnhancedVLLMConfig',
    # 配置管理
    'HandlerConfig',
    'HandlerFeatureConfig',
    # NLP配置模块 (如果存在)
    'NLPEnhancedConfig',
    'NLPPerformanceConfig',
    'NLPConfigTemplates',
    'load_nlp_config',
    'save_nlp_config',
    # 配置工具函数
    'get_default_config',
    'setup_environment',
    'create_default_engine_args',
    'create_demo_engine_args',
    'create_benchmark_engine_args',
    'get_common_handler_config',
    'setup_demo_logging',
    'create_sample_requests',
    'create_nlp_sample_requests',
    # 兼容性函数
    'get_config_manager',
    'get_config'
]