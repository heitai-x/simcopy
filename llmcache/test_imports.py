#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导入测试脚本
用于验证所有模块是否可以正常导入
"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_imports():
    """测试所有关键模块的导入"""
    print("开始测试模块导入...")
    
    # 测试配置模块
    try:
        from src.config.settings import get_default_config, create_benchmark_engine_args
        print("✓ 配置模块导入成功")
    except ImportError as e:
        print(f"✗ 配置模块导入失败: {e}")
    
    # 测试处理器模块
    try:
        from src.handler.nlp_enhanced_vllm_handler import NLPEnhancedVLLMHandler
        print("✓ NLP处理器模块导入成功")
    except ImportError as e:
        print(f"✗ NLP处理器模块导入失败: {e}")
    
    # 测试缓存模块
    try:
        from src.cache import MultiLevelCacheManager
        print("✓ 多级缓存模块导入成功")
    except ImportError as e:
        print(f"✗ 多级缓存模块导入失败: {e}")
    
    # 测试向量搜索模块
    try:
        from src.cache.enhanced_vector_search import VectorSearchManager
        print("✓ 向量搜索模块导入成功")
    except ImportError as e:
        print(f"✗ 向量搜索模块导入失败: {e}")
    
    # 测试日志模块
    try:
        from src.utils.logger import get_logger
        print("✓ 日志模块导入成功")
    except ImportError as e:
        print(f"✗ 日志模块导入失败: {e}")
    
    # 测试NLP模块
    try:
        from src.nlp.async_conjunction_extractor import AsyncAdvancedConjunctionExtractor
        print("✓ NLP连接词提取器模块导入成功")
    except ImportError as e:
        print(f"✗ NLP连接词提取器模块导入失败: {e}")
    
    # 测试基础处理器
    try:
        from src.handler.base_vllm_handler import BaseVLLMHandler
        print("✓ 基础VLLM处理器模块导入成功")
    except ImportError as e:
        print(f"✗ 基础VLLM处理器模块导入失败: {e}")
    
    # 测试增强异步LLM
    try:
        from src.handler.enhanced_async_llm import EnhancedAsyncLLM
        print("✓ 增强异步LLM模块导入成功")
    except ImportError as e:
        print(f"✗ 增强异步LLM模块导入失败: {e}")
    
    # 测试模型模块
    try:
        from src.models.request import SimpleRequest
        from src.models.cache import CacheEntry
        from src.models.enums import RequestStatus, CacheStatus
        print("✓ 模型模块导入成功")
    except ImportError as e:
        print(f"✗ 模型模块导入失败: {e}")
    
    # 测试工具模块
    try:
        from src.utils.similarity_search_helper import SimilaritySearchHelper
        from src.utils.hasher import RequestHasher
        from src.utils.request_manager import RequestManager
        print("✓ 工具模块导入成功")
    except ImportError as e:
        print(f"✗ 工具模块导入失败: {e}")
    
    print("\n模块导入测试完成!")

if __name__ == "__main__":
    test_imports()