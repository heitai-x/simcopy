#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""最终导入测试脚本

测试所有模块的导入是否正常工作
"""

import sys
import traceback
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_import(module_name, description):
    """测试单个模块导入"""
    try:
        exec(f"import {module_name}")
        print(f"✓ {description}: 导入成功")
        return True
    except Exception as e:
        print(f"✗ {description}: 导入失败 - {e}")
        traceback.print_exc()
        return False

def test_from_import(import_statement, description):
    """测试from import语句"""
    try:
        exec(import_statement)
        print(f"✓ {description}: 导入成功")
        return True
    except Exception as e:
        print(f"✗ {description}: 导入失败 - {e}")
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("开始最终导入测试...\n")
    
    success_count = 0
    total_count = 0
    
    # 测试基础模块
    tests = [
        ("src", "主包"),
        ("src.models", "数据模型模块"),
        ("src.config", "配置模块"),
        ("src.cache", "缓存模块"),
        ("src.handler", "处理器模块"),
        ("src.utils", "工具模块"),
        ("src.retriever", "检索模块"),
        ("src.nlp", "NLP模块"),
    ]
    
    for module, desc in tests:
        if test_import(module, desc):
            success_count += 1
        total_count += 1
    
    print("\n测试具体组件导入...")
    
    # 测试具体组件
    component_tests = [
        ("from src.models import SimpleRequest, CacheEntry", "数据模型组件"),
        ("from src.config import VLLMConfig, HandlerConfig", "配置组件"),
        ("from src.cache import RedisManager, MultiLevelCache", "缓存组件"),
        ("from src.handler import BaseVLLMHandler, NLPEnhancedVLLMHandler", "处理器组件"),
        ("from src.utils.similarity_search_helper import SimilaritySearchHelper", "相似度搜索助手"),
        ("from src.utils.hasher import RequestHasher", "哈希工具"),
        ("from src.retriever import EnhancedDocumentRetriever", "文档检索器"),
        ("from src.nlp.async_conjunction_extractor import AsyncAdvancedConjunctionExtractor", "NLP连接词提取器"),
    ]
    
    for import_stmt, desc in component_tests:
        if test_from_import(import_stmt, desc):
            success_count += 1
        total_count += 1
    
    print("\n测试脚本模块...")
    
    # 测试脚本模块
    script_tests = [
        ("scripts.startup_check", "启动检查脚本"),
        ("scripts.health_check", "健康检查脚本"),
        ("scripts.monitor", "监控脚本"),
        ("scripts.benchmark", "基准测试脚本"),
        ("scripts.deployment_validator", "部署验证脚本"),
    ]
    
    for module, desc in script_tests:
        if test_import(module, desc):
            success_count += 1
        total_count += 1
    
    # 输出结果
    print(f"\n=== 测试结果 ===")
    print(f"成功: {success_count}/{total_count}")
    print(f"失败: {total_count - success_count}/{total_count}")
    print(f"成功率: {success_count/total_count*100:.1f}%")
    
    if success_count == total_count:
        print("\n🎉 所有导入测试通过！项目导入结构正常。")
        return True
    else:
        print(f"\n❌ 还有 {total_count - success_count} 个导入问题需要解决。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)