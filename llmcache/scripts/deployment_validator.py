#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
部署验证脚本

验证VLLM llmcache系统的部署是否正确，包括配置、依赖、服务等各个方面。
"""

import os
import sys
import json
import time
import asyncio
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import psutil
except ImportError:
    psutil = None

try:
    import redis
except ImportError:
    redis = None

try:
    import torch
except ImportError:
    torch = None

from src.config.settings import get_default_config
from scripts.startup_check import StartupChecker


@dataclass
class ValidationResult:
    """验证结果"""
    component: str
    status: str  # 'pass', 'warning', 'fail'
    message: str
    details: Optional[Dict[str, Any]] = None
    fix_suggestion: Optional[str] = None


class DeploymentValidator:
    """部署验证器"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = get_default_config()
        self.startup_checker = StartupChecker()
        self.results: List[ValidationResult] = []
        self.project_root = project_root
    
    def add_result(self, component: str, status: str, message: str, 
                   details: Optional[Dict[str, Any]] = None, 
                   fix_suggestion: Optional[str] = None):
        """添加验证结果"""
        result = ValidationResult(
            component=component,
            status=status,
            message=message,
            details=details,
            fix_suggestion=fix_suggestion
        )
        self.results.append(result)
    
    def validate_system_requirements(self) -> bool:
        """验证系统要求"""
        print("验证系统要求...")
        
        # Python版本
        python_version = sys.version_info
        if python_version >= (3, 8):
            self.add_result(
                "Python版本", "pass", 
                f"Python {python_version.major}.{python_version.minor}.{python_version.micro}"
            )
        else:
            self.add_result(
                "Python版本", "fail", 
                f"Python版本过低: {python_version.major}.{python_version.minor}.{python_version.micro}",
                fix_suggestion="请升级到Python 3.8或更高版本"
            )
            return False
        
        # 内存检查
        if psutil:
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            
            if memory_gb >= 8:
                self.add_result(
                    "系统内存", "pass", 
                    f"可用内存: {memory_gb:.1f}GB"
                )
            elif memory_gb >= 4:
                self.add_result(
                    "系统内存", "warning", 
                    f"内存较少: {memory_gb:.1f}GB，可能影响性能",
                    fix_suggestion="建议至少8GB内存以获得最佳性能"
                )
            else:
                self.add_result(
                    "系统内存", "fail", 
                    f"内存不足: {memory_gb:.1f}GB",
                    fix_suggestion="至少需要4GB内存"
                )
                return False
        else:
            self.add_result(
                "系统内存", "warning", 
                "无法检查内存（psutil未安装）",
                fix_suggestion="安装psutil: pip install psutil"
            )
        
        # 磁盘空间检查
        if psutil:
            disk_usage = psutil.disk_usage('/')
            free_gb = disk_usage.free / (1024**3)
            
            if free_gb >= 10:
                self.add_result(
                    "磁盘空间", "pass", 
                    f"可用空间: {free_gb:.1f}GB"
                )
            elif free_gb >= 5:
                self.add_result(
                    "磁盘空间", "warning", 
                    f"磁盘空间较少: {free_gb:.1f}GB",
                    fix_suggestion="建议至少10GB可用空间"
                )
            else:
                self.add_result(
                    "磁盘空间", "fail", 
                    f"磁盘空间不足: {free_gb:.1f}GB",
                    fix_suggestion="至少需要5GB可用空间"
                )
                return False
        
        return True
    
    def validate_dependencies(self) -> bool:
        """验证依赖包"""
        print("验证依赖包...")
        
        required_packages = {
            'torch': 'PyTorch深度学习框架',
            'transformers': 'Hugging Face Transformers',
            'vllm': 'VLLM推理引擎',
            'redis': 'Redis客户端',
            'psutil': '系统监控',
            'numpy': '数值计算',
            'spacy': 'NLP处理',
            'faiss-cpu': '向量搜索（CPU版本）',
            'sentence-transformers': '句子嵌入'
        }
        
        optional_packages = {
            'faiss-gpu': '向量搜索（GPU版本）',
            'GPUtil': 'GPU监控',
            'uvloop': '高性能事件循环',
            'orjson': '高性能JSON处理'
        }
        
        all_good = True
        
        # 检查必需包
        for package, description in required_packages.items():
            try:
                if package == 'faiss-cpu':
                    import faiss
                    self.add_result(
                        f"依赖包-{package}", "pass", 
                        f"{description} 已安装"
                    )
                else:
                    __import__(package)
                    self.add_result(
                        f"依赖包-{package}", "pass", 
                        f"{description} 已安装"
                    )
            except ImportError:
                self.add_result(
                    f"依赖包-{package}", "fail", 
                    f"{description} 未安装",
                    fix_suggestion=f"安装命令: pip install {package}"
                )
                all_good = False
        
        # 检查可选包
        for package, description in optional_packages.items():
            try:
                if package == 'faiss-gpu':
                    import faiss
                    # 检查是否有GPU支持
                    if hasattr(faiss, 'StandardGpuResources'):
                        self.add_result(
                            f"可选包-{package}", "pass", 
                            f"{description} 已安装"
                        )
                    else:
                        self.add_result(
                            f"可选包-{package}", "warning", 
                            f"FAISS无GPU支持",
                            fix_suggestion="安装GPU版本: pip install faiss-gpu"
                        )
                else:
                    __import__(package)
                    self.add_result(
                        f"可选包-{package}", "pass", 
                        f"{description} 已安装"
                    )
            except ImportError:
                self.add_result(
                    f"可选包-{package}", "warning", 
                    f"{description} 未安装（可选）",
                    fix_suggestion=f"可选安装: pip install {package}"
                )
        
        return all_good
    
    def validate_gpu_setup(self) -> bool:
        """验证GPU设置"""
        print("验证GPU设置...")
        
        if torch is None:
            self.add_result(
                "GPU设置", "fail", 
                "PyTorch未安装，无法检查GPU",
                fix_suggestion="安装PyTorch: pip install torch"
            )
            return False
        
        # 检查CUDA可用性
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_device)
            gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / (1024**3)
            
            self.add_result(
                "GPU设置", "pass", 
                f"检测到{gpu_count}个GPU，当前设备: {gpu_name} ({gpu_memory:.1f}GB)",
                details={
                    "gpu_count": gpu_count,
                    "current_device": current_device,
                    "gpu_name": gpu_name,
                    "gpu_memory_gb": gpu_memory
                }
            )
            
            # 检查GPU内存
            if gpu_memory < 4:
                self.add_result(
                    "GPU内存", "warning", 
                    f"GPU内存较少: {gpu_memory:.1f}GB",
                    fix_suggestion="建议至少4GB GPU内存以获得最佳性能"
                )
            else:
                self.add_result(
                    "GPU内存", "pass", 
                    f"GPU内存充足: {gpu_memory:.1f}GB"
                )
            
            return True
        else:
            self.add_result(
                "GPU设置", "warning", 
                "未检测到可用GPU，将使用CPU模式",
                fix_suggestion="安装CUDA版本的PyTorch以启用GPU加速"
            )
            return True  # CPU模式也是可以的
    
    def validate_configuration_files(self) -> bool:
        """验证配置文件"""
        print("验证配置文件...")
        
        config_files = {
            'shared_memory_config.json': 'config/shared_memory_config.json',
            'vllm_config.json': 'config/vllm_config.json',
            'nlp_config.json': 'config/nlp_config.json'
        }
        
        all_good = True
        
        for config_name, config_path in config_files.items():
            full_path = self.project_root / config_path
            
            if full_path.exists():
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        config_data = json.load(f)
                    
                    self.add_result(
                        f"配置文件-{config_name}", "pass", 
                        f"配置文件有效: {config_path}",
                        details={"path": str(full_path), "keys": list(config_data.keys())}
                    )
                    
                    # 验证配置内容
                    if config_name == 'shared_memory_config.json':
                        self._validate_shared_memory_config(config_data)
                    elif config_name == 'vllm_config.json':
                        self._validate_vllm_config(config_data)
                    elif config_name == 'nlp_config.json':
                        self._validate_nlp_config(config_data)
                        
                except json.JSONDecodeError as e:
                    self.add_result(
                        f"配置文件-{config_name}", "fail", 
                        f"JSON格式错误: {e}",
                        fix_suggestion=f"修复JSON格式错误: {config_path}"
                    )
                    all_good = False
                except Exception as e:
                    self.add_result(
                        f"配置文件-{config_name}", "fail", 
                        f"读取配置文件失败: {e}",
                        fix_suggestion=f"检查文件权限和格式: {config_path}"
                    )
                    all_good = False
            else:
                self.add_result(
                    f"配置文件-{config_name}", "warning", 
                    f"配置文件不存在: {config_path}",
                    fix_suggestion=f"创建配置文件或使用默认配置"
                )
        
        return all_good
    
    def _validate_shared_memory_config(self, config: Dict[str, Any]):
        """验证共享内存配置"""
        required_sections = ['shared_memory', 'cache', 'speculative_decoding']
        
        for section in required_sections:
            if section not in config:
                self.add_result(
                    f"共享内存配置-{section}", "warning", 
                    f"缺少配置节: {section}",
                    fix_suggestion=f"添加{section}配置节"
                )
            else:
                self.add_result(
                    f"共享内存配置-{section}", "pass", 
                    f"配置节存在: {section}"
                )
    
    def _validate_vllm_config(self, config: Dict[str, Any]):
        """验证VLLM配置"""
        required_fields = ['model', 'dtype', 'gpu_memory_utilization']
        
        for field in required_fields:
            if field not in config:
                self.add_result(
                    f"VLLM配置-{field}", "warning", 
                    f"缺少配置字段: {field}",
                    fix_suggestion=f"添加{field}配置字段"
                )
            else:
                self.add_result(
                    f"VLLM配置-{field}", "pass", 
                    f"配置字段存在: {field} = {config[field]}"
                )
    
    def _validate_nlp_config(self, config: Dict[str, Any]):
        """验证NLP配置"""
        if 'enabled' in config and config['enabled']:
            required_fields = ['conjunction_extraction', 'enhanced_similarity']
            
            for field in required_fields:
                if field not in config:
                    self.add_result(
                        f"NLP配置-{field}", "warning", 
                        f"缺少配置字段: {field}",
                        fix_suggestion=f"添加{field}配置字段"
                    )
                else:
                    self.add_result(
                        f"NLP配置-{field}", "pass", 
                        f"配置字段存在: {field}"
                    )
        else:
            self.add_result(
                "NLP配置", "pass", 
                "NLP功能已禁用"
            )
    
    def validate_project_structure(self) -> bool:
        """验证项目结构"""
        print("验证项目结构...")
        
        required_dirs = [
            'src',
            'src/cache',
            'src/config',
            'src/nlp',
            'src/utils',
            'config',
            'scripts',
            'tests',
            'examples'
        ]
        
        required_files = [
            'src/__init__.py',
            'src/cache/__init__.py',
            'src/cache/cache_manager.py',
            'src/cache/vector_search.py',
            'src/cache/shared_memory_manager.py',
            'src/config/__init__.py',
            'src/config/settings.py',
            'src/config/config_manager.py',
            'src/nlp/__init__.py',
            'src/nlp/nlp_enhanced_handler.py',
            'src/utils/__init__.py',
            'src/utils/logger.py',
            'main.py',
            'requirements.txt'
        ]
        
        all_good = True
        
        # 检查目录
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if full_path.exists() and full_path.is_dir():
                self.add_result(
                    f"项目结构-目录", "pass", 
                    f"目录存在: {dir_path}"
                )
            else:
                self.add_result(
                    f"项目结构-目录", "fail", 
                    f"目录缺失: {dir_path}",
                    fix_suggestion=f"创建目录: {dir_path}"
                )
                all_good = False
        
        # 检查文件
        for file_path in required_files:
            full_path = self.project_root / file_path
            if full_path.exists() and full_path.is_file():
                self.add_result(
                    f"项目结构-文件", "pass", 
                    f"文件存在: {file_path}"
                )
            else:
                self.add_result(
                    f"项目结构-文件", "fail", 
                    f"文件缺失: {file_path}",
                    fix_suggestion=f"创建文件: {file_path}"
                )
                all_good = False
        
        return all_good
    
    def validate_redis_connection(self) -> bool:
        """验证Redis连接"""
        print("验证Redis连接...")
        
        if redis is None:
            self.add_result(
                "Redis连接", "fail", 
                "Redis客户端未安装",
                fix_suggestion="安装Redis客户端: pip install redis"
            )
            return False
        
        try:
            # 尝试连接Redis
            r = redis.Redis(host='localhost', port=6379, db=0, socket_timeout=5)
            r.ping()
            
            # 获取Redis信息
            info = r.info()
            redis_version = info.get('redis_version', 'unknown')
            used_memory = info.get('used_memory_human', 'unknown')
            
            self.add_result(
                "Redis连接", "pass", 
                f"Redis连接成功，版本: {redis_version}，内存使用: {used_memory}",
                details={
                    "version": redis_version,
                    "memory_usage": used_memory,
                    "connected_clients": info.get('connected_clients', 0)
                }
            )
            
            # 测试基本操作
            test_key = "deployment_test_key"
            test_value = "deployment_test_value"
            
            r.set(test_key, test_value, ex=60)  # 60秒过期
            retrieved_value = r.get(test_key)
            
            if retrieved_value and retrieved_value.decode() == test_value:
                self.add_result(
                    "Redis操作", "pass", 
                    "Redis读写操作正常"
                )
                r.delete(test_key)  # 清理测试数据
            else:
                self.add_result(
                    "Redis操作", "fail", 
                    "Redis读写操作失败",
                    fix_suggestion="检查Redis配置和权限"
                )
                return False
            
            return True
            
        except redis.ConnectionError:
            self.add_result(
                "Redis连接", "fail", 
                "无法连接到Redis服务器",
                fix_suggestion="启动Redis服务器: redis-server"
            )
            return False
        except redis.TimeoutError:
            self.add_result(
                "Redis连接", "fail", 
                "Redis连接超时",
                fix_suggestion="检查Redis服务器状态和网络连接"
            )
            return False
        except Exception as e:
            self.add_result(
                "Redis连接", "fail", 
                f"Redis连接错误: {e}",
                fix_suggestion="检查Redis配置和服务状态"
            )
            return False
    
    def validate_environment_variables(self) -> bool:
        """验证环境变量"""
        print("验证环境变量...")
        
        important_env_vars = {
            'CUDA_VISIBLE_DEVICES': 'GPU设备选择',
            'VLLM_CACHE_DIR': '缓存目录',
            'REDIS_URL': 'Redis连接URL',
            'LOG_LEVEL': '日志级别'
        }
        
        for var_name, description in important_env_vars.items():
            value = os.getenv(var_name)
            if value:
                self.add_result(
                    f"环境变量-{var_name}", "pass", 
                    f"{description}: {value}"
                )
            else:
                self.add_result(
                    f"环境变量-{var_name}", "warning", 
                    f"{description} 未设置（将使用默认值）",
                    fix_suggestion=f"可选设置: export {var_name}=<value>"
                )
        
        return True
    
    async def validate_service_startup(self) -> bool:
        """验证服务启动"""
        print("验证服务启动...")
        
        try:
            # 尝试导入主要模块
            from src.handler.nlp_enhanced_vllm_handler import NLPEnhancedVLLMHandler
            from src.cache import MultiLevelCacheManager as CacheManager
            from src.cache.enhanced_vector_search import VectorSearchManager
            # SharedMemoryManager 不存在，注释掉
            # from src.cache.shared_memory_manager import SharedMemoryManager
            
            self.add_result(
                "模块导入", "pass", 
                "所有核心模块导入成功"
            )
            
            # 尝试初始化配置
            try:
                config = get_default_config()
                self.add_result(
                    "配置管理器", "pass", 
                    "配置初始化成功"
                )
            except Exception as e:
                self.add_result(
                    "配置管理器", "fail", 
                    f"配置初始化失败: {e}",
                    fix_suggestion="检查配置文件格式和路径"
                )
                return False
            
            # 尝试初始化缓存管理器
            try:
                cache_manager = CacheManager()
                self.add_result(
                    "缓存管理器", "pass", 
                    "缓存管理器初始化成功"
                )
            except Exception as e:
                self.add_result(
                    "缓存管理器", "warning", 
                    f"缓存管理器初始化警告: {e}",
                    fix_suggestion="检查Redis连接和配置"
                )
            
            return True
            
        except ImportError as e:
            self.add_result(
                "模块导入", "fail", 
                f"模块导入失败: {e}",
                fix_suggestion="检查项目结构和依赖包"
            )
            return False
        except Exception as e:
            self.add_result(
                "服务启动", "fail", 
                f"服务启动验证失败: {e}",
                fix_suggestion="检查系统配置和依赖"
            )
            return False
    
    def validate_performance_requirements(self) -> bool:
        """验证性能要求"""
        print("验证性能要求...")
        
        # 检查CPU核心数
        if psutil:
            cpu_count = psutil.cpu_count()
            if cpu_count >= 4:
                self.add_result(
                    "CPU核心数", "pass", 
                    f"CPU核心数充足: {cpu_count}核"
                )
            elif cpu_count >= 2:
                self.add_result(
                    "CPU核心数", "warning", 
                    f"CPU核心数较少: {cpu_count}核",
                    fix_suggestion="建议至少4核CPU以获得最佳性能"
                )
            else:
                self.add_result(
                    "CPU核心数", "fail", 
                    f"CPU核心数不足: {cpu_count}核",
                    fix_suggestion="至少需要2核CPU"
                )
                return False
        
        # 检查网络连接
        try:
            import socket
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            self.add_result(
                "网络连接", "pass", 
                "网络连接正常"
            )
        except OSError:
            self.add_result(
                "网络连接", "warning", 
                "网络连接可能有问题",
                fix_suggestion="检查网络连接，某些功能可能需要网络访问"
            )
        
        return True
    
    async def run_full_validation(self) -> Dict[str, Any]:
        """运行完整验证"""
        print("=" * 60)
        print("VLLM llmcache 部署验证")
        print("=" * 60)
        
        validation_steps = [
            ("系统要求", self.validate_system_requirements),
            ("依赖包", self.validate_dependencies),
            ("GPU设置", self.validate_gpu_setup),
            ("项目结构", self.validate_project_structure),
            ("配置文件", self.validate_configuration_files),
            ("环境变量", self.validate_environment_variables),
            ("Redis连接", self.validate_redis_connection),
            ("性能要求", self.validate_performance_requirements),
            ("服务启动", self.validate_service_startup)
        ]
        
        overall_status = "pass"
        
        for step_name, step_func in validation_steps:
            print(f"\n{step_name}...")
            try:
                if asyncio.iscoroutinefunction(step_func):
                    result = await step_func()
                else:
                    result = step_func()
                
                if not result:
                    overall_status = "fail"
            except Exception as e:
                self.add_result(
                    step_name, "fail", 
                    f"验证步骤失败: {e}",
                    fix_suggestion="检查系统配置和依赖"
                )
                overall_status = "fail"
        
        # 生成报告
        report = self.generate_validation_report(overall_status)
        
        return report
    
    def generate_validation_report(self, overall_status: str) -> Dict[str, Any]:
        """生成验证报告"""
        # 统计结果
        pass_count = sum(1 for r in self.results if r.status == 'pass')
        warning_count = sum(1 for r in self.results if r.status == 'warning')
        fail_count = sum(1 for r in self.results if r.status == 'fail')
        
        # 按组件分组
        components = {}
        for result in self.results:
            component = result.component.split('-')[0] if '-' in result.component else result.component
            if component not in components:
                components[component] = []
            components[component].append(result)
        
        # 生成修复建议
        critical_issues = [r for r in self.results if r.status == 'fail']
        warnings = [r for r in self.results if r.status == 'warning']
        
        report = {
            "overall_status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_checks": len(self.results),
                "passed": pass_count,
                "warnings": warning_count,
                "failed": fail_count,
                "success_rate": (pass_count / len(self.results) * 100) if self.results else 0
            },
            "components": components,
            "critical_issues": [
                {
                    "component": issue.component,
                    "message": issue.message,
                    "fix_suggestion": issue.fix_suggestion
                }
                for issue in critical_issues
            ],
            "warnings": [
                {
                    "component": warning.component,
                    "message": warning.message,
                    "fix_suggestion": warning.fix_suggestion
                }
                for warning in warnings
            ],
            "detailed_results": [
                {
                    "component": r.component,
                    "status": r.status,
                    "message": r.message,
                    "details": r.details,
                    "fix_suggestion": r.fix_suggestion
                }
                for r in self.results
            ]
        }
        
        return report
    
    def print_validation_summary(self, report: Dict[str, Any]):
        """打印验证摘要"""
        print("\n" + "=" * 60)
        print("验证摘要")
        print("=" * 60)
        
        summary = report["summary"]
        overall_status = report["overall_status"]
        
        status_emoji = {
            "pass": "✅",
            "warning": "⚠️",
            "fail": "❌"
        }
        
        print(f"总体状态: {status_emoji.get(overall_status, '❓')} {overall_status.upper()}")
        print(f"总检查项: {summary['total_checks']}")
        print(f"通过: {summary['passed']} | 警告: {summary['warnings']} | 失败: {summary['failed']}")
        print(f"成功率: {summary['success_rate']:.1f}%")
        
        # 显示关键问题
        if report["critical_issues"]:
            print("\n❌ 关键问题:")
            for i, issue in enumerate(report["critical_issues"], 1):
                print(f"{i}. {issue['component']}: {issue['message']}")
                if issue['fix_suggestion']:
                    print(f"   修复建议: {issue['fix_suggestion']}")
        
        # 显示警告
        if report["warnings"]:
            print("\n⚠️ 警告:")
            for i, warning in enumerate(report["warnings"], 1):
                print(f"{i}. {warning['component']}: {warning['message']}")
                if warning['fix_suggestion']:
                    print(f"   建议: {warning['fix_suggestion']}")
        
        # 部署建议
        if overall_status == "pass":
            print("\n🎉 部署验证通过！系统已准备就绪。")
        elif overall_status == "warning":
            print("\n⚠️ 部署基本可用，但建议解决警告问题以获得最佳性能。")
        else:
            print("\n❌ 部署验证失败，请解决关键问题后重新验证。")
    
    def save_validation_report(self, report: Dict[str, Any], filename: Optional[str] = None) -> str:
        """保存验证报告"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"deployment_validation_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            return f"验证报告已保存到: {filename}"
        except Exception as e:
            return f"保存报告失败: {e}"


async def main():
    """主函数"""
    print("VLLM llmcache 部署验证工具")
    
    try:
        validator = DeploymentValidator()
        report = await validator.run_full_validation()
        
        # 显示结果
        validator.print_validation_summary(report)
        
        # 保存报告
        save_choice = input("\n是否保存验证报告到文件? (y/n): ").strip().lower()
        if save_choice == 'y':
            filename = input("文件名 (留空使用默认名称): ").strip() or None
            result = validator.save_validation_report(report, filename)
            print(result)
    
    except KeyboardInterrupt:
        print("\n验证被用户中断")
    except Exception as e:
        print(f"验证过程中发生错误: {e}")


if __name__ == "__main__":
    asyncio.run(main())