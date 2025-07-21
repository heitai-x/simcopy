#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统启动检查脚本

在启动系统前检查所有必要的依赖、配置和环境设置。
"""

import os
import sys
import json
import importlib
from pathlib import Path
from typing import List, Dict, Any, Tuple

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class StartupChecker:
    """系统启动检查器"""
    
    def __init__(self):
        self.issues: List[Dict[str, Any]] = []
        self.warnings: List[Dict[str, Any]] = []
        self.project_root = project_root
    
    def add_issue(self, category: str, message: str, severity: str = "error", fix_suggestion: str = ""):
        """添加问题
        
        Args:
            category: 问题类别
            message: 问题描述
            severity: 严重程度 (error, warning, info)
            fix_suggestion: 修复建议
        """
        issue = {
            "category": category,
            "message": message,
            "severity": severity,
            "fix_suggestion": fix_suggestion
        }
        
        if severity == "error":
            self.issues.append(issue)
        else:
            self.warnings.append(issue)
    
    def check_python_version(self) -> bool:
        """检查Python版本"""
        print("检查Python版本...")
        
        version = sys.version_info
        if version < (3, 8):
            self.add_issue(
                "python",
                f"Python版本过低: {version.major}.{version.minor}.{version.micro}，需要3.8+",
                "error",
                "请升级到Python 3.8或更高版本"
            )
            return False
        
        print(f"✓ Python版本: {version.major}.{version.minor}.{version.micro}")
        return True
    
    def check_required_packages(self) -> bool:
        """检查必需的包"""
        print("检查必需的包...")
        
        required_packages = {
            "torch": "PyTorch深度学习框架",
            "transformers": "Hugging Face Transformers",
            "vllm": "VLLM推理引擎",
            "redis": "Redis客户端",
            "numpy": "数值计算库",
            "pandas": "数据处理库",
            "psutil": "系统监控库",
            "pydantic": "数据验证库",
            "spacy": "自然语言处理库",
            "faiss": "向量搜索库 (faiss-cpu或faiss-gpu)",
            "sentence_transformers": "句子嵌入库",
            "aioredis": "异步Redis客户端",
            "loguru": "日志库"
        }
        
        missing_packages = []
        installed_packages = {}
        
        for package, description in required_packages.items():
            try:
                # 特殊处理一些包名
                import_name = package
                if package == "faiss":
                    try:
                        import faiss
                        import_name = "faiss"
                    except ImportError:
                        try:
                            import faiss_cpu as faiss
                            import_name = "faiss_cpu"
                        except ImportError:
                            try:
                                import faiss_gpu as faiss
                                import_name = "faiss_gpu"
                            except ImportError:
                                raise ImportError("No faiss package found")
                
                module = importlib.import_module(import_name)
                version = getattr(module, '__version__', 'unknown')
                installed_packages[package] = version
                print(f"✓ {package}: {version}")
                
            except ImportError:
                missing_packages.append((package, description))
                print(f"✗ {package}: 未安装")
        
        if missing_packages:
            missing_list = "\n".join([f"  - {pkg}: {desc}" for pkg, desc in missing_packages])
            self.add_issue(
                "dependencies",
                f"缺少必需的包:\n{missing_list}",
                "error",
                "运行: pip install -r requirements.txt"
            )
            return False
        
        return True
    
    def check_spacy_models(self) -> bool:
        """检查spaCy模型"""
        print("检查spaCy模型...")
        
        try:
            import spacy
            
            # 检查英文模型
            try:
                nlp = spacy.load("en_core_web_sm")
                print("✓ spaCy英文模型 (en_core_web_sm): 已安装")
                return True
            except OSError:
                self.add_issue(
                    "spacy_models",
                    "spaCy英文模型未安装",
                    "warning",
                    "运行: python -m spacy download en_core_web_sm"
                )
                return False
                
        except ImportError:
            self.add_issue(
                "spacy_models",
                "spaCy未安装，无法检查模型",
                "error",
                "先安装spaCy: pip install spacy"
            )
            return False
    
    def check_project_structure(self) -> bool:
        """检查项目结构"""
        print("检查项目结构...")
        
        required_dirs = [
            "src",
            "src/config",
            "src/handler",
            "src/cache",
            "src/nlp",
            "src/retriever",
            "src/speculative",
            "src/utils",
            "config",
            "examples",
            "tests"
        ]
        
        required_files = [
            "main.py",
            "requirements.txt",
            "src/__init__.py",
            "src/config/settings.py",
            "src/config/config_manager.py",
            "src/utils/logger.py",
            "config/shared_memory_config.json"
        ]
        
        missing_dirs = []
        missing_files = []
        
        # 检查目录
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if not full_path.exists():
                missing_dirs.append(dir_path)
            else:
                print(f"✓ 目录: {dir_path}")
        
        # 检查文件
        for file_path in required_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                missing_files.append(file_path)
            else:
                print(f"✓ 文件: {file_path}")
        
        if missing_dirs:
            self.add_issue(
                "project_structure",
                f"缺少目录: {', '.join(missing_dirs)}",
                "error",
                "请检查项目完整性，重新下载或克隆项目"
            )
        
        if missing_files:
            self.add_issue(
                "project_structure",
                f"缺少文件: {', '.join(missing_files)}",
                "error",
                "请检查项目完整性，重新下载或克隆项目"
            )
        
        return len(missing_dirs) == 0 and len(missing_files) == 0
    
    def check_config_files(self) -> bool:
        """检查配置文件"""
        print("检查配置文件...")
        
        config_file = self.project_root / "config" / "shared_memory_config.json"
        
        if not config_file.exists():
            self.add_issue(
                "config",
                "共享内存配置文件不存在",
                "error",
                "请检查 config/shared_memory_config.json 文件"
            )
            return False
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # 检查必要的配置项
            required_sections = ["shared_memory", "cache_strategy", "performance", "speculative_decoding"]
            
            for section in required_sections:
                if section not in config_data:
                    self.add_issue(
                        "config",
                        f"配置文件缺少 '{section}' 部分",
                        "warning",
                        "请检查配置文件完整性"
                    )
            
            print("✓ 配置文件格式正确")
            return True
            
        except json.JSONDecodeError as e:
            self.add_issue(
                "config",
                f"配置文件JSON格式错误: {e}",
                "error",
                "请检查JSON语法，确保没有多余的逗号或括号"
            )
            return False
    
    def check_environment_variables(self) -> bool:
        """检查环境变量"""
        print("检查环境变量...")
        
        # 检查是否有.env文件
        env_file = self.project_root / ".env"
        env_example_file = self.project_root / ".env.example"
        
        if not env_file.exists() and env_example_file.exists():
            self.add_issue(
                "environment",
                ".env文件不存在",
                "warning",
                "复制 .env.example 为 .env 并根据需要修改配置"
            )
        
        # 检查重要的环境变量
        important_vars = {
            "VLLM_ENV": "运行环境",
            "VLLM_LOG_LEVEL": "日志级别"
        }
        
        for var, description in important_vars.items():
            value = os.getenv(var)
            if value:
                print(f"✓ {var}: {value}")
            else:
                print(f"- {var}: 未设置 (将使用默认值)")
        
        return True
    
    def check_system_resources(self) -> bool:
        """检查系统资源"""
        print("检查系统资源...")
        
        try:
            import psutil
            
            # 检查内存
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            
            if memory_gb < 4:
                self.add_issue(
                    "system_resources",
                    f"系统内存可能不足: {memory_gb:.1f}GB",
                    "warning",
                    "建议至少4GB内存用于运行VLLM"
                )
            else:
                print(f"✓ 系统内存: {memory_gb:.1f}GB")
            
            # 检查磁盘空间
            disk = psutil.disk_usage('.')
            free_gb = disk.free / (1024**3)
            
            if free_gb < 10:
                self.add_issue(
                    "system_resources",
                    f"磁盘空间不足: {free_gb:.1f}GB",
                    "warning",
                    "建议至少10GB可用磁盘空间"
                )
            else:
                print(f"✓ 可用磁盘空间: {free_gb:.1f}GB")
            
            return True
            
        except ImportError:
            self.add_issue(
                "system_resources",
                "无法检查系统资源 (psutil未安装)",
                "warning",
                "安装psutil以启用系统资源检查: pip install psutil"
            )
            return False
    
    def check_gpu_availability(self) -> bool:
        """检查GPU可用性"""
        print("检查GPU可用性...")
        
        try:
            import torch
            
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                gpu_name = torch.cuda.get_device_name(current_device)
                
                print(f"✓ CUDA可用: {gpu_count}个GPU")
                print(f"✓ 当前GPU: {gpu_name}")
                
                # 检查GPU内存
                gpu_memory = torch.cuda.get_device_properties(current_device).total_memory
                gpu_memory_gb = gpu_memory / (1024**3)
                
                if gpu_memory_gb < 8:
                    self.add_issue(
                        "gpu",
                        f"GPU内存可能不足: {gpu_memory_gb:.1f}GB",
                        "warning",
                        "建议至少8GB GPU内存用于运行大型模型"
                    )
                else:
                    print(f"✓ GPU内存: {gpu_memory_gb:.1f}GB")
                
                return True
            else:
                self.add_issue(
                    "gpu",
                    "CUDA不可用，将使用CPU模式",
                    "warning",
                    "安装CUDA和对应的PyTorch版本以启用GPU加速"
                )
                return False
                
        except ImportError:
            self.add_issue(
                "gpu",
                "无法检查GPU (PyTorch未安装)",
                "error",
                "请先安装PyTorch"
            )
            return False
    
    def run_all_checks(self) -> Tuple[bool, Dict[str, Any]]:
        """运行所有检查
        
        Returns:
            Tuple[bool, Dict[str, Any]]: (是否通过所有检查, 检查结果)
        """
        print("=" * 60)
        print("VLLM llmcache 系统启动检查")
        print("=" * 60)
        
        checks = [
            ("Python版本", self.check_python_version),
            ("必需包", self.check_required_packages),
            ("spaCy模型", self.check_spacy_models),
            ("项目结构", self.check_project_structure),
            ("配置文件", self.check_config_files),
            ("环境变量", self.check_environment_variables),
            ("系统资源", self.check_system_resources),
            ("GPU可用性", self.check_gpu_availability)
        ]
        
        results = {}
        all_passed = True
        
        for check_name, check_func in checks:
            print(f"\n--- {check_name} ---")
            try:
                result = check_func()
                results[check_name] = result
                if not result:
                    all_passed = False
            except Exception as e:
                print(f"✗ 检查失败: {e}")
                results[check_name] = False
                all_passed = False
                self.add_issue(
                    "check_error",
                    f"{check_name}检查时发生错误: {e}",
                    "error"
                )
        
        # 输出总结
        print("\n" + "=" * 60)
        print("检查总结")
        print("=" * 60)
        
        if self.issues:
            print(f"\n❌ 发现 {len(self.issues)} 个错误:")
            for i, issue in enumerate(self.issues, 1):
                print(f"{i}. [{issue['category']}] {issue['message']}")
                if issue['fix_suggestion']:
                    print(f"   修复建议: {issue['fix_suggestion']}")
        
        if self.warnings:
            print(f"\n⚠️  发现 {len(self.warnings)} 个警告:")
            for i, warning in enumerate(self.warnings, 1):
                print(f"{i}. [{warning['category']}] {warning['message']}")
                if warning['fix_suggestion']:
                    print(f"   建议: {warning['fix_suggestion']}")
        
        if all_passed and not self.issues:
            print("\n✅ 所有检查通过！系统可以启动。")
        elif not self.issues:
            print("\n✅ 基本检查通过，但有一些警告。系统可以启动。")
        else:
            print("\n❌ 检查未通过，请修复错误后再启动系统。")
        
        return all_passed and len(self.issues) == 0, {
            "passed": all_passed and len(self.issues) == 0,
            "results": results,
            "issues": self.issues,
            "warnings": self.warnings
        }


def main():
    """主函数"""
    checker = StartupChecker()
    passed, results = checker.run_all_checks()
    
    # 返回适当的退出码
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()