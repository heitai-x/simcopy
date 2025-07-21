#!/usr/bin/env python3
"""系统健康检查脚本

用于检查系统各组件的状态和配置
"""

import asyncio
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.handler.nlp_enhanced_vllm_handler import NLPEnhancedVLLMHandler
from src.config.settings import get_default_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HealthCheckResult:
    """健康检查结果"""
    
    def __init__(self, component: str, status: str, message: str = "", details: Dict[str, Any] = None):
        self.component = component
        self.status = status  # "healthy", "warning", "error"
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "component": self.component,
            "status": self.status,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }


class SystemHealthChecker:
    """系统健康检查器"""
    
    def __init__(self, config_path: str = None):
        """初始化健康检查器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = get_default_config()
        self.handler = None
        self.results: List[HealthCheckResult] = []
        
        # 健康检查配置
        self.check_config = {
            "timeout_seconds": 30,
            "max_response_time": 10.0,
            "min_memory_mb": 100,
            "max_cpu_percent": 90,
            "min_disk_space_gb": 1.0
        }
    
    def add_result(self, component: str, status: str, message: str = "", details: Dict[str, Any] = None):
        """添加检查结果
        
        Args:
            component: 组件名称
            status: 状态（healthy/warning/error）
            message: 消息
            details: 详细信息
        """
        result = HealthCheckResult(component, status, message, details)
        self.results.append(result)
        
        # 记录日志
        if status == "healthy":
            logger.info(f"✓ {component}: {message}")
        elif status == "warning":
            logger.warning(f"⚠ {component}: {message}")
        else:  # error
            logger.error(f"✗ {component}: {message}")
    
    def check_system_requirements(self) -> None:
        """检查系统要求"""
        logger.info("检查系统要求...")
        
        try:
            import psutil
            import platform
            
            # 检查Python版本
            python_version = sys.version_info
            if python_version >= (3, 8):
                self.add_result(
                    "python_version", "healthy",
                    f"Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}"
                )
            else:
                self.add_result(
                    "python_version", "error",
                    f"Python版本过低: {python_version.major}.{python_version.minor}.{python_version.micro}，需要3.8+"
                )
            
            # 检查内存
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            if memory_gb >= self.check_config["min_memory_mb"] / 1024:
                self.add_result(
                    "system_memory", "healthy",
                    f"系统内存: {memory_gb:.1f}GB，可用: {memory.available / (1024**3):.1f}GB",
                    {"total_gb": memory_gb, "available_gb": memory.available / (1024**3)}
                )
            else:
                self.add_result(
                    "system_memory", "warning",
                    f"系统内存可能不足: {memory_gb:.1f}GB"
                )
            
            # 检查CPU
            cpu_count = psutil.cpu_count()
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent < self.check_config["max_cpu_percent"]:
                self.add_result(
                    "system_cpu", "healthy",
                    f"CPU: {cpu_count}核，当前使用率: {cpu_percent:.1f}%",
                    {"cpu_count": cpu_count, "cpu_percent": cpu_percent}
                )
            else:
                self.add_result(
                    "system_cpu", "warning",
                    f"CPU使用率较高: {cpu_percent:.1f}%"
                )
            
            # 检查磁盘空间
            disk = psutil.disk_usage('/')
            free_gb = disk.free / (1024**3)
            if free_gb >= self.check_config["min_disk_space_gb"]:
                self.add_result(
                    "disk_space", "healthy",
                    f"磁盘空间: 总计{disk.total / (1024**3):.1f}GB，可用{free_gb:.1f}GB",
                    {"total_gb": disk.total / (1024**3), "free_gb": free_gb}
                )
            else:
                self.add_result(
                    "disk_space", "error",
                    f"磁盘空间不足: 仅剩{free_gb:.1f}GB"
                )
            
            # 检查操作系统
            os_info = platform.platform()
            self.add_result(
                "operating_system", "healthy",
                f"操作系统: {os_info}",
                {"platform": os_info}
            )
            
        except Exception as e:
            self.add_result(
                "system_requirements", "error",
                f"检查系统要求时发生错误: {e}"
            )
    
    def check_dependencies(self) -> None:
        """检查依赖包"""
        logger.info("检查依赖包...")
        
        required_packages = [
            "torch", "transformers", "vllm", "asyncio",
            "psutil", "numpy", "pandas"
        ]
        
        missing_packages = []
        installed_packages = {}
        
        for package in required_packages:
            try:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
                installed_packages[package] = version
            except ImportError:
                missing_packages.append(package)
        
        if not missing_packages:
            self.add_result(
                "dependencies", "healthy",
                f"所有必需依赖包已安装: {len(installed_packages)}个",
                {"installed_packages": installed_packages}
            )
        else:
            self.add_result(
                "dependencies", "error",
                f"缺少依赖包: {', '.join(missing_packages)}",
                {"missing_packages": missing_packages, "installed_packages": installed_packages}
            )
    
    def check_configuration(self) -> None:
        """检查配置"""
        logger.info("检查配置...")
        
        try:
            # 使用默认配置
            self.config = get_default_config()
            
            # 检查配置文件是否存在
            config_files = [
                "config/model_config.json",
                "config/cache_config.json",
                "config/performance_config.json"
            ]
            
            missing_configs = []
            existing_configs = []
            
            for config_file in config_files:
                config_path = project_root / config_file
                if config_path.exists():
                    existing_configs.append(config_file)
                else:
                    missing_configs.append(config_file)
            
            if not missing_configs:
                self.add_result(
                    "configuration_files", "healthy",
                    f"所有配置文件存在: {len(existing_configs)}个",
                    {"existing_configs": existing_configs}
                )
            else:
                self.add_result(
                    "configuration_files", "warning",
                    f"部分配置文件缺失: {', '.join(missing_configs)}",
                    {"missing_configs": missing_configs, "existing_configs": existing_configs}
                )
            
            # 检查配置内容
            try:
                model_config = self.config_manager.get_model_config()
                if model_config:
                    self.add_result(
                        "model_configuration", "healthy",
                        "模型配置加载成功",
                        {"model_name": model_config.get("model_name", "unknown")}
                    )
                else:
                    self.add_result(
                        "model_configuration", "error",
                        "模型配置加载失败"
                    )
            except Exception as e:
                self.add_result(
                    "model_configuration", "error",
                    f"模型配置检查失败: {e}"
                )
            
        except Exception as e:
            self.add_result(
                "configuration", "error",
                f"配置检查失败: {e}"
            )
    
    async def check_handler_initialization(self) -> None:
        """检查处理器初始化"""
        logger.info("检查处理器初始化...")
        
        try:
            if not self.config:
                self.config = get_default_config()
            
            # 初始化处理器
            start_time = time.time()
            self.handler = NLPEnhancedVLLMHandler(self.config)
            
            # 等待初始化完成
            await asyncio.sleep(2)
            
            init_time = time.time() - start_time
            
            if self.handler:
                self.add_result(
                    "handler_initialization", "healthy",
                    f"处理器初始化成功，耗时: {init_time:.2f}秒",
                    {"initialization_time": init_time}
                )
            else:
                self.add_result(
                    "handler_initialization", "error",
                    "处理器初始化失败"
                )
            
        except Exception as e:
            self.add_result(
                "handler_initialization", "error",
                f"处理器初始化异常: {e}"
            )
    
    async def check_basic_functionality(self) -> None:
        """检查基本功能"""
        logger.info("检查基本功能...")
        
        if not self.handler:
            self.add_result(
                "basic_functionality", "error",
                "处理器未初始化，无法测试基本功能"
            )
            return
        
        try:
            # 测试基本请求处理
            start_time = time.time()
            
            response = await asyncio.wait_for(
                self.handler.add_request(
                    request_id="health_check_test",
                    prompt="Hello, this is a health check test.",
                    max_tokens=20,
                    temperature=0.5
                ),
                timeout=self.check_config["timeout_seconds"]
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            if response and "error" not in response:
                if response_time <= self.check_config["max_response_time"]:
                    self.add_result(
                        "basic_functionality", "healthy",
                        f"基本功能正常，响应时间: {response_time:.2f}秒",
                        {"response_time": response_time, "response_preview": str(response)[:100]}
                    )
                else:
                    self.add_result(
                        "basic_functionality", "warning",
                        f"基本功能正常但响应较慢: {response_time:.2f}秒",
                        {"response_time": response_time}
                    )
            else:
                self.add_result(
                    "basic_functionality", "error",
                    f"基本功能测试失败: {response}"
                )
            
        except asyncio.TimeoutError:
            self.add_result(
                "basic_functionality", "error",
                f"基本功能测试超时（>{self.check_config['timeout_seconds']}秒）"
            )
        except Exception as e:
            self.add_result(
                "basic_functionality", "error",
                f"基本功能测试异常: {e}"
            )
    

            )
    

    
    async def check_memory_usage(self) -> None:
        """检查内存使用情况"""
        logger.info("检查内存使用情况...")
        
        try:
            if self.handler:
                memory_stats = self.handler.get_memory_stats()
                
                if memory_stats:
                    # 检查缓存使用情况
                    cache_stats = memory_stats.get('cache_stats', {})
                    cache_size_mb = cache_stats.get('total_size', 0) / (1024 * 1024)
                    
                    if cache_size_mb < 1000:  # 小于1GB
                        self.add_result(
                            "memory_usage", "healthy",
                            f"内存使用正常，缓存大小: {cache_size_mb:.1f}MB",
                            {"cache_size_mb": cache_size_mb, "memory_stats": memory_stats}
                        )
                    else:
                        self.add_result(
                            "memory_usage", "warning",
                            f"缓存使用较多: {cache_size_mb:.1f}MB",
                            {"cache_size_mb": cache_size_mb}
                        )
                else:
                    self.add_result(
                        "memory_usage", "warning",
                        "无法获取内存统计信息"
                    )
            else:
                self.add_result(
                    "memory_usage", "error",
                    "处理器未初始化，无法检查内存使用"
                )
            
        except Exception as e:
            self.add_result(
                "memory_usage", "error",
                f"内存使用检查异常: {e}"
            )
    
    def check_file_permissions(self) -> None:
        """检查文件权限"""
        logger.info("检查文件权限...")
        
        try:
            # 检查关键目录的读写权限
            test_dirs = [
                project_root / "logs",
                project_root / "cache",
                project_root / "config"
            ]
            
            permission_issues = []
            
            for test_dir in test_dirs:
                try:
                    # 创建目录（如果不存在）
                    test_dir.mkdir(exist_ok=True)
                    
                    # 测试写权限
                    test_file = test_dir / "health_check_test.tmp"
                    test_file.write_text("test")
                    test_file.unlink()  # 删除测试文件
                    
                except Exception as e:
                    permission_issues.append(f"{test_dir}: {e}")
            
            if not permission_issues:
                self.add_result(
                    "file_permissions", "healthy",
                    "文件权限检查通过"
                )
            else:
                self.add_result(
                    "file_permissions", "error",
                    f"文件权限问题: {'; '.join(permission_issues)}",
                    {"permission_issues": permission_issues}
                )
            
        except Exception as e:
            self.add_result(
                "file_permissions", "error",
                f"文件权限检查异常: {e}"
            )
    
    async def run_all_checks(self) -> List[HealthCheckResult]:
        """运行所有健康检查
        
        Returns:
            检查结果列表
        """
        logger.info("开始系统健康检查...")
        
        start_time = time.time()
        
        try:
            # 1. 系统要求检查
            self.check_system_requirements()
            
            # 2. 依赖包检查
            self.check_dependencies()
            
            # 3. 配置检查
            self.check_configuration()
            
            # 4. 文件权限检查
            self.check_file_permissions()
            
            # 5. 处理器初始化检查
            await self.check_handler_initialization()
            
            # 6. 基本功能检查
            await self.check_basic_functionality()
            
            # 7. 内存使用检查
            await self.check_memory_usage()
            

            
            end_time = time.time()
            total_time = end_time - start_time
            
            logger.info(f"健康检查完成，总耗时: {total_time:.2f}秒")
            
        except Exception as e:
            logger.error(f"健康检查过程中发生错误: {e}")
            self.add_result(
                "health_check_process", "error",
                f"健康检查过程异常: {e}"
            )
        
        finally:
            # 清理资源
            pass
        
        return self.results
    
    def generate_summary(self) -> Dict[str, Any]:
        """生成检查摘要
        
        Returns:
            检查摘要
        """
        total_checks = len(self.results)
        healthy_count = sum(1 for r in self.results if r.status == "healthy")
        warning_count = sum(1 for r in self.results if r.status == "warning")
        error_count = sum(1 for r in self.results if r.status == "error")
        
        overall_status = "healthy"
        if error_count > 0:
            overall_status = "error"
        elif warning_count > 0:
            overall_status = "warning"
        
        return {
            "overall_status": overall_status,
            "total_checks": total_checks,
            "healthy_count": healthy_count,
            "warning_count": warning_count,
            "error_count": error_count,
            "health_percentage": (healthy_count / total_checks * 100) if total_checks > 0 else 0,
            "timestamp": datetime.now().isoformat()
        }
    
    def export_results(self, output_file: str = None) -> str:
        """导出检查结果
        
        Args:
            output_file: 输出文件路径
            
        Returns:
            输出文件路径
        """
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"health_check_report_{timestamp}.json"
        
        # 确保输出目录存在
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 生成报告数据
        report_data = {
            "summary": self.generate_summary(),
            "results": [result.to_dict() for result in self.results],
            "check_config": self.check_config
        }
        
        # 写入文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"健康检查报告已导出到: {output_file}")
        return output_file


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="系统健康检查")
    parser.add_argument("--config", "-c", help="配置文件路径")
    parser.add_argument("--output", "-o", help="输出文件路径")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    parser.add_argument("--timeout", type=int, default=30, help="超时时间（秒）")
    
    args = parser.parse_args()
    
    # 创建健康检查器
    checker = SystemHealthChecker(args.config)
    
    # 设置超时时间
    if args.timeout:
        checker.check_config["timeout_seconds"] = args.timeout
    
    try:
        # 运行健康检查
        results = await checker.run_all_checks()
        
        # 生成摘要
        summary = checker.generate_summary()
        
        # 输出结果
        print("\n=== 系统健康检查结果 ===")
        print(f"总体状态: {summary['overall_status'].upper()}")
        print(f"检查项目: {summary['total_checks']}")
        print(f"健康: {summary['healthy_count']}, 警告: {summary['warning_count']}, 错误: {summary['error_count']}")
        print(f"健康度: {summary['health_percentage']:.1f}%")
        
        if args.verbose:
            print("\n=== 详细结果 ===")
            for result in results:
                status_symbol = {
                    "healthy": "✓",
                    "warning": "⚠",
                    "error": "✗"
                }.get(result.status, "?")
                
                print(f"{status_symbol} {result.component}: {result.message}")
        
        # 导出结果
        output_file = checker.export_results(args.output)
        print(f"\n详细报告已保存到: {output_file}")
        
        # 设置退出码
        if summary['overall_status'] == "error":
            sys.exit(1)
        elif summary['overall_status'] == "warning":
            sys.exit(2)
        else:
            sys.exit(0)
        
    except KeyboardInterrupt:
        logger.info("用户中断健康检查")
        sys.exit(130)
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())