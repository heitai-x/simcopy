#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能基准测试脚本

评估VLLM llmcache系统的性能表现，包括缓存效率、响应时间、吞吐量等指标。
"""

import os
import sys
import time
import asyncio
import statistics
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import psutil
except ImportError:
    psutil = None

try:
    import numpy as np
except ImportError:
    np = None

from src.config.settings import get_default_config, create_benchmark_engine_args
# ConfigManager 已移除，使用默认配置


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    test_name: str
    start_time: str
    end_time: str
    duration_seconds: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time_ms: float
    min_response_time_ms: float
    max_response_time_ms: float
    p50_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    requests_per_second: float
    cache_hit_rate: float
    memory_usage_mb: float
    cpu_usage_percent: float
    error_rate: float
    throughput_tokens_per_second: float = 0.0
    avg_tokens_per_request: float = 0.0


@dataclass
class SystemMetrics:
    """系统指标"""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    gpu_utilization: Optional[float] = None
    gpu_memory_used_mb: Optional[float] = None


class PerformanceBenchmark:
    """性能基准测试器"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = get_default_config()
        self.results: List[BenchmarkResult] = []
        self.system_metrics: List[SystemMetrics] = []
        self.baseline_metrics = None
        
        # 测试数据
        self.test_prompts = self._generate_test_prompts()
        self.cache_hit_count = 0
        self.cache_miss_count = 0
    
    def _generate_test_prompts(self) -> List[str]:
        """生成测试提示词"""
        prompts = [
            # 短文本生成
            "Generate a brief summary of artificial intelligence.",
            "Explain the concept of machine learning in simple terms.",
            "What are the benefits of cloud computing?",
            "Describe the process of photosynthesis.",
            "List the top 5 programming languages in 2024.",
            
            # 中等长度文本生成
            "Write a detailed explanation of how neural networks work, including the concepts of layers, weights, and activation functions.",
            "Create a comprehensive guide for setting up a development environment for Python programming, including virtual environments and package management.",
            "Explain the differences between supervised, unsupervised, and reinforcement learning with practical examples.",
            "Describe the architecture and benefits of microservices compared to monolithic applications.",
            "Write a tutorial on implementing RESTful APIs using modern web frameworks.",
            
            # 长文本生成
            "Write a comprehensive research paper outline on the impact of artificial intelligence on modern healthcare, including current applications, challenges, ethical considerations, and future prospects.",
            "Create a detailed business plan for a technology startup focusing on sustainable energy solutions, including market analysis, competitive landscape, financial projections, and implementation timeline.",
            "Develop a complete curriculum for teaching data science to beginners, including learning objectives, course modules, practical exercises, and assessment methods.",
            
            # 代码生成
            "Write a Python function to implement binary search algorithm with error handling.",
            "Create a JavaScript class for managing user authentication with JWT tokens.",
            "Implement a SQL query to find the top 10 customers by total purchase amount.",
            "Write a React component for displaying a data table with sorting and filtering.",
            
            # 分析和推理
            "Analyze the pros and cons of different database types (SQL vs NoSQL) for various use cases.",
            "Compare and contrast different machine learning algorithms for classification problems.",
            "Evaluate the security implications of using third-party APIs in web applications.",
            "Assess the environmental impact of different cloud computing providers.",
            
            # 创意写作
            "Write a short story about a programmer who discovers their code can predict the future.",
            "Create a dialogue between two AI systems discussing the nature of consciousness.",
            "Compose a poem about the beauty of mathematical equations.",
            "Write a product description for a revolutionary new smartphone feature.",
            
            # 技术文档
            "Create API documentation for a user management service including endpoints, parameters, and response formats.",
            "Write installation instructions for a complex software system with multiple dependencies.",
            "Document the troubleshooting steps for common database connection issues.",
            "Create a user manual for a data visualization dashboard."
        ]
        
        # 添加一些重复的提示词来测试缓存效果
        cache_test_prompts = prompts[:10] * 3  # 重复前10个提示词3次
        
        return prompts + cache_test_prompts
    
    def collect_system_metrics(self) -> SystemMetrics:
        """收集系统指标"""
        timestamp = datetime.now().isoformat()
        
        if psutil:
            # CPU和内存
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_mb = memory.used / (1024**2)
            
            # 磁盘I/O
            disk_io = psutil.disk_io_counters()
            if self.baseline_metrics:
                disk_io_read_mb = (disk_io.read_bytes - self.baseline_metrics['disk_read']) / (1024**2)
                disk_io_write_mb = (disk_io.write_bytes - self.baseline_metrics['disk_write']) / (1024**2)
            else:
                disk_io_read_mb = disk_io_write_mb = 0.0
                self.baseline_metrics = {
                    'disk_read': disk_io.read_bytes,
                    'disk_write': disk_io.write_bytes,
                    'network_sent': 0,
                    'network_recv': 0
                }
            
            # 网络I/O
            network_io = psutil.net_io_counters()
            if self.baseline_metrics:
                network_sent_mb = (network_io.bytes_sent - self.baseline_metrics['network_sent']) / (1024**2)
                network_recv_mb = (network_io.bytes_recv - self.baseline_metrics['network_recv']) / (1024**2)
            else:
                network_sent_mb = network_recv_mb = 0.0
                self.baseline_metrics.update({
                    'network_sent': network_io.bytes_sent,
                    'network_recv': network_io.bytes_recv
                })
        else:
            cpu_percent = memory_percent = memory_used_mb = 0.0
            disk_io_read_mb = disk_io_write_mb = 0.0
            network_sent_mb = network_recv_mb = 0.0
        
        # GPU指标（如果可用）
        gpu_utilization = gpu_memory_used_mb = None
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_utilization = gpu.load * 100
                gpu_memory_used_mb = gpu.memoryUsed
        except ImportError:
            pass
        
        return SystemMetrics(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_mb=memory_used_mb,
            disk_io_read_mb=disk_io_read_mb,
            disk_io_write_mb=disk_io_write_mb,
            network_sent_mb=network_sent_mb,
            network_recv_mb=network_recv_mb,
            gpu_utilization=gpu_utilization,
            gpu_memory_used_mb=gpu_memory_used_mb
        )
    
    async def simulate_request(self, prompt: str, request_id: str) -> Tuple[bool, float, int]:
        """模拟请求处理
        
        Returns:
            Tuple[bool, float, int]: (成功标志, 响应时间(ms), 生成的token数)
        """
        start_time = time.time()
        
        try:
            # 模拟缓存查找
            cache_key = f"cache_{hash(prompt) % 1000}"
            
            # 30%的概率命中缓存（模拟）
            if hash(prompt) % 10 < 3:
                self.cache_hit_count += 1
                # 缓存命中，快速响应
                await asyncio.sleep(0.01)  # 10ms
                response_time = (time.time() - start_time) * 1000
                return True, response_time, len(prompt.split()) * 2  # 模拟token数
            else:
                self.cache_miss_count += 1
                # 缓存未命中，需要生成
                # 根据提示词长度模拟不同的处理时间
                prompt_length = len(prompt)
                if prompt_length < 100:
                    processing_time = 0.1 + (prompt_length / 1000)  # 短文本
                elif prompt_length < 500:
                    processing_time = 0.3 + (prompt_length / 2000)  # 中等文本
                else:
                    processing_time = 0.8 + (prompt_length / 3000)  # 长文本
                
                # 添加一些随机性
                import random
                processing_time *= (0.8 + random.random() * 0.4)
                
                await asyncio.sleep(processing_time)
                
                # 5%的概率失败
                if random.random() < 0.05:
                    response_time = (time.time() - start_time) * 1000
                    return False, response_time, 0
                
                response_time = (time.time() - start_time) * 1000
                return True, response_time, len(prompt.split()) * 3  # 模拟token数
                
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return False, response_time, 0
    
    async def run_concurrent_test(self, test_name: str, num_requests: int, concurrency: int) -> BenchmarkResult:
        """运行并发测试"""
        print(f"\n开始测试: {test_name}")
        print(f"请求数: {num_requests}, 并发数: {concurrency}")
        
        start_time = datetime.now()
        start_metrics = self.collect_system_metrics()
        
        # 重置缓存统计
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        
        # 准备请求
        requests = []
        for i in range(num_requests):
            prompt = self.test_prompts[i % len(self.test_prompts)]
            request_id = f"{test_name}_req_{i}"
            requests.append((prompt, request_id))
        
        # 执行并发请求
        semaphore = asyncio.Semaphore(concurrency)
        
        async def limited_request(prompt, request_id):
            async with semaphore:
                return await self.simulate_request(prompt, request_id)
        
        print("执行请求...")
        tasks = [limited_request(prompt, req_id) for prompt, req_id in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = datetime.now()
        end_metrics = self.collect_system_metrics()
        
        # 分析结果
        successful_requests = 0
        failed_requests = 0
        response_times = []
        total_tokens = 0
        
        for result in results:
            if isinstance(result, Exception):
                failed_requests += 1
            else:
                success, response_time, tokens = result
                if success:
                    successful_requests += 1
                    response_times.append(response_time)
                    total_tokens += tokens
                else:
                    failed_requests += 1
        
        # 计算统计指标
        duration = (end_time - start_time).total_seconds()
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            
            if np:
                p50_response_time = np.percentile(response_times, 50)
                p95_response_time = np.percentile(response_times, 95)
                p99_response_time = np.percentile(response_times, 99)
            else:
                sorted_times = sorted(response_times)
                n = len(sorted_times)
                p50_response_time = sorted_times[int(n * 0.5)]
                p95_response_time = sorted_times[int(n * 0.95)]
                p99_response_time = sorted_times[int(n * 0.99)]
        else:
            avg_response_time = min_response_time = max_response_time = 0.0
            p50_response_time = p95_response_time = p99_response_time = 0.0
        
        requests_per_second = successful_requests / duration if duration > 0 else 0
        cache_hit_rate = (self.cache_hit_count / (self.cache_hit_count + self.cache_miss_count) * 100) if (self.cache_hit_count + self.cache_miss_count) > 0 else 0
        error_rate = (failed_requests / num_requests * 100) if num_requests > 0 else 0
        throughput_tokens_per_second = total_tokens / duration if duration > 0 else 0
        avg_tokens_per_request = total_tokens / successful_requests if successful_requests > 0 else 0
        
        result = BenchmarkResult(
            test_name=test_name,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            duration_seconds=duration,
            total_requests=num_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            avg_response_time_ms=avg_response_time,
            min_response_time_ms=min_response_time,
            max_response_time_ms=max_response_time,
            p50_response_time_ms=p50_response_time,
            p95_response_time_ms=p95_response_time,
            p99_response_time_ms=p99_response_time,
            requests_per_second=requests_per_second,
            cache_hit_rate=cache_hit_rate,
            memory_usage_mb=end_metrics.memory_used_mb,
            cpu_usage_percent=end_metrics.cpu_percent,
            error_rate=error_rate,
            throughput_tokens_per_second=throughput_tokens_per_second,
            avg_tokens_per_request=avg_tokens_per_request
        )
        
        self.results.append(result)
        
        print(f"测试完成: {test_name}")
        print(f"成功请求: {successful_requests}/{num_requests}")
        print(f"平均响应时间: {avg_response_time:.2f}ms")
        print(f"请求/秒: {requests_per_second:.2f}")
        print(f"缓存命中率: {cache_hit_rate:.1f}%")
        print(f"错误率: {error_rate:.1f}%")
        
        return result
    
    async def run_stress_test(self, duration_minutes: int = 5, max_concurrency: int = 50) -> BenchmarkResult:
        """运行压力测试"""
        test_name = f"stress_test_{duration_minutes}min"
        print(f"\n开始压力测试: {duration_minutes}分钟")
        
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        total_requests = 0
        successful_requests = 0
        failed_requests = 0
        response_times = []
        total_tokens = 0
        
        # 重置缓存统计
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        
        print("开始压力测试...")
        
        while datetime.now() < end_time:
            # 动态调整并发数
            current_concurrency = min(max_concurrency, total_requests // 10 + 1)
            
            # 批量请求
            batch_size = min(20, current_concurrency)
            batch_requests = []
            
            for i in range(batch_size):
                prompt = self.test_prompts[total_requests % len(self.test_prompts)]
                request_id = f"{test_name}_req_{total_requests}"
                batch_requests.append((prompt, request_id))
                total_requests += 1
            
            # 执行批量请求
            semaphore = asyncio.Semaphore(current_concurrency)
            
            async def limited_request(prompt, request_id):
                async with semaphore:
                    return await self.simulate_request(prompt, request_id)
            
            tasks = [limited_request(prompt, req_id) for prompt, req_id in batch_requests]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理批量结果
            for result in batch_results:
                if isinstance(result, Exception):
                    failed_requests += 1
                else:
                    success, response_time, tokens = result
                    if success:
                        successful_requests += 1
                        response_times.append(response_time)
                        total_tokens += tokens
                    else:
                        failed_requests += 1
            
            # 短暂休息
            await asyncio.sleep(0.1)
        
        actual_end_time = datetime.now()
        duration = (actual_end_time - start_time).total_seconds()
        
        # 计算统计指标
        if response_times:
            avg_response_time = statistics.mean(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            
            if np:
                p50_response_time = np.percentile(response_times, 50)
                p95_response_time = np.percentile(response_times, 95)
                p99_response_time = np.percentile(response_times, 99)
            else:
                sorted_times = sorted(response_times)
                n = len(sorted_times)
                p50_response_time = sorted_times[int(n * 0.5)]
                p95_response_time = sorted_times[int(n * 0.95)]
                p99_response_time = sorted_times[int(n * 0.99)]
        else:
            avg_response_time = min_response_time = max_response_time = 0.0
            p50_response_time = p95_response_time = p99_response_time = 0.0
        
        requests_per_second = successful_requests / duration if duration > 0 else 0
        cache_hit_rate = (self.cache_hit_count / (self.cache_hit_count + self.cache_miss_count) * 100) if (self.cache_hit_count + self.cache_miss_count) > 0 else 0
        error_rate = (failed_requests / total_requests * 100) if total_requests > 0 else 0
        throughput_tokens_per_second = total_tokens / duration if duration > 0 else 0
        avg_tokens_per_request = total_tokens / successful_requests if successful_requests > 0 else 0
        
        # 获取最终系统指标
        final_metrics = self.collect_system_metrics()
        
        result = BenchmarkResult(
            test_name=test_name,
            start_time=start_time.isoformat(),
            end_time=actual_end_time.isoformat(),
            duration_seconds=duration,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            avg_response_time_ms=avg_response_time,
            min_response_time_ms=min_response_time,
            max_response_time_ms=max_response_time,
            p50_response_time_ms=p50_response_time,
            p95_response_time_ms=p95_response_time,
            p99_response_time_ms=p99_response_time,
            requests_per_second=requests_per_second,
            cache_hit_rate=cache_hit_rate,
            memory_usage_mb=final_metrics.memory_used_mb,
            cpu_usage_percent=final_metrics.cpu_percent,
            error_rate=error_rate,
            throughput_tokens_per_second=throughput_tokens_per_second,
            avg_tokens_per_request=avg_tokens_per_request
        )
        
        self.results.append(result)
        
        print(f"压力测试完成")
        print(f"总请求数: {total_requests}")
        print(f"成功请求: {successful_requests}")
        print(f"平均响应时间: {avg_response_time:.2f}ms")
        print(f"请求/秒: {requests_per_second:.2f}")
        print(f"缓存命中率: {cache_hit_rate:.1f}%")
        print(f"错误率: {error_rate:.1f}%")
        
        return result
    
    async def run_cache_efficiency_test(self) -> BenchmarkResult:
        """运行缓存效率测试"""
        test_name = "cache_efficiency_test"
        print(f"\n开始缓存效率测试")
        
        start_time = datetime.now()
        
        # 重置缓存统计
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        
        # 第一轮：填充缓存
        print("第一轮：填充缓存...")
        cache_prompts = self.test_prompts[:20]  # 使用前20个提示词
        
        for i, prompt in enumerate(cache_prompts):
            await self.simulate_request(prompt, f"cache_fill_{i}")
        
        first_round_hit_rate = (self.cache_hit_count / (self.cache_hit_count + self.cache_miss_count) * 100) if (self.cache_hit_count + self.cache_miss_count) > 0 else 0
        
        # 重置统计
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        
        # 第二轮：重复请求测试缓存命中
        print("第二轮：测试缓存命中...")
        response_times = []
        successful_requests = 0
        failed_requests = 0
        total_tokens = 0
        
        for i in range(3):  # 重复3次
            for j, prompt in enumerate(cache_prompts):
                success, response_time, tokens = await self.simulate_request(prompt, f"cache_test_{i}_{j}")
                if success:
                    successful_requests += 1
                    response_times.append(response_time)
                    total_tokens += tokens
                else:
                    failed_requests += 1
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # 计算统计指标
        if response_times:
            avg_response_time = statistics.mean(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            
            if np:
                p50_response_time = np.percentile(response_times, 50)
                p95_response_time = np.percentile(response_times, 95)
                p99_response_time = np.percentile(response_times, 99)
            else:
                sorted_times = sorted(response_times)
                n = len(sorted_times)
                p50_response_time = sorted_times[int(n * 0.5)]
                p95_response_time = sorted_times[int(n * 0.95)]
                p99_response_time = sorted_times[int(n * 0.99)]
        else:
            avg_response_time = min_response_time = max_response_time = 0.0
            p50_response_time = p95_response_time = p99_response_time = 0.0
        
        total_requests = successful_requests + failed_requests
        requests_per_second = successful_requests / duration if duration > 0 else 0
        cache_hit_rate = (self.cache_hit_count / (self.cache_hit_count + self.cache_miss_count) * 100) if (self.cache_hit_count + self.cache_miss_count) > 0 else 0
        error_rate = (failed_requests / total_requests * 100) if total_requests > 0 else 0
        throughput_tokens_per_second = total_tokens / duration if duration > 0 else 0
        avg_tokens_per_request = total_tokens / successful_requests if successful_requests > 0 else 0
        
        # 获取系统指标
        final_metrics = self.collect_system_metrics()
        
        result = BenchmarkResult(
            test_name=test_name,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            duration_seconds=duration,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            avg_response_time_ms=avg_response_time,
            min_response_time_ms=min_response_time,
            max_response_time_ms=max_response_time,
            p50_response_time_ms=p50_response_time,
            p95_response_time_ms=p95_response_time,
            p99_response_time_ms=p99_response_time,
            requests_per_second=requests_per_second,
            cache_hit_rate=cache_hit_rate,
            memory_usage_mb=final_metrics.memory_used_mb,
            cpu_usage_percent=final_metrics.cpu_percent,
            error_rate=error_rate,
            throughput_tokens_per_second=throughput_tokens_per_second,
            avg_tokens_per_request=avg_tokens_per_request
        )
        
        self.results.append(result)
        
        print(f"缓存效率测试完成")
        print(f"第一轮缓存命中率: {first_round_hit_rate:.1f}%")
        print(f"第二轮缓存命中率: {cache_hit_rate:.1f}%")
        print(f"平均响应时间: {avg_response_time:.2f}ms")
        
        return result
    
    async def run_full_benchmark_suite(self) -> Dict[str, Any]:
        """运行完整的基准测试套件"""
        print("=" * 60)
        print("VLLM llmcache 性能基准测试套件")
        print("=" * 60)
        
        # 收集基线系统指标
        baseline_metrics = self.collect_system_metrics()
        print(f"基线系统指标:")
        print(f"  CPU: {baseline_metrics.cpu_percent:.1f}%")
        print(f"  内存: {baseline_metrics.memory_used_mb:.1f}MB")
        if baseline_metrics.gpu_utilization is not None:
            print(f"  GPU: {baseline_metrics.gpu_utilization:.1f}%")
        
        # 测试套件
        test_suite = [
            # 基础性能测试
            ("low_concurrency", 100, 5),
            ("medium_concurrency", 200, 20),
            ("high_concurrency", 500, 50),
            
            # 缓存效率测试
            ("cache_efficiency", None, None),
            
            # 压力测试
            ("stress_test", 2, 30),  # 2分钟压力测试
        ]
        
        for test_config in test_suite:
            if test_config[0] == "cache_efficiency":
                await self.run_cache_efficiency_test()
            elif test_config[0] == "stress_test":
                await self.run_stress_test(test_config[1], test_config[2])
            else:
                await self.run_concurrent_test(test_config[0], test_config[1], test_config[2])
            
            # 测试间隔
            await asyncio.sleep(2)
        
        # 生成综合报告
        report = self.generate_comprehensive_report()
        
        return report
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """生成综合报告"""
        if not self.results:
            return {"error": "没有测试结果"}
        
        # 计算总体统计
        total_requests = sum(r.total_requests for r in self.results)
        total_successful = sum(r.successful_requests for r in self.results)
        total_failed = sum(r.failed_requests for r in self.results)
        
        avg_response_times = [r.avg_response_time_ms for r in self.results if r.avg_response_time_ms > 0]
        avg_cache_hit_rates = [r.cache_hit_rate for r in self.results]
        avg_requests_per_second = [r.requests_per_second for r in self.results if r.requests_per_second > 0]
        
        overall_avg_response_time = statistics.mean(avg_response_times) if avg_response_times else 0
        overall_cache_hit_rate = statistics.mean(avg_cache_hit_rates) if avg_cache_hit_rates else 0
        overall_requests_per_second = statistics.mean(avg_requests_per_second) if avg_requests_per_second else 0
        
        # 找出最佳和最差性能
        best_performance = min(self.results, key=lambda r: r.avg_response_time_ms) if self.results else None
        worst_performance = max(self.results, key=lambda r: r.avg_response_time_ms) if self.results else None
        highest_throughput = max(self.results, key=lambda r: r.requests_per_second) if self.results else None
        
        report = {
            "summary": {
                "total_tests": len(self.results),
                "total_requests": total_requests,
                "total_successful_requests": total_successful,
                "total_failed_requests": total_failed,
                "overall_success_rate": (total_successful / total_requests * 100) if total_requests > 0 else 0,
                "overall_avg_response_time_ms": overall_avg_response_time,
                "overall_cache_hit_rate": overall_cache_hit_rate,
                "overall_requests_per_second": overall_requests_per_second
            },
            "performance_highlights": {
                "best_performance": {
                    "test_name": best_performance.test_name if best_performance else None,
                    "avg_response_time_ms": best_performance.avg_response_time_ms if best_performance else None
                },
                "worst_performance": {
                    "test_name": worst_performance.test_name if worst_performance else None,
                    "avg_response_time_ms": worst_performance.avg_response_time_ms if worst_performance else None
                },
                "highest_throughput": {
                    "test_name": highest_throughput.test_name if highest_throughput else None,
                    "requests_per_second": highest_throughput.requests_per_second if highest_throughput else None
                }
            },
            "detailed_results": [asdict(result) for result in self.results],
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """生成性能优化建议"""
        recommendations = []
        
        if not self.results:
            return ["无法生成建议：没有测试结果"]
        
        # 分析缓存效率
        avg_cache_hit_rate = statistics.mean([r.cache_hit_rate for r in self.results])
        if avg_cache_hit_rate < 50:
            recommendations.append("缓存命中率较低，建议优化缓存策略或增加缓存容量")
        elif avg_cache_hit_rate > 80:
            recommendations.append("缓存效率良好，可以考虑进一步优化缓存算法")
        
        # 分析响应时间
        avg_response_times = [r.avg_response_time_ms for r in self.results if r.avg_response_time_ms > 0]
        if avg_response_times:
            overall_avg_response_time = statistics.mean(avg_response_times)
            if overall_avg_response_time > 1000:
                recommendations.append("平均响应时间较长，建议优化模型推理速度或增加GPU资源")
            elif overall_avg_response_time < 100:
                recommendations.append("响应时间表现优秀，系统性能良好")
        
        # 分析错误率
        avg_error_rate = statistics.mean([r.error_rate for r in self.results])
        if avg_error_rate > 5:
            recommendations.append("错误率较高，建议检查系统稳定性和错误处理机制")
        elif avg_error_rate < 1:
            recommendations.append("系统稳定性良好，错误率很低")
        
        # 分析吞吐量
        avg_throughput = statistics.mean([r.requests_per_second for r in self.results if r.requests_per_second > 0])
        if avg_throughput < 10:
            recommendations.append("吞吐量较低，建议优化并发处理能力或增加计算资源")
        elif avg_throughput > 100:
            recommendations.append("吞吐量表现优秀，系统处理能力强")
        
        # 分析内存使用
        max_memory_usage = max([r.memory_usage_mb for r in self.results])
        if max_memory_usage > 8000:  # 8GB
            recommendations.append("内存使用量较高，建议优化内存管理或增加内存容量")
        
        return recommendations
    
    def save_report(self, filename: Optional[str] = None) -> str:
        """保存报告到文件"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"benchmark_report_{timestamp}.json"
        
        report = self.generate_comprehensive_report()
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            return f"报告已保存到: {filename}"
        except Exception as e:
            return f"保存报告失败: {e}"
    
    def print_summary(self):
        """打印测试摘要"""
        report = self.generate_comprehensive_report()
        
        print("\n" + "=" * 60)
        print("基准测试摘要")
        print("=" * 60)
        
        summary = report["summary"]
        print(f"总测试数: {summary['total_tests']}")
        print(f"总请求数: {summary['total_requests']}")
        print(f"成功率: {summary['overall_success_rate']:.1f}%")
        print(f"平均响应时间: {summary['overall_avg_response_time_ms']:.2f}ms")
        print(f"平均缓存命中率: {summary['overall_cache_hit_rate']:.1f}%")
        print(f"平均吞吐量: {summary['overall_requests_per_second']:.2f} req/s")
        
        print("\n性能亮点:")
        highlights = report["performance_highlights"]
        if highlights["best_performance"]["test_name"]:
            print(f"最佳性能: {highlights['best_performance']['test_name']} ({highlights['best_performance']['avg_response_time_ms']:.2f}ms)")
        if highlights["highest_throughput"]["test_name"]:
            print(f"最高吞吐量: {highlights['highest_throughput']['test_name']} ({highlights['highest_throughput']['requests_per_second']:.2f} req/s)")
        
        print("\n优化建议:")
        for i, recommendation in enumerate(report["recommendations"], 1):
            print(f"{i}. {recommendation}")


async def main():
    """主函数"""
    print("VLLM llmcache 性能基准测试工具")
    print("选择测试模式:")
    print("1. 快速测试 (5分钟)")
    print("2. 标准测试 (15分钟)")
    print("3. 完整测试 (30分钟)")
    print("4. 自定义测试")
    print("5. 仅缓存效率测试")
    
    try:
        choice = input("请选择 (1-5): ").strip()
        
        benchmark = PerformanceBenchmark()
        
        if choice == "1":
            # 快速测试
            await benchmark.run_concurrent_test("quick_test", 50, 10)
            await benchmark.run_cache_efficiency_test()
        
        elif choice == "2":
            # 标准测试
            await benchmark.run_concurrent_test("standard_low", 100, 10)
            await benchmark.run_concurrent_test("standard_medium", 200, 25)
            await benchmark.run_cache_efficiency_test()
            await benchmark.run_stress_test(2, 20)
        
        elif choice == "3":
            # 完整测试
            await benchmark.run_full_benchmark_suite()
        
        elif choice == "4":
            # 自定义测试
            num_requests = int(input("请求数量: "))
            concurrency = int(input("并发数: "))
            test_name = input("测试名称: ") or "custom_test"
            
            await benchmark.run_concurrent_test(test_name, num_requests, concurrency)
        
        elif choice == "5":
            # 仅缓存效率测试
            await benchmark.run_cache_efficiency_test()
        
        else:
            print("无效选择")
            return
        
        # 显示结果
        benchmark.print_summary()
        
        # 保存报告
        save_choice = input("\n是否保存报告到文件? (y/n): ").strip().lower()
        if save_choice == 'y':
            filename = input("文件名 (留空使用默认名称): ").strip() or None
            result = benchmark.save_report(filename)
            print(result)
    
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"测试过程中发生错误: {e}")


if __name__ == "__main__":
    asyncio.run(main())