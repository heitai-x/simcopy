#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统监控脚本

实时监控VLLM系统的运行状态、性能指标和资源使用情况。
"""

import os
import sys
import time
import json
import asyncio
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import deque

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import psutil
    import GPUtil
except ImportError:
    print("警告: psutil 或 GPUtil 未安装，某些监控功能可能不可用")
    psutil = None
    GPUtil = None

try:
    import redis
except ImportError:
    redis = None

try:
    import torch
except ImportError:
    torch = None


@dataclass
class SystemMetrics:
    """系统指标数据类"""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_percent: float
    disk_free_gb: float
    gpu_utilization: Optional[float] = None
    gpu_memory_percent: Optional[float] = None
    gpu_memory_used_gb: Optional[float] = None
    gpu_temperature: Optional[float] = None
    network_sent_mb: Optional[float] = None
    network_recv_mb: Optional[float] = None
    process_count: Optional[int] = None
    load_average: Optional[List[float]] = None


@dataclass
class VLLMMetrics:
    """VLLM相关指标"""
    timestamp: str
    active_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    tokens_per_second: float = 0.0
    cache_hit_rate: float = 0.0
    shared_memory_usage: float = 0.0
    redis_connections: int = 0
    redis_memory_usage: Optional[float] = None


class SystemMonitor:
    """系统监控器"""
    
    def __init__(self, history_size: int = 100, update_interval: float = 5.0):
        self.history_size = history_size
        self.update_interval = update_interval
        self.system_history = deque(maxlen=history_size)
        self.vllm_history = deque(maxlen=history_size)
        self.is_running = False
        self.monitor_thread = None
        self.redis_client = None
        
        # 网络统计基线
        self.network_baseline = None
        
        # 初始化Redis连接
        self._init_redis()
    
    def _init_redis(self):
        """初始化Redis连接"""
        if redis:
            try:
                self.redis_client = redis.Redis(
                    host='localhost',
                    port=6379,
                    decode_responses=True,
                    socket_connect_timeout=5
                )
                # 测试连接
                self.redis_client.ping()
                print("✓ Redis连接成功")
            except Exception as e:
                print(f"⚠️ Redis连接失败: {e}")
                self.redis_client = None
        else:
            print("⚠️ Redis库未安装")
    
    def collect_system_metrics(self) -> SystemMetrics:
        """收集系统指标"""
        timestamp = datetime.now().isoformat()
        
        # CPU和内存
        cpu_percent = psutil.cpu_percent(interval=1) if psutil else 0.0
        
        if psutil:
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024**3)
            memory_total_gb = memory.total / (1024**3)
            
            # 磁盘
            disk = psutil.disk_usage('.')
            disk_percent = (disk.used / disk.total) * 100
            disk_free_gb = disk.free / (1024**3)
            
            # 网络
            net_io = psutil.net_io_counters()
            if self.network_baseline is None:
                self.network_baseline = (net_io.bytes_sent, net_io.bytes_recv)
                network_sent_mb = 0.0
                network_recv_mb = 0.0
            else:
                network_sent_mb = (net_io.bytes_sent - self.network_baseline[0]) / (1024**2)
                network_recv_mb = (net_io.bytes_recv - self.network_baseline[1]) / (1024**2)
            
            # 进程数
            process_count = len(psutil.pids())
            
            # 负载平均值 (仅Linux)
            try:
                load_average = list(os.getloadavg())
            except (OSError, AttributeError):
                load_average = None
        else:
            memory_percent = memory_used_gb = memory_total_gb = 0.0
            disk_percent = disk_free_gb = 0.0
            network_sent_mb = network_recv_mb = 0.0
            process_count = 0
            load_average = None
        
        # GPU指标
        gpu_utilization = gpu_memory_percent = gpu_memory_used_gb = gpu_temperature = None
        
        if GPUtil:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # 使用第一个GPU
                    gpu_utilization = gpu.load * 100
                    gpu_memory_percent = gpu.memoryUtil * 100
                    gpu_memory_used_gb = gpu.memoryUsed / 1024
                    gpu_temperature = gpu.temperature
            except Exception as e:
                print(f"GPU监控错误: {e}")
        
        return SystemMetrics(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_gb=memory_used_gb,
            memory_total_gb=memory_total_gb,
            disk_percent=disk_percent,
            disk_free_gb=disk_free_gb,
            gpu_utilization=gpu_utilization,
            gpu_memory_percent=gpu_memory_percent,
            gpu_memory_used_gb=gpu_memory_used_gb,
            gpu_temperature=gpu_temperature,
            network_sent_mb=network_sent_mb,
            network_recv_mb=network_recv_mb,
            process_count=process_count,
            load_average=load_average
        )
    
    def collect_vllm_metrics(self) -> VLLMMetrics:
        """收集VLLM相关指标"""
        timestamp = datetime.now().isoformat()
        
        # Redis指标
        redis_connections = 0
        redis_memory_usage = None
        cache_hit_rate = 0.0
        
        if self.redis_client:
            try:
                info = self.redis_client.info()
                redis_connections = info.get('connected_clients', 0)
                redis_memory_usage = info.get('used_memory', 0) / (1024**2)  # MB
                
                # 缓存命中率 (如果有统计信息)
                keyspace_hits = info.get('keyspace_hits', 0)
                keyspace_misses = info.get('keyspace_misses', 0)
                total_requests = keyspace_hits + keyspace_misses
                if total_requests > 0:
                    cache_hit_rate = (keyspace_hits / total_requests) * 100
                    
            except Exception as e:
                print(f"Redis指标收集错误: {e}")
        
        # TODO: 从实际的VLLM handler收集指标
        # 这里使用模拟数据，实际应该从handler获取
        active_requests = 0
        completed_requests = 0
        failed_requests = 0
        avg_response_time = 0.0
        tokens_per_second = 0.0
        shared_memory_usage = 0.0
        
        return VLLMMetrics(
            timestamp=timestamp,
            active_requests=active_requests,
            completed_requests=completed_requests,
            failed_requests=failed_requests,
            avg_response_time=avg_response_time,
            tokens_per_second=tokens_per_second,
            cache_hit_rate=cache_hit_rate,
            shared_memory_usage=shared_memory_usage,
            redis_connections=redis_connections,
            redis_memory_usage=redis_memory_usage
        )
    
    def _monitor_loop(self):
        """监控循环"""
        while self.is_running:
            try:
                # 收集系统指标
                system_metrics = self.collect_system_metrics()
                self.system_history.append(system_metrics)
                
                # 收集VLLM指标
                vllm_metrics = self.collect_vllm_metrics()
                self.vllm_history.append(vllm_metrics)
                
                # 检查警告条件
                self._check_alerts(system_metrics, vllm_metrics)
                
            except Exception as e:
                print(f"监控循环错误: {e}")
            
            time.sleep(self.update_interval)
    
    def _check_alerts(self, system_metrics: SystemMetrics, vllm_metrics: VLLMMetrics):
        """检查警告条件"""
        alerts = []
        
        # 系统资源警告
        if system_metrics.cpu_percent > 90:
            alerts.append(f"CPU使用率过高: {system_metrics.cpu_percent:.1f}%")
        
        if system_metrics.memory_percent > 90:
            alerts.append(f"内存使用率过高: {system_metrics.memory_percent:.1f}%")
        
        if system_metrics.disk_percent > 90:
            alerts.append(f"磁盘使用率过高: {system_metrics.disk_percent:.1f}%")
        
        if system_metrics.gpu_utilization and system_metrics.gpu_utilization > 95:
            alerts.append(f"GPU使用率过高: {system_metrics.gpu_utilization:.1f}%")
        
        if system_metrics.gpu_temperature and system_metrics.gpu_temperature > 80:
            alerts.append(f"GPU温度过高: {system_metrics.gpu_temperature:.1f}°C")
        
        # VLLM相关警告
        if vllm_metrics.cache_hit_rate < 50 and len(self.vllm_history) > 10:
            alerts.append(f"缓存命中率过低: {vllm_metrics.cache_hit_rate:.1f}%")
        
        # 输出警告
        for alert in alerts:
            print(f"⚠️ 警告: {alert}")
    
    def start(self):
        """启动监控"""
        if self.is_running:
            print("监控已在运行")
            return
        
        print("启动系统监控...")
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("✓ 监控已启动")
    
    def stop(self):
        """停止监控"""
        if not self.is_running:
            print("监控未运行")
            return
        
        print("停止系统监控...")
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        print("✓ 监控已停止")
    
    def get_current_status(self) -> Dict[str, Any]:
        """获取当前状态"""
        if not self.system_history or not self.vllm_history:
            return {"error": "暂无监控数据"}
        
        latest_system = self.system_history[-1]
        latest_vllm = self.vllm_history[-1]
        
        return {
            "system": asdict(latest_system),
            "vllm": asdict(latest_vllm),
            "monitoring": {
                "is_running": self.is_running,
                "history_size": len(self.system_history),
                "update_interval": self.update_interval
            }
        }
    
    def get_history(self, minutes: int = 10) -> Dict[str, List[Dict]]:
        """获取历史数据"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        system_data = []
        vllm_data = []
        
        for metrics in self.system_history:
            if datetime.fromisoformat(metrics.timestamp) >= cutoff_time:
                system_data.append(asdict(metrics))
        
        for metrics in self.vllm_history:
            if datetime.fromisoformat(metrics.timestamp) >= cutoff_time:
                vllm_data.append(asdict(metrics))
        
        return {
            "system": system_data,
            "vllm": vllm_data
        }
    
    def generate_report(self) -> str:
        """生成监控报告"""
        if not self.system_history or not self.vllm_history:
            return "暂无监控数据"
        
        latest_system = self.system_history[-1]
        latest_vllm = self.vllm_history[-1]
        
        # 计算平均值
        recent_system = list(self.system_history)[-10:]  # 最近10个数据点
        avg_cpu = sum(m.cpu_percent for m in recent_system) / len(recent_system)
        avg_memory = sum(m.memory_percent for m in recent_system) / len(recent_system)
        
        report = f"""
=== VLLM 系统监控报告 ===
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

--- 系统资源 ---
CPU使用率: {latest_system.cpu_percent:.1f}% (平均: {avg_cpu:.1f}%)
内存使用率: {latest_system.memory_percent:.1f}% ({latest_system.memory_used_gb:.1f}GB / {latest_system.memory_total_gb:.1f}GB)
磁盘使用率: {latest_system.disk_percent:.1f}% (可用: {latest_system.disk_free_gb:.1f}GB)
进程数: {latest_system.process_count or 'N/A'}
"""
        
        if latest_system.gpu_utilization is not None:
            report += f"""
--- GPU状态 ---
GPU使用率: {latest_system.gpu_utilization:.1f}%
GPU内存: {latest_system.gpu_memory_percent:.1f}% ({latest_system.gpu_memory_used_gb:.1f}GB)
GPU温度: {latest_system.gpu_temperature:.1f}°C
"""
        
        if latest_system.load_average:
            report += f"""
--- 系统负载 ---
负载平均值: {latest_system.load_average[0]:.2f}, {latest_system.load_average[1]:.2f}, {latest_system.load_average[2]:.2f}
"""
        
        report += f"""
--- VLLM状态 ---
活跃请求: {latest_vllm.active_requests}
完成请求: {latest_vllm.completed_requests}
失败请求: {latest_vllm.failed_requests}
平均响应时间: {latest_vllm.avg_response_time:.2f}ms
令牌/秒: {latest_vllm.tokens_per_second:.1f}
缓存命中率: {latest_vllm.cache_hit_rate:.1f}%
"""
        
        if latest_vllm.redis_memory_usage is not None:
            report += f"""
--- Redis状态 ---
连接数: {latest_vllm.redis_connections}
内存使用: {latest_vllm.redis_memory_usage:.1f}MB
"""
        
        report += f"""
--- 监控状态 ---
运行状态: {'运行中' if self.is_running else '已停止'}
数据点数: {len(self.system_history)}
更新间隔: {self.update_interval}秒
"""
        
        return report
    
    def save_report(self, filename: Optional[str] = None) -> str:
        """保存监控报告到文件"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"monitor_report_{timestamp}.txt"
        
        report = self.generate_report()
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report)
            return f"报告已保存到: {filename}"
        except Exception as e:
            return f"保存报告失败: {e}"


def main():
    """主函数 - 交互式监控"""
    monitor = SystemMonitor()
    
    print("VLLM 系统监控工具")
    print("可用命令:")
    print("  start    - 启动监控")
    print("  stop     - 停止监控")
    print("  status   - 显示当前状态")
    print("  report   - 生成监控报告")
    print("  save     - 保存报告到文件")
    print("  history  - 显示历史数据")
    print("  quit     - 退出")
    print()
    
    try:
        while True:
            command = input("monitor> ").strip().lower()
            
            if command == "start":
                monitor.start()
            
            elif command == "stop":
                monitor.stop()
            
            elif command == "status":
                status = monitor.get_current_status()
                print(json.dumps(status, indent=2, ensure_ascii=False))
            
            elif command == "report":
                print(monitor.generate_report())
            
            elif command == "save":
                result = monitor.save_report()
                print(result)
            
            elif command == "history":
                minutes = input("输入历史数据分钟数 (默认10): ").strip()
                try:
                    minutes = int(minutes) if minutes else 10
                except ValueError:
                    minutes = 10
                
                history = monitor.get_history(minutes)
                print(f"最近{minutes}分钟的数据:")
                print(json.dumps(history, indent=2, ensure_ascii=False))
            
            elif command in ["quit", "exit", "q"]:
                break
            
            elif command == "help":
                print("可用命令: start, stop, status, report, save, history, quit")
            
            else:
                print(f"未知命令: {command}，输入 'help' 查看帮助")
    
    except KeyboardInterrupt:
        print("\n正在退出...")
    
    finally:
        monitor.stop()
        print("监控已停止")


if __name__ == "__main__":
    main()