#!/usr/bin/env python3
"""GPU工具模块 - 提供GPU检测、配置和优化功能"""

import torch
import os
from typing import Dict, List, Optional, Tuple
from loguru import logger


class GPUManager:
    """GPU管理器 - 负责GPU检测、配置和优化"""
    
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        self.gpu_count = torch.cuda.device_count() if self.gpu_available else 0
        self.current_device = None
        self.device_info = {}
        
        if self.gpu_available:
            self._initialize_gpu_info()
    
    def _initialize_gpu_info(self):
        """初始化GPU信息"""
        try:
            for i in range(self.gpu_count):
                device_props = torch.cuda.get_device_properties(i)
                self.device_info[i] = {
                    'name': device_props.name,
                    'total_memory': device_props.total_memory,
                    'major': device_props.major,
                    'minor': device_props.minor,
                    'multi_processor_count': device_props.multi_processor_count
                }
            
            self.current_device = torch.cuda.current_device()
            logger.info(f"GPU初始化完成，当前设备: {self.current_device}")
            
        except Exception as e:
            logger.error(f"GPU信息初始化失败: {e}")
    
    def get_gpu_status(self) -> Dict:
        """获取GPU状态信息"""
        status = {
            'gpu_available': self.gpu_available,
            'gpu_count': self.gpu_count,
            'current_device': self.current_device,
            'pytorch_version': torch.__version__,
            'cuda_version': torch.version.cuda if self.gpu_available else None,
            'devices': []
        }
        
        if self.gpu_available:
            for i in range(self.gpu_count):
                try:
                    memory_allocated = torch.cuda.memory_allocated(i)
                    memory_reserved = torch.cuda.memory_reserved(i)
                    total_memory = self.device_info[i]['total_memory']
                    
                    device_status = {
                        'device_id': i,
                        'name': self.device_info[i]['name'],
                        'compute_capability': f"{self.device_info[i]['major']}.{self.device_info[i]['minor']}",
                        'total_memory_gb': total_memory / (1024**3),
                        'allocated_memory_gb': memory_allocated / (1024**3),
                        'reserved_memory_gb': memory_reserved / (1024**3),
                        'free_memory_gb': (total_memory - memory_reserved) / (1024**3),
                        'utilization_percent': (memory_allocated / total_memory) * 100,
                        'multi_processor_count': self.device_info[i]['multi_processor_count']
                    }
                    status['devices'].append(device_status)
                    
                except Exception as e:
                    logger.warning(f"获取设备 {i} 状态失败: {e}")
        
        return status
    
    def print_gpu_status(self):
        """打印GPU状态信息"""
        status = self.get_gpu_status()
        
        print("\n" + "="*60)
        print("🔧 GPU 状态检测")
        print("="*60)
        print(f"PyTorch版本: {status['pytorch_version']}")
        print(f"CUDA可用: {status['gpu_available']}")
        
        if status['gpu_available']:
            print(f"CUDA版本: {status['cuda_version']}")
            print(f"GPU数量: {status['gpu_count']}")
            print(f"当前设备: {status['current_device']}")
            
            for device in status['devices']:
                print(f"\n📱 设备 {device['device_id']}: {device['name']}")
                print(f"   计算能力: {device['compute_capability']}")
                print(f"   总内存: {device['total_memory_gb']:.2f} GB")
                print(f"   已分配: {device['allocated_memory_gb']:.2f} GB")
                print(f"   已保留: {device['reserved_memory_gb']:.2f} GB")
                print(f"   可用内存: {device['free_memory_gb']:.2f} GB")
                print(f"   利用率: {device['utilization_percent']:.1f}%")
                print(f"   多处理器数量: {device['multi_processor_count']}")
        else:
            print("❌ 未检测到可用的GPU")
        
        print("="*60)
    
    def get_optimal_device(self) -> Optional[int]:
        """获取最优的GPU设备ID"""
        if not self.gpu_available:
            return None
        
        status = self.get_gpu_status()
        best_device = None
        max_free_memory = 0
        
        for device in status['devices']:
            if device['free_memory_gb'] > max_free_memory:
                max_free_memory = device['free_memory_gb']
                best_device = device['device_id']
        
        return best_device
    
    def set_device(self, device_id: int) -> bool:
        """设置当前GPU设备"""
        if not self.gpu_available or device_id >= self.gpu_count:
            return False
        
        try:
            torch.cuda.set_device(device_id)
            self.current_device = device_id
            logger.info(f"已切换到GPU设备 {device_id}")
            return True
        except Exception as e:
            logger.error(f"切换GPU设备失败: {e}")
            return False
    
    def optimize_for_benepar(self) -> Dict[str, str]:
        """为benepar模型优化GPU配置"""
        recommendations = {
            'status': 'success',
            'recommendations': [],
            'warnings': []
        }
        
        if not self.gpu_available:
            recommendations['status'] = 'warning'
            recommendations['warnings'].append(
                "未检测到GPU，benepar_en3_large在CPU上运行会比标准版本慢约3倍"
            )
            recommendations['recommendations'].extend([
                "考虑使用benepar_en3标准版本以提高CPU性能",
                "如果可能，请配置GPU环境以获得最佳性能"
            ])
            return recommendations
        
        status = self.get_gpu_status()
        best_device = self.get_optimal_device()
        
        if best_device is not None:
            device_info = status['devices'][best_device]
            
            # 检查内存是否足够
            if device_info['free_memory_gb'] < 2.0:
                recommendations['warnings'].append(
                    f"GPU {best_device} 可用内存较少 ({device_info['free_memory_gb']:.2f} GB)，可能影响性能"
                )
            
            # 检查计算能力
            compute_capability = float(device_info['compute_capability'])
            if compute_capability < 6.0:
                recommendations['warnings'].append(
                    f"GPU {best_device} 计算能力较低 ({device_info['compute_capability']})，建议使用更新的GPU"
                )
            
            recommendations['recommendations'].extend([
                f"建议使用GPU {best_device}: {device_info['name']}",
                f"可用内存: {device_info['free_memory_gb']:.2f} GB",
                "设置环境变量 CUDA_VISIBLE_DEVICES 指定GPU设备",
                "使用批处理方式处理多个文本以提高效率"
            ])
            
            # 设置最优设备
            if self.set_device(best_device):
                recommendations['recommendations'].append(
                    f"已自动切换到最优GPU设备 {best_device}"
                )
        
        return recommendations
    
    def get_batch_size_recommendation(self, model_type: str = "benepar") -> int:
        """根据GPU内存推荐批处理大小"""
        if not self.gpu_available:
            return 8  # CPU默认批处理大小
        
        status = self.get_gpu_status()
        current_device_info = None
        
        for device in status['devices']:
            if device['device_id'] == self.current_device:
                current_device_info = device
                break
        
        if not current_device_info:
            return 16  # 默认值
        
        free_memory_gb = current_device_info['free_memory_gb']
        
        # 根据可用内存推荐批处理大小
        if model_type == "benepar":
            if free_memory_gb >= 8:
                return 64
            elif free_memory_gb >= 4:
                return 32
            elif free_memory_gb >= 2:
                return 16
            else:
                return 8
        
        return 16  # 默认值
    
    def clear_cache(self):
        """清理GPU缓存"""
        if self.gpu_available:
            try:
                torch.cuda.empty_cache()
                logger.info("GPU缓存已清理")
            except Exception as e:
                logger.error(f"清理GPU缓存失败: {e}")


def check_gpu_environment() -> GPUManager:
    """检查GPU环境并返回GPU管理器"""
    gpu_manager = GPUManager()
    gpu_manager.print_gpu_status()
    return gpu_manager


def setup_optimal_gpu_environment() -> Tuple[GPUManager, Dict]:
    """设置最优GPU环境"""
    gpu_manager = GPUManager()
    recommendations = gpu_manager.optimize_for_benepar()
    
    print("\n" + "="*60)
    print("🚀 GPU优化建议")
    print("="*60)
    
    if recommendations['warnings']:
        print("⚠️  警告:")
        for warning in recommendations['warnings']:
            print(f"   • {warning}")
        print()
    
    if recommendations['recommendations']:
        print("💡 建议:")
        for rec in recommendations['recommendations']:
            print(f"   • {rec}")
    
    print("="*60)
    
    return gpu_manager, recommendations


if __name__ == "__main__":
    # 测试GPU工具
    gpu_manager, recommendations = setup_optimal_gpu_environment()
    
    print(f"\n推荐批处理大小: {gpu_manager.get_batch_size_recommendation()}")