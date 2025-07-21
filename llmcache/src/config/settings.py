"""核心配置设置"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union
from vllm import EngineArgs
from vllm.config import VllmConfig


@dataclass
class SharedMemoryConfig:
    """共享内存配置"""
    enabled: bool = True
    memory_size: int = 1024 * 1024 * 100  # 100MB
    max_entries: int = 10000
    cleanup_interval: int = 300  # 5分钟
    enable_compression: bool = True
    shared_name: str = "vllm_cache_shared_memory"


@dataclass
class CacheConfig:
    """缓存配置"""
    max_memory_entries: int = 1000
    memory_ttl: int = 3600  # 1小时
    redis_ttl: int = 86400  # 24小时
    enable_redis: bool = True
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_prefix: str = "vllm_cache:"


@dataclass
class ModelConfig:
    """模型配置"""
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: Optional[int] = None
    trust_remote_code: bool = False


@dataclass
class VectorSearchConfig:
    """向量搜索配置"""
    enabled: bool = True
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    similarity_threshold: float = 0.7
    max_results: int = 10
    use_gpu: bool = True


@dataclass
class NLPConfig:
    """NLP处理配置"""
    enabled: bool = True
    max_concurrent_tasks: int = 10
    spacy_model: str = "en_core_web_sm"
    enable_typo_correction: bool = True


@dataclass
class VLLMConfig:
    """VLLM主配置"""
    shared_memory: SharedMemoryConfig = field(default_factory=SharedMemoryConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    vector_search: VectorSearchConfig = field(default_factory=VectorSearchConfig)
    nlp: NLPConfig = field(default_factory=NLPConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'shared_memory': self.shared_memory.__dict__,
            'cache': self.cache.__dict__,
            'model': self.model.__dict__,
            'vector_search': self.vector_search.__dict__,
            'nlp': self.nlp.__dict__
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VLLMConfig':
        """从字典创建配置"""
        return cls(
            shared_memory=SharedMemoryConfig(**data.get('shared_memory', {})),
            cache=CacheConfig(**data.get('cache', {})),
            model=ModelConfig(**data.get('model', {})),
            vector_search=VectorSearchConfig(**data.get('vector_search', {})),
            nlp=NLPConfig(**data.get('nlp', {}))
        )


@dataclass
class EnhancedVLLMConfig:
    """增强的 VLLM 配置，适配 enhanced_async_llm.py"""
    
    # 核心配置组件
    shared_memory: SharedMemoryConfig = field(default_factory=SharedMemoryConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    vector_search: VectorSearchConfig = field(default_factory=VectorSearchConfig)
    nlp: NLPConfig = field(default_factory=NLPConfig)
    
    # VLLM 引擎参数
    engine_args: Optional[EngineArgs] = None
    
    # 新增：投机解码配置
    speculative_config: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """初始化后处理"""
        if self.engine_args is None:
            self.engine_args = self._create_default_engine_args()
    
    def _create_default_engine_args(self) -> EngineArgs:
        """创建默认的引擎参数"""
        engine_args = EngineArgs(
            model=self.model.model_name,
            tensor_parallel_size=self.model.tensor_parallel_size,
            gpu_memory_utilization=self.model.gpu_memory_utilization,
            max_model_len=self.model.max_model_len,
            trust_remote_code=self.model.trust_remote_code
        )
        
        # 添加投机解码配置
        if self.speculative_config:
            engine_args.speculative_config = self.speculative_config
            
        return engine_args
    
    def update_engine_args(self, **kwargs) -> None:
        """更新引擎参数"""
        if self.engine_args is None:
            self.engine_args = self._create_default_engine_args()
        
        # 更新引擎参数
        for key, value in kwargs.items():
            if hasattr(self.engine_args, key):
                setattr(self.engine_args, key, value)
        
        # 特殊处理 speculative_config
        if 'speculative_config' in kwargs:
            self.speculative_config = kwargs['speculative_config']
            self.engine_args.speculative_config = kwargs['speculative_config']
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            'shared_memory': self.shared_memory.__dict__,
            'cache': self.cache.__dict__,
            'model': self.model.__dict__,
            'vector_search': self.vector_search.__dict__,
            'nlp': self.nlp.__dict__
        }
        
        # 添加引擎参数
        if self.engine_args:
            result['engine_args'] = {
                'model': self.engine_args.model,
                'tensor_parallel_size': self.engine_args.tensor_parallel_size,
                'gpu_memory_utilization': self.engine_args.gpu_memory_utilization,
                'max_model_len': self.engine_args.max_model_len,
                'trust_remote_code': self.engine_args.trust_remote_code
            }
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnhancedVLLMConfig':
        """从字典创建配置"""
        config = cls(
            shared_memory=SharedMemoryConfig(**data.get('shared_memory', {})),
            cache=CacheConfig(**data.get('cache', {})),
            model=ModelConfig(**data.get('model', {})),
            vector_search=VectorSearchConfig(**data.get('vector_search', {})),
            nlp=NLPConfig(**data.get('nlp', {}))
        )
        
        # 处理引擎参数
        engine_args_data = data.get('engine_args', {})
        if engine_args_data:
            config.engine_args = EngineArgs(**engine_args_data)
        
        return config
    
    @classmethod
    def from_vllm_config(cls, vllm_config: 'VLLMConfig') -> 'EnhancedVLLMConfig':
        """从标准 VLLMConfig 创建增强配置"""
        return cls(
            shared_memory=vllm_config.shared_memory,
            cache=vllm_config.cache,
            model=vllm_config.model,
            vector_search=vllm_config.vector_search,
            nlp=vllm_config.nlp
        )
    
    def to_vllm_config(self) -> 'VLLMConfig':
        """转换为标准 VLLMConfig"""
        return VLLMConfig(
            shared_memory=self.shared_memory,
            cache=self.cache,
            model=self.model,
            vector_search=self.vector_search,
            nlp=self.nlp
        )


# 全局函数，用于scripts中的导入
def get_default_config() -> VLLMConfig:
    """获取默认配置"""
    return VLLMConfig()


def create_benchmark_engine_args(**kwargs) -> EngineArgs:
    """创建基准测试引擎参数"""
    default_config = get_default_config()
    engine_args = EngineArgs(
        model=default_config.model.model_name,
        tensor_parallel_size=default_config.model.tensor_parallel_size,
        gpu_memory_utilization=default_config.model.gpu_memory_utilization,
        max_model_len=default_config.model.max_model_len,
        trust_remote_code=default_config.model.trust_remote_code
    )
    
    # 应用自定义参数
    for key, value in kwargs.items():
        if hasattr(engine_args, key):
            setattr(engine_args, key, value)
    
    return engine_args
