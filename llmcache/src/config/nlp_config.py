"""NLP增强功能配置模块"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any


@dataclass
class NLPEnhancedConfig:
    """NLP增强功能配置类"""
    
    # NLP处理配置
    enable_nlp_processing: bool = True
    enable_conjunction_extraction: bool = True
    enable_subsentence_analysis: bool = True
    
    # 连接词提取配置
    max_token_limit: int = 512
    spacy_model_large: str = "en_core_web_sm"
    spacy_model_sent: str = "en_core_web_sm"
    
    # 支持的连接词列表
    conjunctions: tuple = (
        'and', 'or', 'but', 'yet', 'so', 'for', 'nor',
        'however', 
        ',', ';'
    )
    
    # 相似度搜索配置
    enable_enhanced_similarity: bool = True
    original_text_weight: float = 0.7
    subsentence_weight: float = 0.3
    min_subsentence_length: int = 10
    max_subsentences_for_search: int = 3
    
    # 缓存策略配置
    enable_enhanced_hashing: bool = True
    include_conjunction_types: bool = True
    include_subsentence_order: bool = False
    hash_normalization: bool = True
    
    # 性能优化配置
    enable_parallel_processing: bool = False
    max_parallel_workers: int = 2
    
    # 调试和日志配置
    enable_nlp_logging: bool = True
    log_extraction_results: bool = False
    log_similarity_scores: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'enable_nlp_processing': self.enable_nlp_processing,
            'enable_conjunction_extraction': self.enable_conjunction_extraction,
            'enable_subsentence_analysis': self.enable_subsentence_analysis,
            'max_token_limit': self.max_token_limit,
            'spacy_model_large': self.spacy_model_large,
            'spacy_model_sent': self.spacy_model_sent,
            'conjunctions': self.conjunctions,
            'enable_enhanced_similarity': self.enable_enhanced_similarity,
            'original_text_weight': self.original_text_weight,
            'subsentence_weight': self.subsentence_weight,
            'min_subsentence_length': self.min_subsentence_length,
            'max_subsentences_for_search': self.max_subsentences_for_search,
            'enable_enhanced_hashing': self.enable_enhanced_hashing,
            'include_conjunction_types': self.include_conjunction_types,
            'include_subsentence_order': self.include_subsentence_order,
            'hash_normalization': self.hash_normalization,
            'enable_parallel_processing': self.enable_parallel_processing,
            'max_parallel_workers': self.max_parallel_workers,
            'enable_nlp_logging': self.enable_nlp_logging,
            'log_extraction_results': self.log_extraction_results,
            'log_similarity_scores': self.log_similarity_scores
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'NLPEnhancedConfig':
        """从字典创建配置对象"""
        return cls(**config_dict)
    
    def validate(self) -> List[str]:
        """验证配置的有效性"""
        errors = []
        
        if self.max_token_limit <= 0:
            errors.append("max_token_limit必须大于0")
        
        if self.original_text_weight < 0 or self.original_text_weight > 1:
            errors.append("original_text_weight必须在0-1之间")
        
        if self.subsentence_weight < 0 or self.subsentence_weight > 1:
            errors.append("subsentence_weight必须在0-1之间")
        
        if abs(self.original_text_weight + self.subsentence_weight - 1.0) > 0.01:
            errors.append("original_text_weight和subsentence_weight之和应该等于1.0")
        
        if self.min_subsentence_length < 5:
            errors.append("min_subsentence_length不应小于5")
        
        if self.max_subsentences_for_search <= 0:
            errors.append("max_subsentences_for_search必须大于0")
        
        if self.max_parallel_workers <= 0:
            errors.append("max_parallel_workers必须大于0")
        
        return errors
    
    def get_extractor_config(self):
        """获取连接词提取器配置（已弃用 - 配置现在内置在ConjunctionExtractor中）"""
        import warnings
        warnings.warn(
            "get_extractor_config方法已弃用，配置现在内置在ConjunctionExtractor中",
            DeprecationWarning,
            stacklevel=2
        )
        return None


@dataclass
class NLPPerformanceConfig:
    """NLP性能配置类"""
    
    # 模型加载配置
    lazy_model_loading: bool = True
    model_cache_timeout: int = 3600  # 秒
    enable_model_sharing: bool = True
    
    # 批处理配置 (已弃用 - 保留用于向后兼容)
    enable_batch_processing: bool = False  # 已弃用，不再使用
    batch_size: int = 8  # 已弃用，不再使用
    batch_timeout: float = 0.1  # 已弃用，不再使用
    
    # 内存管理配置
    max_memory_usage: int = 2048  # MB
    enable_memory_monitoring: bool = True
    memory_cleanup_threshold: float = 0.8
    
    # GPU配置
    enable_gpu_acceleration: bool = False
    gpu_device_id: int = 0
    gpu_memory_fraction: float = 0.3
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'lazy_model_loading': self.lazy_model_loading,
            'model_cache_timeout': self.model_cache_timeout,
            'enable_model_sharing': self.enable_model_sharing,
            'enable_batch_processing': self.enable_batch_processing,
            'batch_size': self.batch_size,
            'batch_timeout': self.batch_timeout,
            'max_memory_usage': self.max_memory_usage,
            'enable_memory_monitoring': self.enable_memory_monitoring,
            'memory_cleanup_threshold': self.memory_cleanup_threshold,
            'enable_gpu_acceleration': self.enable_gpu_acceleration,
            'gpu_device_id': self.gpu_device_id,
            'gpu_memory_fraction': self.gpu_memory_fraction
        }


# 预定义配置模板
class NLPConfigTemplates:
    """NLP配置模板"""
    
    @staticmethod
    def get_development_config() -> NLPEnhancedConfig:
        """开发环境配置"""
        return NLPEnhancedConfig(
            enable_nlp_processing=True,
            enable_conjunction_extraction=True,
            enable_subsentence_analysis=True,
            enable_enhanced_similarity=True,
            enable_enhanced_hashing=True,
            enable_nlp_logging=True,
            log_extraction_results=True,
            log_similarity_scores=True,
            max_token_limit=256  # 较小的限制用于开发
        )
    
    @staticmethod
    def get_production_config() -> NLPEnhancedConfig:
        """生产环境配置"""
        return NLPEnhancedConfig(
            enable_nlp_processing=True,
            enable_conjunction_extraction=True,
            enable_subsentence_analysis=True,
            enable_enhanced_similarity=True,
            enable_enhanced_hashing=True,
            enable_nlp_logging=False,
            log_extraction_results=False,
            log_similarity_scores=False,
            max_token_limit=512,
            enable_parallel_processing=True,
            max_parallel_workers=4
        )
    
    @staticmethod
    def get_minimal_config() -> NLPEnhancedConfig:
        """最小化配置（仅基础功能）"""
        return NLPEnhancedConfig(
            enable_nlp_processing=True,
            enable_conjunction_extraction=True,
            enable_subsentence_analysis=False,
            enable_enhanced_similarity=False,
            enable_enhanced_hashing=False,
            enable_nlp_logging=False,
            log_extraction_results=False,
            log_similarity_scores=False,
            max_token_limit=256
        )
    
    @staticmethod
    def get_high_performance_config() -> NLPEnhancedConfig:
        """高性能配置"""
        return NLPEnhancedConfig(
            enable_nlp_processing=True,
            enable_conjunction_extraction=True,
            enable_subsentence_analysis=True,
            enable_enhanced_similarity=True,
            enable_enhanced_hashing=True,
            enable_nlp_logging=False,
            log_extraction_results=False,
            log_similarity_scores=False,
            max_token_limit=1024,
            enable_parallel_processing=True,
            max_parallel_workers=8
        )


def load_nlp_config(config_path: Optional[str] = None, 
                    template: Optional[str] = None) -> NLPEnhancedConfig:
    """加载NLP配置
    
    Args:
        config_path: 配置文件路径
        template: 配置模板名称 ('development', 'production', 'minimal', 'high_performance')
    
    Returns:
        NLPEnhancedConfig: 配置对象
    """
    if config_path:
        import json
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            return NLPEnhancedConfig.from_dict(config_dict)
        except Exception as e:
            print(f"加载配置文件失败: {e}，使用默认配置")
    
    if template:
        template_map = {
            'development': NLPConfigTemplates.get_development_config,
            'production': NLPConfigTemplates.get_production_config,
            'minimal': NLPConfigTemplates.get_minimal_config,
            'high_performance': NLPConfigTemplates.get_high_performance_config
        }
        
        if template in template_map:
            return template_map[template]()
        else:
            print(f"未知的配置模板: {template}，使用默认配置")
    
    # 返回默认配置
    return NLPEnhancedConfig()


def save_nlp_config(config: NLPEnhancedConfig, config_path: str) -> bool:
    """保存NLP配置到文件
    
    Args:
        config: 配置对象
        config_path: 保存路径
    
    Returns:
        bool: 是否保存成功
    """
    try:
        import json
        import os
        
        # 确保目录存在
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        print(f"保存配置文件失败: {e}")
        return False