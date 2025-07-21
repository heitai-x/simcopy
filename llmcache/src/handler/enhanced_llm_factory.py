"""增强 LLM 工厂类

该模块提供了创建和配置 EnhancedAsyncLLM 和 EnhancedVLLMHandler 的工厂方法。
主要特性：
1. 统一的创建接口
2. 自动依赖注入
3. 配置验证和优化
4. 资源管理
"""

import asyncio
from typing import Optional, Dict, Any

from loguru import logger

from .enhanced_async_llm import EnhancedAsyncLLM
# from .enhanced_vllm_handler import EnhancedVLLMHandler  # 模块不存在，已注释
from ..cache import MultiLevelCacheManager
from ..utils.similarity_search_helper import SimilaritySearchHelper
from ..config.settings import VLLMConfig
from ..config.handler_config import HandlerConfig
# from ..config.cache_config import CacheConfig
# from ..config.similarity_config import SimilarityConfig


class EnhancedLLMFactory:
    """增强 LLM 工厂类
    
    提供创建和配置增强 LLM 组件的统一接口。
    """
    
    def __init__(self):
        self._cache_managers: Dict[str, MultiLevelCacheManager] = {}
        self._similarity_helpers: Dict[str, SimilaritySearchHelper] = {}
        self._llm_instances: Dict[str, EnhancedAsyncLLM] = {}
        # self._handler_instances: Dict[str, EnhancedVLLMHandler] = {}  # EnhancedVLLMHandler不存在
    
    # async def create_cache_manager(
    #     self,
    #     cache_config: CacheConfig,
    #     instance_name: str = "default"
    # ) -> MultiLevelCacheManager:
        # """创建缓存管理器
        # 
        # Args:
        #     cache_config: 缓存配置
        #     instance_name: 实例名称
        #     
        # Returns:
        #     MultiLevelCacheManager 实例
        # """
        # if instance_name in self._cache_managers:
        #     logger.info(f"复用现有缓存管理器: {instance_name}")
        #     return self._cache_managers[instance_name]
        # 
        # try:
        #     cache_manager = MultiLevelCacheManager(cache_config)
        #     await cache_manager.initialize()
        #     
        #     self._cache_managers[instance_name] = cache_manager
        #     logger.info(f"缓存管理器创建成功: {instance_name}")
        #     
        #     return cache_manager
        #     
        # except Exception as e:
        #     logger.error(f"创建缓存管理器失败 [{instance_name}]: {e}")
        #     raise
        pass
    
    # async def create_similarity_helper(
    #     self,
    #     similarity_config: SimilarityConfig,
    #     cache_manager: MultiLevelCacheManager,
    #     instance_name: str = "default"
    # ) -> SimilaritySearchHelper:
        # """创建相似度搜索助手
        # 
        # Args:
        #     similarity_config: 相似度搜索配置
        #     cache_manager: 缓存管理器
        #     instance_name: 实例名称
        #     
        # Returns:
        #     SimilaritySearchHelper 实例
        # """
        # if instance_name in self._similarity_helpers:
        #     logger.info(f"复用现有相似度搜索助手: {instance_name}")
        #     return self._similarity_helpers[instance_name]
        # 
        # try:
        #     similarity_helper = SimilaritySearchHelper(
        #         config=similarity_config,
        #         cache_manager=cache_manager
        #     )
        #     await similarity_helper.initialize()
        #     
        #     self._similarity_helpers[instance_name] = similarity_helper
        #     logger.info(f"相似度搜索助手创建成功: {instance_name}")
        #     
        #     return similarity_helper
        #     
        # except Exception as e:
        #     logger.error(f"创建相似度搜索助手失败 [{instance_name}]: {e}")
        #     raise
        pass
    
    async def create_enhanced_llm(
        self,
        vllm_config: VLLMConfig,
        cache_manager: MultiLevelCacheManager,
        similarity_helper: SimilaritySearchHelper,
        handler_config: Optional[HandlerConfig] = None,
        instance_name: str = "default",
        **kwargs
    ) -> EnhancedAsyncLLM:
        """创建增强 AsyncLLM
        
        Args:
            vllm_config: VLLM 配置
            cache_manager: 缓存管理器
            similarity_helper: 相似度搜索助手
            handler_config: 处理器配置
            instance_name: 实例名称
            **kwargs: 其他参数
            
        Returns:
            EnhancedAsyncLLM 实例
        """
        if instance_name in self._llm_instances:
            logger.info(f"复用现有增强 LLM: {instance_name}")
            return self._llm_instances[instance_name]
        
        try:
            # 验证配置
            self._validate_vllm_config(vllm_config)
            
            # 创建增强 LLM
            enhanced_llm = EnhancedAsyncLLM.from_custom_config(
                custom_config=vllm_config,
                cache_manager=cache_manager,
                similarity_search_helper=similarity_helper,
                handler_config=handler_config or HandlerConfig(),
                **kwargs
            )
            
            self._llm_instances[instance_name] = enhanced_llm
            logger.info(f"增强 LLM 创建成功: {instance_name}")
            
            return enhanced_llm
            
        except Exception as e:
            logger.error(f"创建增强 LLM 失败 [{instance_name}]: {e}")
            raise
    
    # async def create_enhanced_handler(
    #     self,
    #     vllm_config: VLLMConfig,
    #     cache_manager: MultiLevelCacheManager,
    #     similarity_helper: SimilaritySearchHelper,
    #     handler_config: Optional[HandlerConfig] = None,
    #     instance_name: str = "default",
    #     **kwargs
    # ) -> EnhancedVLLMHandler:
    #     """创建增强 VLLM 处理器
    #     
    #     Args:
    #         vllm_config: VLLM 配置
    #         cache_manager: 缓存管理器
    #         similarity_helper: 相似度搜索助手
    #         handler_config: 处理器配置
    #         instance_name: 实例名称
    #         **kwargs: 其他参数
    #         
    #     Returns:
    #         EnhancedVLLMHandler 实例
    #     """
    #     if instance_name in self._handler_instances:
    #         logger.info(f"复用现有增强处理器: {instance_name}")
    #         return self._handler_instances[instance_name]
    #     
    #     try:
    #         # 验证配置
    #         self._validate_vllm_config(vllm_config)
    #         
    #         # 创建增强处理器
    #         enhanced_handler = EnhancedVLLMHandler(
    #             vllm_config=vllm_config,
    #             cache_manager=cache_manager,
    #             similarity_search_helper=similarity_helper,
    #             handler_config=handler_config or HandlerConfig(),
    #             **kwargs
    #         )
    #         
    #         self._handler_instances[instance_name] = enhanced_handler
    #         logger.info(f"增强处理器创建成功: {instance_name}")
    #         
    #         return enhanced_handler
    #         
    #     except Exception as e:
    #         logger.error(f"创建增强处理器失败 [{instance_name}]: {e}")
    #         raise
    pass  # EnhancedVLLMHandler类不存在，方法已注释
    
    # async def create_complete_system(
    #     self,
    #     vllm_config: VLLMConfig,
    #     cache_config: CacheConfig,
    #     similarity_config: SimilarityConfig,
    #     handler_config: Optional[HandlerConfig] = None,
    #     instance_name: str = "default",
    #     **kwargs
    # ) -> EnhancedVLLMHandler:
        # """创建完整的增强系统
        # 
        # Args:
        #     vllm_config: VLLM 配置
        #     cache_config: 缓存配置
        #     similarity_config: 相似度搜索配置
        #     handler_config: 处理器配置
        #     instance_name: 实例名称
        #     **kwargs: 其他参数
        #     
        # Returns:
        #     EnhancedVLLMHandler 实例
        # """
        # try:
        #     logger.info(f"开始创建完整增强系统: {instance_name}")
        #     
        #     # 1. 创建缓存管理器
        #     cache_manager = await self.create_cache_manager(
        #         cache_config=cache_config,
        #         instance_name=f"{instance_name}_cache"
        #     )
        #     
        #     # 2. 创建相似度搜索助手
        #     similarity_helper = await self.create_similarity_helper(
        #         similarity_config=similarity_config,
        #         cache_manager=cache_manager,
        #         instance_name=f"{instance_name}_similarity"
        #     )
        #     
        #     # 3. 创建增强处理器
        #     enhanced_handler = await self.create_enhanced_handler(
        #         vllm_config=vllm_config,
        #         cache_manager=cache_manager,
        #         similarity_helper=similarity_helper,
        #         handler_config=handler_config,
        #         instance_name=instance_name,
        #         **kwargs
        #     )
        #     
        #     logger.info(f"完整增强系统创建成功: {instance_name}")
        #     return enhanced_handler
        #     
        # except Exception as e:
        #     logger.error(f"创建完整增强系统失败 [{instance_name}]: {e}")
        #     raise
    pass
    
    def _validate_vllm_config(self, vllm_config: VLLMConfig) -> None:
        """验证 VLLM 配置
        
        Args:
            vllm_config: VLLM 配置
            
        Raises:
            ValueError: 配置无效时抛出
        """
        if not vllm_config:
            raise ValueError("VLLM 配置不能为空")
        
        if not hasattr(vllm_config, 'engine_args'):
            raise ValueError("VLLM 配置缺少 engine_args")
        
        # 检查模型路径
        if not vllm_config.engine_args.model:
            raise ValueError("VLLM 配置缺少模型路径")
        
        logger.debug(f"VLLM 配置验证通过: {vllm_config.engine_args.model}")
    
    # async def start_handler(
    #     self,
    #     instance_name: str = "default"
    # ) -> None:
    #     """启动处理器
    #     
    #     Args:
    #         instance_name: 实例名称
    #     """
    #     handler = self._handler_instances.get(instance_name)
    #     if not handler:
    #         raise ValueError(f"处理器实例不存在: {instance_name}")
    #     
    #     await handler.start()
    #     logger.info(f"处理器启动成功: {instance_name}")
    pass  # EnhancedVLLMHandler不存在
    
    # async def stop_handler(
    #     self,
    #     instance_name: str = "default"
    # ) -> None:
    #     """停止处理器
    #     
    #     Args:
    #         instance_name: 实例名称
    #     """
    #     handler = self._handler_instances.get(instance_name)
    #     if not handler:
    #         logger.warning(f"处理器实例不存在: {instance_name}")
    #         return
    #     
    #     await handler.stop()
    #     logger.info(f"处理器停止成功: {instance_name}")
    pass  # EnhancedVLLMHandler不存在
    
    async def cleanup_instance(
        self,
        instance_name: str
    ) -> None:
        """清理指定实例
        
        Args:
            instance_name: 实例名称
        """
        try:
            # 停止并清理处理器
            # if instance_name in self._handler_instances:
            #     handler = self._handler_instances.pop(instance_name)
            #     await handler.stop()  # EnhancedVLLMHandler不存在
            
            # 清理 LLM
            if instance_name in self._llm_instances:
                llm = self._llm_instances.pop(instance_name)
                await llm.cleanup()
            
            # 清理相似度助手
            similarity_name = f"{instance_name}_similarity"
            if similarity_name in self._similarity_helpers:
                similarity_helper = self._similarity_helpers.pop(similarity_name)
                if hasattr(similarity_helper, 'cleanup'):
                    await similarity_helper.cleanup()
            
            # 清理缓存管理器
            cache_name = f"{instance_name}_cache"
            if cache_name in self._cache_managers:
                cache_manager = self._cache_managers.pop(cache_name)
                if hasattr(cache_manager, 'cleanup'):
                    await cache_manager.cleanup()
            
            logger.info(f"实例清理完成: {instance_name}")
            
        except Exception as e:
            logger.error(f"清理实例失败 [{instance_name}]: {e}")
    
    async def cleanup_all(self) -> None:
        """清理所有实例"""
        try:
            # 获取所有实例名称
            all_instances = set()
            # all_instances.update(self._handler_instances.keys())  # EnhancedVLLMHandler不存在
            all_instances.update(self._llm_instances.keys())
            
            # 清理所有实例
            cleanup_tasks = []
            for instance_name in all_instances:
                cleanup_tasks.append(self.cleanup_instance(instance_name))
            
            if cleanup_tasks:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            
            # 清理剩余的组件
            self._cache_managers.clear()
            self._similarity_helpers.clear()
            self._llm_instances.clear()
            # self._handler_instances.clear()  # EnhancedVLLMHandler不存在
            
            logger.info("所有实例清理完成")
            
        except Exception as e:
            logger.error(f"清理所有实例失败: {e}")
    
    def get_instance_stats(self) -> Dict[str, Any]:
        """获取实例统计信息
        
        Returns:
            实例统计信息
        """
        return {
            'cache_managers': len(self._cache_managers),
            'similarity_helpers': len(self._similarity_helpers),
            'llm_instances': len(self._llm_instances),
            'handler_instances': len(self._handler_instances),
            'instance_names': {
                'cache_managers': list(self._cache_managers.keys()),
                'similarity_helpers': list(self._similarity_helpers.keys()),
                'llm_instances': list(self._llm_instances.keys()),
                'handler_instances': list(self._handler_instances.keys())
            }
        }


# 全局工厂实例
_global_factory = None


def get_global_factory() -> EnhancedLLMFactory:
    """获取全局工厂实例
    
    Returns:
        EnhancedLLMFactory 实例
    """
    global _global_factory
    if _global_factory is None:
        _global_factory = EnhancedLLMFactory()
    return _global_factory


async def create_enhanced_system(
    vllm_config: VLLMConfig,
    cache_config: CacheConfig,
    similarity_config: SimilarityConfig,
    handler_config: Optional[HandlerConfig] = None,
    instance_name: str = "default",
    **kwargs
) -> EnhancedVLLMHandler:
    """便捷函数：创建完整的增强系统
    
    Args:
        vllm_config: VLLM 配置
        cache_config: 缓存配置
        similarity_config: 相似度搜索配置
        handler_config: 处理器配置
        instance_name: 实例名称
        **kwargs: 其他参数
        
    Returns:
        EnhancedVLLMHandler 实例
    """
    factory = get_global_factory()
    return await factory.create_complete_system(
        vllm_config=vllm_config,
        cache_config=cache_config,
        similarity_config=similarity_config,
        handler_config=handler_config,
        instance_name=instance_name,
        **kwargs
    )