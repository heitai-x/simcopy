
"""增强的 AsyncLLM 实现

该模块继承 VLLM 的 AsyncLLM，集成缓存、NLP 分析和相似度搜索功能。
主要特性：
1. 智能缓存策略：精确匹配和相似度匹配
2. 异步 NLP 处理：连接词提取和子句分解
3. 相似度搜索：基于向量的语义相似度匹配
4. 流式输出：支持实时流式生成
5. 性能优化：减少重复计算和等待时间
"""
import os
import asyncio
import uuid
import time
from typing import Dict, Any, Optional, List, Tuple, AsyncGenerator, Union, Mapping
from copy import copy

from loguru import logger
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.sampling_params import SamplingParams
from vllm.config import VllmConfig
from vllm.v1.executor.abstract import Executor
from vllm.usage.usage_lib import UsageContext
from vllm.multimodal import MultiModalRegistry, MULTIMODAL_REGISTRY
from vllm.logger import init_logger
from vllm.v1.request import Request
from vllm.outputs import RequestOutput

from vllm.lora.request import LoRARequest
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.parallel_sampling import ParentRequest

from ..models.request import SimpleRequest
from ..models.cache import CacheEntry
from ..models.enums import RequestStatus, CacheStatus
from ..cache import MultiLevelCacheManager
from ..cache.enhanced_request_output_collector import EnhancedRequestOutputCollector
from ..utils.similarity_search_helper import SimilaritySearchHelper
from ..utils.hasher import RequestHasher
from ..config.settings import VLLMConfig as CustomVLLMConfig
from ..config.handler_config import HandlerConfig
from ..nlp.async_conjunction_extractor import AsyncAdvancedConjunctionExtractor as AsyncConjunctionExtractor
from vllm.outputs import RequestOutput as CustomRequestOutput
from vllm.v1.metrics.loggers import (StatLoggerBase, StatLoggerFactory,
                                     setup_default_loggers)
from .similar_request_memory import SimilarRequestMemoryManager, SimilarRequestMemoryConfig
class EnhancedAsyncLLM(AsyncLLM):
    """增强的 AsyncLLM，集成缓存、NLP 和相似度搜索功能
    
    继承自 VLLM 的 AsyncLLM，在保持原有功能的基础上，添加：
    - 智能缓存管理
    - 异步 NLP 处理
    - 相似度搜索和上下文管理
    - 优化的请求处理流程
    """
    
    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        cache_manager: MultiLevelCacheManager,
        similarity_search_helper: SimilaritySearchHelper,
        handler_config: HandlerConfig = None,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[list] = None,
        disable_log_requests: bool = False,
        disable_log_stats: bool = False,
        client_addresses: Optional[dict[str, str]] = None,
        client_index: int = 0,
        nlp_max_concurrent: int = 50,
    ) -> "EnhancedAsyncLLM":
        """从VllmConfig创建EnhancedAsyncLLM实例，参考父类AsyncLLM的from_vllm_config方法
        
        Args:
            vllm_config: VLLM配置
            cache_manager: 多级缓存管理器
            similarity_search_helper: 相似度搜索助手
            handler_config: 处理器配置
            start_engine_loop: 是否启动引擎循环
            usage_context: 使用上下文
            stat_loggers: 统计日志记录器
            disable_log_requests: 是否禁用请求日志
            disable_log_stats: 是否禁用统计日志
            client_addresses: 客户端地址
            client_index: 客户端索引
            nlp_max_concurrent: NLP最大并发数
            
        Returns:
            EnhancedAsyncLLM实例
        """
        from vllm import envs
        from vllm.v1.executor.abstract import Executor
        # 检查VLLM_USE_V1环境变量
        if not envs.VLLM_USE_V1:
            raise ValueError(
                "Using V1 EnhancedAsyncLLM, but envs.VLLM_USE_V1=False. "
                "This should not happen. As a workaround, try setting "
                "VLLM_USE_V1=1 and report this issue on Github.")
        
        # 直接在构造函数调用中获取执行器类，与原始 AsyncLLM 保持一致
        return cls(
            vllm_config=vllm_config,
            executor_class=Executor.get_class(vllm_config),
            cache_manager=cache_manager,
            similarity_search_helper=similarity_search_helper,
            handler_config=handler_config,
            log_stats=not disable_log_stats,
            usage_context=usage_context,
            use_cached_outputs=False,
            log_requests=not disable_log_requests,
            start_engine_loop=start_engine_loop,
            stat_loggers=stat_loggers,
            client_addresses=client_addresses,
            client_index=client_index,
            nlp_max_concurrent=nlp_max_concurrent,
        )
    
    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        cache_manager: MultiLevelCacheManager,
        similarity_search_helper: SimilaritySearchHelper,
        handler_config: HandlerConfig = None,
        log_stats: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        use_cached_outputs: bool = False,
        log_requests: bool = True,
        start_engine_loop: bool = True,
        stat_loggers: Optional[list[StatLoggerFactory]] = None,
        client_addresses: Optional[dict[str, str]] = None,
        client_index: int = 0,
        nlp_max_concurrent: int = 50,
    ):
        # 先初始化增强组件的属性，但不进行任何可能影响CUDA的操作
        self.cache_manager = cache_manager
        self.similarity_search_helper = similarity_search_helper
        self.handler_config = handler_config or HandlerConfig()
        self._nlp_max_concurrent = nlp_max_concurrent
        self._active_tasks: Dict[str, asyncio.Task] = {}
        
        # 初始化推理中断统计
        self._interrupt_stats = {
            'successful_interrupts': 0,
            'failed_interrupts': 0,
            'total_interrupt_time': 0.0,
            'total_attempts': 0
        }
        if self.handler_config.features.enable_nlp_enhancement:
            self.conjunction_extractor = AsyncConjunctionExtractor()
            self.conjunction_extractor.set_max_concurrent_tasks(self._nlp_max_concurrent)
        else:
            self.conjunction_extractor = None
        self.hasher = RequestHasher()
        self.vector_search_manager = self.similarity_search_helper.vector_search
                
        config = SimilarRequestMemoryConfig(
            request_mapping_memory_size=8 * 1024 * 1024,  # 8MB
            token_mapping_memory_size=8 * 1024 * 1024,    # 8MB
            max_entries=5000,
            request_mapping_shared_name="vllm_request_mappings",
            token_mapping_shared_name="vllm_token_mappings"
        )
        self._shared_memory_manager = SimilarRequestMemoryManager(config)
        
        
        super().__init__(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=log_stats,
            usage_context=usage_context,
            mm_registry=mm_registry,
            use_cached_outputs=use_cached_outputs,
            log_requests=log_requests,
            start_engine_loop=start_engine_loop,
            stat_loggers=stat_loggers,
            client_addresses=client_addresses,
            client_index=client_index,
        )
        
    def _generate_hash_id(self, prompt: str, sampling_params: SamplingParams) -> str:
        """生成请求的哈希ID"""
        return self.hasher.compute_hash_id(prompt, sampling_params)
    
    def _create_task(self, coro, task_name: str) -> asyncio.Task:
        """创建并管理异步任务，确保与输出处理器协调"""
        # 检查输出处理器状态
        if hasattr(self, 'output_handler') and self.output_handler and self.output_handler.done():
            logger.warning(f"输出处理器已停止，跳过任务创建: {task_name}")
            raise RuntimeError("Output handler is not running")
        
        # 检查是否已有同名任务
        if task_name in self._active_tasks:
            existing_task = self._active_tasks[task_name]
            if not existing_task.done():
                logger.debug(f"任务 {task_name} 已存在，取消旧任务")
                existing_task.cancel()
        
        task = asyncio.create_task(coro)
        self._active_tasks[task_name] = task
        
        def cleanup_task(task_name):
            self._active_tasks.pop(task_name, None)
            if not task.cancelled():
                try:
                    exception = task.exception()
                    if exception:
                        logger.error(f"关键任务失败 {task_name}: {exception}")
                except Exception as e:
                    logger.warning(f"获取任务异常信息失败 {task_name}: {e}")
        
        task.add_done_callback(lambda t: cleanup_task(task_name))
        return task
    
    def _cancel_task(self, task_name: str) -> None:
        """取消指定任务"""
        task = self._active_tasks.get(task_name)
        if task and not task.done():
            task.cancel()
            self._active_tasks.pop(task_name, None)
    
    async def _check_cache(self,  request_id: str, cache_entry: CacheEntry,prompt:str) -> Optional[CustomRequestOutput]:
        """检查缓存，返回缓存结果或None，兼容旧缓存格式"""
        try:
            hash_id=cache_entry.hash_id
            if cache_entry and cache_entry.status == CacheStatus.COMPLETED:
                # 缓存命中
                logger.debug(f"缓存命中: {request_id} -> {hash_id}")
                
                # 更新访问时间
                cache_entry.update_access_time()
                await self.cache_manager.put(hash_id, cache_entry)
                
                # 兼容旧缓存格式处理
                cached_result = cache_entry.result
                cached_tokens = cache_entry.result_tokens or []
                
                # 检查是否为旧格式（直接的字符串或token列表）
                if isinstance(cached_result, str):
                    # 旧格式：直接文本
                    result_text = cached_result
                elif isinstance(cached_result, list):
                    # 旧格式：token ID列表，需要转换为文本
                    result_text = " ".join(map(str, cached_result))
                    if not cached_tokens:
                        cached_tokens = cached_result
                elif hasattr(cached_result, 'outputs') and cached_result.outputs:
                    # 新格式：RequestOutput对象
                    result_text = "".join([comp.text for comp in cached_result.outputs])
                    cached_tokens = []
                    for comp in cached_result.outputs:
                        cached_tokens.extend(comp.token_ids)
                else:
                    # 未知格式，尝试转换为字符串
                    result_text = str(cached_result)
                
                # 构建标准化的缓存结果输出
                from vllm.outputs import CompletionOutput
                
                completion_output = CompletionOutput(
                    index=0,
                    text=result_text,
                    token_ids=cached_tokens,
                    cumulative_logprob=None,
                    logprobs=None,
                    finish_reason="stop"
                )
                
                return CustomRequestOutput(
                    request_id=request_id,
                    prompt=prompt,
                    prompt_token_ids=None,
                    prompt_logprobs=None,
                    outputs=[completion_output],
                    finished=True,
                    metrics=None
                )
            
            return None
            
        except Exception as e:
            logger.error(f"检查缓存失败 [{request_id}]: {e}")
            return None
    
    async def _perform_subsentence_similarity_search(self, nlp_result: Dict[str, Any], 
                                                    top_k: int = 10, 
                                                    similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """执行子句相似度搜索"""
        subsentences = nlp_result.get('subsentences', [])
        if not subsentences or not self.vector_search_manager:
            return []
        
        try:
            # 批量搜索子句
            batch_results = await self.vector_search_manager.batch_search_similar_questions(
                query_texts=subsentences, top_k=top_k, similarity_threshold=similarity_threshold
            )
            
            # 收集所有结果
            all_results = []
            for results in batch_results:
                all_results.extend(results)
            
            # 去重并排序
            return self._deduplicate_search_results(all_results, top_k)
            
        except Exception as e:
            logger.error(f"子句相似度搜索失败: {e}")
            return []
    
    def _deduplicate_and_sort_results(self, results: List[Dict[str, Any]], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """去重并排序结果"""
        if not results:
            return []
        
        unique_results = {}
        for result in results:
            hash_id = result['hash_id']
            if hash_id not in unique_results or result['similarity'] > unique_results[hash_id]['similarity']:
                unique_results[hash_id] = result
        
        sorted_results = sorted(unique_results.values(), key=lambda x: x['similarity'], reverse=True)
        return sorted_results[:top_k] if top_k else sorted_results
    
    async def _process_initial_similarity_search(self, initial_similarity_task, request_id: str, 
                                                task_prefix: str) -> Tuple[bool, List[Dict[str, Any]]]:
        """处理初步相似度搜索结果
        
        Args:
            initial_similarity_task: 初步相似度搜索任务
            request_id: 请求ID
            task_prefix: 任务前缀
            
        Returns:
            (是否直接复用, 相似答案列表)
        """
        if not initial_similarity_task:
            return False, []
        
        try:
            direct_reuse, similar_answers = await initial_similarity_task
            if direct_reuse:
                # 高相似度结果，直接返回，取消NLP任务
                self._cancel_task(f"{task_prefix}_nlp")
                if self.handler_config.features.enable_detailed_logging:
                    logger.info("通过并行原始问题相似度搜索找到高相似度结果")
                return True, similar_answers
            elif similar_answers:
                logger.debug(f"原始问题找到 {len(similar_answers)} 个中等相似度结果")
                return False, similar_answers
        except asyncio.CancelledError:
            logger.debug("初步相似度搜索被取消")
        except Exception as e:
            logger.error(f"初步相似度搜索失败: {e}")
        
        return False, []
    
    async def _process_nlp_task(self, nlp_task, request_id: str) -> Optional[Dict[str, Any]]:
        """处理NLP任务
        
        Args:
            nlp_task: NLP任务
            request_id: 请求ID
            
        Returns:
            NLP处理结果或None
        """
        if not nlp_task:
            return None
        
        try:
            nlp_result = await nlp_task
            # 修复：添加 None 检查，避免在 None 对象上调用 get 方法
            if nlp_result is not None and self.handler_config.features.enable_detailed_logging:
                logger.debug(f"异步NLP处理完成 [{request_id}]: 找到连接词 {nlp_result.get('has_conjunctions', False)}")
            return nlp_result
        except asyncio.CancelledError:
            if self.handler_config.features.enable_detailed_logging:
                logger.debug(f"异步NLP任务被取消 [{request_id}]")
        except Exception as e:
            logger.error(f"异步NLP分析失败 [{request_id}]: {e}")
        
        return None
    
    async def _store_similarity_context(self, request_id: str, similar_answers: List[Dict[str, Any]]) -> None:
        """存储相似度上下文到共享内存"""
        if not similar_answers:
            return
        
        try:
            # 提取相似请求的 hash_id 列表
            similar_hash_ids = [answer['hash_id'] for answer in similar_answers]
            
            # 存储请求映射
            self._shared_memory_manager.store_similar_request_mapping(request_id, similar_hash_ids)
            
            # 批量准备 token 映射数据
            token_mappings = {}
            for similar_hash_id in similar_hash_ids:
                try:
                    cache_entry = await self.cache_manager.get(similar_hash_id)
                    if cache_entry and cache_entry.status.name == 'COMPLETED':
                        if hasattr(cache_entry, 'result_tokens') and cache_entry.result_tokens is not None:
                            token_mappings[similar_hash_id] = cache_entry.result_tokens
                except Exception as e:
                    logger.warning(f"准备相似答案 token 数据失败 {similar_hash_id}: {e}")
                    continue
            
            # 批量存储 token 映射
            if token_mappings:
                self._shared_memory_manager.store_multiple_answer_tokens(token_mappings)
            
            logger.debug(f"相似度上下文存储完成 [{request_id}]: {len(similar_hash_ids)} 个相似答案")
            
        except Exception as e:
            logger.error(f"存储相似度上下文失败 [{request_id}]: {e}")

    
    async def _do_cache_update(self, prompt: str, sampling_params: SamplingParams, 
                              request_id: str,hash_id:str, output: CustomRequestOutput) -> None:
        """实际的缓存更新操作"""
        try:
            cache_entry = await self.cache_manager.get(hash_id)
            logger.info("开始更新缓存")
            # 从RequestOutput中提取文本和token信息
            output_text = ""
            output_tokens = []
            if output.outputs:
                logger.info("合并所有输出的文本和tokens")
                
                output_text = "".join([comp.text for comp in output.outputs])
                output_tokens = []
                for comp in output.outputs:
                    output_tokens.extend(comp.token_ids)
            
            if cache_entry:
                # 更新缓存条目
                logger.info("更新缓存条目")
                cache_entry.status = CacheStatus.COMPLETED
                cache_entry.result = output_text
                cache_entry.result_tokens = output_tokens
                cache_entry.update_access_time()
                
                # 保存到缓存
                await self.cache_manager.put(hash_id, cache_entry)
                
                # 更新向量搜索映射
                if self.vector_search_manager:
                    await self._update_vector_search_mapping(prompt, hash_id)
                
                logger.debug(f"缓存已更新: {hash_id}")
            else:
                # 创建新的缓存条目
                logger.info("创建新的Cache")
                cache_entry = CacheEntry(
                    original_request_id=request_id,
                    result=output_text,
                    hash_id=hash_id,
                    result_tokens=output_tokens,
                    status=CacheStatus.COMPLETED
                )
                await self.cache_manager.put(hash_id, cache_entry)
                
                # 更新向量搜索映射
                if self.vector_search_manager:
                    await self._update_vector_search_mapping(prompt, hash_id)
                
                logger.debug(f"新缓存条目已创建: {hash_id}")
                
        except Exception as e:
            logger.error(f"缓存更新失败 [{request_id}]: {e}")
    
    async def _update_vector_search_mapping(self, prompt: str, hash_id: str) -> None:
        """更新向量搜索映射"""
        try:
            if self.vector_search_manager:
                await self.vector_search_manager.add_question_mapping(prompt, hash_id)
                logger.info(f"已添加到向量搜索映射: {hash_id}")
            else:
                logger.info("无vector_search_manager")
        except Exception as e:
            logger.error(f"更新向量搜索映射失败: {e}")
    
    async def _process_nlp_async(self, prompt: str, request_id: str) -> Dict[str, Any]:
        """异步处理NLP任务"""
        try:
            # 修复：添加初始化检查
            if not self.conjunction_extractor.is_ready():
                logger.warning(f"提取器未初始化 [{request_id}]，尝试重新初始化")
                await self.conjunction_extractor.initialize()
                
            if not self.conjunction_extractor.is_ready():
                logger.error(f"提取器初始化失败 [{request_id}]")
                return {'has_conjunctions': False, 'subsentences': []}
                
            return await self.conjunction_extractor.extract_and_generate_variants(prompt)
        except Exception as e:
            logger.error(f"NLP处理失败 [{request_id}]: {e}")
            return {'has_conjunctions': False, 'subsentences': []}
    
    async def _start_async_tasks(self, prompt: str, request_id: str):
        """启动异步任务"""
        task_prefix = f"req_{request_id}"
        nlp_task = None
        initial_similarity_task = None

        if self.handler_config.features.enable_nlp_enhancement:
            nlp_task = self._create_task(
                self._process_nlp_async(str(prompt), request_id),
                f"{task_prefix}_nlp"
            )

        if self.handler_config.features.enable_similarity_search and self.vector_search_manager:
            initial_similarity_task = self._create_task(
                self.similarity_search_helper.try_layered_similarity_search(
                    prompt, self.cache_manager
                ),
                f"{task_prefix}_initial_similarity"
            )
        
        return nlp_task, initial_similarity_task
    
    async def _process_nlp_and_subsearch(self, nlp_task, request_id: str, initial_similar_answers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """处理NLP任务和子句搜索"""
        nlp_result = await self._process_nlp_task(nlp_task, request_id)
        all_similar_answers = list(initial_similar_answers)
        # 子句级相似度搜索
        # 修复：添加更严格的 None 检查
        if nlp_result is not None and nlp_result.get('has_conjunctions', False) and self.vector_search_manager:
            try:
                subsentence_results = await self._perform_subsentence_similarity_search(
                    nlp_result, top_k=10, similarity_threshold=0.7
                )
                all_similar_answers.extend(subsentence_results)
                logger.debug(f"子句搜索完成 [{request_id}]，收集到 {len(subsentence_results)} 个相似答案")
            except Exception as e:
                logger.error(f"子句搜索失败 [{request_id}]: {e}")
        
        # 去重和排序
        if all_similar_answers:
            all_similar_answers = self._deduplicate_and_sort_results(all_similar_answers)
        
        return all_similar_answers
    
    async def _add_to_engine(self, prompt: str, params: SamplingParams, request_id: str,
                           arrival_time: Optional[float], lora_request: Optional[Any],
                           tokenization_kwargs: Optional[dict[str, Any]], 
                           trace_headers: Optional[dict[str, str]],
                           prompt_adapter_request: Optional[Any], priority: int,
                           collector: 'EnhancedRequestOutputCollector') -> None:
        """将请求添加到引擎"""
        # Convert Input --> Request (same as parent)
        prompt_str, request = self.processor.process_inputs(
            request_id, prompt, params, arrival_time, lora_request,
            tokenization_kwargs, trace_headers, prompt_adapter_request,
            priority)
        
        # Handle single vs multiple requests
        if params.n == 1:
            await self._add_enhanced_request(request, prompt_str, None, 0, collector)
        else:
            # Fan out child requests (for n>1)
            parent_request = ParentRequest(request_id, params)
            for idx in range(params.n):
                child_request_id, child_params = parent_request.get_child_info(idx)
                child_request = request if idx == params.n - 1 else copy(request)
                child_request.request_id = child_request_id
                child_request.sampling_params = child_params
                await self._add_enhanced_request(child_request, prompt_str, parent_request,
                                               idx, collector)
    
    async def add_request(
        self,
        request_id: str,
        hash_id: str,
        prompt: str,
        params: SamplingParams,
        arrival_time: Optional[float] = None,
        lora_request: Optional[Any] = None,
        tokenization_kwargs: Optional[dict[str, Any]] = None,
        trace_headers: Optional[dict[str, str]] = None,
        prompt_adapter_request: Optional[Any] = None,
        priority: int = 0,
    ) -> 'EnhancedRequestOutputCollector':
        """Enhanced add_request with parallel processing - 保持原有流程但不等待完成"""
        
        start_time = time.time()
        
        try:

            # Step 1: Check cache first (保持不变)
            cache_entry = await self.cache_manager.get(hash_id)
            
            if cache_entry:
                cached_result = await self._check_cache(request_id, cache_entry, prompt)
                collector = EnhancedRequestOutputCollector(
                    output_kind=params.output_kind,
                    hash_id=hash_id
                )
                collector.put_cached_result(cached_result)
                return collector

            # 立即创建收集器
            collector = EnhancedRequestOutputCollector(
                output_kind=params.output_kind,
                hash_id=hash_id
            )
            
            # 立即启动推理
            inference_task = self._create_task(
                self._add_to_engine(
                    prompt, params, request_id, arrival_time, lora_request,
                    tokenization_kwargs, trace_headers, prompt_adapter_request,
                    priority, collector
                ),
                f"inference_{request_id}"
            )
            
            # 并行启动NLP和相似度搜索
            background_task = self._create_task(
                self._process_background_nlp_and_similarity_with_interrupt(
                    prompt, request_id, hash_id, params, collector, inference_task
                ),
                f"background_processing_{request_id}"
            )
            
            # 立即返回收集器
            return collector
            
        except Exception as e:
            logger.error(f"Enhanced add_request failed [{request_id}]: {e}")
            # Fallback to parent implementation
            hash_id = self._generate_hash_id(prompt, params)
            fallback_collector = EnhancedRequestOutputCollector(
                output_kind=params.output_kind,
                hash_id=hash_id
            )
            parent_collector = await super().add_request(
                request_id=request_id,
                prompt=prompt,
                params=params,
                arrival_time=arrival_time,
                lora_request=lora_request,
                tokenization_kwargs=tokenization_kwargs,
                trace_headers=trace_headers,
                prompt_adapter_request=prompt_adapter_request,
                priority=priority
            )
            fallback_collector.set_parent_collector(parent_collector)
            return fallback_collector
    
    async def _process_background_nlp_and_similarity(
        self,
        prompt: str,
        request_id: str,
        hash_id: str,
        params: SamplingParams
    ) -> None:
        """后台处理NLP和相似度搜索"""
        try:
            nlp_task, initial_similarity_task = await self._start_async_tasks(prompt, request_id)
        
            direct_reuse, initial_similar_answers = await self._process_initial_similarity_search(
                initial_similarity_task, request_id, f"req_{request_id}"
            )
        
            # 如果发现高相似度结果，记录但不中断推理
            if direct_reuse:
                if initial_similar_answers:
                    best_match = initial_similar_answers[0]["cache_entry"]
                    if best_match and best_match.status == CacheStatus.COMPLETED:
                        logger.info(f"发现高相似度缓存，但推理已启动 [{request_id}]")
                else:
                    logger.warning(f"相似度重用失败，继续正常处理 [{request_id}]")
            
            all_similar_answers = await self._process_nlp_and_subsearch(
                nlp_task, request_id, initial_similar_answers
            )
            
            # Step 7: Process and store similarity context
            if all_similar_answers:
                await self._store_similarity_context(request_id, all_similar_answers)

            # 创建缓存条目
            cache_entry = CacheEntry(
                original_request_id=request_id,
                hash_id=hash_id,
                result=None,
                result_tokens=None,
                status=CacheStatus.PROCESSING
            )
            await self.cache_manager.put(hash_id, cache_entry)
            
        except Exception as e:
            logger.error(f"后台NLP和相似度处理失败 [{request_id}]: {e}")
    
    async def _process_background_nlp_and_similarity_with_interrupt(
        self,
        prompt: str,
        request_id: str,
        hash_id: str,
        params: SamplingParams,
        collector: 'EnhancedRequestOutputCollector',
        inference_task: asyncio.Task
    ) -> None:
        """后台处理NLP和相似度搜索 - 支持推理中断优化"""
        try:
            nlp_task, initial_similarity_task = await self._start_async_tasks(prompt, request_id)
        
            direct_reuse, initial_similar_answers = await self._process_initial_similarity_search(
                initial_similarity_task, request_id, f"req_{request_id}"
            )
        
            # 推理中断优化：如果发现高相似度结果，尝试中断推理
            if direct_reuse and initial_similar_answers:
                best_match = initial_similar_answers[0]["cache_entry"]
                if best_match and best_match.status == CacheStatus.COMPLETED:
                    similarity_score = initial_similar_answers[0].get("similarity", 0.0)
                    logger.info(f"发现高相似度缓存 (相似度: {similarity_score:.3f})，尝试中断推理 [{request_id}]")
                    
                    if not inference_task.done() or (await self._should_interrupt_inference(request_id, similarity_score) and self.output_processor.request_states.get(request_id)):
                        interrupt_success = await self._attempt_inference_interrupt(
                            request_id, inference_task, collector, best_match, prompt
                        )
                        if interrupt_success:
                            return

                    else:
                        if inference_task.done():
                            logger.info(f"推理已完成，无法中断 [{request_id}]")
                        else:
                            logger.info(f"相似度不足以中断推理 [{request_id}]")
            
            all_similar_answers = await self._process_nlp_and_subsearch(
                nlp_task, request_id, initial_similar_answers
            )
            
            # Step 7: Process and store similarity context
            if all_similar_answers:
                await self._store_similarity_context(request_id, all_similar_answers)

            # 创建缓存条目
            cache_entry = CacheEntry(
                original_request_id=request_id,
                hash_id=hash_id,
                result=None,
                result_tokens=None,
                status=CacheStatus.PROCESSING
            )
            await self.cache_manager.put(hash_id, cache_entry)
            
        except Exception as e:
            logger.error(f"后台NLP和相似度处理失败 [{request_id}]: {e}")
    
    async def _should_interrupt_inference(self, request_id: str, similarity_score: float) -> bool:
        """判断是否应该中断推理
        
        Args:
            request_id: 请求ID
            similarity_score: 相似度分数
            
        Returns:
            是否应该中断推理
        """
        # 更新中断尝试统计
        self._interrupt_stats['total_attempts'] += 1
        
        # 中断阈值：只有极高相似度才中断（避免频繁中断）
        INTERRUPT_THRESHOLD = 0.95
        
        if similarity_score < INTERRUPT_THRESHOLD:
            logger.debug(f"相似度 {similarity_score:.3f} 低于中断阈值 {INTERRUPT_THRESHOLD} [{request_id}]")
            return False

        return True
    
    async def _attempt_inference_interrupt(
        self,
        request_id: str,
        inference_task: asyncio.Task,
        collector: 'EnhancedRequestOutputCollector',
        best_match: 'CacheEntry',
        prompt: str
    ) -> bool:
        """尝试中断推理并返回缓存结果
        
        Args:
            request_id: 请求ID
            inference_task: 推理任务
            collector: 输出收集器
            best_match: 最佳匹配的缓存条目
            prompt: 原始提示
            
        Returns:
            是否成功中断
        """
        try:
            # 记录中断开始时间
            interrupt_start = time.time()
            
            # 1. 中断推理任务
            inference_task.cancel()
            logger.info(f"推理任务已取消 [{request_id}]")
            
            # 2. 中断底层VLLM请求
            await self._abort_vllm_request(request_id)
            
            # 3. 将缓存结果放入收集器
            cached_result = await self._check_cache(request_id, best_match, prompt)
            collector.put_cached_result(cached_result)
            
            # 4. 记录中断性能
            interrupt_time = time.time() - interrupt_start
            logger.info(f"推理中断成功，耗时 {interrupt_time:.3f}s，使用缓存结果 [{request_id}]")
            
            # 5. 更新统计信息
            self._interrupt_stats['successful_interrupts'] += 1
            self._interrupt_stats['total_interrupt_time'] += interrupt_time
            
            return True
            
        except Exception as interrupt_error:
            logger.warning(f"推理中断失败，继续正常处理 [{request_id}]: {interrupt_error}")
            
            # 更新失败统计
            self._interrupt_stats['failed_interrupts'] += 1
            
            return False
    
    async def _abort_vllm_request(self, request_id: str) -> None:
        """中断VLLM底层请求"""
        try:
            # 调用父类的abort方法中断VLLM请求
            await super().abort(request_id)
            logger.info(f"VLLM请求中断成功 [{request_id}]")
        except Exception as e:
            logger.error(f"VLLM请求中断失败 [{request_id}]: {e}")
            raise
    
    def get_interrupt_stats(self) -> Dict[str, Any]:
        """获取推理中断统计信息
        
        Returns:
            包含中断统计信息的字典
        """
        stats = self._interrupt_stats.copy()
        
        # 计算成功率
        if stats['total_attempts'] > 0:
            stats['success_rate'] = stats['successful_interrupts'] / stats['total_attempts']
        else:
            stats['success_rate'] = 0.0
        
        # 计算平均中断时间
        if stats['successful_interrupts'] > 0:
            stats['avg_interrupt_time'] = stats['total_interrupt_time'] / stats['successful_interrupts']
        else:
            stats['avg_interrupt_time'] = 0.0
        
        return stats
    
    def reset_interrupt_stats(self) -> None:
        """重置推理中断统计信息"""
        self._interrupt_stats = {
            'successful_interrupts': 0,
            'failed_interrupts': 0,
            'total_interrupt_time': 0.0,
            'total_attempts': 0
        }
    
    async def _add_enhanced_request(
        self, 
        request: EngineCoreRequest,
        prompt: Optional[str],
        parent_req: Optional[ParentRequest], 
        index: int,
        collector: 'EnhancedRequestOutputCollector'
    ):
        """Enhanced version of _add_request with additional functionality."""
        
        # Add the request to OutputProcessor (this process).
        self.output_processor.add_request(request, prompt, parent_req, index, collector)

        # Add the EngineCoreRequest to EngineCore (separate process).
        await self.engine_core.add_request_async(request)

        if self.log_requests:
            logger.info("Added enhanced request %s.", request.request_id)
    
    async def generate_enhanced(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        request_id: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[CustomRequestOutput, None]:
        """增强的生成方法，集成缓存、NLP 和相似度搜索
        
        Args:
            prompt: 输入提示
            sampling_params: 采样参数
            request_id: 请求ID（可选）
            **kwargs: 其他参数
            
        Yields:
            CustomRequestOutput: 增强的输出对象
        """
        if request_id is None:
            request_id = str(uuid.uuid4())
        hash_id = self._generate_hash_id(prompt, sampling_params)
        collector = None
        
        try:
            # Use enhanced add_request method
            collector = await self.add_request(
                request_id=request_id,
                hash_id=hash_id,
                prompt=prompt,
                params=sampling_params,
                priority=kwargs.get('priority', 0)
            )

            finished = False
            while not finished:
                # Note: drain queue without await if possible (avoids
                # task switching under load which helps performance).
                output = await collector.get()
                
                # Note: both OutputProcessor and EngineCore handle their
                # own request cleanup based on finished.
                finished = output.finished
                if finished:
                    # 直接调用缓存更新，避免双重任务创建
                    try:
                        await self._do_cache_update(prompt, sampling_params, request_id, hash_id, output)
                    except Exception as e:
                        logger.error(f"缓存更新失败 [{request_id}]: {e}")
                yield output
                
                # Update cache on completion
                # Update cache on completion
            
        except Exception as e:
            logger.error(f"增强生成失败 [{request_id}]: {e}")
            
            # 清理相关任务
            task_prefix = f"req_{request_id}"
            for task_name in list(self._active_tasks.keys()):
                if task_name.startswith(task_prefix):
                    self._cancel_task(task_name)
            
            # 清理缓存占位符
            try:
                hash_id = self._generate_hash_id(prompt, sampling_params)
                cache_entry = await self.cache_manager.get(hash_id)
                if cache_entry and cache_entry.status == CacheStatus.PENDING:
                    await self.cache_manager.remove(hash_id)
            except Exception as cleanup_error:
                logger.debug(f"清理缓存占位符失败 [{request_id}]: {cleanup_error}")
            
            # 重新抛出异常
            raise
    
    async def cleanup(self) -> None:
        """清理资源"""
        # 取消所有活跃任务
        cancelled_tasks = []
        for task_name, task in list(self._active_tasks.items()):
            if not task.done():
                task.cancel()
                cancelled_tasks.append(task)
        
        # 等待任务完成（带超时）
        if cancelled_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*cancelled_tasks, return_exceptions=True),
                    timeout=5.0  # 5秒超时
                )
            except asyncio.TimeoutError:
                logger.warning("部分任务清理超时")
        
        self._active_tasks.clear()
        
        # 清理 NLP 组件
        if hasattr(self.conjunction_extractor, 'cleanup'):
            try:
                if asyncio.iscoroutinefunction(self.conjunction_extractor.cleanup):
                    await self.conjunction_extractor.cleanup()
                else:
                    self.conjunction_extractor.cleanup()
            except Exception as e:
                logger.error(f"清理 NLP 组件失败: {e}")
        
        # 清理缓存管理器
        if hasattr(self.cache_manager, 'cleanup'):
            try:
                if asyncio.iscoroutinefunction(self.cache_manager.cleanup):
                    await self.cache_manager.cleanup()
                else:
                    self.cache_manager.cleanup()
            except Exception as e:
                logger.error(f"清理缓存管理器失败: {e}")
        
        # 清理共享内存管理器
        if self._shared_memory_manager:
            try:
                self._shared_memory_manager.cleanup()
            except Exception as e:
                logger.error(f"清理共享内存管理器失败: {e}")

    @classmethod
    def from_custom_config(
        cls,
        custom_config: CustomVLLMConfig,
        cache_manager: MultiLevelCacheManager,
        similarity_search_helper: SimilaritySearchHelper,
        handler_config: HandlerConfig = None,
        **kwargs
    ) -> "EnhancedAsyncLLM":
        """从自定义配置创建增强 AsyncLLM
        
        Args:
            custom_config: 自定义 VLLM 配置
            cache_manager: 缓存管理器
            similarity_search_helper: 相似度搜索助手
            handler_config: 处理器配置
            **kwargs: 其他参数
            
        Returns:
            EnhancedAsyncLLM 实例
        """
        # 创建 VLLM 配置
        vllm_config = custom_config.engine_args.create_engine_config()
        executor_class = Executor.get_class(vllm_config)
        
        return cls(
            vllm_config=vllm_config,
            executor_class=executor_class,
            cache_manager=cache_manager,
            similarity_search_helper=similarity_search_helper,
            handler_config=handler_config,
            **kwargs
        )
    
    def shutdown(self):
        """增强的关闭方法，确保所有组件正确清理"""
        # 首先清理增强组件
        try:
            # 取消所有活跃任务
            for task_name, task in list(self._active_tasks.items()):
                if not task.done():
                    task.cancel()
            
            # 清理 NLP 组件
            if hasattr(self.conjunction_extractor, 'cleanup'):
                # 如果是异步清理，需要在事件循环中执行
                loop = None
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    pass
                
                if loop and loop.is_running():
                    # 在运行的事件循环中创建任务
                    asyncio.create_task(self.conjunction_extractor.cleanup())
                else:
                    # 如果没有运行的事件循环，尝试同步清理
                    if hasattr(self.conjunction_extractor, 'cleanup_sync'):
                        self.conjunction_extractor.cleanup_sync()
            
            # 清理缓存连接
            if hasattr(self.cache_manager, 'disconnect'):
                if asyncio.iscoroutinefunction(self.cache_manager.disconnect):
                    # 异步断开连接
                    loop = None
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        pass
                    
                    if loop and loop.is_running():
                        asyncio.create_task(self.cache_manager.disconnect())
                else:
                    # 同步断开连接
                    self.cache_manager.disconnect()
            
            # 清理共享内存管理器
            if self._shared_memory_manager:
                self._shared_memory_manager.cleanup()
                
        except Exception as e:
            logger.error(f"增强组件清理失败: {e}")
        
        # 清理 PyTorch 分布式进程组
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception as e:
            logger.error(f"清理 PyTorch 分布式进程组失败: {e}")
        
        # 最后调用父类清理
        super().shutdown()

