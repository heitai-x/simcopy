"""基础VLLM处理器模块

基于 VLLM AsyncLLM 架构重构的缓存处理器
"""

import asyncio
import uuid
import numpy as np
import os
import time
from typing import Dict, Any, Optional, List, AsyncGenerator
from abc import ABC, abstractmethod

from loguru import logger

# 修改导入，使用 AsyncLLM 替代 AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.sampling_params import SamplingParams
from vllm.config import VllmConfig

# 导入新的缓存组件
from ..cache.enhanced_request_output_collector import EnhancedRequestOutputCollector as CacheRequestOutputCollector
# from ..cache.enhanced_request_output_collector import RequestOutput  # 导入冲突，已注释
from vllm.outputs import RequestOutput
# 这些模块不存在，暂时注释掉
# from .cache_output_processor import CacheOutputProcessor, CacheIterationStats
# from .cache_request_manager import CacheRequestManager, RequestState

from ..models.request import SimpleRequest
from ..models.cache import CacheEntry
from ..models.enums import RequestStatus, CacheStatus, Priority
from ..cache import MultiLevelCacheManager
from ..utils.similarity_search_helper import SimilaritySearchHelper
from ..utils.hasher import RequestHasher

from ..config.settings import VLLMConfig
from ..config.handler_config import HandlerConfig
from ..utils.request_manager import RequestManager
from .similar_request_memory import SimilarRequestMemoryConfig



class BaseVLLMHandler(ABC):
    """VLLM处理器基类
    
    基于 VLLM AsyncLLM 架构重构，提供:
    - RequestOutputCollector 模式的异步输出处理
    - 分离式 OutputProcessor 和 RequestManager
    - 分块处理防止阻塞
    - 统一的错误处理和性能统计
    - 缓存管理和内存优化
    """
    
    def __init__(self, config: VLLMConfig, cache_manager: MultiLevelCacheManager, 
                 similarity_search_helper: SimilaritySearchHelper, 
                 handler_config: HandlerConfig = None):
        # 基础配置
        self.engine_args = config.engine_args
        self.max_concurrent = config.max_concurrent
        
        # Handler配置
        self.handler_config = handler_config or HandlerConfig.create_default()
        
        # 依赖注入
        self.cache_manager = cache_manager
        self.similarity_search_helper = similarity_search_helper
        
        # 从注入的组件中获取依赖
        self.redis_manager = self.cache_manager.redis_manager
        self.vector_search_manager = self.similarity_search_helper.vector_search_manager
        
        # 统一的哈希生成器
        self.request_hasher = RequestHasher()
        
        # 新的缓存组件（基于 VLLM AsyncLLM 架构）- 暂时注释掉，因为模块不存在
        # self.output_processor = CacheOutputProcessor(
        #     cache_manager=self.cache_manager,
        #     max_collectors=self.handler_config.features.max_concurrent_tasks
        # )
        # 
        # self.request_manager_v2 = CacheRequestManager(
        #     cache_manager=self.cache_manager,
        #     output_processor=self.output_processor,
        #     max_concurrent_requests=self.max_concurrent,
        #     request_timeout=self.handler_config.features.request_timeout,
        #     max_retry_count=3
        # )
        
        # 保留原有的请求管理器以兼容现有代码
        similar_memory_config = SimilarRequestMemoryConfig(
            request_mapping_memory_size=8 * 1024 * 1024,  # 8MB for request mappings
            token_mapping_memory_size=8 * 1024 * 1024,    # 8MB for token mappings
            max_entries=5000,
            request_mapping_shared_name="vllm_request_mappings",
            token_mapping_shared_name="vllm_token_mappings"
        )
        self.request_manager = RequestManager(similar_memory_config)
        
        # 运行时状态
        self.llm: Optional[AsyncLLM] = None
        self.is_running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        self._output_handler_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        
        # 异步任务管理
        self._active_tasks: Dict[str, asyncio.Task] = {}
        
        # 性能统计 - 暂时注释掉，因为CacheIterationStats不存在
        # self._stats = CacheIterationStats()
        self._last_stats_log_time = time.time()
        
        logger.info(f"{self.__class__.__name__} 初始化完成（基于 VLLM AsyncLLM 架构）")

    def _setup_environment(self) -> None:
        """检查环境变量设置"""
        required_vars = ["VLLM_USE_V1", "TRANSFORMERS_CACHE"]
        for var in required_vars:
            if var not in os.environ:
                logger.warning(f"环境变量 {var} 未设置，这可能导致初始化问题")
        
        logger.info(f"当前环境变量: VLLM_USE_V1={os.environ.get('VLLM_USE_V1')}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    
    async def start(self):
        """启动处理器"""
        if self.is_running:
            logger.warning("处理器已在运行中")
            return
        
        try:
            # 设置环境变量
            self._setup_environment()
            
            # 连接Redis
            if self.redis_manager:
                await self.redis_manager.connect()
                logger.info("Redis连接成功")
            
            # 初始化VLLM引擎
            vllm_config = self.engine_args.create_engine_config()
            self.llm = AsyncLLM.from_vllm_config(vllm_config)
            logger.info("VLLM引擎初始化成功")

            # 启动新的缓存组件
            # 注意：output_processor 和 request_manager_v2 不需要显式启动
            # 它们在处理请求时会自动激活
            
            # 启动调度器和输出处理器
            self.is_running = True
            self._scheduler_task = asyncio.create_task(self._scheduler_loop())
            self._output_handler_task = asyncio.create_task(self._output_handler_loop())
            
            logger.info(f"{self.__class__.__name__} 启动成功（包含新的缓存架构）")
            
        except Exception as e:
            logger.error(f"处理器启动失败: {e}")
            await self.stop()
            raise
    
    async def stop(self) -> None:
        """停止处理器"""
        if not self.is_running:
            return
        
        logger.info("正在停止处理器...")
        self.is_running = False
        
        # 停止调度器
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        
        # 停止输出处理器任务
        if self._output_handler_task:
            self._output_handler_task.cancel()
            try:
                await self._output_handler_task
            except asyncio.CancelledError:
                pass
        
        # 关闭新的缓存组件 - 暂时注释掉，因为模块不存在
        # try:
        #     await self.request_manager_v2.shutdown()
        #     await self.output_processor.shutdown()
        #     logger.info("新缓存组件已关闭")
        # except Exception as e:
        #     logger.error(f"关闭新缓存组件失败: {e}")
        
        # 清理所有活跃任务
        await self._cleanup_all_tasks()
        
        # 关闭vLLM引擎
        if self.llm:
            self.llm.shutdown()
            self.llm = None
            logger.info("vLLM引擎已关闭")
        
        # 断开Redis连接
        if self.redis_manager:
            await self.redis_manager.disconnect()
            logger.info("Redis连接已断开")
        
        # 保存向量搜索数据
        if self.vector_search_manager:
            self.vector_search_manager.save_data()
            logger.info("向量搜索数据已保存")
        
        logger.info("处理器已停止（包含新的缓存架构）")
    
    async def get_result(self, request_id: str) -> str:
        """获取请求结果"""
        request = await self.request_manager.get_request(request_id)
        
        if request is None:
            raise ValueError(f"请求 {request_id} 不存在")
        
        try:
            result = await request.get_result()
            if request.is_completed():
                await self.request_manager.remove_completed_request(request_id)
            return result
        except Exception as e:
            if request.is_completed() or request.status == RequestStatus.FAILED:
                await self.request_manager.remove_completed_request(request_id)
            raise e
    
    async def _scheduler_loop(self):
        """调度器主循环（兼容原有逻辑）"""
        while self.is_running:
            # 检查是否有空闲的运行槽位
            if self.request_manager.get_running_requests_count() < self.max_concurrent:
                request = await self.request_manager.get_from_waiting_queue()
                if request:
                    # 标记为运行中
                    await self.request_manager.add_running_request(request)
                    
                    # 异步处理请求
                    asyncio.create_task(self._process_request(request))
            
            await asyncio.sleep(0.01)  # 短暂休眠，避免CPU占用过高
    
    async def _output_handler_loop(self):
        """输出处理器主循环（基于 VLLM AsyncLLM 架构）- 暂时注释掉，因为模块不存在"""
        # while self.is_running:
        #     try:
        #         # 获取下一个待处理请求
        #         context = await self.request_manager_v2.get_next_request()
        #         if context:
        #             # 异步处理请求
        #             asyncio.create_task(self._process_request_v2(context))
        #         
        #         # 定期记录统计信息
        #         await self._log_stats_periodically()
        #         
        #         await asyncio.sleep(0.01)  # 短暂休眠，避免CPU占用过高
        #         
        #     except Exception as e:
        #         logger.error(f"输出处理器循环失败: {e}")
        #         # 传播错误到输出处理器
        #         self.output_processor.propagate_error(e)
        #         await asyncio.sleep(1.0)  # 错误时等待更长时间
        pass  # 暂时什么都不做
    
    async def _process_request(self, request: SimpleRequest) -> None:
        """处理请求"""
        try:
            async for output in self.llm.generate(
                prompt=request.prompt,
                sampling_params=request.sampling_params,
                request_id=request.request_id
            ):
                if output.outputs:
                    final_result_tokens = output.outputs[0].token_ids
                    final_result = output.outputs[0].text

            await self._update_cache_on_success(request, final_result, final_result_tokens)
            request.set_result(final_result)
            request.status = RequestStatus.COMPLETED
            
            # 请求完成后清理共享内存中的相关资源
            await self._cleanup_shared_memory_resources(request)
            
        except Exception as e:
            self._handle_standard_error(e, "请求处理")
            await self._handle_request_failure(request, e)
            # 失败时也需要清理共享内存资源
            await self._cleanup_shared_memory_resources(request)
    
    async def _process_request_v2(self, context) -> None:
        """处理请求（基于 RequestOutputCollector 模式）
        
        Args:
            context: 请求上下文，包含请求和输出收集器
        """
        request = context.request
        collector = context.collector
        
        try:
            # 检查缓存
            cache_hit = await self._check_cache_v2(request, collector)
            if cache_hit:
                await self.request_manager_v2.complete_request(request.request_id)
                return
            
            # 使用 VLLM AsyncLLM 生成
            async for output in self.llm.generate(
                prompt=request.prompt,
                sampling_params=request.sampling_params,
                request_id=request.request_id
            ):
                # 构建 RequestOutput
                request_output = self._build_request_output_from_vllm(output)
                
                # 推送到收集器
                await collector.put(request_output)
                
                # 如果完成，更新缓存
                if request_output.finished:
                    await self._update_cache_on_success_v2(request, request_output)
                    await self.request_manager_v2.complete_request(request.request_id, request_output)
                    break
            
        except Exception as e:
            logger.error(f"请求处理失败 [{request.request_id}]: {e}")
            
            # 构建错误输出
            from vllm.outputs import CompletionOutput
            error_completion = CompletionOutput(
                index=0,
                text="",
                token_ids=[],
                cumulative_logprob=None,
                logprobs=None,
                finish_reason="error"
            )
            error_output = RequestOutput(
                request_id=request.request_id,
                prompt="",
                prompt_token_ids=[],
                prompt_logprobs=None,
                outputs=[error_completion],
                finished=True,
                metrics=None
            )
            error_output.error = e
            
            # 推送错误到收集器
            await collector.put(error_output)
            
            # 标记请求失败
            await self.request_manager_v2.fail_request(request.request_id, e)
    
    async def _check_cache_v2(self, request: SimpleRequest, collector: CacheRequestOutputCollector) -> bool:
        """检查缓存（V2版本）
        
        Args:
            request: 请求对象
            collector: 输出收集器
            
        Returns:
            是否命中缓存
        """
        try:
            hash_id = self._generate_hash_id(request.prompt, request.sampling_params)
            cache_entry = await self.cache_manager.get(hash_id)
            
            if cache_entry and cache_entry.status == CacheStatus.COMPLETED:
                # 缓存命中
                self._stats.add_cache_hit()
                
                # 构建缓存结果输出
                from vllm.outputs import CompletionOutput
                cached_completion = CompletionOutput(
                    index=0,
                    text=cache_entry.result,
                    token_ids=cache_entry.result_tokens or [],
                    cumulative_logprob=None,
                    logprobs=None,
                    finish_reason="stop"
                )
                cached_output = RequestOutput(
                    request_id=request.request_id,
                    prompt="",
                    prompt_token_ids=[],
                    prompt_logprobs=None,
                    outputs=[cached_completion],
                    finished=True,
                    metrics=None
                )
                cached_output.metadata = {
                    'cache_hit': True,
                    'cache_timestamp': cache_entry.access_time
                }
                
                # 推送到收集器
                await collector.put(cached_output)
                
                # 更新缓存访问时间
                cache_entry.update_access_time()
                await self.cache_manager.put(hash_id, cache_entry)
                
                logger.debug(f"缓存命中: {request.request_id}")
                return True
            
            # 缓存未命中
            self._stats.add_cache_miss()
            
            # 创建新的缓存条目
            if not cache_entry:
                cache_entry = CacheEntry(
                    hash_id=hash_id,
                    prompt=request.prompt,
                    sampling_params=request.sampling_params,
                    status=CacheStatus.PROCESSING
                )
                await self.cache_manager.put(hash_id, cache_entry)
            
            return False
            
        except Exception as e:
            logger.error(f"检查缓存失败 [{request.request_id}]: {e}")
            return False
    
    def _build_request_output_from_vllm(self, vllm_output) -> RequestOutput:
        """从 VLLM 输出构建 RequestOutput
        
        Args:
            vllm_output: VLLM 的输出对象
            
        Returns:
            RequestOutput 对象
        """
        request_id = vllm_output.request_id
        finished = vllm_output.finished
        
        # 提取文本和token
        text = ""
        token_ids = []
        
        if vllm_output.outputs:
            first_output = vllm_output.outputs[0]
            text = getattr(first_output, 'text', '')
            token_ids = getattr(first_output, 'token_ids', [])
        
        from vllm.outputs import CompletionOutput
        completion_output = CompletionOutput(
            index=0,
            text=text,
            token_ids=token_ids,
            cumulative_logprob=None,
            logprobs=None,
            finish_reason="stop" if finished else None
        )
        
        result = RequestOutput(
            request_id=request_id,
            prompt="",
            prompt_token_ids=[],
            prompt_logprobs=None,
            outputs=[completion_output],
            finished=finished,
            metrics=None
        )
        result.metadata = {
            'timestamp': time.time(),
            'vllm_output_type': type(vllm_output).__name__
        }
        return result
    
    async def _update_cache_on_success_v2(self, request: SimpleRequest, request_output: RequestOutput) -> None:
        """成功时更新缓存（V2版本）
        
        Args:
            request: 请求对象
            request_output: 请求输出
        """
        try:
            hash_id = self._generate_hash_id(request.prompt, request.sampling_params)
            cache_entry = await self.cache_manager.get(hash_id)
            
            if cache_entry:
                # 从RequestOutput中提取文本和token信息
                output_text = ""
                output_tokens = []
                if request_output.outputs:
                    # 合并所有输出的文本和tokens
                    output_text = "".join([comp.text for comp in request_output.outputs])
                    output_tokens = []
                    for comp in request_output.outputs:
                        output_tokens.extend(comp.token_ids)
                
                # 更新缓存条目
                cache_entry.status = CacheStatus.COMPLETED
                cache_entry.result = output_text
                cache_entry.result_tokens = output_tokens
                cache_entry.update_access_time()
                
                # 保存到缓存
                await self.cache_manager.put(hash_id, cache_entry)
                
                # 更新向量搜索映射
                await self._update_vector_search_mapping(request, hash_id)
                
                logger.debug(f"缓存已更新: {hash_id}")
                
        except Exception as e:
            logger.error(f"更新缓存失败 [{request.request_id}]: {e}")
    
    async def _log_stats_periodically(self) -> None:
        """定期记录统计信息"""
        current_time = time.time()
        if current_time - self._last_stats_log_time > 60.0:  # 每分钟记录一次
            try:
                # 获取统计信息
                output_stats = self.output_processor.get_stats()
                request_stats = self.request_manager_v2.get_stats()
                
                logger.info(
                    f"缓存统计 - 命中率: {output_stats.cache_hit_rate:.2%}, "
                    f"活跃收集器: {self.output_processor.active_collectors_count}, "
                    f"队列大小: {self.request_manager_v2.queue_size}, "
                    f"吞吐量: {request_stats.throughput:.2f} req/s"
                )
                
                self._last_stats_log_time = current_time
                
            except Exception as e:
                logger.error(f"记录统计信息失败: {e}")
    
    async def _update_cache_on_success(self, request: SimpleRequest, result: str, result_tokens: List[int]) -> None:
        """成功时更新缓存"""
        async with self._lock:
            hash_id = request.hash_id
            cache_entry = await self.cache_manager.get(hash_id)
            
            if cache_entry:
                # 更新缓存状态
                cache_entry.status = CacheStatus.COMPLETED
                cache_entry.result = result
                cache_entry.result_tokens = result_tokens  # 使用新的字段名和列表格式
                cache_entry.update_access_time()
                
                # 通知等待的相同请求
                waiting_requests = cache_entry.get_waiting_requests()
                completed_request_ids = []  # 只包含等待的请求，当前请求在外层处理
                
                for waiting_request_id in waiting_requests:
                    waiting_request = self.request_manager.all_requests.get(waiting_request_id)
                    if waiting_request:
                        waiting_request.status = RequestStatus.COMPLETED
                        waiting_request.set_result(result)
                        completed_request_ids.append(waiting_request_id)
                
                # 清空等待队列并更新缓存
                cache_entry.clear_waiting_requests()
                await self.cache_manager.put(hash_id, cache_entry)
                     
                # 更新向量搜索映射
                await self._update_vector_search_mapping(request, hash_id)
    
    async def _handle_request_failure(self, request: SimpleRequest, error: Exception) -> None:
        """处理请求失败"""
        request.set_error(error)
        request.status = RequestStatus.FAILED
        
        # 更新缓存状态为失败
        async with self._lock:
            hash_id = request.hash_id
            cache_entry = await self.cache_manager.get(hash_id)
            
            if cache_entry:
                # 通知等待的相同请求失败
                waiting_requests = cache_entry.get_waiting_requests()
                failed_request_ids = []  # 只包含等待的请求，当前请求在外层处理
                
                for waiting_request_id in waiting_requests:
                    waiting_request = self.request_manager.all_requests.get(waiting_request_id)
                    if waiting_request:
                        waiting_request.status = RequestStatus.FAILED
                        waiting_request.set_error(error)
                        failed_request_ids.append(waiting_request_id)
                
                # 从缓存中移除失败的条目
                await self.cache_manager.remove(hash_id)
                
                # 失败处理完成后，立即清理等待的失败请求
                for failed_request_id in failed_request_ids:
                    await self.request_manager.remove_completed_request(failed_request_id)
        
        # 清理当前失败的请求
        await self.request_manager.remove_completed_request(request.request_id)
    
    async def add_request_v2(
        self, 
        prompt: str, 
        sampling_params: SamplingParams,
        priority: int = Priority.NORMAL.value
    ) -> AsyncGenerator[RequestOutput, None]:
        """添加请求到处理器（基于 RequestOutputCollector 模式）
        
        Args:
            prompt: 输入提示
            sampling_params: 采样参数
            priority: 请求优先级
            
        Yields:
            RequestOutput: 请求输出流
        """
        request_id = str(uuid.uuid4())
        
        try:
            # 创建请求对象
            request = SimpleRequest(
                request_id=request_id,
                prompt=prompt,
                sampling_params=sampling_params,
                hash_id=self._generate_hash_id(prompt, sampling_params)
            )
            
            # 创建输出收集器
            collector = CacheRequestOutputCollector(
                request_id=request_id,
                timeout=self.handler_config.features.request_timeout
            )
            
            # 添加到请求管理器
            await self.request_manager_v2.add_request(request, collector, priority)
            
            # 流式返回结果
            finished = False
            while not finished:
                try:
                    # 获取输出（带超时）
                    output = await asyncio.wait_for(
                        collector.get(),
                        timeout=self.handler_config.features.request_timeout
                    )
                    
                    finished = output.finished
                    yield output
                    
                    # 如果有错误，抛出异常
                    if output.error:
                        raise output.error
                        
                except asyncio.TimeoutError:
                    # 超时处理
                    await self.request_manager_v2.abort_request(request_id)
                    raise asyncio.TimeoutError(f"请求超时: {request_id}")
                    
        except asyncio.CancelledError:
            # 请求被取消
            await self.request_manager_v2.abort_request(request_id)
            logger.info(f"请求被取消: {request_id}")
            raise
            
        except Exception as e:
            # 其他错误
            await self.request_manager_v2.abort_request(request_id)
            logger.error(f"请求处理失败: {request_id}, 错误: {e}")
            raise
    
    @abstractmethod
    async def add_request(self, prompt: str, sampling_params: SamplingParams) -> str:
        """添加请求到处理器（子类需要实现，兼容原有接口）"""
        pass
    
    def _generate_hash_id(self, prompt: str, sampling_params: SamplingParams) -> str:
        """统一的哈希ID生成方法"""
        return self.request_hasher.compute_hash_id(prompt, sampling_params)
    
    @abstractmethod
    async def _update_vector_search_mapping(self, request: SimpleRequest, hash_id: str) -> None:
        """更新向量搜索映射（子类需要实现）"""
        pass
    
    async def _standard_vector_search_mapping(self, request: SimpleRequest, hash_id: str) -> None:
        """标准化的向量搜索映射更新"""
        if not self.handler_config.features.enable_vector_search:
            return
            
        try:
            # 检查是否有NLP结果
            if hasattr(request, 'nlp_result') and request.nlp_result:
                # 使用NLP结果进行映射，传入nlp_result参数
                await self.vector_search_manager.add_question_mapping(
                    request.prompt, hash_id, None
                )
            else:
                # 使用基础映射
                await self.vector_search_manager.add_question_mapping(
                    request.prompt, hash_id
                )
            
            if self.handler_config.features.enable_detailed_logging:
                logger.info(f"向量搜索映射已更新: {hash_id}")
        except Exception as e:
            self._handle_standard_error(e, "更新向量搜索映射")
    
    def _create_task(self, coro, task_name: str) -> asyncio.Task:
        """创建并管理异步任务"""
        # 检查任务数量限制
        if len(self._active_tasks) >= self.handler_config.features.max_concurrent_tasks:
            logger.warning(f"活跃任务数量已达上限: {len(self._active_tasks)}")
        
        task = asyncio.create_task(coro)
        self._active_tasks[task_name] = task
        
        # 添加完成回调以清理任务
        def cleanup_task(t):
            self._active_tasks.pop(task_name, None)
            if self.handler_config.features.enable_detailed_logging:
                logger.debug(f"任务已完成并清理: {task_name}")
        task.add_done_callback(cleanup_task)
        
        return task
    
    def _cancel_task(self, task_name: str) -> bool:
        """取消指定的异步任务"""
        task = self._active_tasks.get(task_name)
        if task and not task.done():
            task.cancel()
            # 记录任务取消
            pass
            return True
        return False
    
    async def _cleanup_all_tasks(self) -> None:
        """清理所有活跃的异步任务"""
        if self._active_tasks:
            logger.info(f"正在清理 {len(self._active_tasks)} 个活跃任务")
            for task_name, task in list(self._active_tasks.items()):
                if not task.done():
                    task.cancel()
            
            # 等待所有任务完成或被取消
            await asyncio.gather(*self._active_tasks.values(), return_exceptions=True)
            self._active_tasks.clear()
    
    def _handle_standard_error(self, error: Exception, context: str) -> None:
        """标准化错误处理"""
        error_msg = f"{context}: {str(error)}"
        
        # 根据错误类型进行分类处理
        if isinstance(error, asyncio.CancelledError):
            logger.debug(f"任务被取消 - {error_msg}")
        elif isinstance(error, ConnectionError):
            logger.error(f"连接错误 - {error_msg}")
        elif isinstance(error, TimeoutError):
            logger.warning(f"超时错误 - {error_msg}")
        elif isinstance(error, ValueError):
            logger.warning(f"参数错误 - {error_msg}")
        else:
            logger.error(f"未知错误 - {error_msg}")
    
    
    async def _cleanup_shared_memory_resources(self, request: SimpleRequest) -> None:
        """清理共享内存中与当前请求相关的资源"""
        try:
            if hasattr(self.request_manager, 'memory_manager') and self.request_manager.memory_manager:
                # 清理当前请求的映射
                self.request_manager.memory_manager.remove_request_mapping(request.hash_id)
                
                if self.handler_config.features.enable_detailed_logging:
                    logger.debug(f"已清理请求 {request.hash_id} 的共享内存资源")
        except Exception as e:
            logger.warning(f"清理共享内存资源失败 [{request.request_id}]: {e}")
    
    async def _find_similar_requests(self, request: SimpleRequest) -> List[str]:
        """查找相似的历史请求"""
        try:
            # 使用向量搜索查找相似请求
            if self.vector_search_manager and self.handler_config.features.enable_vector_search:
                similar_hash_ids = await self.vector_search_manager.search_similar_questions(
                    request.prompt, top_k=5  # 获取最相似的5个请求
                )
                return similar_hash_ids
            return []
        except Exception as e:
            logger.warning(f"查找相似请求失败 [{request.request_id}]: {e}")
            return []
    
    async def _safe_execute(self, coro, context: str, default_return=None):
        """安全执行协程，统一错误处理"""
        try:
            return await coro
        except Exception as e:
            self._handle_standard_error(e, context)
            return default_return
    