"""VLLM处理器模块"""

import asyncio
import uuid
import time
from typing import Dict, Any, Optional, List, Tuple

from loguru import logger
from vllm.sampling_params import SamplingParams
from ..models.request import SimpleRequest
from ..models.cache import CacheEntry
from ..models.enums import RequestStatus, CacheStatus
from ..cache import MultiLevelCacheManager
from ..utils.similarity_search_helper import SimilaritySearchHelper
from ..utils.request_manager import RequestManager
from ..utils.hasher import RequestHasher
from ..config.settings import VLLMConfig
from ..config.handler_config import HandlerConfig
from .base_vllm_handler import BaseVLLMHandler
from ..nlp.async_conjunction_extractor import AsyncAdvancedConjunctionExtractor as AsyncConjunctionExtractor
# 移除全局资源管理器依赖，改为使用 RequestManager 的共享内存功能



# 常量定义
class NLPConstants:
    """NLP处理相关常量"""
    DEFAULT_MAX_CONCURRENT = 50
    DEFAULT_TOP_K = 10
    DEFAULT_SIMILARITY_THRESHOLD = 0.7
    SUBSENTENCE_TOP_K = 10


class NLPEnhancedVLLMHandler(BaseVLLMHandler):
    """集成NLP连接词提取功能的增强VLLM处理器
    
    特性:
    - 异步连接词提取和子句处理
    - 高并发NLP处理支持
    - 结合原始请求和分解文本进行相似度查询
    - 智能缓存策略优化
    - 多层次缓存支持
    - 请求级相似度上下文管理
    - 可配置的并发控制
    """
    
    def __init__(self, config: VLLMConfig, cache_manager: MultiLevelCacheManager, 
                 similarity_search_helper: SimilaritySearchHelper, 
                 handler_config: HandlerConfig = None,
                 nlp_max_concurrent: int = NLPConstants.DEFAULT_MAX_CONCURRENT):
        # 移除全局资源管理器设置，改为直接使用各组件的功能
        
        super().__init__(config, cache_manager, similarity_search_helper, handler_config)
        
        # 异步NLP连接词提取器
        self.conjunction_extractor = AsyncConjunctionExtractor()
        
        # 设置NLP处理的最大并发数（仅在提取器层面控制）
        self.nlp_max_concurrent = nlp_max_concurrent
        self.conjunction_extractor.set_max_concurrent_tasks(nlp_max_concurrent)

        
        logger.info(f"NLP增强VLLM处理器初始化完成，NLP最大并发数: {nlp_max_concurrent}")
    
    
    def set_nlp_max_concurrent(self, max_concurrent: int) -> None:
        """设置NLP处理的最大并发数。
        
        Args:
            max_concurrent: 最大并发数
        """
        self.nlp_max_concurrent = max_concurrent
        self.conjunction_extractor.set_max_concurrent_tasks(max_concurrent)
        logger.info(f"NLP最大并发数已更新为: {max_concurrent}")
    
    
    def _create_default_nlp_result(self, prompt: str) -> Dict[str, Any]:
        """创建默认的NLP处理结果
        
        Args:
            prompt: 输入文本
            
        Returns:
            默认NLP结果字典
        """
        return {
            'original_text': prompt,
            'extraction_results': [],
            'subsentences': [prompt],
            'conjunctions': [],
            'has_conjunctions': False
        }
    
    def _build_nlp_result_from_extraction(self, prompt: str, extraction_result: Dict[str, Any]) -> Dict[str, Any]:
        """从提取结果构建NLP结果
        
        Args:
            prompt: 原始文本
            extraction_result: 提取结果
            
        Returns:
            构建的NLP结果字典
        """
        subsentences = []
        conjunctions = []
        has_conjunctions = False
        extraction_results = []
        
        if extraction_result:
            extraction_results = [extraction_result]
            
            # 提取变体作为子句
            if 'variants' in extraction_result:
                subsentences.extend(extraction_result['variants'])
            
            # 提取连接词
            if 'conjunctions' in extraction_result:
                conjunctions.extend(extraction_result['conjunctions'])
                if extraction_result['conjunctions']:
                    has_conjunctions = True
        
        # 如果没有子句，使用原始文本
        if not subsentences:
            subsentences = []
        
        return {
            'original_text': prompt,
            'extraction_results': extraction_results,
            'subsentences': subsentences,
            'conjunctions': conjunctions,
            'has_conjunctions': has_conjunctions
        }
    
    async def _process_text_with_nlp_async(self, prompt: str, request_id: str = None) -> Dict[str, Any]:
        """使用异步NLP处理文本，提取连接词和子句。
        
        Args:
            prompt: 输入文本
            request_id: 请求ID（用于日志追踪）
            
        Returns:
            NLP处理结果字典
        """
        try:
            # 直接使用异步连接词提取器（内部已有并发控制）
            extraction_result = await self.conjunction_extractor.extract_async(prompt, id=request_id)
            
            # 构造与原process_for_cache方法兼容的返回格式
            nlp_result = self._build_nlp_result_from_extraction(prompt, extraction_result)
            
            if self.handler_config.features.enable_detailed_logging:
                logger.debug(f"异步NLP处理结果 [{request_id}]: {nlp_result}")
            return nlp_result
            
        except Exception as e:
            logger.error(f"异步NLP处理失败 [{request_id}]: {e}")
            return self._create_default_nlp_result(prompt)
    

    async def _store_similarity_context(self, request_id: str, request_hash_id: str, similar_answers: List[Dict[str, Any]]) -> None:
        """直接将相似度上下文存储到共享内存中，不通过请求管理器
        
        Args:
            request_id: 请求ID
            request_hash_id: 请求的哈希ID
            similar_answers: 相似答案列表
        """
        try:
            # 提取相似请求的hash_id列表
            similar_hash_ids = [answer['hash_id'] for answer in similar_answers]
            
            # 确保共享内存管理器存在
            if not hasattr(self, '_shared_memory_manager') or self._shared_memory_manager is None:
                from .similar_request_memory import SimilarRequestMemoryManager, SimilarRequestMemoryConfig
                config = SimilarRequestMemoryConfig(
                    request_mapping_memory_size=8 * 1024 * 1024,  # 8MB
                    token_mapping_memory_size=8 * 1024 * 1024,    # 8MB
                    max_entries=5000,
                    request_mapping_shared_name="vllm_request_mappings",
                    token_mapping_shared_name="vllm_token_mappings"
                )
                self._shared_memory_manager = SimilarRequestMemoryManager(config)
            
            # 直接存储请求映射到共享内存
            self._shared_memory_manager.store_similar_request_mapping(
                request_hash_id, similar_hash_ids
            )
            
            # 批量准备token映射数据
            token_mappings = {}
            for similar_hash_id in similar_hash_ids:
                try:
                    cache_entry = await self.cache_manager.get(similar_hash_id)
                    if cache_entry and cache_entry.status.name == 'COMPLETED':
                        # 使用新的result_tokens字段获取token序列
                        if hasattr(cache_entry, 'result_tokens') and cache_entry.result_tokens is not None:
                            token_mappings[similar_hash_id] = cache_entry.result_tokens
                except Exception as e:
                    logger.warning(f"准备相似答案token数据失败 {similar_hash_id}: {e}")
                    continue
            
            # 批量存储token映射到共享内存
            if token_mappings:
                self._shared_memory_manager.store_multiple_answer_tokens(token_mappings)
            
            logger.debug(f"成功将相似度上下文直接存储到共享内存 [{request_id}]: {len(similar_hash_ids)} 个相似答案，{len(token_mappings)} 个token序列")
            
        except Exception as e:
            logger.error(f"直接存储相似度上下文到共享内存失败 [{request_id}]: {e}")

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
    
    async def _process_nlp_task(self, nlp_task, request_id: str, request: SimpleRequest) -> Optional[Dict[str, Any]]:
        """处理NLP任务
        
        Args:
            nlp_task: NLP任务
            request_id: 请求ID
            request: 请求对象
            
        Returns:
            NLP处理结果或None
        """
        if not nlp_task:
            return None
        
        try:
            nlp_result = await nlp_task
            # 将NLP结果存储到请求对象中，供后续使用
            request.nlp_result = nlp_result
            if self.handler_config.features.enable_detailed_logging:
                logger.debug(f"异步NLP处理完成 [{request_id}]: 找到连接词 {nlp_result.get('has_conjunctions', False)}")
            return nlp_result
        except asyncio.CancelledError:
            if self.handler_config.features.enable_detailed_logging:
                logger.debug(f"异步NLP任务被取消 [{request_id}]")
        except Exception as e:
            self._handle_standard_error(e, f"异步NLP分析 [{request_id}]")
        
        return None
    
    def _deduplicate_and_sort_answers(self, all_similar_answers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """去重并排序相似答案
        
        Args:
            all_similar_answers: 所有相似答案列表
            
        Returns:
            去重并排序后的答案列表
        """
        if not all_similar_answers:
            return []
        
        # 去重并按相似度排序
        unique_answers = {}
        for answer in all_similar_answers:
            hash_id_key = answer['hash_id']
            if hash_id_key not in unique_answers or answer['similarity'] > unique_answers[hash_id_key]['similarity']:
                unique_answers[hash_id_key] = answer
        
        return sorted(unique_answers.values(), key=lambda x: x['similarity'], reverse=True)
    
    async def add_request(self, prompt: str, sampling_params: SamplingParams) -> str:
        """添加请求到处理器，使用异步NLP分析和原始问题相似度搜索"""
        start_time = time.time()
        
        # 1. 生成请求ID（先生成，用于后续处理）
        request_id = str(uuid.uuid4())
        
        try:

            # 2. 生成统一哈希ID
            hash_id = self._generate_hash_id(prompt, sampling_params)
            
            # 3. 首先尝试精确缓存匹配
            cache_entry = await self.cache_manager.get(hash_id)
            request = SimpleRequest(
                request_id=request_id,
                prompt=prompt,
                sampling_params=sampling_params,
                hash_id=hash_id
            )
            
            if cache_entry:
                if cache_entry.is_completed():
                    # 已完成的缓存，直接返回结果
                    logger.info(f"缓存命中，直接返回结果: {request_id} -> {hash_id}")
                    request.status = RequestStatus.COMPLETED
                    request.set_result(cache_entry.result)
                    await self.request_manager.add_request(request)
                    return request_id
                elif cache_entry.is_processing():
                    # 正在处理中的缓存，将当前请求加入等待队列
                    cache_entry.add_waiting_request(request_id)
                    await self.cache_manager.put(hash_id, cache_entry)
                    logger.info(f"发现相同请求正在处理，加入等待队列: {request_id} -> {hash_id}")
                    # 将请求添加到请求管理器，但不加入处理队列
                    request.status = RequestStatus.WAITING_FOR_DUPLICATE
                    await self.request_manager.add_request(request)
                    return request_id
            else:
                # 4. 创建新的PROCESSING状态缓存条目，占位以便后续相同请求能够发现
                cache_entry = CacheEntry(
                    hash_id=hash_id,
                    status=CacheStatus.PROCESSING,
                    original_request_id=request_id,
                )
                await self.cache_manager.put(hash_id, cache_entry)
                logger.info(f"创建新的处理中缓存条目: {request_id} -> {hash_id}")
            
            # 5. 并行执行异步NLP处理和初步相似度搜索
            task_prefix = f"req_{request_id}"
            nlp_task = None
            if self.handler_config.features.enable_nlp_enhancement:
                nlp_task = self._create_task(
                    self._process_text_with_nlp_async(prompt, request_id),
                    f"{task_prefix}_nlp"
                )
            
            # 6. 同时进行原始问题的相似度搜索（如果向量搜索可用）
            initial_similarity_task = None
            if self.vector_search_manager:
                initial_similarity_task = self._create_task(
                    self.similarity_search_helper.try_layered_similarity_search(
                        prompt, request, self.cache_manager  # 暂时传入None，后续会更新
                    ),
                    f"{task_prefix}_similarity"
                )
            
            
            # 7. 获取初步相似度搜索结果
            direct_reuse, initial_similar_answers = await self._process_initial_similarity_search(
                initial_similarity_task, request_id, task_prefix
            )
            
            if direct_reuse:
                await self.request_manager.add_request(request)
                return request_id
            await self.request_manager.add_request(request)
            # 8. 等待异步NLP处理完成，然后进行子句级别的相似度搜索
            nlp_result = await self._process_nlp_task(nlp_task, request_id, request)
            # 9. 收集所有相似度结果
            all_similar_answers = list(initial_similar_answers)  # 复制初始结果
            
            if nlp_result and nlp_result.get('has_conjunctions', False) and self.vector_search_manager:
                try:
                    # 仅对子句进行搜索，不包含原始文本
                    subsentence_results = await self._perform_subsentence_similarity_search(
                        nlp_result, top_k=NLPConstants.SUBSENTENCE_TOP_K, 
                        similarity_threshold=NLPConstants.DEFAULT_SIMILARITY_THRESHOLD
                    )
                    
                    # 将结果添加到all_similar_answers中
                    all_similar_answers.extend(subsentence_results)
                    
                    logger.debug(f"子句搜索完成 [{request_id}]，收集到 {len(subsentence_results)} 个相似答案")
                    
                except Exception as e:
                    logger.error(f"子句搜索失败 [{request_id}]: {e}")
                
            
            
            # 11. 处理收集到的相似度结果并更新请求的相似信息
            if all_similar_answers:
                final_similar_answers = self._deduplicate_and_sort_answers(all_similar_answers)
                
                await self._store_similarity_context(request_id, final_similar_answers)
                
                logger.debug(f"完成相似度搜索和上下文存储 [{request_id}]: {len(final_similar_answers)} 个去重后的相似答案")
            
            # 12. 将请求状态设置为等待处理
            await self.request_manager.update_request_status(request_id, RequestStatus.WAITING)
            processing_time = time.time() - start_time
            logger.info(f"请求预处理完成 [{request_id}]: {processing_time:.3f}s")
            
            return request_id
            
        except Exception as e:
            logger.error(f"添加请求失败 [{request_id}]: {e}")
            # 清理可能创建的缓存条目
            try:
                await self.cache_manager.remove(hash_id)
            except:
                pass
            raise e
    

    async def _update_vector_search_mapping(self, request: SimpleRequest, hash_id: str) -> None:
        """更新向量搜索映射（使用标准化方法）"""
        try:
            await self._standard_vector_search_mapping(request, hash_id)
            logger.debug(f"已添加请求到向量搜索映射: {hash_id}")
        except Exception as e:
            logger.error(f"更新向量搜索映射失败: {e}")
    
    def get_nlp_analysis(self, request_id: str) -> Optional[Dict[str, Any]]:
        """获取请求的NLP分析结果"""
        request = self.request_manager.all_requests.get(request_id)
        if request and hasattr(request, 'nlp_result'):
            return request.nlp_result
        return None
    
    def _deduplicate_search_results(self, results: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """去重搜索结果
        
        Args:
            results: 搜索结果列表
            top_k: 返回的最大结果数
            
        Returns:
            去重后的结果列表
        """
        unique_results = {}
        for result in results:
            hash_id = result['hash_id']
            if hash_id not in unique_results or result['similarity'] > unique_results[hash_id]['similarity']:
                unique_results[hash_id] = result
        
        final_results = list(unique_results.values())
        final_results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return final_results[:top_k]
    
    async def _perform_subsentence_similarity_search(self, nlp_result: Dict[str, Any], 
                                                    top_k: int = NLPConstants.DEFAULT_TOP_K, 
                                                    similarity_threshold: float = NLPConstants.DEFAULT_SIMILARITY_THRESHOLD) -> List[Dict[str, Any]]:
        """执行子句相似度搜索（仅搜索分解后的内容，使用批量检索）
        
        Args:
            nlp_result: NLP分析结果
            top_k: 返回结果数量
            similarity_threshold: 相似度阈值
            
        Returns:
            子句相似度搜索结果列表
        """
        if not self.vector_search_manager:
            return []
        
        try:
            results = []
            
            # 使用批量搜索处理多个子句
            if nlp_result.get('has_conjunctions', False):
                subsentences = nlp_result.get('subsentences', [])

                
                if subsentences:
                    # 使用批量搜索API
                    query_texts = subsentences
                    query_ids = [f"subsentence_{i}" for i in range(len(subsentences))]
                    
                    batch_results = await self.vector_search_manager.batch_search_similar_questions(
                        query_texts=query_texts,
                        query_ids=query_ids,
                        k=top_k,
                        similarity_threshold=similarity_threshold
                    )
                    
                    for i, query_results in enumerate(batch_results):
                        if query_results:
                            for result in query_results:
                                cache_entry = await self.cache_manager.get(result.get('id', ''))
                                if cache_entry and cache_entry.status == CacheStatus.COMPLETED:
                                    result_item = {
                                        'hash_id': result.get('id'),
                                        'similarity': result.get('similarity', 0.0)
                                    }
                                    results.append(result_item)
            
            # 去重并按相似度排序
            return self._deduplicate_search_results(results, top_k)
            
        except Exception as e:
            logger.error(f"子句批量相似度搜索失败: {e}")
            return []
