"""相似性搜索助手"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from ..models.request import SimpleRequest
from ..models.cache import CacheEntry
from ..models.enums import RequestStatus
from ..cache.enhanced_vector_search import EnhancedVectorSearchManager

logger = logging.getLogger(__name__)


@dataclass
class SimilarityResult:
    """相似性搜索结果"""
    request_id: str
    similarity_score: float
    cache_entry: Optional[CacheEntry] = None
    is_exact_match: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'request_id': self.request_id,
            'similarity_score': self.similarity_score,
            'cache_entry': self.cache_entry.to_dict() if self.cache_entry else None,
            'is_exact_match': self.is_exact_match
        }


class SimilaritySearchHelper:
    """相似性搜索助手"""
    
    def __init__(self, 
                 high_similarity_threshold: float = 0.95,
                 medium_similarity_threshold: float = 0.7,
                 max_similar_requests: int = 10):
        self.high_similarity_threshold = high_similarity_threshold
        self.medium_similarity_threshold = medium_similarity_threshold
        self.max_similar_requests = max_similar_requests
        
        # 向量搜索管理器
        self.vector_search: Optional[EnhancedVectorSearchManager] = None
        
        # 初始化状态
        self._initialized = False
        
        # 统计信息
        self._stats = {
            'total_searches': 0,
            'high_similarity_hits': 0,
            'medium_similarity_hits': 0,
            'cache_reuse_count': 0,
            'avg_search_time': 0.0
        }
    
    def initialize(self, vector_search_config: Optional[Dict[str, Any]] = None) -> bool:
        """初始化相似性搜索助手"""
        try:
            
            # 使用正确的参数创建 EnhancedVectorSearchManager
            self.vector_search = EnhancedVectorSearchManager(
                config=vector_search_config
            )
            self._initialized = True
            logger.info("相似性搜索助手初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"初始化相似性搜索助手失败: {e}")
            return False
    
    async def cleanup(self) -> None:
        """清理资源"""
        try:
            if self.vector_search:
                await self.vector_search.cleanup()
            self._initialized = False
            logger.info("相似性搜索助手已清理")
            
        except Exception as e:
            logger.error(f"清理相似性搜索助手失败: {e}")
    
    async def try_layered_similarity_search(self, prompt: str, cache_manager) -> Tuple[bool, List[Dict[str, Any]]]:
        """分层相似度搜索：相似度>0.95直接复用，相似度>0.7获取答案用于后续推理
        
        Args:
            prompt: 查询文本
            request: 请求对象
            cache_manager: 缓存管理器
            
        Returns:
            Tuple[bool, List[Dict]]: (是否直接复用, 相似答案列表)
        """
        if not self.vector_search:
            return False, []

        try:
            # 使用向量搜索查找相似文本
            similar_results = await self.vector_search.search_similar_questions_with_scores(
                query_text=prompt,
                k=10,
                similarity_threshold=0.7
            )
            
            high_similarity_results = []  # 相似度>0.95的结果
            medium_similarity_results = []  # 相似度0.7-0.95的结果
            
            # 分层处理相似度结果
            for result in similar_results:
                similarity = result['similarity']
                request_id = result['id']
                
                # 获取缓存条目
                cache_entry = await cache_manager.get(request_id)
                if not cache_entry:
                    continue
                
                result_with_mapping = {
                    'hash_id': request_id,
                    'similarity': similarity,
                    'cache_entry': cache_entry
                }
                
                if similarity >= 0.95:
                    high_similarity_results.append(result_with_mapping)
                elif similarity >= 0.7:
                    medium_similarity_results.append(result_with_mapping)
            
            # 如果有相似度>0.95的结果，直接复用第一个
            if high_similarity_results:
                best_result = high_similarity_results[0]
                logger.info(f"找到高相似度结果(相似度: {best_result['similarity']:.3f})，直接复用: {best_result['hash_id']}")
                return True, high_similarity_results
            
            # 如果只有中等相似度结果，返回这些结果用于后续推理
            if medium_similarity_results:
                logger.info(f"找到 {len(medium_similarity_results)} 个中等相似度结果，将用于后续推理")
                return False, medium_similarity_results
            
            return False, []

        except Exception as e:
            logger.error(f"分层相似度搜索失败: {e}")
            return False, []
    
    def _update_search_stats(self, results: List[SimilarityResult], search_time: float) -> None:
        """更新搜索统计"""
        self._stats['total_searches'] += 1
        
        # 更新平均搜索时间
        total_searches = self._stats['total_searches']
        current_avg = self._stats['avg_search_time']
        self._stats['avg_search_time'] = (
            (current_avg * (total_searches - 1) + search_time) / total_searches
        )

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        vector_stats = self.vector_search.get_stats() if self.vector_search else {}
        
        return {
            'initialized': self._initialized,
            'total_cached_requests': len(self.request_cache),
            'high_similarity_threshold': self.high_similarity_threshold,
            'medium_similarity_threshold': self.medium_similarity_threshold,
            'max_similar_requests': self.max_similar_requests,
            'search_stats': self._stats.copy(),
            'vector_search_stats': vector_stats
        }
    
    def is_ready(self) -> bool:
        """检查是否准备就绪"""
        return (
            self._initialized and 
            self.vector_search is not None and 
            self.vector_search.is_ready()
        )
    
    async def clear_cache(self) -> None:
        """清空缓存"""
        try:
            if self.vector_search:
                await self.vector_search.clear_index()
            
            self.request_cache.clear()
            
            # 重置统计
            self._stats = {
                'total_searches': 0,
                'high_similarity_hits': 0,
                'medium_similarity_hits': 0,
                'cache_reuse_count': 0,
                'avg_search_time': 0.0
            }
            
            logger.info("相似性搜索缓存已清空")
            
        except Exception as e:
            logger.error(f"清空相似性搜索缓存失败: {e}")
    