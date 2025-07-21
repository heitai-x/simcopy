import os
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Union
from loguru import logger

from ..retriever.enhanced_document_retriever import EnhancedDocumentRetriever
from ..config.settings import VectorSearchConfig

# 导入配置管理器
try:
    from ..config import get_config
except ImportError:
    # 如果导入失败，使用默认配置
    logger.warning("无法导入配置管理器，将使用默认配置")
    get_config = None


class EnhancedVectorSearchManager:
    """增强的向量搜索管理器
    
    基于FAISS和Sentence Transformers的高性能向量搜索系统
    支持GPU加速、异步操作和智能缓存
    """
    
    def __init__(self, config: VectorSearchConfig):
        """初始化增强向量搜索管理器
        
        Args:
            config: 向量搜索配置
            use_gpu: 是否使用GPU加速
        """
        self.config = config
        self.use_gpu = self.config.use_gpu
        self.retriever: Optional[EnhancedDocumentRetriever] = None
        self.enabled = config.enabled
        print("enabled:",self.enabled)
        self.similarity_threshold = config.similarity_threshold
        if self.enabled:
            self._init_retriever()
        # 性能统计
        self.stats = {
            "total_searches": 0,
            "successful_searches": 0,
            "cache_hits": 0,
            "gpu_searches": 0,
            "cpu_searches": 0,
            "average_search_time": 0.0,
            "total_documents": 0
        }
    
    def _load_config(self, config_path: Optional[str] = None):
        """
        加载配置
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置对象或默认配置
        """
        if get_config is not None:
            try:
                return get_config(config_path)
            except Exception as e:
                logger.warning(f"加载配置失败: {e}，使用默认配置")
        
        # 返回默认配置
        class DefaultConfig:
            def __init__(self):
                self.model = type('ModelConfig', (), {
                    'name': 'sentence-transformers/all-MiniLM-L6-v2',
                    'device': 'auto',
                    'batch_size': 32,
                })()
                self.search = type('SearchConfig', (), {
                    'similarity_threshold': 0.7
                })()
        
        return DefaultConfig()
    
    def _init_retriever(self) -> None:
        """异步初始化文档检索器"""
        try:
            # 创建数据目录
            data_dir = "llmcache/data/enhanced_vector_search"
            os.makedirs(data_dir, exist_ok=True)
            
            metadata_path = os.path.join(data_dir, "metadata.json")
            index_path = os.path.join(data_dir, "vector_index_cosine.faiss")
            
            # 使用增强的文档检索器
            self.retriever = EnhancedDocumentRetriever(
                metadata_path=metadata_path,
                index_path=index_path,
                model_name=self.config.model_name,
                auto_init=True,
                enable_logging=True,
                use_gpu=self.use_gpu
            )
            
            

        except Exception as e:
            logger.error(f"增强向量搜索初始化失败: {e}")
            self.enabled = False
    
    async def search_similar_questions(self, 
                                     query_text: str, 
                                     k: int = 5, 
                                     similarity_threshold: Optional[float] = None,
                                     exclude_query_id: Optional[str] = None) -> List[str]:
        """搜索相似问题并返回哈希ID列表
        
        Args:
            query_text: 查询文本
            k: 返回结果数量
            similarity_threshold: 相似度阈值
            exclude_query_id: 要排除的查询ID
            
        Returns:
            相似问题的哈希ID列表
        """
        if not self.enabled or not self.retriever:
            return []
        
        threshold = similarity_threshold or self.similarity_threshold
        self.stats["total_searches"] += 1
        
        try:
            import time
            start_time = time.time()
            
            # 执行相似度搜索
            results = await self.retriever.search_async(
                query_text=query_text,
                query_id=exclude_query_id,
                k=k,
                similarity_threshold=threshold
            )
            
            search_time = time.time() - start_time
            self._update_search_stats(search_time, len(results) > 0)
            
            # 提取相似问题的哈希ID
            similar_hash_ids = []
            for result in results:
                similar_hash_id = result['id']
                similarity = result['similarity']
                similar_hash_ids.append(similar_hash_id)
                logger.debug(f"找到相似问题 (相似度: {similarity:.3f}): {similar_hash_id}")
            
            return similar_hash_ids
            
        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            return []
    
    async def search_similar_questions_with_scores(self, 
                                                  query_text: str, 
                                                  k: int = 10, 
                                                  similarity_threshold: Optional[float] = None,
                                                  exclude_query_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """搜索相似问题并返回详细信息（包含相似度分数）
        
        Args:
            query_text: 查询文本
            k: 返回结果数量
            similarity_threshold: 相似度阈值
            exclude_query_id: 要排除的查询ID
            
        Returns:
            包含详细信息的结果列表
        """
        if not self.enabled or not self.retriever:
            return []
        
        threshold = similarity_threshold or self.similarity_threshold
        self.stats["total_searches"] += 1
        
        try:
            import time
            start_time = time.time()
            
            # 执行相似度搜索
            results = await self.retriever.search_async(
                query_text=query_text,
                query_id=exclude_query_id,
                k=k,
                similarity_threshold=threshold
            )
            
            search_time = time.time() - start_time
            self._update_search_stats(search_time, len(results) > 0)
            
            # 返回详细结果
            detailed_results = []
            for result in results:
                detailed_results.append({
                    'id': result['id'],
                    'text': result['text'],
                    'similarity': result['similarity'],
                    'index': result['index']
                })
            
            return detailed_results
            
        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            return []
    
    async def batch_search_similar_questions(self,
                                           query_texts: List[str],
                                           query_ids: List[str],
                                           top_k: int = 10,
                                           similarity_threshold: Optional[float] = None) -> List[List[Dict[str, Any]]]:
        """批量搜索相似问题
        
        Args:
            query_texts: 查询文本列表
            query_ids: 查询ID列表
            k: 每个查询返回的结果数量
            similarity_threshold: 相似度阈值
            
        Returns:
            批量搜索结果
        """
        if not self.enabled or not self.retriever:
            return [[] for _ in query_texts]
        
        threshold = similarity_threshold or self.similarity_threshold
        self.stats["total_searches"] += len(query_texts)
        
        try:
            import time
            start_time = time.time()
            
            # 执行批量搜索
            batch_results = await self.retriever.search_async(
                query_text=query_texts,
                query_id=query_ids,
                k=k,
                similarity_threshold=threshold
            )
            
            search_time = time.time() - start_time
            successful_searches = sum(1 for results in batch_results if len(results) > 0)
            self._update_batch_search_stats(search_time, successful_searches, len(query_texts))
            
            return batch_results
            
        except Exception as e:
            logger.error(f"批量向量搜索失败: {e}")
            return [[] for _ in query_texts]
    
    def _update_search_stats(self, search_time: float, success: bool):
        """更新搜索统计信息"""
        if success:
            self.stats["successful_searches"] += 1
        
        if self.use_gpu and self.retriever and self.retriever.use_gpu:
            self.stats["gpu_searches"] += 1
        else:
            self.stats["cpu_searches"] += 1
        
        # 更新平均搜索时间
        total_searches = self.stats["total_searches"]
        current_avg = self.stats["average_search_time"]
        self.stats["average_search_time"] = (current_avg * (total_searches - 1) + search_time) / total_searches
    
    def _update_batch_search_stats(self, search_time: float, successful_count: int, total_count: int):
        """更新批量搜索统计信息"""
        self.stats["successful_searches"] += successful_count
        
        if self.use_gpu and self.retriever and self.retriever.use_gpu:
            self.stats["gpu_searches"] += total_count
        else:
            self.stats["cpu_searches"] += total_count
        
        # 更新平均搜索时间（按单个查询计算）
        avg_time_per_query = search_time / total_count if total_count > 0 else 0
        total_searches = self.stats["total_searches"]
        current_avg = self.stats["average_search_time"]
        
        # 加权平均
        new_total_time = current_avg * (total_searches - total_count) + search_time
        self.stats["average_search_time"] = new_total_time / total_searches
    
    async def add_question_mapping(self, question_text: str, hash_id: str, nlp_result: Optional[Dict[str, Any]] = None) -> bool:
        """添加新的问题-哈希映射关系
        
        Args:
            question_text: 问题文本
            hash_id: 哈希ID
            nlp_result: NLP分析结果，包含子句信息
            
        Returns:
            是否添加成功
        """
        try:
            if self.enabled and self.retriever:
                
                # 准备要添加的文本列表和对应的ID列表
                texts_to_add = [question_text]  # 始终包含原始问题
                ids_to_add = [hash_id]  # 原始问题使用原hash_id
                
                # 如果有NLP结果，添加子句
                # if nlp_result and nlp_result.get('subsentences'):
                #     subsentences = nlp_result['subsentences']
                #     # 过滤掉与原始文本相同的子句，避免重复
                #     unique_subsentences = [
                #         sub for sub in subsentences 
                #         if sub.strip()
                #     ]
                    
                #     if unique_subsentences:
                #         texts_to_add.extend(unique_subsentences)
                #         # 所有子句都使用相同的hash_id
                #         ids_to_add.extend([hash_id] * len(unique_subsentences))
                
                # 批量添加到检索器
                success = await self.retriever.update_index_async(texts_to_add, ids_to_add)
                if success:
                    self.stats["total_documents"] += len(texts_to_add)
                    logger.info(f"添加问题映射: {hash_id} -> {question_text[:50]}... (包含 {len(texts_to_add)} 个文本)")
                return success
            
            return True
            
        except Exception as e:
            logger.error(f"添加问题映射失败: {e}")
            return False
            
    async def add_multiple_question_mappings(self, question_texts: List[str], hash_ids: List[str]) -> bool:
        """批量添加问题-哈希映射关系
        
        Args:
            question_texts: 问题文本列表
            hash_ids: 哈希ID列表
            
        Returns:
            是否添加成功
        """
        if len(question_texts) != len(hash_ids):
            logger.error("问题文本和哈希ID列表长度不匹配")
            return False
        
        try:
            # 如果启用了检索器，批量添加到检索器中
            if self.enabled and self.retriever:
                success = await self.retriever.update_index_async(question_texts, hash_ids)
                if success:
                    self.stats["total_documents"] += len(question_texts)
                    logger.info(f"批量添加 {len(question_texts)} 个问题映射")
                return success
            
            return True
            
        except Exception as e:
            logger.error(f"批量添加问题映射失败: {e}")
            return False

    
    async def save_data(self) -> bool:
        """保存数据
        
        Returns:
            是否保存成功
        """
        try:
            if self.retriever:
                self.retriever.save_index_and_metadata(os.path.dirname(self.retriever.metadata_path))
                logger.info("向量搜索数据保存成功")
                return True
            return False
        except Exception as e:
            logger.error(f"保存向量搜索数据失败: {e}")
            return False
    
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        base_stats = {
            "enabled": self.enabled,
            "similarity_threshold": self.similarity_threshold,
            "use_gpu": self.use_gpu,
            "config": {
                "embedding_model": self.config.embedding_model,
                "max_search_results": self.config.max_search_results,
                "high_similarity_threshold": self.config.high_similarity_threshold
            }
        }
        
        # 合并性能统计
        base_stats.update(self.stats)
        
        # 添加检索器统计
        if self.retriever:
            retriever_stats = self.retriever.get_stats()
            base_stats["retriever_stats"] = retriever_stats
        
        return base_stats
        
    async def cleanup(self):
        """清理资源（与 close 方法功能相同，提供统一接口）"""
        await self.close()
    async def close(self):
        """关闭并释放资源"""
        try:
            if self.retriever:
                self.retriever.close()
            logger.info("增强向量搜索管理器已关闭")
        except Exception as e:
            logger.error(f"关闭增强向量搜索管理器失败: {e}")
    
    def __del__(self):
        """析构函数"""
        if hasattr(self, 'retriever') and self.retriever:
            try:
                self.retriever.close()
            except:
                pass