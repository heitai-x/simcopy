import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import faiss
import datetime
import asyncio
from typing import List, Dict, Tuple, Optional, Union, Any
from loguru import logger


class EnhancedDocumentRetriever:
    """增强的文档检索器
    
    基于FAISS向量索引和Sentence Transformers的高性能文档检索系统
    支持GPU加速、批量处理和增量更新
    """
    
    def __init__(self, metadata_path: str, index_path: str, model_name: str = 'all-MiniLM-L6-v2', 
                 auto_init: bool = True, init_texts: List[str] = None, init_ids: List[Union[str, int]] = None,
                 enable_logging: bool = True, use_gpu: bool = True):
        """初始化增强文档检索器
        
        Args:
            metadata_path: 元数据文件路径
            index_path: FAISS索引文件路径
            model_name: 使用的Sentence Transformer模型名称
            auto_init: 是否在数据库文件不存在时自动初始化
            init_texts: 初始化数据库时使用的文档文本列表
            init_ids: 初始化数据库时使用的文档ID列表
            enable_logging: 是否启用日志记录
            use_gpu: 是否使用GPU加速
        """
        self.metadata_path = metadata_path
        self.index_path = index_path
        self.use_gpu = use_gpu
        self.enable_logging = enable_logging
        
        # 初始化组件
        self._init_cuda()
        self._init_model(model_name)
        
        # 检查数据库文件是否存在
        metadata_exists = os.path.exists(metadata_path)
        index_exists = os.path.exists(index_path)
        
        # 如果文件不存在且设置了自动初始化
        if (not metadata_exists or not index_exists) and auto_init and init_texts and init_ids:
            self._log(f"数据库文件不存在，正在初始化新数据库...")
            
            # 获取目录路径
            output_dir = os.path.dirname(metadata_path)
            
            # 构建新索引
            self.build_index(init_texts, init_ids, output_dir)
            
            # 重新加载元数据
            self._load_metadata(metadata_path)
            
            # 加载FAISS索引
            self._init_faiss(index_path)
        elif metadata_exists and index_exists:
            # 正常加载现有数据库
            self._load_metadata(metadata_path)
            self._init_faiss(index_path)
        elif (not metadata_exists or not index_exists) and auto_init:
            # 如果文件不存在且设置了自动初始化，但没有提供初始文本和ID，则创建空数据库
            self._log(f"数据库文件不存在，正在创建空数据库...")
            self._create_empty_database()
        else:
            raise FileNotFoundError(f"数据库文件不存在: {metadata_path} 或 {index_path}。请设置auto_init=True来自动初始化数据库。")
    
    def _log(self, message: str, level: str = 'info'):
        """记录日志
        
        Args:
            message: 日志消息
            level: 日志级别，可选值为'info', 'warning', 'error', 'debug'
        """
        if self.enable_logging:
            if level == 'info':
                logger.info(message)
            elif level == 'warning':
                logger.warning(message)
            elif level == 'error':
                logger.error(message)
            elif level == 'debug':
                logger.debug(message)
        else:
            print(message)
    
    def _init_cuda(self):
        """初始化并检查CUDA状态"""
        self._log("=== CUDA状态检测 ===")
        self._log(f"PyTorch版本: {torch.__version__}")
        self._log(f"CUDA可用: {torch.cuda.is_available()}")
        self._log(f"GPU数量: {torch.cuda.device_count()}")
        if torch.cuda.is_available() and self.use_gpu:
            self._log(f"当前设备: {torch.cuda.current_device()}")
            self._log(f"设备名称: {torch.cuda.get_device_name(0)}")
        else:
            self.use_gpu = False
            self._log("将使用CPU模式")
    
    def _init_model(self, model_name: str):
        """初始化模型"""
        self._log("初始化模型...")
        self.device = "cuda" if torch.cuda.is_available() and self.use_gpu else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        self._log(f"模型加载完成（设备：{self.device}）")
    
    def _load_metadata(self, metadata_path: str):
        """加载元数据"""
        self._log("加载元数据...")
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        self.ids = metadata["ids"]
        self.texts = metadata["texts"]
        self._log(f"已加载 {len(self.ids)} 条元数据")
    
    def _init_faiss(self, index_path: str):
        """初始化FAISS索引"""
        self._log("加载FAISS索引...")
        self.cpu_index = faiss.read_index(index_path)
        
        if self.use_gpu and torch.cuda.is_available():
            try:
                self.res = faiss.StandardGpuResources()
                self.res.setTempMemory(8 * 1024 * 1024 * 1024)  # 8GB
                self.gpu_index = faiss.index_cpu_to_gpu(self.res, 0, self.cpu_index)
                self._log(f"GPU索引加载完成，包含 {self.gpu_index.ntotal} 条数据")
            except Exception as e:
                self._log(f"GPU索引初始化失败，回退到CPU: {e}", level='warning')
                self.use_gpu = False
                self.gpu_index = self.cpu_index
        else:
            self.gpu_index = self.cpu_index
            self._log(f"CPU索引加载完成，包含 {self.cpu_index.ntotal} 条数据")
    
    def _create_empty_database(self):
        """创建空数据库"""
        # 获取目录路径
        output_dir = os.path.dirname(self.metadata_path)
        
        # 创建空的元数据文件
        metadata = {
            "ids": [],
            "texts": [],
            "total_count": 0,
            "created_at": datetime.datetime.now().isoformat()
        }
        
        # 确保目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存元数据
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # 获取模型的嵌入维度
        dimension = self.model.encode("", convert_to_tensor=True).shape[0]
        
        # 创建空的FAISS索引
        index = faiss.IndexFlatIP(dimension)  # 使用内积相似度
        
        # 保存索引
        faiss.write_index(index, self.index_path)
        
        # 加载元数据和索引
        self.ids = []
        self.texts = []
        self.cpu_index = index
        
        if self.use_gpu and torch.cuda.is_available():
            try:
                self.res = faiss.StandardGpuResources()
                self.res.setTempMemory(8 * 1024 * 1024 * 1024)
                self.gpu_index = faiss.index_cpu_to_gpu(self.res, 0, self.cpu_index)
            except Exception as e:
                self._log(f"GPU索引初始化失败，使用CPU: {e}", level='warning')
                self.use_gpu = False
                self.gpu_index = self.cpu_index
        else:
            self.gpu_index = self.cpu_index
        
        self._log(f"空数据库创建完成")

    @classmethod
    def create_or_load(cls, metadata_path: str, index_path: str, texts: List[str] = None, 
                      ids: List[Union[str, int]] = None, model_name: str = 'all-MiniLM-L6-v2',
                      enable_logging: bool = True, use_gpu: bool = True) -> 'EnhancedDocumentRetriever':
        """创建或加载文档检索器"""
        metadata_exists = os.path.exists(metadata_path)
        index_exists = os.path.exists(index_path)
        
        if (not metadata_exists or not index_exists) and texts and ids:
            # 创建目录
            output_dir = os.path.dirname(metadata_path)
            os.makedirs(output_dir, exist_ok=True)
            
            # 初始化检索器（会自动创建数据库）
            return cls(metadata_path, index_path, model_name, auto_init=True, 
                       init_texts=texts, init_ids=ids, enable_logging=enable_logging, use_gpu=use_gpu)
        else:
            # 加载现有数据库
            return cls(metadata_path, index_path, model_name, enable_logging=enable_logging, use_gpu=use_gpu)
            
    def build_index(self, texts: List[str], ids: List[Union[str, int]], output_dir: str) -> Tuple[str, str]:
        """从文本列表构建新的FAISS索引"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # 编码文本
            self._log("Encoding documents...")
            embeddings = self.model.encode(
                texts,
                device=self.device,
                batch_size=32,
                show_progress_bar=True,
                convert_to_tensor=True,
                normalize_embeddings=True
            ).float().cpu().numpy()
            
            # 创建FAISS索引
            self._log("Creating FAISS index...")
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)  # 使用内积相似度
            index.add(embeddings)
            
            # 保存索引
            index_path = os.path.join(output_dir, 'vector_index_cosine.faiss')
            faiss.write_index(index, index_path)
            
            # 保存元数据
            metadata = {
                "ids": ids,
                "texts": texts,
                "total_count": len(ids),
                "created_at": datetime.datetime.now().isoformat()
            }
            metadata_path = os.path.join(output_dir, 'metadata.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            self._log(f"Index built and saved to {output_dir}")
            return index_path, metadata_path
            
        except Exception as e:
            self._log(f"构建索引失败: {str(e)}", level='error')
            return None, None
    
    async def search_async(self, query_text: Union[str, List[str]], query_id: Union[str, int, List[Union[str, int]]], 
                          k: int = 5, similarity_threshold: float = 0.7) -> Union[List[Dict], List[List[Dict]]]:
        """异步执行文档检索"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.search, query_text, k, similarity_threshold)
    
    def search(self, query_text: Union[str, List[str]], 
               k: int = 5, similarity_threshold: float = 0.7) -> Union[List[Dict], List[List[Dict]]]:
        """执行文档检索"""
        try:
            # 判断是单个查询还是批量查询
            is_batch = isinstance(query_text, list)
            
            if is_batch:
                return self._batch_search(query_text,  k, similarity_threshold)
            else:
                return self._single_search(query_text,  k, similarity_threshold)
                
        except Exception as e:
            self._log(f"检索失败: {str(e)}", level='error')
            return [] if not is_batch else [[] for _ in range(len(query_text))]
    
    def _single_search(self, query_text: str, 
                      k: int = 5, similarity_threshold: float = 0.7) -> List[Dict]:
        """执行单个文档检索"""
        # 检查必要的属性是否存在
        if not hasattr(self, 'ids') or not hasattr(self, 'texts'):
            self._log("警告：ids或texts属性不存在，无法执行搜索", level='warning')
            return []
            
        query_embedding = self.model.encode(
            query_text,
            device=self.device,
            show_progress_bar=False,
            convert_to_tensor=True,
            normalize_embeddings=True
        ).float().cpu().numpy()
        
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # FAISS检索
        distances, indices = self.gpu_index.search(query_embedding, k + 10)  # 多检索一些以过滤
        
        # 整理结果
        results = []
        for i in range(min(k + 10, len(indices[0]))):
            idx = indices[0][i]
            if idx < len(self.ids) and idx < len(self.texts) and distances[0][i] >= similarity_threshold:
                results.append({
                    'index': int(idx),
                    'id': self.ids[idx],
                    'similarity': float(distances[0][i]),
                    'text': self.texts[idx]
                })
                
                if len(results) >= k:
                    break
        
        return results
    
    def _batch_search(self, query_texts: List[str], 
                     k: int = 5, similarity_threshold: float = 0.7, 
                     batch_size: int = 1000) -> List[List[Dict]]:
        """批量执行文档检索"""
        # 检查必要的属性是否存在
        if not hasattr(self, 'ids') or not hasattr(self, 'texts'):
            self._log("警告：ids或texts属性不存在，无法执行批量搜索", level='warning')
            return [[] for _ in range(len(query_texts))]
            
        all_results = []
        
        for start in range(0, len(query_texts), batch_size):
            end = min(start + batch_size, len(query_texts))
            batch_texts = query_texts[start:end]
            batch_ids = query_ids[start:end]
            
            # 生成批量查询嵌入
            query_embeddings = self.model.encode(
                batch_texts,
                device=self.device,
                batch_size=512,
                show_progress_bar=False,
                convert_to_tensor=True,
                normalize_embeddings=True
            ).float().cpu().numpy()
            
            # FAISS检索
            distances, indices = self.gpu_index.search(query_embeddings, k + 10)
            
            # 处理结果
            for i in range(len(batch_texts)):
                query_results = []
                for j in range(min(k + 10, len(indices[i]))):
                    idx = indices[i][j]
                    if idx < len(self.ids) and idx < len(self.texts) and distances[i][j] >= similarity_threshold:
                        query_results.append({
                            'index': int(idx),
                            'id': self.ids[idx],
                            'similarity': float(distances[i][j]),
                            'text': self.texts[idx]
                        })
                        
                        if len(query_results) >= k:
                            break
                
                all_results.append(query_results)
            
            self._log(f"已处理 {end}/{len(query_texts)} 条查询")
        
        return all_results
    
    async def update_index_async(self, new_texts: List[str], new_ids: List[Union[str, int]], 
                                save_path: Optional[str] = None) -> bool:
        """异步增量更新FAISS索引"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.update_index, new_texts, new_ids, save_path)
    
    def update_index(self, new_texts: List[str], new_ids: List[Union[str, int]], 
                    save_path: Optional[str] = None) -> bool:
        """增量更新FAISS索引"""
        try:
            if len(new_texts) != len(new_ids):
                self._log("错误：文本列表和ID列表长度不匹配", level='error')
                return False
                
            if len(new_texts) == 0:
                self._log("警告：没有提供新文档，无需更新", level='warning')
                return True
            
            # 检查ID是否重复
            duplicate_ids = set(new_ids) & set(self.ids)
            if duplicate_ids:
                self._log(f"发现重复ID: {duplicate_ids}", level='warning')
                return False
                
            self._log(f"正在增量更新索引，添加{len(new_texts)}个新文档...")
            
            # 编码新文本
            self._log("Encoding new documents...")
            new_embeddings = self.model.encode(
                new_texts,
                device=self.device,
                batch_size=32,
                show_progress_bar=True,
                convert_to_tensor=True,
                normalize_embeddings=True
            ).float().cpu().numpy()
            
            # 更新CPU索引
            self._log("Updating FAISS index...")
            self.cpu_index.add(new_embeddings)
            
            # 更新GPU索引
            if self.use_gpu and hasattr(self, 'res'):
                del self.gpu_index  # 释放旧的GPU索引
                self.gpu_index = faiss.index_cpu_to_gpu(self.res, 0, self.cpu_index)
            else:
                self.gpu_index = self.cpu_index
            
            # 更新元数据
            self.ids.extend(new_ids)
            self.texts.extend(new_texts)
            
            # 保存更新后的索引和元数据
            save_dir = save_path if save_path else os.path.dirname(self.metadata_path)
            self.save_index_and_metadata(save_dir)
                
            self._log(f"索引已成功更新，当前共有{len(self.ids)}个文档")
            return True
            
        except Exception as e:
            self._log(f"更新索引失败: {str(e)}", level='error')
            return False
    
    def save_index_and_metadata(self, output_dir: str):
        """保存索引和元数据"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # 检查必要的属性是否存在
            if not hasattr(self, 'ids'):
                self.ids = []
            if not hasattr(self, 'texts'):
                self.texts = []
            if not hasattr(self, 'cpu_index'):
                self._log("错误：CPU索引不存在，无法保存", level='error')
                return
            
            # 保存索引
            index_path = os.path.join(output_dir, os.path.basename(self.index_path))
            faiss.write_index(self.cpu_index, index_path)
            
            # 保存元数据
            metadata = {
                "ids": self.ids,
                "texts": self.texts,
                "total_count": len(self.ids),
                "updated_at": datetime.datetime.now().isoformat()
            }
            metadata_path = os.path.join(output_dir, os.path.basename(self.metadata_path))
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            self._log(f"索引和元数据已保存到: {output_dir}")
            
        except Exception as e:
            self._log(f"保存失败: {str(e)}", level='error')
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_documents": len(self.ids),
            "index_size": self.cpu_index.ntotal if hasattr(self, 'cpu_index') else 0,
            "model_name": self.model.get_sentence_embedding_dimension() if hasattr(self.model, 'get_sentence_embedding_dimension') else "unknown",
            "device": self.device,
            "use_gpu": self.use_gpu,
            "metadata_path": self.metadata_path,
            "index_path": self.index_path
        }
    
    def batch_search_with_format(self, queries: List[Dict], k: int = 10, 
                               similarity_threshold: float = 0.8, 
                               batch_size: int = 1000) -> List[Dict]:
        """批量执行文档检索（兼容BatchDocumentRetriever格式）
        
        Args:
            queries: 包含id和sentences的字典列表
            k: 每个查询返回的结果数量
            similarity_threshold: 相似度阈值
            batch_size: 批处理大小
            
        Returns:
            List[Dict]: 检索结果列表，每个结果包含query_id、query_text和top_k字段
        """
        results = []
        query_texts = [q["sentences"][0] for q in queries]
        query_ids = [q["id"] for q in queries]
        
        batch_results = self._batch_search(query_texts, query_ids, k, similarity_threshold, batch_size)
        
        for i, query_results in enumerate(batch_results):
            top_k_results = []
            for result in query_results:
                top_k_results.append({
                    "index": result["index"],
                    "id": result["id"],
                    "cosine_similarity": result["similarity"]
                })
            
            results.append({
                "query_id": query_ids[i],
                "query_text": query_texts[i],
                "top_k": top_k_results
            })
        
        return results
    
    def add_documents(self, new_docs: List[Dict], save_path: Optional[str] = None) -> bool:
        """将新文档添加到检索系统中（兼容BatchDocumentRetriever格式）
        
        Args:
            new_docs: 包含id和sentences的字典列表，格式与查询相同
            save_path: 保存更新后索引的路径，若为None则使用原路径
            
        Returns:
            bool: 更新是否成功
        """
        try:
            # 提取新文档的文本和ID
            new_texts = [doc["sentences"][0] for doc in new_docs]
            new_ids = [doc["id"] for doc in new_docs]
            
            # 调用update_index方法
            return self.update_index(new_texts, new_ids, save_path)
            
        except Exception as e:
            self._log(f"添加文档失败: {str(e)}", level='error')
            return False
    
    def save_results(self, results: List[Dict], output_path: str):
        """保存检索结果到文件"""
        self._log("保存搜索结果...")
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            self._log(f"搜索结果已保存至 {output_path}")
        except Exception as e:
            self._log(f"保存结果失败: {str(e)}", level='error')
    
    def close(self):
        """释放资源"""
        if hasattr(self, 'res') and self.res:
            del self.res
        if hasattr(self, 'gpu_index'):
            del self.gpu_index
        if hasattr(self, 'cpu_index'):
            del self.cpu_index
        self._log("资源已释放")