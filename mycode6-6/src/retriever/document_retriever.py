import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import faiss
import datetime
import logging
from typing import List, Dict, Tuple, Optional, Union

class DocumentRetriever:
    def __init__(self, metadata_path: str, index_path: str, model_name: str = 'all-MiniLM-L6-v2', 
                 auto_init: bool = True, init_texts: List[str] = None, init_ids: List[Union[str, int]] = None,
                 enable_logging: bool = True):
        """初始化文档检索器
        
        Args:
            metadata_path: 元数据文件路径
            index_path: FAISS索引文件路径
            model_name: 使用的Sentence Transformer模型名称
            auto_init: 是否在数据库文件不存在时自动初始化，默认为False
            init_texts: 初始化数据库时使用的文档文本列表，仅当auto_init为True且数据库不存在时使用
            init_ids: 初始化数据库时使用的文档ID列表，仅当auto_init为True且数据库不存在时使用
            enable_logging: 是否启用日志记录，默认为True
        """
        self.metadata_path = metadata_path
        self.index_path = index_path
        
        # 设置日志
        if enable_logging:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = None
        
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
            
            # 获取目录路径
            output_dir = os.path.dirname(metadata_path)
            
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
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            # 获取模型的嵌入维度
            dimension = self.model.encode("", convert_to_tensor=True).shape[0]
            
            # 创建空的FAISS索引
            index = faiss.IndexFlatIP(dimension)  # 使用内积相似度
            
            # 保存索引
            faiss.write_index(index, index_path)
            
            # 加载元数据和索引
            self.ids = []
            self.texts = []
            self.cpu_index = index
            self.res = faiss.StandardGpuResources()
            self.res.setTempMemory(8 * 1024 * 1024 * 1024)
            self.gpu_index = faiss.index_cpu_to_gpu(self.res, 0, self.cpu_index)
            
            self._log(f"空数据库创建完成")
        else:
            raise FileNotFoundError(f"数据库文件不存在: {metadata_path} 或 {index_path}。请设置auto_init=True来自动初始化数据库。")

    def _log(self, message: str, level: str = 'info'):
        """记录日志
        
        Args:
            message: 日志消息
            level: 日志级别，可选值为'info', 'warning', 'error', 'debug'
        """
        if self.logger:
            if level == 'info':
                self.logger.info(message)
            elif level == 'warning':
                self.logger.warning(message)
            elif level == 'error':
                self.logger.error(message)
            elif level == 'debug':
                self.logger.debug(message)
        else:
            print(message)
    
    def _init_cuda(self):
        """初始化并检查CUDA状态"""
        self._log("=== CUDA状态检测 ===")
        self._log(f"PyTorch版本: {torch.__version__}")
        self._log(f"CUDA可用: {torch.cuda.is_available()}")
        self._log(f"GPU数量: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            self._log(f"当前设备: {torch.cuda.current_device()}")
            self._log(f"设备名称: {torch.cuda.get_device_name(0)}")
    
    def _init_model(self, model_name: str):
        """初始化模型"""
        self._log("初始化模型...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
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
        self.res = faiss.StandardGpuResources()
        self.res.setTempMemory(8 * 1024 * 1024 * 1024)
        self.gpu_index = faiss.index_cpu_to_gpu(self.res, 0, self.cpu_index)
        self._log(f"索引加载完成，包含 {self.gpu_index.ntotal} 条数据")

    @classmethod
    def create_or_load(cls, metadata_path: str, index_path: str, texts: List[str] = None, 
                      ids: List[Union[str, int]] = None, model_name: str = 'all-MiniLM-L6-v2',
                      enable_logging: bool = True) -> 'DocumentRetriever':
        """创建或加载文档检索器
        
        如果指定路径的数据库文件存在，则加载现有数据库；
        如果不存在且提供了texts和ids，则创建新数据库。
        
        Args:
            metadata_path: 元数据文件路径
            index_path: FAISS索引文件路径
            texts: 文档文本列表，用于创建新数据库
            ids: 文档ID列表，用于创建新数据库
            model_name: 使用的Sentence Transformer模型名称
            enable_logging: 是否启用日志记录
            
        Returns:
            DocumentRetriever: 文档检索器实例
        """
        metadata_exists = os.path.exists(metadata_path)
        index_exists = os.path.exists(index_path)
        
        if (not metadata_exists or not index_exists) and texts and ids:
            # 创建目录
            output_dir = os.path.dirname(metadata_path)
            os.makedirs(output_dir, exist_ok=True)
            
            # 初始化检索器（会自动创建数据库）
            return cls(metadata_path, index_path, model_name, auto_init=True, 
                       init_texts=texts, init_ids=ids, enable_logging=enable_logging)
        else:
            # 加载现有数据库
            return cls(metadata_path, index_path, model_name, enable_logging=enable_logging)
            
    def build_index(self, texts: List[str], ids: List[Union[str, int]], output_dir: str) -> Tuple[str, str]:
        """从文本列表构建新的FAISS索引
        
        Args:
            texts: 文档文本列表
            ids: 文档ID列表
            output_dir: 输出目录路径
            
        Returns:
            Tuple[str, str]: 保存的索引和元数据文件路径
        """
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
    
    def search(self, query_text: Union[str, List[str]], query_id: Union[str, int, List[Union[str, int]]], 
               k: int = 5, similarity_threshold: float = 0.7) -> Union[List[Dict], List[List[Dict]]]:
        """执行文档检索
        
        支持单个查询或批量查询。当输入为单个字符串时，执行单个查询；
        当输入为字符串列表时，执行批量查询。
        
        Args:
            query_text: 查询文本或查询文本列表
            query_id: 查询文档ID或查询文档ID列表，用于排除自身
            k: 返回的最大结果数量
            similarity_threshold: 相似度阈值，只返回相似度大于等于此值的结果
            
        Returns:
            Union[List[Dict], List[List[Dict]]]: 检索结果列表，每个结果包含index、id、similarity和text字段
        """
        try:
            # 判断是单个查询还是批量查询
            is_batch = isinstance(query_text, list)
            self._single_search(query_text, query_id, k, similarity_threshold)
            if is_batch:
                return self._batch_search(query_text, query_id, k, similarity_threshold)
            else:
                return self._single_search(query_text, query_id, k, similarity_threshold)
                
        except Exception as e:
            self._log(f"检索失败: {str(e)}", level='error')
            return [] if not is_batch else [[] for _ in range(len(query_text))]
    
    def _single_search(self, query_text: str, query_id: Union[str, int], 
                      k: int = 5, similarity_threshold: float = 0.7) -> List[Dict]:
        """执行单个文档检索
        
        Args:
            query_text: 查询文本
            query_id: 查询文档ID，用于排除自身
            k: 返回的最大结果数量
            similarity_threshold: 相似度阈值，只返回相似度大于等于此值的结果
            
        Returns:
            List[Dict]: 检索结果列表，每个结果包含index、id、similarity和text字段
        """
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
        distances, indices = self.gpu_index.search(query_embedding, k)
        
        # 整理结果
        results = []
        for i in range(k):
            idx = indices[0][i]
            if idx < len(self.ids) and distances[0][i] >= similarity_threshold and self.ids[idx]!=query_id:
                results.append({
                    'index': int(idx),
                    'id': self.ids[idx],
                    'similarity': float(distances[0][i]),
                    'text': self.texts[idx]
                })
        return results
    
    def _batch_search(self, query_texts: List[str], query_ids: List[Union[str, int]], 
                     k: int = 5, similarity_threshold: float = 0.7, 
                     batch_size: int = 1000) -> List[List[Dict]]:
        """批量执行文档检索
        
        Args:
            query_texts: 查询文本列表
            query_ids: 查询文档ID列表，用于排除自身
            k: 每个查询返回的结果数量
            similarity_threshold: 相似度阈值
            batch_size: 批处理大小
            
        Returns:
            List[List[Dict]]: 检索结果列表的列表，每个内部列表对应一个查询的结果
        """
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
            distances, indices = self.gpu_index.search(query_embeddings, k)
            
            # 处理结果
            for i in range(len(batch_texts)):
                query_results = []
                for j in range(k):
                    idx = indices[i][j]
                    if idx < len(self.ids)  and distances[i][j] >= similarity_threshold:
                        query_results.append({
                            'index': int(idx),
                            'id': self.ids[idx],
                            'similarity': float(distances[i][j]),
                            'text': self.texts[idx]
                        })
                
                all_results.append(query_results[:k])
            
            self._log(f"已处理 {end}/{len(query_texts)} 条查询")
        
        return all_results
    
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
    
    def save_results(self, results: List[Dict], output_path: str):
        """保存检索结果到文件"""
        self._log("保存搜索结果...")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        self._log(f"搜索结果已保存至 {output_path}")
    
    def update_index(self, new_texts: List[str], new_ids: List[Union[str, int]], 
                    save_path: Optional[str] = None) -> bool:
        """增量更新FAISS索引
        
        将新的文档添加到现有的FAISS索引中，而不是创建一个新的索引
        
        Args:
            new_texts: 新增的文档文本列表
            new_ids: 新增的文档ID列表，必须与new_texts长度相同
            save_path: 保存更新后索引的路径，若为None则使用原路径
            
        Returns:
            bool: 更新是否成功
        """
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
            del self.gpu_index  # 释放旧的GPU索引
            self.gpu_index = faiss.index_cpu_to_gpu(self.res, 0, self.cpu_index)
            
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
    
    def save_index_and_metadata(self, save_dir: str, prefix: str = "") -> None:
        """保存FAISS索引和元数据到指定目录
        
        Args:
            save_dir: 保存目录路径
            prefix: 文件名前缀，默认为空
        """
        try:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                
            # 保存FAISS索引
            index_filename = f"{prefix}vector_index_cosine.faiss" if prefix else "vector_index_cosine.faiss"
            index_path = os.path.join(save_dir, index_filename)
            self._log(f"正在保存索引到: {index_path}")
            faiss.write_index(self.cpu_index, index_path)
            
            # 保存元数据
            metadata_filename = f"{prefix}metadata.json" if prefix else "metadata.json"
            metadata_path = os.path.join(save_dir, metadata_filename)
            self._log(f"正在保存元数据到: {metadata_path}")
            metadata = {
                "ids": self.ids,
                "texts": self.texts,
                "total_count": len(self.ids),
                "created_at": datetime.datetime.now().isoformat(),
                "last_updated_at": datetime.datetime.now().isoformat()
            }
            
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
                
            self._log(f"索引和元数据已成功保存至: {save_dir}")
            
        except Exception as e:
            self._log(f"保存失败: {str(e)}", level='error')
            raise

    @classmethod
    def create_empty_database(cls, metadata_path: str, index_path: str, model_name: str = 'all-MiniLM-L6-v2',
                             enable_logging: bool = True) -> 'DocumentRetriever':
        """创建一个空的文档检索数据库
        
        创建一个不包含任何文档的空FAISS索引和元数据文件
        
        Args:
            metadata_path: 元数据文件路径
            index_path: FAISS索引文件路径
            model_name: 使用的Sentence Transformer模型名称
            enable_logging: 是否启用日志记录
            
        Returns:
            DocumentRetriever: 文档检索器实例
        """
        # 创建目录
        output_dir = os.path.dirname(metadata_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化临时实例以获取嵌入维度
        temp_retriever = cls(metadata_path, index_path, model_name, auto_init=False, 
                            enable_logging=enable_logging)
        
        try:
            # 获取模型的嵌入维度
            dimension = temp_retriever.model.encode("", convert_to_tensor=True).shape[0]
            
            # 创建空的FAISS索引
            temp_retriever._log("Creating empty FAISS index...")
            index = faiss.IndexFlatIP(dimension)  # 使用内积相似度
            
            # 保存索引
            faiss.write_index(index, index_path)
            
            # 保存空的元数据
            metadata = {
                "ids": [],
                "texts": [],
                "total_count": 0,
                "created_at": datetime.datetime.now().isoformat()
            }
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            temp_retriever._log(f"Empty index built and saved to {output_dir}")
            
            # 重新加载元数据和索引
            temp_retriever._load_metadata(metadata_path)
            temp_retriever._init_faiss(index_path)
            
            return temp_retriever
            
        except Exception as e:
            temp_retriever._log(f"创建空数据库失败: {str(e)}", level='error')
            raise

    def __del__(self):
        """析构函数，清理GPU资源"""
        try:
            if hasattr(self, 'gpu_index'):
                del self.gpu_index
                # 添加同步点，确保GPU操作完成
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            if hasattr(self, 'res'):
                del self.res
        except Exception as e:
            if hasattr(self, '_log'):
                self._log(f"清理资源时出错: {str(e)}", level='error')
            pass

