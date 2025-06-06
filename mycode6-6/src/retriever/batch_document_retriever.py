import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict, Optional
import logging
import datetime  # 添加到文件顶部的导入部分

class BatchDocumentRetriever:
    def __init__(self, metadata_path: str, index_path: str, model_name: str = 'all-MiniLM-L6-v2'):
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 设置环境变量
       
        
        # 初始化CUDA
        self._init_cuda()
        
        # 加载数据和模型
        self._load_metadata(metadata_path)
        self._init_faiss(index_path)
        self._init_model(model_name)
        
    def _init_cuda(self):
        """初始化并检查CUDA状态"""
        self.logger.info("=== CUDA状态检测 ===")
        self.logger.info(f"PyTorch版本: {torch.__version__}")
        self.logger.info(f"CUDA可用: {torch.cuda.is_available()}")
        self.logger.info(f"GPU数量: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            self.logger.info(f"当前设备: {torch.cuda.current_device()}")
            self.logger.info(f"设备名称: {torch.cuda.get_device_name(0)}")
    
    def _load_metadata(self, metadata_path: str):
        """加载元数据"""
        self.logger.info("加载元数据...")
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        self.ids = metadata["ids"]
        self.texts = metadata["texts"]
        self.logger.info(f"已加载 {len(self.ids)} 条元数据")
    
    def _init_faiss(self, index_path: str):
        """初始化FAISS索引"""
        self.logger.info("加载FAISS索引...")
        self.cpu_index = faiss.read_index(index_path)
        self.res = faiss.StandardGpuResources()
        self.res.setTempMemory(8 * 1024 * 1024 * 1024)
        self.gpu_index = faiss.index_cpu_to_gpu(self.res, 0, self.cpu_index)
        self.logger.info(f"索引加载完成，包含 {self.gpu_index.ntotal} 条数据")
    
    def _init_model(self, model_name: str):
        """初始化模型"""
        self.logger.info("初始化模型...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        self.logger.info(f"模型加载完成（设备：{self.device}）")
    
    def batch_search(self, 
                    queries: List[Dict],
                    k: int = 10,
                    similarity_threshold: float = 0.7,
                    batch_size: int = 1000) -> List[Dict]:
        """
        批量执行文档检索
        
        Args:
            queries: 包含id和sentences的字典列表
            k: 每个查询返回的结果数量
            similarity_threshold: 相似度阈值
            batch_size: 批处理大小
        """
        results = []
        query_texts = [q["sentences"][0] for q in queries]
        query_ids = [q["id"] for q in queries]
        
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
            distances, indices = self.gpu_index.search(query_embeddings, k + 1)
            
            # 处理结果
            for i in range(len(batch_texts)):
                top_k_results = []
                for j in range(k + 1):
                    idx = indices[i][j]
                    similarity = float(distances[i][j])
                    
                    # 排除自身并应用相似度阈值
                    if (self.ids[idx] != batch_ids[i] and 
                        similarity >= similarity_threshold):
                        top_k_results.append({
                            "index": int(idx),
                            "id": self.ids[idx],
                            "cosine_similarity": similarity
                        })
                
                results.append({
                    "query_id": batch_ids[i],
                    "query_text": batch_texts[i],
                    "top_k": top_k_results[:k]
                })
            
            self.logger.info(f"已处理 {end}/{len(query_texts)} 条数据")
        
        return results
    
    def save_results(self, results: List[Dict], output_path: str):
        """保存检索结果到文件"""
        self.logger.info("保存搜索结果...")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        self.logger.info(f"搜索结果已保存至 {output_path}")
    
    def add_documents(self, new_docs: List[Dict], save_path: Optional[str] = None) -> None:
        """
        将新文档添加到检索系统中
        
        Args:
            new_docs: 包含id和sentences的字典列表，格式与查询相同
            save_path: 保存更新后索引的路径，若为None则不保存
        """
        self.logger.info(f"开始添加 {len(new_docs)} 条新文档...")
        
        try:
            # 提取新文档的文本和ID
            new_texts = [doc["sentences"][0] for doc in new_docs]
            new_ids = [doc["id"] for doc in new_docs]
            
            # 检查ID是否重复
            duplicate_ids = set(new_ids) & set(self.ids)
            if duplicate_ids:
                self.logger.warning(f"发现重复ID: {duplicate_ids}")
                return
            
            # 生成新文档的嵌入向量
            new_embeddings = self.model.encode(
                new_texts,
                device=self.device,
                batch_size=512,
                show_progress_bar=True,
                convert_to_tensor=True,
                normalize_embeddings=True
            ).float().cpu().numpy()
            
            # 更新FAISS索引
            self.gpu_index = faiss.index_gpu_to_cpu(self.gpu_index)  # 转回CPU
            self.cpu_index.add(new_embeddings)  # 添加新向量
            self.gpu_index = faiss.index_cpu_to_gpu(self.res, 0, self.cpu_index)  # 重新转到GPU
            
            # 更新metadata
            self.ids.extend(new_ids)
            self.texts.extend(new_texts)
            
            self.logger.info(f"成功添加 {len(new_docs)} 条文档")
            self.logger.info(f"当前索引总量: {self.gpu_index.ntotal}")
            
            # 保存更新后的索引和元数据
            if save_path:
                self.save_index_and_metadata(save_path)
                
        except Exception as e:
            self.logger.error(f"添加文档失败: {str(e)}")
            raise
    
    def save_index_and_metadata(self, save_dir: str, prefix: str = "updated") -> None:
        """
        保存FAISS索引和元数据到指定目录
        
        Args:
            save_dir: 保存目录路径
            prefix: 文件名前缀
        """
        try:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                
            # 保存FAISS索引
            index_path = os.path.join(save_dir, f"{prefix}_vector_index.faiss")
            self.logger.info(f"正在保存索引到: {index_path}")
            faiss.write_index(self.cpu_index, index_path)
            
            # 保存元数据
            metadata_path = os.path.join(save_dir, f"{prefix}_metadata.json")
            self.logger.info(f"正在保存元数据到: {metadata_path}")
            metadata = {
                "ids": self.ids,
                "texts": self.texts,
                "total_count": len(self.ids),
                "last_updated": datetime.datetime.now().isoformat()
            }
            
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=4)
                
            self.logger.info(f"索引和元数据已成功保存至: {save_dir}")
            
        except Exception as e:
            self.logger.error(f"保存失败: {str(e)}")
            raise
    
    def __del__(self):
        """清理GPU资源"""
        if hasattr(self, 'gpu_index'):
            del self.gpu_index
        if hasattr(self, 'res'):
            del self.res