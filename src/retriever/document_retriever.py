import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import faiss

class DocumentRetriever:
    def __init__(self, metadata_path, index_path, model_name='all-MiniLM-L6-v2'):
        # 设置环境变量
        
        
        # 加载元数据
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        self.ids = metadata["ids"]
        self.texts = metadata["texts"]
        
        # 加载FAISS索引
        print("Loading FAISS index...")
        self.cpu_index = faiss.read_index(index_path)
        self.res = faiss.StandardGpuResources()
        self.res.setTempMemory(8 * 1024 * 1024 * 1024)
        self.gpu_index = faiss.index_cpu_to_gpu(self.res, 0, self.cpu_index)
        
        # 初始化模型
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        
    def search(self, query_text,query_id, k=5):
        """执行文档检索"""
        query_embedding = self.model.encode(
            query_text,
            device=self.device,
            batch_size=1,
            show_progress_bar=False,
            convert_to_tensor=True,
            normalize_embeddings=True
        ).float().cpu().numpy()
        # FAISS检索
        distances, indices = self.gpu_index.search(query_embedding, k)
        
        # 整理结果
        results = []
        for i in range(k):
            idx = indices[0][i]
            if query_id!=self.ids[idx] and distances[0][i] >= 0.7:
                results.append({
                'index': idx,
                'id': self.ids[idx],
                'similarity': float(distances[0][i]),
                'text': self.texts[idx]
              })
        
        return results
    
    def __del__(self):
        """清理GPU资源"""
        if hasattr(self, 'gpu_index'):
            del self.gpu_index
        if hasattr(self, 'res'):
            del self.res
# DocumentRetriever = DocumentRetriever(
#     metadata_path='src/faiss_dataset/metadata/metadata.json',
#     index_path='src/faiss_dataset/metadata/vector_index_cosine.faiss'
# )
# # Example usage
# if __name__ == "__main__":
#     query_text = "check if this works"
#     query_id = 1
#     results = DocumentRetriever.search(query_text, query_id)
#     for result in results:
#         print(f"ID: {result['id']}, Similarity: {result['similarity']}, Text: {result['text']}")