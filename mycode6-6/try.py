import torch
import faiss
import gc
import time

# 测试函数
def test_gpu_resource_cleanup(iterations=10):
    print("开始测试GPU资源释放...")
    
    for i in range(iterations):
        print(f"迭代 {i+1}/{iterations}")
        
        # 创建资源
        dimension = 128
        index = faiss.IndexFlatIP(dimension)
        
        # 添加一些随机数据
        data = torch.randn(100, dimension).float().cpu().numpy()
        index.add(data)
        
        # 创建GPU资源
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        
        # 执行一些搜索操作
        query = torch.randn(1, dimension).float().cpu().numpy()
        distances, indices = gpu_index.search(query, 5)
        
        # 释放资源 - 使用修改后的释放顺序
        del gpu_index
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        del res
        
        # 强制垃圾回收
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # 短暂暂停
        time.sleep(1)
    
    print("测试完成，没有发生崩溃")

# 执行测试
if __name__ == "__main__":
    test_gpu_resource_cleanup(20)  # 增加迭代次数可以更严格地测试