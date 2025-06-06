import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
cache_dir = "/mnt/sevenT/debinx/huggingface_models"
import torch.cuda
import re
from collections import defaultdict
from sentence_transformers import CrossEncoder
from transformers import Qwen2ForCausalLM, AutoTokenizer
import torch.nn.functional as F
import torch
import json
from tqdm import tqdm
from retriever import DocumentRetriever
from sentence_transformers import SentenceTransformer
import time
import argparse
import spacy
import numpy as np
from tree import ConjunctionExtractor
from transformers.cache_utils import DynamicCache

import threading
from threading import Lock, Event  # 添加这行导入
import queue  # 添加这行导入
import hashlib
import random
import json
import os
from datetime import datetime
# 添加全局变量定义

sim_threshold = 0.7  # 相似度阈值

def generate_random_processing_time():
    """生成2-15秒之间偏重于较大数的随机执行时间"""
    # 使用beta分布，参数设置为偏重较大值
    # alpha=2, beta=5 会产生偏向较小值的分布
    # alpha=5, beta=2 会产生偏向较大值的分布
    beta_sample = np.random.beta(5, 2)  # 偏向较大值
    # 将[0,1]的beta分布映射到[2,15]区间
    processing_time = 2 + beta_sample * 13  # 2 + [0,1] * 13 = [2,15]
    return processing_time
def initialize_retriever():
    """安全地初始化retriever"""
    try:

        # 检查必要的文件是否存在
        index_path = "llma/src/faiss_dataset/metadata/vector_index_cosine.faiss"
        metadata_path = "llma/src/faiss_dataset/metadata/metadata.json"
        
        retriever = DocumentRetriever(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            index_path=index_path,
            metadata_path=metadata_path
        )
        print("DocumentRetriever 初始化成功")
        return retriever
        
    except ImportError as e:
        print(f"导入错误: {str(e)}，请检查 retriever 模块是否正确安装")
        return None
    except Exception as e:
        print(f"初始化retriever失败: {str(e)}")
        return None


# 请求类定义
retriever = initialize_retriever()
class Request:
    def __init__(self, query_text, query_id, arrival_time):
        self.query_text = query_text
        self.query_id = query_id
        self.arrival_time = arrival_time
        self.start_time = None
        self.end_time = None
        self.status = "waiting"
        self.similarity_score = 0.0
        self.has_similar = False
        self.processing_time = generate_random_processing_time()   # 固定处理时间
        
    def start_processing(self, current_time):
        self.start_time = current_time
        self.status = "processing"
        
    def is_finished(self, current_time):
        if self.start_time is None:
            return False
        return (current_time - self.start_time) >= self.processing_time
        
    def complete_processing(self, current_time):
        self.end_time = current_time
        self.status = "completed"

class SimilarRequestStatsLogger:
    def __init__(self, log_file_path="logs/similar_requests_stats_new.jsonl"):
        self.log_file_path = log_file_path
        self.ensure_log_directory()
        
    def ensure_log_directory(self):
        """确保日志目录存在"""
        log_dir = os.path.dirname(self.log_file_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    
    def log_similar_stats(self, similar_count, total_executing, request_id=None):
        """记录相似请求统计信息到文件"""
        stats_entry = {
            "timestamp": datetime.now().isoformat(),
            "similar_requests_count": similar_count,
            "total_executing_count": total_executing,
            "similar_ratio": similar_count / total_executing if total_executing > 0 else 0,
            "trigger_request_id": request_id
        }
        
        try:
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(stats_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[SimilarStatsLogger] 写入统计文件失败: {str(e)}")

# 线程安全的批处理队列管理器
class ThreadSafeBatchManager:
    def __init__(self, max_batch_size=64):
        self.max_batch_size = max_batch_size
        self.waiting_queue = queue.Queue()  # 等待队列
        self.executing_requests = {}  # 正在执行的请求字典 {request_id: request}
        self.completed_queue = queue.Queue()  # 已完成的请求队列
        self.lock = Lock()  # 保护执行队列的锁
        self.stats_lock = Lock()  # 统计数据锁
        
        # 统计信息
        self.total_generated = 0
        self.total_completed = 0
        self.total_processing = 0
        
        # 添加相似请求统计记录器
        self.similar_stats_logger = SimilarRequestStatsLogger()
        
    def add_request(self, request):
        """添加新请求到等待队列"""
        self.waiting_queue.put(request)
        with self.stats_lock:
            self.total_generated += 1
        
    def get_waiting_request(self):
        """从等待队列获取请求（非阻塞）"""
        try:
            return self.waiting_queue.get_nowait()
        except queue.Empty:
            return None
            
    def add_to_executing(self, request):
        """添加请求到执行队列"""
        with self.lock:
            if len(self.executing_requests) < self.max_batch_size:
                request.start_processing(time.time())
                self.executing_requests[request.query_id] = request
                with self.stats_lock:
                    self.total_processing += 1
                
                # 统计执行队列中存在相似问题的请求数量
                similar_count = self.count_similar_requests_in_executing()
                total_executing = len(self.executing_requests)
                
                # 记录到文件
                self.similar_stats_logger.log_similar_stats(
                    similar_count=similar_count,
                    total_executing=total_executing,
                    request_id=request.query_id
                )
                
                print(f"[BatchManager] 新请求加入执行队列，当前执行队列中有相似问题的请求数量: {similar_count}/{total_executing}")
                
                return True
            return False
    def check_and_remove_finished(self):
        """检查并移除已完成的请求"""
        finished_requests = []
        current_time = time.time()
        
        with self.lock:
            # 找出已完成的请求
            completed_ids = []
            for request_id, request in self.executing_requests.items():
                if request.is_finished(current_time):
                    request.complete_processing(current_time)  # 修改这里：finish_processing -> complete_processing
                    finished_requests.append(request)
                    completed_ids.append(request_id)
            
            # 从执行队列中移除已完成的请求
            for request_id in completed_ids:
                del self.executing_requests[request_id]
            
            # 更新统计信息
            with self.stats_lock:
                self.total_completed += len(finished_requests)
        
        return finished_requests
    def count_similar_requests_in_executing(self):
        """统计执行队列中存在相似问题的请求数量（调用时已持有lock）"""
        similar_count = 0
        for request in self.executing_requests.values():
            if hasattr(request, 'has_similar') and request.has_similar:
                similar_count += 1
        return similar_count
    
    def get_status(self):
        """获取当前状态"""
        with self.stats_lock:
            waiting_count = self.waiting_queue.qsize()
            with self.lock:
                executing_count = len(self.executing_requests)
                similar_count = self.count_similar_requests_in_executing()
            return {
                "waiting": waiting_count,
                "executing": executing_count,
                "executing_similar": similar_count,  # 新增：执行队列中相似请求数量
                "completed": self.total_completed,
                "generated": self.total_generated,
                "processing": self.total_processing
            }

# 统计数据收集器
class SimulationStats:
    def __init__(self):
        self.lock = Lock()
        self.request_arrival_times = []
        self.request_completion_times = []
        self.request_processing_times = []
        self.similarity_scores = []
        self.queue_lengths_over_time = []
        self.start_time = None
        
    def record_arrival(self, request):
        with self.lock:
            self.request_arrival_times.append(request.arrival_time)
            self.similarity_scores.append(request.similarity_score)
            
    def record_completion(self, request):
        with self.lock:
            self.request_completion_times.append(request.end_time)
            actual_processing_time = request.end_time - request.start_time
            self.request_processing_times.append(actual_processing_time)
            
    def record_queue_status(self, timestamp, status):
        with self.lock:
            self.queue_lengths_over_time.append({
                "timestamp": timestamp,
                "waiting": status["waiting"],
                "executing": status["executing"],
                "completed": status["completed"]
            })
            
    def get_summary(self):
        with self.lock:
            if not self.request_completion_times:
                return {"message": "No completed requests yet"}
                
            total_requests = len(self.request_completion_times)
            avg_processing_time = np.mean(self.request_processing_times)
            avg_similarity = np.mean(self.similarity_scores) if self.similarity_scores else 0
            
            return {
                "total_completed": total_requests,
                "avg_processing_time": avg_processing_time,
                "avg_similarity_score": avg_similarity,
                "min_processing_time": min(self.request_processing_times),
                "max_processing_time": max(self.request_processing_times)
            }

# 请求发射线程
class RequestGeneratorThread(threading.Thread):
    def __init__(self, batch_manager, stats, lambda_rate, max_requests, simulation_duration, test_data):
        super().__init__(name="RequestGenerator")
        self.batch_manager = batch_manager
        self.stats = stats
        self.lambda_rate = lambda_rate
        self.max_requests = max_requests
        self.simulation_duration = simulation_duration
        self.test_data = test_data
        self.stop_event = Event()
        self.request_count = 0
        
    # 在 RequestGeneratorThread 的 run 方法中，替换第235-236行
    def run(self):
        print(f"[{self.name}] 请求发射线程启动，λ={self.lambda_rate}")
        start_time = time.time()
        next_request_time = start_time
        
        while (not self.stop_event.is_set() and 
               (time.time() - start_time) < self.simulation_duration and 
               self.request_count < self.max_requests):
            
            current_time = time.time()
            
            # 检查是否需要生成新请求
            if current_time >= next_request_time:
                # 生成新请求
                query_item = random.choice(self.test_data)
                query_text = query_item["query"][0]
                query_id = self.generate_unique_hash_id(query_text)
                
                # 创建请求对象
                request = Request(query_text, query_id, current_time)
                
                # 使用DocumentRetriever进行真实相似度查询
                search_results = retriever.search(
                    query_text=query_text,
                    query_id=query_id,
                    k=1,  # 只需要最相似的一个结果
                    similarity_threshold=0.7  # 不设置阈值，获取最高相似度
                )
                    
                if search_results and len(search_results) > 0:
                        # 获取最高相似度分数
                    request.similarity_score = search_results[0]['similarity']
                    request.has_similar = request.similarity_score > sim_threshold  # 使用全局的sim_threshold (0.7)
                else:
                        # 如果没有搜索结果，设置默认值
                    request.similarity_score = 0.0
                    request.has_similar = False
                    
                
                # 添加到批处理管理器
                self.batch_manager.add_request(request)
                self.stats.record_arrival(request)
                
                self.request_count += 1
                
                if self.request_count % 10 == 0:
                    status = self.batch_manager.get_status()
                    print(f"[{self.name}] 已发射 {self.request_count} 个请求，队列状态: {status}")
                
                # 生成下一个请求的时间（泊松分布）
                interval = np.random.exponential(1.0 / self.lambda_rate)
                next_request_time = current_time + interval
            
            time.sleep(0.001)  # 短暂休眠，避免过度占用CPU
            
        print(f"[{self.name}] 完成，共发射 {self.request_count} 个请求")
        
    def stop(self):
        self.stop_event.set()
        
    def generate_unique_hash_id(self, text):
        text_bytes = text.encode('utf-8')
        hash_object = hashlib.md5(text_bytes)
        return hash_object.hexdigest()[:16]

# 模型执行线程
class ModelExecutionThread(threading.Thread):
    def __init__(self, batch_manager, stats):
        super().__init__(name="ModelExecution")
        self.batch_manager = batch_manager
        self.stats = stats
        self.stop_event = Event()
        
    def run(self):
        print(f"[{self.name}] 模型执行线程启动")
        
        while not self.stop_event.is_set():
            # 尝试从等待队列获取请求
            request = self.batch_manager.get_waiting_request()
            
            if request is not None:
                # 尝试添加到执行队列
                if self.batch_manager.add_to_executing(request):
                    similar_status = "有相似" if (hasattr(request, 'has_similar') and request.has_similar) else "无相似"
                    print(f"[{self.name}] 开始处理请求: {request.query_text[:30]}... (ID: {request.query_id}, {similar_status})")
                else:
                    # 执行队列满了，重新放回等待队列
                    self.batch_manager.add_request(request)
            
            time.sleep(0.01)  # 短暂休眠
            
        print(f"[{self.name}] 停止")
        
    def stop(self):
        self.stop_event.set()

# 请求调度线程
class RequestSchedulerThread(threading.Thread):
    def __init__(self, batch_manager, stats):
        super().__init__(name="RequestScheduler")
        self.batch_manager = batch_manager
        self.stats = stats
        self.stop_event = Event()
    
    def run(self):
        print(f"[{self.name}] 请求调度线程启动")
        last_status_time = time.time()
        
        while not self.stop_event.is_set():
            # 检查并移除已完成的请求
            finished_requests = self.batch_manager.check_and_remove_finished()
            
            if finished_requests:
                print(f"[{self.name}] 完成 {len(finished_requests)} 个请求")
                
                # 记录完成的请求
                for request in finished_requests:
                    self.stats.record_completion(request)
                    
                # 批量更新数据库（如果需要）
                self.update_database_batch(finished_requests)
            
            # 定期记录队列状态
            current_time = time.time()
            if current_time - last_status_time >= 1.0:  # 每秒记录一次
                status = self.batch_manager.get_status()
                self.stats.record_queue_status(current_time, status)
                
                if status["executing"] > 0 or status["waiting"] > 0:
                    print(f"[{self.name}] 队列状态 - 等待: {status['waiting']}, 执行中: {status['executing']}, 执行中相似: {status['executing_similar']}, 已完成: {status['completed']}")
                
                last_status_time = current_time
            
            time.sleep(0.05)  # 调度线程可以稍微慢一点
            
        print(f"[{self.name}] 停止")
        
    def stop(self):
        self.stop_event.set()
        
    def update_database_batch(self, finished_requests):
        """批量更新数据库"""
        try:
            query_texts = [req.query_text for req in finished_requests]
            query_ids = [req.query_id for req in finished_requests]
            
            # 这里可以调用实际的数据库更新逻辑
            success = retriever.update_index(new_texts=query_texts, new_ids=query_ids, save_path='llma/src/faiss_dataset/metadata/')
            
            print(f"[{self.name}] 模拟数据库更新: {len(finished_requests)} 个请求")
            return True
        except Exception as e:
            print(f"[{self.name}] 数据库更新失败: {str(e)}")
            return False

# 主要的多线程模拟函数
def run_batch_processing_simulation(lambda_rate=1.0, simulation_duration=60, max_requests=100, batch_size=64):
    """
    运行多线程批处理模拟
    
    Args:
        lambda_rate: 泊松分布的λ参数（每秒平均请求数）
        simulation_duration: 模拟持续时间（秒）
        max_requests: 最大请求数
        batch_size: 批处理队列大小
    """
    print(f"\n=== 开始多线程批处理模拟 ===")
    print(f"参数: λ={lambda_rate}, 持续时间={simulation_duration}秒, 最大请求数={max_requests}, 批处理大小={batch_size}")

    
    # 初始化组件
    batch_manager = ThreadSafeBatchManager(max_batch_size=batch_size)
    stats = SimulationStats()
    stats.start_time = time.time()
    
    # 加载测试数据
    try:
        with open("llma/sample_data/output50.json", "r", encoding="utf-8") as f:
            test_data = json.load(f)
        print(f"加载测试数据: {len(test_data)} 条")
    except FileNotFoundError:
        print("测试数据文件未找到，使用模拟数据")
        test_data = [{"query": [f"测试查询 {i}"]} for i in range(100)]
    
    # 创建三个线程
    generator_thread = RequestGeneratorThread(
        batch_manager, stats, lambda_rate, max_requests, simulation_duration, test_data
    )
    execution_thread = ModelExecutionThread(batch_manager, stats)
    scheduler_thread = RequestSchedulerThread(batch_manager, stats)
    
    # 启动线程
    print("\n启动所有线程...")
    generator_thread.start()
    execution_thread.start()
    scheduler_thread.start()
    
    try:
        # 等待发射线程完成
        generator_thread.join()
        print("\n请求发射完成，等待处理完成...")
        
        # 等待所有请求处理完成
        while True:
            status = batch_manager.get_status()
            if status["executing"] == 0 and status["waiting"] == 0:
                break
            print(f"等待处理完成... 执行中: {status['executing']}, 等待中: {status['waiting']}")
            time.sleep(2)
        
        print("\n所有请求处理完成，停止线程...")
        
    except KeyboardInterrupt:
        print("\n收到中断信号，停止模拟...")
    finally:
        # 停止所有线程
        execution_thread.stop()
        scheduler_thread.stop()
        
        execution_thread.join(timeout=5)
        scheduler_thread.join(timeout=5)
    
    # 输出最终统计结果
    final_status = batch_manager.get_status()
    stats_summary = stats.get_summary()
    
    print(f"\n=== 多线程批处理模拟完成 ===")
    print(f"总发射请求数: {final_status['generated']}")
    print(f"总完成请求数: {final_status['completed']}")
    print(f"模拟总时长: {time.time() - stats.start_time:.2f}秒")
    
    if 'total_completed' in stats_summary:
        print(f"平均处理时间: {stats_summary['avg_processing_time']:.2f}秒")
        print(f"平均相似度: {stats_summary['avg_similarity_score']:.3f}")
        print(f"处理时间范围: {stats_summary['min_processing_time']:.2f}s - {stats_summary['max_processing_time']:.2f}s")
    
    # 保存结果
    save_simulation_results(stats, final_status, {
        "lambda_rate": lambda_rate,
        "simulation_duration": simulation_duration,
        "max_requests": max_requests,
        "batch_size": batch_size,
        "inference_time_constant": (2-15)
    })
    
    return stats, batch_manager

def save_simulation_results(stats, final_status, params):
    """保存模拟结果到文件"""
    results = {
        "timestamp": datetime.now().isoformat(),
        "parameters": params,
        "final_status": final_status,
        "statistics": stats.get_summary(),
        "queue_history": stats.queue_lengths_over_time
    }
    
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    output_file = os.path.join(log_dir, f"batch_simulation_{int(time.time())}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"模拟结果已保存到: {output_file}")

# 保持向后兼容的函数
def generate_unique_hash_id(text):
    if not isinstance(text, str):
        text = str(text)
    text_bytes = text.encode('utf-8')
    hash_object = hashlib.md5(text_bytes)
    return hash_object.hexdigest()[:16]

def update_database_batch(query_texts, query_ids):
    try:
        success = retriever.update_index(
            new_texts=query_texts,
            new_ids=query_ids,
            save_path='llma/src/faiss_dataset/metadata/'
        )
        return success
    except Exception as e:
        print(f"批量更新向量数据库失败: {str(e)}")
        return False

# 向后兼容的函数
def run_poisson_simulation(lambda_rate=1.0, simulation_duration=60, max_requests=100):
    return run_batch_processing_simulation(lambda_rate, simulation_duration, max_requests, 64)

if __name__ == "__main__":
    # 设置模拟参数
    lambda_rate = 2.0  # 每秒平均2个请求
    simulation_duration = 12002000 # 模拟120秒
    max_requests = 200000  # 最多200个请求
    batch_size = 32  # 批处理队列大小
    
    # 运行多线程批处理模拟
    stats, batch_manager = run_batch_processing_simulation(
        lambda_rate=lambda_rate,
        simulation_duration=simulation_duration,
        max_requests=max_requests,
        batch_size=batch_size
    )
    
    print("\n多线程批处理模拟完成！")