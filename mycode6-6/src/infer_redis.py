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

from decode import CopyModel
from tree import ConjunctionExtractor
import threading
import hashlib

# 导入Redis操作模块
import redis_operations as redis_ops
sim_threshold=0.7
trigger_N=2
# 初始化 DocumentRetriever
retriever = DocumentRetriever(
    metadata_path='llma/src/faiss_dataset/metadata/metadata.json',
    index_path='llma/src/faiss_dataset/metadata/vector_index_cosine.faiss'
)
model = CopyModel(
    model_path="Qwen/Qwen2.5-7B-Instruct",
    trigger_N=2,
    block_K=20,
    min_block_K=5,
    max_n=5
    
)
subq=ConjunctionExtractor()
def update_database_batch(query_texts, query_ids):
    """
    批量更新向量数据库
    
    Args:
        query_texts: 查询文本列表
        query_ids: 查询ID列表
    """
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

        
def get_with_retrieval(query_text, query_id, k=3, docs=None):
    """
    获取检索结果
    
    Args:
        query_text: 查询文本
        query_id: 查询ID
        k: 检索结果数量
        docs: 文档字典
    
    Returns:
        similarity_docs: 相似文档
        doc_ngrams: 文档n-grams
        sim_sentences: 相似句子
    """
    results = {"sub_queries": None, "similar_docs": None}
    
    if retriever is not None:
        # 使用线程并行执行子查询提取和相似文档搜索
        def extract_sub_queries():
            results["sub_queries"] = subq.extract(sentence=query_text, doc=None, id=query_id)
            
        def search_similar_docs():
            results["similar_docs"] = retriever.search(query_text, query_id, k=k)
            
        # 创建并启动线程
        extract_thread = threading.Thread(target=extract_sub_queries)
        search_thread = threading.Thread(target=search_similar_docs)
        
        extract_thread.start()
        search_thread.start()
        
        # 等待线程完成
        extract_thread.join()
        search_thread.join()
    if results["similar_docs"]!=[] and results["similar_docs"][0]["similarity"] >=0.95:
        similar_doc_id = results["similar_docs"][0]["id"]
        if docs and similar_doc_id in docs:
            return docs[similar_doc_id]["answer"],[],None,True
        else:
            print(f"在内存缓存中未找到ID为{similar_doc_id}的文档，尝试从Redis获取...")
            doc = redis_ops.get_data(similar_doc_id)
            if doc is not None and "answer" in doc:
                # 如果从Redis获取成功，返回答案
                print(f"成功从Redis获取ID为{similar_doc_id}的文档")
                # 更新内存缓存
                if docs is not None:
                    docs[similar_doc_id] = doc
                return doc["answer"],[],None,True
            else:
                print(f"警告：无法从Redis获取ID为{similar_doc_id}的文档，将继续使用相似文档")
    # 没有一样的问题的话，就利用伪分解出的子问题来丰富相似问题，获取相似问题的文本和id
    # 处理检索结果
    all_similar_docs = []
    sub_queries=None
    if len(results["similar_docs"]) <= 3 and results["sub_queries"] is not None:
        sub_queries = results["sub_queries"]['variants']
        seen_texts = set()
        for similar_doc in results["similar_docs"]:
            if similar_doc['similarity'] >= sim_threshold and similar_doc['text'] not in seen_texts:
                all_similar_docs.append(similar_doc)
                seen_texts.add(similar_doc['text'])
                
        # 添加子查询的相似文档
        if sub_queries is not None:
            for sub_query in sub_queries:
                sub_similar_doc = retriever.search(sub_query, query_id, k=2)
                if sub_similar_doc and len(sub_similar_doc) > 0:
                    if sub_similar_doc[0]['similarity'] >= sim_threshold and sub_similar_doc[0]['text'] not in seen_texts:
                        all_similar_docs.append(sub_similar_doc[0])
                        seen_texts.add(sub_similar_doc[0]['text'])
    elif results["similar_docs"]:
        all_similar_docs = results["similar_docs"]
    
    # 准备返回数据
    doc_ngrams = {
        "doc_ngrams": [],
        "doc_token_id_list": []
    }
    similarity_docs = {"sim_query": [], "similar_answer": []}
    
    # 处理相似文档
    for similar_doc in all_similar_docs:
        doc_id = similar_doc["id"]
        
        # 检查文档是否在内存缓存中
        if not docs or doc_id not in docs:
            # 尝试从Redis获取文档
            print(f"在内存缓存中未找到ID为{doc_id}的文档，尝试从Redis获取...")
            doc = redis_ops.get_data(doc_id)
            if doc is None or "answer" not in doc:
                print(f"警告：无法从Redis获取ID为{doc_id}的文档，跳过此文档")
                continue
            print(f"成功从Redis获取ID为{doc_id}的文档")
            if docs is not None:
                docs[doc_id] = doc
        else:
            doc = docs[doc_id]
            
        similarity_docs["sim_query"].append(similar_doc["text"])
        similarity_docs["similar_answer"].append(doc["answer"])
        
        # if doc.get("sim_sentence"):
        #     sim_sentences.extend(doc["sim_sentence"])
        # else:
        #     doc["sim_sentence"] = model.prepare_sentences([doc["answer"]])
        #     sim_sentences.extend(doc["sim_sentence"])
        #     # 更新Redis中的文档
        #     redis_ops.set_data(similar_doc["id"], doc)
            
        if doc.get("n_grams"):
            doc_ngrams["doc_ngrams"].extend(doc["n_grams"]["doc_ngrams"])
            doc_ngrams["doc_token_id_list"].extend(doc["n_grams"]["doc_token_id_list"])
        else:
            doc["n_grams"] = model.prepare_ngrams([doc["answer"]], n=1)
            doc_ngrams["doc_ngrams"].extend(doc["n_grams"]["doc_ngrams"])
            doc_ngrams["doc_token_id_list"].extend(doc["n_grams"]["doc_token_id_list"])
            redis_ops.set_data(similar_doc["id"], doc)
    
    return similarity_docs, doc_ngrams,sub_queries,False

# 添加日志记录函数
def log_query_info(query_id, query_text, answer, time_info):
    """
    记录查询ID、查询文本、答案和运行时间到日志文件
    
    Args:
        query_id: 查询ID
        query_text: 查询文本
        answer: 生成的答案
        time_info: 包含各种运行时间的字典
    """
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, "query_logs.jsonl")
    
    # 创建日志记录
    log_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "query_id": query_id,
        "query_text": query_text,
        "answer": answer,
        "time_info": time_info
    }
    
    # 将日志写入文件
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    
    print(f"查询信息已记录到 {log_file}")

def generate_with_retrieval(query_text, query_id, max_new_tokens=1024, docs=None, update_db=True,sync_update=False):
    start_time_total = time.time()
    time_info = {}
    
    # 记录检索开始时间
    start_time_retrieval = time.time()
    similarity_docs,doc_ngrams,sub_queries,Copy= get_with_retrieval(query_text, query_id, k=3, docs=docs)
    end_time_retrieval = time.time()
    time_info["retrieval_time"] = round(end_time_retrieval - start_time_retrieval, 4)
    outputs=None
    if Copy is False:
    # 生成答案
        if similarity_docs["sim_query"]!=[]:
            start_time_branch1 = time.time()
            input_ids = model.get_input_text(query_text, similarity_docs if similarity_docs["sim_query"] else None)
            outputs,generate_ids_num, llma_num,number = model.copy_generate(input_ids=input_ids.input_ids, max_new_tokens=max_new_tokens, ngrams_cache=doc_ngrams)

            answer = model.tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            end_time_branch1 = time.time()
            branch_time = end_time_branch1 - start_time_branch1
            time_info["branch"] = "有相似文档"
            time_info["branch_time"] = round(branch_time, 4)
            time_info["generate_ids_num"] = generate_ids_num
            time_info["llma_num"] = llma_num
            time_info["number"] = number
            time_info["speed"] = round(generate_ids_num/branch_time,4)
            print(f"分支1（有相似文档）执行时间: {branch_time:.4f} 秒")
            print("speed:",round(generate_ids_num/branch_time,4),"tokens/s")
        else:
            start_time_branch2 = time.time()
            input_ids = model.get_input_text(query_text, similarity_docs if similarity_docs["sim_query"] else None)
            
            outputs,generate_ids_num = model.base_generate(input_ids=input_ids.input_ids, max_new_tokens=max_new_tokens)
            answer = model.tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            end_time_branch2 = time.time()
            branch_time = end_time_branch2 - start_time_branch2
            time_info["branch"] = "无相似文档"
            time_info["branch_time"] = round(branch_time, 4)
            time_info["generate_ids_num"] = generate_ids_num
            time_info["speed"] = round(generate_ids_num/branch_time,4)
            print(f"分支2（无相似文档）执行时间: {branch_time:.4f} 秒")
            print("speed:",round(generate_ids_num/branch_time,4),"tokens/s")
    else:
        answer=similarity_docs
        time_info["branch"] = "完全匹配"
        print("完全匹配")
        
    end_time_total = time.time()
    total_time = end_time_total - start_time_total
    time_info["total_time"] = round(total_time, 4)
    print(f"generate_with_retrieval 总执行时间: {total_time:.4f} 秒")
    
    # 记录查询信息
    log_query_info(query_id, query_text, answer, time_info)
    def feature_generator(answer_text):
        n_grams_result = [None]
        # sim_sentence_result = [None]
        
        def generate_ngrams():
            n_grams_result[0] = model.prepare_ngrams([answer_text], n=trigger_N)
            
        # def generate_sim_sentence():
        #     sim_sentence_result[0] = model.prepare_sentences([answer_text])
        
        # 创建并启动线程
        ngrams_thread = threading.Thread(target=generate_ngrams)
        # sim_sentence_thread = threading.Thread(target=generate_sim_sentence)
        
        ngrams_thread.start()
        # sim_sentence_thread.start()
        
        # 等待线程完成
        ngrams_thread.join()
        # sim_sentence_thread.join()
        
        # return n_grams_result[0], sim_sentence_result[0]
        return n_grams_result[0]
    # 添加到Redis（支持同步和异步模式）
    if update_db:
        if sync_update:
            # 同步模式：等待数据更新完成
            print("正在同步保存查询结果和嵌入向量...")
            # n_grams, sim_sentence = feature_generator(answer,outputs)
            # 第一步：添加数据到Redis
            n_grams=feature_generator(answer)
            query_id = redis_ops.add_data(
                query_text=query_text,
                answer=answer,
                n_grams=n_grams
            )
            print(f"查询结果已保存到Redis，ID: {query_id}")
            
            # 第二步：更新向量数据库
            print("正在更新向量数据库...")
            texts = [query_text]
            ids = [query_id]
            if sub_queries:
                for sub_q in sub_queries:
                    texts.append(sub_q)
                    ids.append(query_id) 
            success = update_database_batch(texts, ids)
            if success:
                print(f"向量数据库更新成功，ID: {query_id}")
            else:
                print(f"警告：向量数据库更新可能失败，ID: {query_id}")
                
            print(f"查询结果和嵌入向量已完全同步保存，ID: {query_id}")
            # 确保两个操作都完成后才继续执行
        else:
            # 异步模式：后台处理更新
            redis_ops.add_data_async(
                query_text=query_text,
                answer=answer,
                feature_generator=feature_generator,
                update_retriever=update_database_batch
            )
    
    return answer


# 使用示例
if __name__ == "__main__":
    # 使用Redis操作模块获取所有文档
    docs = redis_ops.get_all_data(load_embeddings=True)
    print(f"已加载文档数量: {len(docs)}")
    sync_update = True
    # 检查是否成功加载嵌入向量
    # embedding_count = 0
    # for doc_id, doc in docs.items():
    #     if "sim_sentence" in doc and isinstance(doc["sim_sentence"], list):
    #         for item in doc["sim_sentence"]:
    #             if "sim_embeddings" in item and isinstance(item["sim_embeddings"], np.ndarray):
    #                 embedding_count += 1
    
    # if embedding_count > 0:
    #     print(f"成功加载 {embedding_count} 个嵌入向量")
    # else:
    #     print("未找到嵌入向量数据，请确保数据已正确保存")
    average_arrival_rate_per_second = 0.5 
    # 计算指数分布的 scale 参数（平均间隔时间）
    average_interval_seconds = 1.0 / average_arrival_rate_per_second
    with open("llma/sample_data/output_sample.json", "r", encoding="utf-8") as f:
        data=json.load(f)
    previous_arrival_time = time.time()
    for item in data:
        query = item["query"][0]
        delay_time=0
        now_time = time.time()
        if now_time>previous_arrival_time:
            delay_time = now_time - previous_arrival_time
            print(f"当前请求到达需要等待,延迟时间: {delay_time:.4f} 秒")
        else:
            delay_time=0
            print(f"当前请求到达不需要等待")
        next_arrival_time = previous_arrival_time + np.random.exponential(scale=average_interval_seconds)
        
        previous_arrival_time = next_arrival_time
        
        # query ="I want to play a game and sing a song"
        query_id = redis_ops.generate_unique_hash_id(query)

        # 定义同步更新模式变量
        start_time_main = time.time()
        response = generate_with_retrieval(query, query_id, max_new_tokens=1024, docs=docs, sync_update=sync_update)
        end_time_main = time.time()
        main_time = end_time_main - start_time_main
        print(f"主程序调用generate_with_retrieval执行时间: {main_time:.4f} 秒")
        print(f"查询ID: {query_id}, 查询文本: {query}, 生成的答案: {response}")
        # 将主程序执行时间也记录到日志
        main_time_info = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "delay_time": round(delay_time, 4),
            "query_id": query_id,
            "query_text": query,
            "main_execution_time": round(main_time, 4),
            "update_mode": "异步" if not sync_update else "同步"
        }
    
        # 将主程序时间信息追加到日志
        log_dir = "logs"
        log_file = os.path.join(log_dir, "main_execution_logs.jsonl")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(main_time_info, ensure_ascii=False) + "\n")
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()