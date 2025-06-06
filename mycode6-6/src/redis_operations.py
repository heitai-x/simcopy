import redis
import json
import pickle
import time
import threading
import hashlib
import os
import numpy as np
from typing import Dict, List, Any, Optional, Union

# Redis连接配置
REDIS_CONFIG = {
    'host': 'localhost',
    'port': 6379,
    'db': 0,
    'password': None,
    'decode_responses': False  # 不自动解码，因为我们存储的是序列化后的数据
}

# 键前缀，用于区分不同类型的数据
KEY_PREFIX = 'query:'

# 缓存配置
CACHE_LIMIT = 1  # 缓存限制，达到此数量时触发批量更新
UPDATE_INTERVAL = 300  # 更新间隔（秒），即使未达到缓存限制也会触发更新

# 本地文件备份路径
BACKUP_FILE_PATH = "llma/src/faiss_dataset/query_answers.json"

# 嵌入向量存储路径
EMBEDDINGS_DB_PATH = "llma/src/faiss_dataset/embeddings.lmdb"

# 全局变量
redis_client = None
answer_cache = {}  # 本地缓存
cache_count = 0  # 缓存计数
last_update_time = time.time()  # 最后更新时间
db_lock = threading.Lock()  # 线程锁

# 嵌入向量内存缓存
embeddings_cache = {}
MAX_EMBEDDINGS_CACHE_SIZE = 1000  # 最大缓存条目数

# 初始化LMDB环境
lmdb_env = None

def init_embeddings_db():
    """初始化嵌入向量数据库"""
    global lmdb_env
    
    # 确保目录存在
    os.makedirs(os.path.dirname(EMBEDDINGS_DB_PATH), exist_ok=True)
    
    # 初始化LMDB环境
    if lmdb_env is None:
        try:
            # 创建LMDB环境，设置较大的map_size以适应大量向量数据
            # 1TB的map_size，可以根据实际需求调整
            lmdb_env = lmdb.open(EMBEDDINGS_DB_PATH, map_size=1099511627776)
            print(f"已初始化LMDB环境: {EMBEDDINGS_DB_PATH}")
        except Exception as e:
            print(f"初始化LMDB环境失败: {str(e)}")
            lmdb_env = None
    
    return lmdb_env is not None

def save_embedding(embedding_id, embeddings):
    """保存嵌入向量到LMDB
    
    Args:
        embedding_id: 嵌入向量ID
        embeddings: 嵌入向量数据
    
    Returns:
        是否成功
    """
    global lmdb_env, embeddings_cache
    
    if lmdb_env is None:
        if not init_embeddings_db():
            return False
    
    try:
        # 序列化嵌入向量
        serialized = pickle.dumps(embeddings)
        
        # 写入LMDB
        with lmdb_env.begin(write=True) as txn:
            txn.put(embedding_id.encode(), serialized)
        
        # 更新内存缓存
        embeddings_cache[embedding_id] = embeddings
        
        # 如果缓存过大，移除最早的条目
        if len(embeddings_cache) > MAX_EMBEDDINGS_CACHE_SIZE:
            # 移除最早添加的条目（简单实现，实际可能需要LRU策略）
            oldest_key = next(iter(embeddings_cache))
            del embeddings_cache[oldest_key]
        
        return True
    except Exception as e:
        print(f"保存嵌入向量失败: {str(e)}")
        return False

def load_embedding(embedding_id):
    """从LMDB加载嵌入向量
    
    Args:
        embedding_id: 嵌入向量ID
    
    Returns:
        嵌入向量数据，如果不存在则返回None
    """
    global lmdb_env, embeddings_cache
    
    # 首先检查内存缓存
    if embedding_id in embeddings_cache:
        return embeddings_cache[embedding_id]
    
    if lmdb_env is None:
        if not init_embeddings_db():
            return None
    
    try:
        # 从LMDB读取
        with lmdb_env.begin() as txn:
            serialized = txn.get(embedding_id.encode())
            
            if serialized is None:
                return None
            
            # 反序列化
            embeddings = pickle.loads(serialized)
            
            # 更新内存缓存
            embeddings_cache[embedding_id] = embeddings
            
            # 如果缓存过大，移除最早的条目
            if len(embeddings_cache) > MAX_EMBEDDINGS_CACHE_SIZE:
                oldest_key = next(iter(embeddings_cache))
                del embeddings_cache[oldest_key]
            
            return embeddings
    except Exception as e:
        print(f"加载嵌入向量失败: {str(e)}")
        return None

def init_redis():
    """初始化Redis连接"""
    global redis_client
    try:
        redis_client = redis.Redis(**REDIS_CONFIG)
        redis_client.ping()  # 测试连接
        print("成功连接到Redis服务器")
        return True
    except redis.ConnectionError as e:
        print(f"无法连接到Redis服务器: {str(e)}")
        redis_client = None
        return False
    except Exception as e:
        print(f"初始化Redis时发生错误: {str(e)}")
        redis_client = None
        return False


def generate_unique_hash_id(text):
    """
    为查询文本生成唯一的哈希ID
    
    Args:
        text: 查询文本
    
    Returns:
        16位哈希ID
    """
    if not isinstance(text, str):
        text = str(text)
    
    text_bytes = text.encode('utf-8')
    hash_object = hashlib.md5(text_bytes)
    hash_hex = hash_object.hexdigest()
    
    return hash_hex[:16]


def _get_key(key):
    """
    获取带前缀的键
    """
    return f"{KEY_PREFIX}{key}"


def set_data(key, value):
    """
    设置键值对
    
    Args:
        key: 键
        value: 值（将被序列化）
    
    Returns:
        是否成功
    """
    if redis_client is None:
        if not init_redis():
            return False
    
    try:
        # 序列化值
        serialized_value = pickle.dumps(value)
        
        # 设置键值对
        redis_client.set(_get_key(key), serialized_value)
        return True
    except Exception as e:
        print(f"设置Redis键值对失败: {str(e)}")
        return False


def get_data(key):
    """
    获取键对应的值
    
    Args:
        key: 键
    
    Returns:
        值（已反序列化），如果不存在则返回None
    """
    if redis_client is None:
        if not init_redis():
            return None
    
    try:
        # 获取值
        value = redis_client.get(_get_key(key))
        
        # 如果值不存在，返回None
        if value is None:
            return None
        
        # 反序列化值
        return pickle.loads(value)
    except Exception as e:
        print(f"获取Redis键值对失败: {str(e)}")
        return None


def delete_data(key):
    """
    删除键值对
    
    Args:
        key: 键
    
    Returns:
        是否成功
    """
    if redis_client is None:
        if not init_redis():
            return False
    
    try:
        # 删除键值对
        redis_client.delete(_get_key(key))
        return True
    except Exception as e:
        print(f"删除Redis键值对失败: {str(e)}")
        return False


def exists_data(key):
    """
    检查键是否存在
    
    Args:
        key: 键
    
    Returns:
        是否存在
    """
    if redis_client is None:
        if not init_redis():
            return False
    
    try:
        # 检查键是否存在
        return redis_client.exists(_get_key(key)) > 0
    except Exception as e:
        print(f"检查Redis键是否存在失败: {str(e)}")
        return False


def set_batch_data(data):
    """
    批量设置键值对
    
    Args:
        data: 键值对字典
    
    Returns:
        是否成功
    """
    if redis_client is None:
        if not init_redis():
            return False
    
    if not data:
        return False
    
    try:
        # 创建管道，提高批量操作效率
        pipeline = redis_client.pipeline()
        
        # 批量设置键值对
        for key, value in data.items():
            serialized_value = pickle.dumps(value)
            pipeline.set(_get_key(key), serialized_value)
        
        # 执行管道操作
        pipeline.execute()
        return True
    except Exception as e:
        print(f"批量设置Redis键值对失败: {str(e)}")
        return False


def get_all_keys():
    """
    获取所有键
    
    Returns:
        键列表（已去除前缀）
    """
    if redis_client is None:
        if not init_redis():
            return []
    
    try:
        # 获取所有键
        pattern = f"{KEY_PREFIX}*"
        keys = redis_client.keys(pattern)
        
        # 去除前缀
        prefix_len = len(KEY_PREFIX)
        return [key.decode('utf-8')[prefix_len:] for key in keys]
    except Exception as e:
        print(f"获取所有Redis键失败: {str(e)}")
        return []


def get_all_data(load_embeddings=True):
    """
    获取所有键值对，包括嵌入向量
    
    Args:
        load_embeddings: 是否加载嵌入向量，默认为True
    
    Returns:
        键值对字典
    """
    if redis_client is None:
        if not init_redis():
            return {}
    
    try:
        # 获取所有键
        keys = get_all_keys()
        
        # 获取所有值
        result = {}
        for key in keys:
            value = get_data(key)
            if value is not None:
                # 处理嵌入向量
                if load_embeddings and "sim_sentence" in value and isinstance(value["sim_sentence"], list):
                    for i, result_item in enumerate(value["sim_sentence"]):
                        if "sim_embeddings_id" in result_item:
                            # 从LMDB加载嵌入向量
                            embedding_id = result_item["sim_embeddings_id"]
                            embeddings = load_embedding(embedding_id)
                            if embeddings is not None:
                                result_item["sim_embeddings"] = embeddings
                
                result[key] = value
        
        print(f"已加载 {len(result)} 个文档" + ("(包含嵌入向量)" if load_embeddings else ""))
        return result
    except Exception as e:
        print(f"获取所有Redis键值对失败: {str(e)}")
        return {}


def clear_all_data():
    """
    清空所有键值对
    
    Returns:
        是否成功
    """
    if redis_client is None:
        if not init_redis():
            return False
    
    try:
        # 获取所有键
        pattern = f"{KEY_PREFIX}*"
        keys = redis_client.keys(pattern)
        
        # 如果有键，则删除
        if keys:
            redis_client.delete(*keys)
        
        return True
    except Exception as e:
        print(f"清空Redis键值对失败: {str(e)}")
        return False


def backup_to_file(data=None):
    """
    备份数据到文件
    
    Args:
        data: 要备份的数据，如果为None则从Redis获取所有数据
    
    Returns:
        是否成功
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(BACKUP_FILE_PATH), exist_ok=True)
        
        # 如果没有提供数据，则从Redis获取所有数据
        if data is None:
            data = get_all_data()
        
        # 打开文件以追加模式
        with open(BACKUP_FILE_PATH, 'a', encoding='utf-8') as f:
            for query_id, doc_data in data.items():
                # 处理嵌入向量数据
                sim_sentence = doc_data.get("sim_sentence")
                
                if sim_sentence and isinstance(sim_sentence, list):
                    for i, result in enumerate(sim_sentence):
                        if "sim_embeddings" in result:
                            # 生成嵌入向量ID
                            embedding_id = f"{query_id}_{i}"
                            
                            # 保存嵌入向量到LMDB
                            save_embedding(embedding_id, result["sim_embeddings"])
                            
                            # 在sim_sentence中保存引用ID而非实际数据
                            result["sim_embeddings_id"] = embedding_id
                            del result["sim_embeddings"]
                
                # 创建新数据条目
                new_data = {
                    query_id: {
                        "query": doc_data.get("query"),
                        "answer": doc_data.get("answer"),
                        "n-grams": doc_data.get("n_grams"),
                        "sim_sentence": sim_sentence,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                }
                # 将新数据转换为JSON字符串并写入文件
                json.dump(new_data, f, ensure_ascii=False)
                f.write('\n')  # 添加换行符以分隔条目
        
        print(f"已备份 {len(data)} 条记录到文件")
        return True
    except Exception as e:
        print(f"备份到文件失败: {str(e)}")
        return False

def load_from_file():
    docs = {}
    os.makedirs(os.path.dirname(BACKUP_FILE_PATH), exist_ok=True)
    if os.path.exists(BACKUP_FILE_PATH):
        
        try:
            with open(BACKUP_FILE_PATH, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():  # 确保行不为空
                        try:
                            entry = json.loads(line)
                            for qid, data in entry.items():
                                docs[qid] = {
                                    "answer": data.get("answer"),
                                    "n_grams": data.get("n-grams"),
                                    "query": data.get("query")
                                }
                        except json.JSONDecodeError as json_err:
                            print(f"解析行失败: {line.strip()}, 错误: {json_err}")
                        except Exception as line_err:
                            print(f"处理行时发生错误: {line.strip()}, 错误: {line_err}")
            print(f"从文件加载了 {len(docs)} 个文档")
        except Exception as e:
            print(f"加载文档失败: {str(e)}")
            docs = {}
    else:
        print(f"文档文件不存在，将创建新文件: {BACKUP_FILE_PATH}")
    return docs
# def load_from_file():
#     """
#     从文件加载数据
    
#     Returns:
#         文档字典，键为查询ID
#     """
#     docs = {}
    
#     # 确保目录存在
#     os.makedirs(os.path.dirname(BACKUP_FILE_PATH), exist_ok=True)
    
#     if os.path.exists(BACKUP_FILE_PATH):
#         try:
#             with open(BACKUP_FILE_PATH, 'r', encoding='utf-8') as f:
#                 for line in f:
#                     if line.strip():  # 确保行不为空
#                         try:
#                             entry = json.loads(line)
#                             for qid, data in entry.items():
#                                 # 处理sim_sentence中的嵌入向量引用
#                                 sim_sentence = data.get("sim_sentence")
#                                 if sim_sentence and isinstance(sim_sentence, list):
#                                     for result in sim_sentence:
#                                         if "sim_embeddings_id" in result:
#                                             # 从LMDB加载嵌入向量
#                                             embedding_id = result["sim_embeddings_id"]
#                                             embeddings = load_embedding(embedding_id)
#                                             if embeddings is not None:
#                                                 result["sim_embeddings"] = embeddings
#                                                 del result["sim_embeddings_id"]
                                
#                                 docs[qid] = {
#                                     "answer": data.get("answer"),
#                                     "sim_sentence": sim_sentence,
#                                     "n_grams": data.get("n-grams"),
#                                     "query": data.get("query")
#                                 }
#                         except json.JSONDecodeError as json_err:
#                             print(f"解析行失败: {line.strip()}, 错误: {json_err}")
#                         except Exception as line_err:
#                             print(f"处理行时发生错误: {line.strip()}, 错误: {line_err}")
#             print(f"从文件加载了 {len(docs)} 个文档")
#         except Exception as e:
#             print(f"加载文档失败: {str(e)}")
#             docs = {}
#     else:
#         print(f"文档文件不存在，将创建新文件: {BACKUP_FILE_PATH}")
    
#     return docs


def add_data(query_text, answer, n_grams=None, sim_sentence=None, query_id=None):
    """
    添加新的查询结果
    
    Args:
        query_text: 查询文本
        answer: 生成的答案
        n_grams: 生成的n-grams，如果为None则不添加
        sim_sentence: 生成的相似句子，如果为None则不添加
        query_id: 查询ID，如果为None则自动生成
    
    Returns:
        查询ID
    """
    global cache_count, last_update_time, answer_cache
    
    if query_id is None:
        query_id = generate_unique_hash_id(query_text)
    
    if exists_data(query_id):
        print(f"查询ID {query_id} 已存在，跳过添加操作")
        return query_id
    with db_lock:
        # 准备数据
        data = {
            "query": query_text,
            "answer": answer,
            "n_grams": n_grams,
            # "sim_sentence": sim_sentence,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 添加到缓存
        answer_cache[query_id] = data
        
        # 立即更新Redis
        set_data(query_id, data)
        
        cache_count += 1
        
        # 检查是否需要批量备份到文件
        current_time = time.time()
        if cache_count >= CACHE_LIMIT or (current_time - last_update_time) >= UPDATE_INTERVAL:
            # 启动一个新线程进行批量备份，避免阻塞主线程
            backup_thread = threading.Thread(target=process_cache_backup)
            backup_thread.daemon = True
            backup_thread.start()
    
    return query_id


def add_data_async(query_text, answer, feature_generator, update_retriever=None):
    """
    异步添加新的查询结果，特征生成在后台线程中进行
    
    Args:
        query_text: 查询文本
        answer: 生成的答案
        feature_generator: 特征生成函数，接收answer参数，返回(n_grams, sim_sentence)元组
        update_retriever: 更新检索器的回调函数
    
    Returns:
        查询ID
    """
    query_id = generate_unique_hash_id(query_text)
        # 检查query_id是否已存在
    if exists_data(query_id):
        print(f"查询ID {query_id} 已存在，跳过异步添加操作")
        return query_id
    def delayed_features():
        try:
            # 生成特征
            # n_grams, sim_sentence = feature_generator(answer)
            n_grams=feature_generator(answer)
            # 添加到Redis
            # add_data(query_text, answer, n_grams, sim_sentence, query_id)
            add_data(query_text, answer, n_grams, query_id=query_id)
            # 更新检索器
            if update_retriever is not None:
                update_retriever([query_text], [query_id])
            
            print(f"已完成查询 '{query_text[:30]}...' 的特征生成和缓存")
        except Exception as e:
            print(f"特征生成或缓存过程中发生错误: {str(e)}")
    
    # 启动异步处理线程
    feature_thread = threading.Thread(target=delayed_features)
    feature_thread.daemon = True
    feature_thread.start()
    
    return query_id


def process_cache_backup():
    """
    处理缓存备份，包括保存到文件
    """
    global cache_count, last_update_time, answer_cache
    
    with db_lock:
        if not answer_cache:
            return
        
        # 备份到文件
        backup_success = backup_to_file(answer_cache)
        
        if backup_success:
            print(f"已批量备份 {len(answer_cache)} 条记录到文件")
        
        # 清空缓存
        answer_cache.clear()
        cache_count = 0
        last_update_time = time.time()


def start_periodic_check():
    """
    启动定期检查缓存是否需要更新
    """
    def check_cache_update():
        current_time = time.time()
        if answer_cache and (current_time - last_update_time) >= UPDATE_INTERVAL:
            print(f"定时更新触发，已经 {(current_time - last_update_time):.1f} 秒未更新")
            process_cache_backup()
        
        # 设置下一次检查
        threading.Timer(60, check_cache_update).start()  # 每分钟检查一次
    
    # 启动定期检查
    check_cache_update()


def flush_cache(update_retriever=None):
    """
    刷新缓存，确保所有数据都被保存
    
    Args:
        update_retriever: 更新检索器的回调函数
    """
    if answer_cache:
        process_cache_backup()
        
        # 更新检索器
        if update_retriever is not None:
            query_texts = [data["query"] for data in answer_cache.values()]
            query_ids = list(answer_cache.keys())
            
            if query_texts and query_ids:
                update_success = update_retriever(query_texts, query_ids)
                if update_success:
                    print(f"已更新 {len(query_texts)} 条记录到向量数据库")
        
        print(f"已刷新缓存并保存所有数据")


# 初始化Redis连接
init_redis()

# 启动定期检查
start_periodic_check()

# 注册退出时的清理函数
import atexit
atexit.register(flush_cache)

# 从文件加载数据并同步到Redis
def init_data():
    # 从文件加载数据
    docs = load_from_file()
    
    # 同步到Redis
    if docs and redis_client is not None:
        set_batch_data(docs)
        print(f"已将 {len(docs)} 个文档同步到Redis")
    
    return docs

# 初始化数据
init_data()
clear_all_data()