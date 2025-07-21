# LLMCache - 增强型 LLM 缓存系统

基于 VLLM AsyncLLM 的高性能缓存系统，集成智能缓存、NLP 分析和相似度搜索功能，为大语言模型提供更高效的推理服务。

## 🚀 核心特性

### 1. 智能缓存系统
- **多级缓存架构**: 内存缓存 + Redis 缓存，提供毫秒级响应
- **精确匹配**: 基于请求哈希的精确缓存命中
- **相似度匹配**: 基于语义向量的智能缓存复用
- **自动管理**: 缓存生命周期和清理机制

### 2. 增强型 AsyncLLM
- **真正继承**: 完全继承 VLLM v1 的 `AsyncLLM`，保持原生兼容性
- **重写核心方法**: 增强 `add_request` 方法，集成缓存和 NLP 功能
- **流式输出**: 支持实时流式生成和元数据传递
- **无缝替换**: 可直接替换现有 AsyncLLM 实例

### 3. NLP 增强处理
- **异步分析**: 并行执行 NLP 处理，不阻塞生成流程
- **连接词提取**: 自动识别复杂问题中的连接词
- **子句分解**: 将复杂问题分解为多个子问题
- **上下文增强**: 基于 NLP 结果提供更丰富的上下文

### 4. 向量相似度搜索
- **语义搜索**: 基于 Sentence Transformers 的语义向量搜索
- **FAISS 索引**: 高效的向量索引和检索
- **批量处理**: 支持批量相似度搜索
- **智能去重**: 自动去重和排序相似结果

### 5. 共享内存管理
- **高性能存储**: 基于共享内存的高速数据交换
- **请求映射**: 智能的请求到响应映射
- **内存优化**: 自动内存清理和压缩

## 📁 项目结构

```
llmcache/
├── src/
│   ├── cache/                     # 缓存系统
│   │   ├── multi_level_cache.py   # 多级缓存管理
│   │   ├── redis_manager.py       # Redis 缓存管理
│   │   └── enhanced_vector_search.py # 向量搜索
│   ├── handler/                   # 处理器模块
│   │   ├── enhanced_async_llm.py  # 增强 AsyncLLM
│   │   └── similar_request_memory.py # 相似请求内存管理
│   ├── nlp/                      # NLP 处理
│   │   └── async_conjunction_extractor.py # 连接词提取
│   ├── config/                   # 配置管理
│   │   ├── settings.py           # 主配置
│   │   ├── handler_config.py     # 处理器配置
│   │   └── nlp_config.py         # NLP 配置
│   ├── models/                   # 数据模型
│   │   ├── request.py            # 请求模型
│   │   ├── cache.py              # 缓存模型
│   │   └── enums.py              # 枚举定义
│   └── utils/                    # 工具模块
│       ├── hasher.py             # 哈希工具
│       ├── logger.py             # 日志工具
│       └── similarity_search_helper.py # 相似度搜索助手
├── examples/                     # 使用示例
│   ├── enhanced_llm_usage.py     # 基本使用示例
│   └── complete_speculative_example.py # 完整示例
├── config/                       # 配置文件
    └── shared_memory_config.json # 共享内存配置
```

## 🛠️ 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repository-url>
cd llmcache

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件，配置模型路径和 Redis 连接
```

### 2. 基本使用

```python
import asyncio
from vllm.sampling_params import SamplingParams
from src.handler.enhanced_async_llm import EnhancedAsyncLLM
from src.config.handler_config import HandlerConfig

async def main():
    # 配置
    config = HandlerConfig(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        cache_enabled=True,
        nlp_enabled=True
    )
    
    # 创建增强的 AsyncLLM
    llm = EnhancedAsyncLLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        handler_config=config
    )
    
    # 采样参数
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=200
    )
    
    # 使用继承的 add_request 方法
    collector = await llm.add_request(
        request_id="my_request_1",
        prompt="什么是人工智能？",
        params=sampling_params
    )
    
    # 流式获取结果
    async for output in collector.aiter():
        print(f"结果: {output.text}")
        if output.finished:
            print(f"缓存命中: {output.metadata.get('cached', False)}")
            print(f"处理时间: {output.metadata.get('processing_time', 0):.3f}s")
            break

if __name__ == "__main__":
    asyncio.run(main())
```

## ⚙️ 配置说明

### 环境变量配置

主要配置项（详见 `.env.example`）：

```bash
# 模型配置
VLLM_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
VLLM_GPU_MEMORY_UTILIZATION=0.90
VLLM_MAX_MODEL_LEN=4096

# 缓存配置
VLLM_REDIS_URL=redis://localhost:6379
VLLM_ENABLE_REDIS=true
VLLM_MEMORY_CACHE_SIZE=100

# 向量搜索配置
VLLM_VECTOR_SEARCH_ENABLED=true
VLLM_SIMILARITY_THRESHOLD=0.7
VLLM_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# NLP 配置
VLLM_NLP_ENABLED=true
VLLM_NLP_CONJUNCTION_EXTRACTION=true
```

### 代码配置

```python
# VLLM 配置
vllm_config = VLLMConfig()
vllm_config.engine_args.model = "your-model-path"
vllm_config.engine_args.max_model_len = 4096
vllm_config.engine_args.gpu_memory_utilization = 0.8

# 缓存配置
cache_config = CacheConfig(
    cache_type=CacheType.REDIS,
    redis_host="localhost",
    redis_port=6379,
    memory_cache_size=1000,
    ttl=3600
)

# 相似度搜索配置
similarity_config = SimilarityConfig(
    enable_vector_search=True,
    similarity_threshold=0.8,
    max_similar_results=5
)
```

## 🔧 高级功能

### 1. 推测解码

```python
# 启用推测解码以提高性能
config.features.enable_speculative_decoding = True
config.speculative.num_tokens = 25
config.speculative.similarity_threshold = 0.7
```

### 2. 并发控制

```python
# 设置最大并发请求数
config.performance.max_concurrent_requests = 4
config.performance.nlp_max_concurrent = 50
```

### 3. 监控和日志

```python
# 启用详细监控
config.features.enable_monitoring = True
config.features.enable_detailed_logging = True
```

## 📊 性能优势

- **缓存命中率**: 
- **响应时间**: 
- **吞吐量**: 
- **资源利用**: 

## 🧪 测试和验证

```bash
# 运行健康检查
python scripts/health_check.py

# 运行性能测试
python scripts/benchmark.py

# 运行完整示例
python examples/enhanced_llm_usage.py

# 运行导入测试
python test_imports.py
```

## 🔧 依赖要求

### 核心依赖
- Python >= 3.8
- VLLM >= 0.6.0
- Redis >= 6.0
- CUDA >= 11.8 (GPU 推理)

### Python 包
```
numpy>=1.21.0
redis>=4.0.0
faiss-cpu>=1.7.0
sentence-transformers>=2.2.0
spacy>=3.4.0
vllm==9.1
transformers>=4.20.0
torch>=1.12.0
aioredis>=2.0.0
loguru>=0.6.0
```

## 🚀 部署指南



### 生产环境部署


## 🤝 贡献指南


## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- [VLLM](https://github.com/vllm-project/vllm) - 高性能 LLM 推理引擎
- [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) - 语义向量模型
- [FAISS](https://github.com/facebookresearch/faiss) - 向量相似度搜索
- [Redis](https://redis.io/) - 高性能缓存数据库

## 📞 支持


**LLMCache** - 让 LLM 推理更快、更智能！ 🚀
