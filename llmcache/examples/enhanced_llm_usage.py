"""增强 AsyncLLM 使用示例

该示例展示了如何使用新的 EnhancedAsyncLLM 架构来处理请求。
包含以下功能演示：
1. 基本的增强 LLM 使用
2. 完整系统的创建和配置
3. 缓存和相似度搜索功能
4. 流式输出处理
5. 错误处理和资源清理
"""

import asyncio
import time
from typing import List, Dict, Any

from loguru import logger
from vllm.sampling_params import SamplingParams

# 导入增强组件
from src.handler.enhanced_llm_factory import create_enhanced_system, get_global_factory
from src.config.settings import VLLMConfig
from src.config.cache_config import CacheConfig
from src.config.similarity_config import SimilarityConfig
from src.config.handler_config import HandlerConfig
from src.models.enums import CacheType


class EnhancedLLMDemo:
    """增强 LLM 演示类"""
    
    def __init__(self):
        self.handler = None
        self.factory = get_global_factory()
    
    def create_configs(self) -> tuple:
        """创建配置对象
        
        Returns:
            (vllm_config, cache_config, similarity_config, handler_config)
        """
        # VLLM 配置
        vllm_config = VLLMConfig()
        vllm_config.engine_args.model = "microsoft/DialoGPT-medium"  # 示例模型
        vllm_config.engine_args.max_model_len = 2048
        vllm_config.engine_args.gpu_memory_utilization = 0.8
        vllm_config.engine_args.tensor_parallel_size = 1
        
        # 缓存配置
        cache_config = CacheConfig(
            cache_type=CacheType.REDIS,
            redis_host="localhost",
            redis_port=6379,
            redis_db=0,
            memory_cache_size=1000,
            ttl=3600,  # 1小时
            enable_compression=True
        )
        
        # 相似度搜索配置
        similarity_config = SimilarityConfig(
            enable_vector_search=True,
            similarity_threshold=0.8,
            max_similar_results=5,
            vector_dimension=768,
            index_type="faiss"
        )
        
        # 处理器配置
        handler_config = HandlerConfig()
        handler_config.features.enable_nlp_enhancement = True
        handler_config.features.enable_similarity_search = True
        handler_config.features.enable_caching = True
        handler_config.features.enable_detailed_logging = True
        
        return vllm_config, cache_config, similarity_config, handler_config
    
    async def setup_system(self) -> None:
        """设置增强系统"""
        try:
            logger.info("开始设置增强 LLM 系统...")
            
            # 创建配置
            vllm_config, cache_config, similarity_config, handler_config = self.create_configs()
            
            # 创建完整系统
            self.handler = await create_enhanced_system(
                vllm_config=vllm_config,
                cache_config=cache_config,
                similarity_config=similarity_config,
                handler_config=handler_config,
                instance_name="demo",
                nlp_max_concurrent=20
            )
            
            # 启动处理器
            await self.handler.start()
            
            logger.info("增强 LLM 系统设置完成")
            
        except Exception as e:
            logger.error(f"设置系统失败: {e}")
            raise
    
    async def demo_basic_generation(self) -> None:
        """演示基本生成功能"""
        logger.info("=== 基本生成演示 ===")
        
        try:
            prompt = "请解释什么是人工智能？"
            sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.9,
                max_tokens=200
            )
            
            # 使用新版本接口
            collector = await self.handler.add_request_v2(
                prompt=prompt,
                sampling_params=sampling_params
            )
            
            # 流式获取结果
            logger.info("开始生成...")
            start_time = time.time()
            
            async for output in collector.get_outputs():
                if output.finished:
                    end_time = time.time()
                    logger.info(f"生成完成，耗时: {end_time - start_time:.2f}s")
                    logger.info(f"结果: {output.text}")
                    logger.info(f"元数据: {output.metadata}")
                    break
                else:
                    # 流式输出
                    logger.debug(f"流式输出: {output.text}")
            
        except Exception as e:
            logger.error(f"基本生成演示失败: {e}")
    
    async def demo_cache_functionality(self) -> None:
        """演示缓存功能"""
        logger.info("=== 缓存功能演示 ===")
        
        try:
            prompt = "什么是机器学习？"
            sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.9,
                max_tokens=150
            )
            
            # 第一次请求（应该会生成新结果）
            logger.info("第一次请求（新生成）...")
            start_time = time.time()
            
            collector1 = await self.handler.add_request_v2(
                prompt=prompt,
                sampling_params=sampling_params
            )
            
            async for output in collector1.get_outputs():
                if output.finished:
                    first_time = time.time() - start_time
                    logger.info(f"第一次生成完成，耗时: {first_time:.2f}s")
                    logger.info(f"缓存命中: {output.metadata.get('cache_hit', False)}")
                    break
            
            # 等待一秒
            await asyncio.sleep(1)
            
            # 第二次请求（应该命中缓存）
            logger.info("第二次请求（缓存命中）...")
            start_time = time.time()
            
            collector2 = await self.handler.add_request_v2(
                prompt=prompt,
                sampling_params=sampling_params
            )
            
            async for output in collector2.get_outputs():
                if output.finished:
                    second_time = time.time() - start_time
                    logger.info(f"第二次生成完成，耗时: {second_time:.2f}s")
                    logger.info(f"缓存命中: {output.metadata.get('cache_hit', False)}")
                    logger.info(f"性能提升: {(first_time - second_time) / first_time * 100:.1f}%")
                    break
            
        except Exception as e:
            logger.error(f"缓存功能演示失败: {e}")
    
    async def demo_nlp_enhancement(self) -> None:
        """演示 NLP 增强功能"""
        logger.info("=== NLP 增强功能演示 ===")
        
        try:
            # 包含连接词的复杂问题
            prompt = "请解释深度学习和机器学习的区别，并且说明它们的应用场景。"
            sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.9,
                max_tokens=300
            )
            
            logger.info("处理包含连接词的复杂问题...")
            start_time = time.time()
            
            collector = await self.handler.add_request_v2(
                prompt=prompt,
                sampling_params=sampling_params
            )
            
            async for output in collector.get_outputs():
                if output.finished:
                    end_time = time.time()
                    logger.info(f"NLP 增强生成完成，耗时: {end_time - start_time:.2f}s")
                    
                    # 显示 NLP 分析结果
                    nlp_result = output.metadata.get('nlp_result', {})
                    if nlp_result:
                        logger.info(f"检测到连接词: {nlp_result.get('conjunctions', [])}")
                        logger.info(f"子句数量: {len(nlp_result.get('subsentences', []))}")
                        logger.info(f"相似上下文数量: {output.metadata.get('similar_contexts_count', 0)}")
                    
                    logger.info(f"结果: {output.text[:200]}...")
                    break
            
        except Exception as e:
            logger.error(f"NLP 增强功能演示失败: {e}")
    
    async def demo_batch_processing(self) -> None:
        """演示批量处理"""
        logger.info("=== 批量处理演示 ===")
        
        try:
            prompts = [
                "什么是自然语言处理？",
                "解释计算机视觉的基本概念。",
                "深度学习有哪些主要的网络架构？",
                "什么是强化学习？",
                "解释神经网络的工作原理。"
            ]
            
            sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.9,
                max_tokens=150
            )
            
            logger.info(f"开始批量处理 {len(prompts)} 个请求...")
            start_time = time.time()
            
            # 并发提交所有请求
            collectors = []
            for i, prompt in enumerate(prompts):
                collector = await self.handler.add_request_v2(
                    prompt=prompt,
                    sampling_params=sampling_params
                )
                collectors.append((i, prompt, collector))
            
            # 并发处理所有请求
            async def process_request(index, prompt, collector):
                async for output in collector.get_outputs():
                    if output.finished:
                        logger.info(f"请求 {index + 1} 完成: {prompt[:30]}...")
                        logger.info(f"  缓存命中: {output.metadata.get('cache_hit', False)}")
                        logger.info(f"  相似上下文: {output.metadata.get('similar_contexts_count', 0)}")
                        return output
            
            # 等待所有请求完成
            tasks = [process_request(i, prompt, collector) for i, prompt, collector in collectors]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            end_time = time.time()
            logger.info(f"批量处理完成，总耗时: {end_time - start_time:.2f}s")
            logger.info(f"平均每个请求: {(end_time - start_time) / len(prompts):.2f}s")
            
            # 统计结果
            successful = sum(1 for r in results if not isinstance(r, Exception))
            logger.info(f"成功处理: {successful}/{len(prompts)} 个请求")
            
        except Exception as e:
            logger.error(f"批量处理演示失败: {e}")
    
    async def demo_health_check(self) -> None:
        """演示健康检查"""
        logger.info("=== 健康检查演示 ===")
        
        try:
            health_status = await self.handler.health_check()
            logger.info(f"系统状态: {health_status['status']}")
            logger.info(f"组件状态: {health_status['components']}")
            logger.info(f"统计信息: {health_status['stats']}")
            
            # 显示工厂统计信息
            factory_stats = self.factory.get_instance_stats()
            logger.info(f"工厂统计: {factory_stats}")
            
        except Exception as e:
            logger.error(f"健康检查演示失败: {e}")
    
    async def cleanup(self) -> None:
        """清理资源"""
        try:
            if self.handler:
                await self.handler.stop()
            
            # 清理工厂中的所有实例
            await self.factory.cleanup_all()
            
            logger.info("资源清理完成")
            
        except Exception as e:
            logger.error(f"清理资源失败: {e}")
    
    async def run_all_demos(self) -> None:
        """运行所有演示"""
        try:
            # 设置系统
            await self.setup_system()
            
            # 运行各种演示
            await self.demo_basic_generation()
            await asyncio.sleep(2)
            
            await self.demo_cache_functionality()
            await asyncio.sleep(2)
            
            await self.demo_nlp_enhancement()
            await asyncio.sleep(2)
            
            await self.demo_batch_processing()
            await asyncio.sleep(2)
            
            await self.demo_health_check()
            
        except Exception as e:
            logger.error(f"演示运行失败: {e}")
        finally:
            await self.cleanup()


async def main():
    """主函数"""
    logger.info("开始增强 AsyncLLM 演示")
    
    demo = EnhancedLLMDemo()
    await demo.run_all_demos()
    
    logger.info("演示完成")


if __name__ == "__main__":
    # 配置日志
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # 运行演示
    asyncio.run(main())