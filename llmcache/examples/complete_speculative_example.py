"""完整的投机解码配置案例

这是一个完整的示例，展示如何在llmcache项目中使用投机解码配置来提升推理性能。
包含：
1. 完整的配置设置
2. 系统初始化
3. 性能测试
4. 错误处理
5. 资源清理
"""
import os
os.environ["VLLM_USE_V1"] = "1"
os.environ['TRANSFORMERS_CACHE'] = '/home/xudebin'
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import asyncio
import sys
import time
import json
import uuid
from typing import Dict, Any, List, Optional
from pathlib import Path

# Add the parent directory to Python path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from vllm.sampling_params import SamplingParams
from vllm.config import VllmConfig
from vllm.v1.executor.abstract import Executor

# 导入增强组件
from src.handler.enhanced_async_llm import EnhancedAsyncLLM
from src.cache.multi_level_cache import MultiLevelCache
from src.utils.similarity_search_helper import SimilaritySearchHelper
from src.config.settings import CacheConfig, VectorSearchConfig
from src.config.handler_config import HandlerConfig, HandlerFeatureConfig
from src.models.enums import CacheLevel


class CompleteSpeculativeDemo:
    """完整的投机解码演示案例"""
    
    def __init__(self):
        self.enhanced_llm: Optional[EnhancedAsyncLLM] = None
        self.cache_manager: Optional[MultiLevelCache] = None
        self.similarity_search_helper: Optional[SimilaritySearchHelper] = None
        self.performance_stats = []
    
    def create_speculative_config(self) -> Dict[str, Any]:
        """创建投机解码配置
        
        Returns:
            投机解码配置字典
        """
        return {
            "method": "myngram",
            "num_speculative_tokens": 100,
            "prompt_lookup_max": 5,
            "prompt_lookup_min": 1,
        }
    
    def create_vllm_config(self) -> VllmConfig:
        """创建VLLM配置
        
        Returns:
            配置好的VllmConfig实例
        """
        from vllm.engine.arg_utils import AsyncEngineArgs
        
        # 投机解码配置
        speculative_config = self.create_speculative_config()
        
        # 创建AsyncEngineArgs并设置投机解码
        engine_args = AsyncEngineArgs(
            model="Qwen/Qwen2.5-7B-Instruct",
            dtype="bfloat16",
            gpu_memory_utilization=0.90,
            speculative_config=speculative_config
        )
        
        # 创建VllmConfig
        return engine_args.create_engine_config()
    
    def create_cache_config(self) -> CacheConfig:
        """创建缓存配置"""
        return CacheConfig(
            enable_redis=True,
            redis_host="localhost",
            redis_port=6379,
            redis_db=0,
            redis_password=None,
            max_memory_entries=2000,
            memory_ttl=7200,  # 2小时
            redis_ttl=7200
        )
    
    def create_similarity_config(self) -> VectorSearchConfig:
        """创建相似度搜索配置"""
        return VectorSearchConfig(
            enabled=True,
            similarity_threshold=0.7,
            max_results=10,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            use_gpu=True
        )
    
    def create_handler_config(self) -> HandlerConfig:
        """创建处理器配置"""
        config = HandlerConfig()
        
        # 功能配置
        config.features = HandlerFeatureConfig(
            enable_nlp_enhancement=True,
            enable_similarity_search=True,
            enable_caching=True,
            enable_detailed_logging=True,
            enable_vector_search=True
            # 移除了 enable_async_processing=True 这个不存在的参数
        )
        
        # 性能配置
        config.max_concurrent_tasks = 50
        config.request_timeout = 300  # 5分钟
        config.batch_size = 16
        config.enable_streaming = True
        
        return config
    
    async def setup_system(self) -> None:
        """设置完整的增强系统"""
        try:
            logger.info("🚀 开始设置完整的投机解码增强系统...")
            
            # 创建所有配置
            vllm_config = self.create_vllm_config()
            cache_config = self.create_cache_config()
            similarity_config = self.create_similarity_config()
            handler_config = self.create_handler_config()
            
            # 显示配置信息
            speculative_config = self.create_speculative_config()
            logger.info(f"📋 投机解码配置: {json.dumps(speculative_config, indent=2)}")
            logger.info(f"🎯 模型: {vllm_config.model_config.model}")
            logger.info(f"💾 GPU内存利用率: {vllm_config.cache_config.gpu_memory_utilization}")
            logger.info(f"🔄 缓存启用: {cache_config.enable_redis}")
            logger.info(f"🔍 相似度阈值: {similarity_config.similarity_threshold}")

            self.cache_manager = MultiLevelCache()
        
            # 创建相似度搜索助手
            self.similarity_search_helper = SimilaritySearchHelper()
            self.similarity_search_helper.initialize(similarity_config)
            # 使用新的from_vllm_config类方法创建增强的AsyncLLM
            self.enhanced_llm = EnhancedAsyncLLM.from_vllm_config(
                vllm_config=vllm_config,
                cache_manager=self.cache_manager,
                similarity_search_helper=self.similarity_search_helper,
                handler_config=handler_config,
                start_engine_loop=True,
                nlp_max_concurrent=30
            )
            
            logger.info("✅ 完整的投机解码增强系统设置完成")
            
        except Exception as e:
            logger.error(f"❌ 系统设置失败: {e}")
            raise

    async def run_performance_test(self, test_name: str, prompts: List[str], 
                                 sampling_params: SamplingParams) -> Dict[str, Any]:
        """运行性能测试
        
        Args:
            test_name: 测试名称
            prompts: 测试提示列表
            sampling_params: 采样参数
            
        Returns:
            性能统计结果
        """
        logger.info(f"🧪 开始性能测试: {test_name}")
        
        start_time = time.time()
        total_tokens = 0
        cache_hits = 0
        similar_contexts = 0
        
        for i, prompt in enumerate(prompts, 1):
            logger.info(f"  📝 处理第 {i}/{len(prompts)} 个请求...")
            
            request_start = time.time()
            
            try:
                # 使用EnhancedAsyncLLM的generate_enhanced方法
                request_id = str(uuid.uuid4())
                
                async for output in self.enhanced_llm.generate_enhanced(
                    prompt=prompt,
                    sampling_params=sampling_params,
                    request_id=request_id
                ):
                    if output.finished:
                        request_time = time.time() - request_start
                        
                        # 计算token数量
                        token_count = 0
                        output_text = ""
                        if output.outputs:
                            for comp in output.outputs:
                                token_count += len(comp.token_ids) if comp.token_ids else len(comp.text.split())
                                output_text += comp.text
                        
                        total_tokens += token_count
                        
                        # 检查是否为缓存命中（通过输出时间判断）
                        if request_time < 0.1:  # 小于100ms认为是缓存命中
                            cache_hits += 1
                        logger.info(f"    ⏱️  耗时: {request_time:.2f}s, tokens: {token_count}")
                        logger.info(f"    📝 问题: {prompt}")
                        logger.info(f"    💬 答案: {output_text}")
                        logger.info(f"    {'='*60}")
                        break
                        
            except Exception as e:
                logger.error(f"    ❌ 请求失败: {e}")
                continue
            
            # 短暂延迟避免过载
            await asyncio.sleep(0.2)
        
        total_time = time.time() - start_time
        avg_speed = total_tokens / total_time if total_time > 0 else 0
        
        stats = {
            'test_name': test_name,
            'total_requests': len(prompts),
            'total_time': total_time,
            'total_tokens': total_tokens,
            'avg_speed': avg_speed,
            'cache_hits': cache_hits,
            'cache_hit_rate': cache_hits / len(prompts) if prompts else 0,
            'similar_contexts': similar_contexts,
            'avg_similar_contexts': similar_contexts / len(prompts) if prompts else 0
        }
        
        self.performance_stats.append(stats)
        
        logger.info(f"📊 {test_name} 完成:")
        logger.info(f"    🚀 平均速度: {avg_speed:.2f} tokens/s")
        logger.info(f"    💾 缓存命中率: {stats['cache_hit_rate']:.1%}")
        logger.info(f"    🔍 平均相似上下文: {stats['avg_similar_contexts']:.1f}")
        
        return stats
    
    async def demo_basic_generation(self) -> None:
        """演示基本生成功能"""
        logger.info("\n=== 🎯 基本生成演示 ===")
        
        prompts = [
            "Please explain what artificial intelligence is?",
            "What is machine learning?",
            "What are the basic concepts of deep learning?"
        ]
        
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=200
        )
        
        await self.run_performance_test("基本生成", prompts, sampling_params)
    
    async def demo_cache_performance(self) -> None:
        """演示缓存性能"""
        logger.info("\n=== 💾 缓存性能演示 ===")
        
        # 重复相同的问题来测试缓存
        prompts = [
            "What is natural language processing?",
            "What is natural language processing?",  # 重复
            "Explain the basic concepts of computer vision.",
            "Explain the basic concepts of computer vision.",  # 重复
            "What is natural language processing?",  # 再次重复
        ]
        
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=150
        )
        
        await self.run_performance_test("缓存性能", prompts, sampling_params)
    
    async def demo_complex_generation(self) -> None:
        """演示复杂生成任务"""
        logger.info("\n=== 🧠 复杂生成演示 ===")
        
        prompts = [
            "Please explain in detail the differences between deep learning and machine learning, and describe their respective application scenarios and advantages/disadvantages.",
            "Analyze the current application status of artificial intelligence in healthcare, finance, and education, as well as future development trends.",
            "Compare different neural network architectures (CNN, RNN, Transformer) and explain their working principles."
        ]
        
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=500  # 更长的输出以体现投机解码优势
        )
        
        await self.run_performance_test("复杂生成", prompts, sampling_params)
    
    async def demo_batch_processing(self) -> None:
        """演示批量处理"""
        logger.info("\n=== 📦 批量处理演示 ===")
        
        prompts = [
            "What is reinforcement learning?",
            "Explain the backpropagation algorithm in neural networks.",
            "What is the attention mechanism?",
            "Explain how Generative Adversarial Networks (GANs) work.",
            "What is transfer learning?",
            "Explain the basic structure of convolutional neural networks.",
            "What are recurrent neural networks?",
            "Explain the innovations of the Transformer architecture.",
            "Explain how Generative Adversarial Networks (GANs)."
        ]
        
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=250
        )
        
        await self.run_performance_test("批量处理", prompts, sampling_params)
    
    async def demo_health_check(self) -> None:
        """演示系统健康检查"""
        logger.info("\n=== 🏥 系统健康检查 ===")
        
        try:
            # 检查各组件状态
            logger.info("📋 系统状态报告:")
            
            # 检查缓存管理器
            if self.cache_manager:
                cache_stats = await self.cache_manager.get_stats()
                logger.info(f"  🟢 缓存管理器: 正常运行")
                logger.info(f"    📊 缓存统计: {cache_stats}")
            else:
                logger.info("  🔴 缓存管理器: 未初始化")
            
            # 检查相似度搜索助手
            if self.similarity_search_helper:
                logger.info(f"  🟢 相似度搜索: 正常运行")
            else:
                logger.info("  🔴 相似度搜索: 未初始化")
            
            # 检查增强LLM
            if self.enhanced_llm:
                logger.info(f"  🟢 增强LLM: 正常运行")
                logger.info(f"    🔧 活跃任务数: {len(self.enhanced_llm._active_tasks)}")
            else:
                logger.info("  🔴 增强LLM: 未初始化")
            
        except Exception as e:
            logger.error(f"❌ 健康检查失败: {e}")
    
    def generate_performance_report(self) -> None:
        """生成性能报告"""
        logger.info("\n=== 📊 性能报告 ===")
        
        if not self.performance_stats:
            logger.warning("⚠️  没有性能数据")
            return
        
        logger.info("📈 详细性能统计:")
        logger.info("-" * 80)
        logger.info(f"{'测试名称':<15} {'请求数':<8} {'总耗时':<10} {'平均速度':<12} {'缓存命中率':<12} {'相似上下文':<12}")
        logger.info("-" * 80)
        
        total_requests = 0
        total_time = 0
        total_tokens = 0
        total_cache_hits = 0
        
        for stats in self.performance_stats:
            logger.info(
                f"{stats['test_name']:<15} "
                f"{stats['total_requests']:<8} "
                f"{stats['total_time']:<10.2f} "
                f"{stats['avg_speed']:<12.2f} "
                f"{stats['cache_hit_rate']:<12.1%} "
                f"{stats['avg_similar_contexts']:<12.1f}"
            )
            
            total_requests += stats['total_requests']
            total_time += stats['total_time']
            total_tokens += stats['total_tokens']
            total_cache_hits += stats['cache_hits']
        
        logger.info("-" * 80)
        
        overall_speed = total_tokens / total_time if total_time > 0 else 0
        overall_cache_rate = total_cache_hits / total_requests if total_requests > 0 else 0
        
        logger.info("🎯 总体性能:")
        logger.info(f"  📝 总请求数: {total_requests}")
        logger.info(f"  ⏱️  总耗时: {total_time:.2f}s")
        logger.info(f"  🚀 总体平均速度: {overall_speed:.2f} tokens/s")
        logger.info(f"  💾 总体缓存命中率: {overall_cache_rate:.1%}")
        logger.info(f"  📊 总生成tokens: {total_tokens}")
    
    async def cleanup(self) -> None:
        """清理系统资源"""
        logger.info("\n=== 🧹 清理系统资源 ===")
        
        try:
            if self.enhanced_llm:
                logger.info("🛑 停止增强LLM...")
                await self.enhanced_llm.cleanup()
                self.enhanced_llm.shutdown()
            
            if self.cache_manager:
                logger.info("💾 清理缓存管理器...")
                await self.cache_manager.clear()
            
            if self.similarity_search_helper:
                logger.info("🔍 清理相似度搜索助手...")
                await self.similarity_search_helper.cleanup()
            
            logger.info("✅ 资源清理完成")
            
        except Exception as e:
            logger.error(f"❌ 清理资源失败: {e}")
    
    async def run_complete_demo(self) -> None:
        """运行完整演示"""
        try:
            # 设置系统
            await self.setup_system()
            
            # 运行各种演示
            await self.demo_basic_generation()
            await asyncio.sleep(1)
            
            await self.demo_cache_performance()
            await asyncio.sleep(1)
            
            await self.demo_complex_generation()
            await asyncio.sleep(1)
            
            await self.demo_batch_processing()
            await asyncio.sleep(1)
            
            await self.demo_health_check()
            
            # 生成性能报告
            self.generate_performance_report()
            
        except Exception as e:
            logger.error(f"❌ 演示运行失败: {e}")
            raise
        finally:
            # 确保清理资源
            await self.cleanup()


async def main():
    """主函数"""
    logger.info("🎬 开始完整的投机解码配置演示")
    logger.info("=" * 60)
    
    demo = CompleteSpeculativeDemo()
    
    try:
        await demo.run_complete_demo()
        logger.info("\n🎉 演示完成！")
        
    except KeyboardInterrupt:
        logger.info("\n⏹️  用户中断演示")
        await demo.cleanup()
        
    except Exception as e:
        logger.error(f"\n💥 演示失败: {e}")
        await demo.cleanup()
        raise


if __name__ == "__main__":
    # 配置日志
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level="INFO"
    )
    
    # 运行演示
    asyncio.run(main())