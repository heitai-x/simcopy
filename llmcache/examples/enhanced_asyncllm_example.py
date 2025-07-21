"""Example demonstrating the Enhanced AsyncLLM with true inheritance.

This example shows how to use the new EnhancedAsyncLLM that properly inherits
from VLLM's AsyncLLM and integrates caching, NLP processing, and similarity search.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to the path instead of just src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from vllm.v1.request import SamplingParams
# Use absolute imports from the src package
from src.handler.enhanced_async_llm import EnhancedAsyncLLM
from src.config.handler_config import HandlerConfig, CacheConfig, NLPConfig, SimilarityConfig


async def main():
    """Main example function."""
    print("=== Enhanced AsyncLLM Example ===")
    
    # Configure the enhanced LLM
    config = HandlerConfig(
        model_name="microsoft/DialoGPT-medium",
        cache=CacheConfig(
            enabled=True,
            max_size=1000,
            ttl_seconds=3600
        ),
        nlp=NLPConfig(
            enabled=True,
            language="en"
        ),
        similarity=SimilarityConfig(
            enabled=True,
            threshold=0.8
        )
    )
    
    try:
        # Initialize the enhanced AsyncLLM
        print("Initializing Enhanced AsyncLLM...")
        enhanced_llm = EnhancedAsyncLLM(
            model=config.model_name,
            handler_config=config
        )
        
        # Test prompts
        test_prompts = [
            "What is the capital of France?",
            "Explain quantum computing in simple terms.",
            "What is the capital of France?",  # Duplicate to test caching
        ]
        
        # Sampling parameters
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=100
        )
        
        print("\n=== Testing Enhanced add_request Method ===")
        
        for i, prompt in enumerate(test_prompts):
            print(f"\n--- Request {i+1}: {prompt[:50]}... ---")
            
            try:
                # Use the enhanced add_request method
                collector = await enhanced_llm.add_request(
                    request_id=f"test_request_{i+1}",
                    prompt=prompt,
                    params=sampling_params
                )
                
                print(f"✓ Request added successfully")
                print(f"  - Collector type: {type(collector).__name__}")
                print(f"  - Is enhanced: {getattr(collector, 'is_enhanced', False)}")
                
                # Stream the results
                print("  - Streaming results:")
                async for output in collector.aiter():
                    if hasattr(output, 'text'):
                        text_preview = output.text[:100] + "..." if len(output.text) > 100 else output.text
                        print(f"    Text: {text_preview}")
                    
                    if hasattr(output, 'metadata'):
                        metadata = output.metadata
                        print(f"    Metadata: cached={metadata.get('cached', False)}, "
                              f"processing_time={metadata.get('processing_time', 0):.3f}s")
                    
                    if output.finished:
                        print(f"    ✓ Request completed")
                        break
                
                # Cleanup
                if hasattr(collector, 'cleanup'):
                    collector.cleanup()
                    
            except Exception as e:
                print(f"✗ Error processing request: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n=== Testing Legacy generate_enhanced Method ===")
        
        # Test the legacy method for comparison
        try:
            print("\nTesting legacy generate_enhanced method...")
            async for output in enhanced_llm.generate_enhanced(
                prompt="What is machine learning?",
                sampling_params=sampling_params,
                request_id="legacy_test"
            ):
                if hasattr(output, 'text'):
                    text_preview = output.text[:100] + "..." if len(output.text) > 100 else output.text
                    print(f"Legacy output: {text_preview}")
                
                if output.finished:
                    print("✓ Legacy method completed")
                    break
                    
        except Exception as e:
            print(f"✗ Legacy method error: {e}")
        
        print("\n=== Performance Comparison ===")
        
        # Compare performance between cached and non-cached requests
        import time
        
        test_prompt = "Explain artificial intelligence."
        
        # First request (no cache)
        start_time = time.time()
        collector1 = await enhanced_llm.add_request(
            request_id="perf_test_1",
            prompt=test_prompt,
            params=sampling_params
        )
        
        async for output in collector1.aiter():
            if output.finished:
                first_time = time.time() - start_time
                break
        
        # Second request (should use cache)
        start_time = time.time()
        collector2 = await enhanced_llm.add_request(
            request_id="perf_test_2",
            prompt=test_prompt,
            params=sampling_params
        )
        
        async for output in collector2.aiter():
            if output.finished:
                second_time = time.time() - start_time
                break
        
        print(f"First request time: {first_time:.3f}s")
        print(f"Second request time: {second_time:.3f}s")
        print(f"Speedup: {first_time/second_time:.2f}x" if second_time > 0 else "N/A")
        
        # Cleanup
        for collector in [collector1, collector2]:
            if hasattr(collector, 'cleanup'):
                collector.cleanup()
        
    except Exception as e:
        print(f"\n✗ Example failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== Example Complete ===")


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())