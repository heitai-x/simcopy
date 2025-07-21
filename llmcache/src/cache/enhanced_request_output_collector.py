"""Enhanced Request Output Collector for VLLM AsyncLLM Integration.

This module provides a simple extension of VLLM's RequestOutputCollector
that adds hash_id for history storage.
"""

from typing import Optional
from vllm.v1.engine.output_processor import RequestOutputCollector
from vllm.outputs import RequestOutput

class EnhancedRequestOutputCollector(RequestOutputCollector):
    """Enhanced RequestOutputCollector with hash_id for history storage.
    
    This class extends VLLM's RequestOutputCollector to add:
    - hash_id for history request storage
    - cached result handling
    """
    
    def __init__(
        self,
        output_kind: str = "delta",
        hash_id: Optional[str] = None
    ):
        """Initialize EnhancedRequestOutputCollector.
        
        Args:
            output_kind: Type of output ("delta" or "final")
            hash_id: Hash identifier for history storage
        """
        super().__init__(output_kind=output_kind)
        self.hash_id = hash_id
    
    def put_cached_result(self, cached_result) -> None:
        """Put cached result directly into the collector, compatible with old cache formats.
        
        Args:
            cached_result: The cached result (RequestOutput, CustomRequestOutput, or legacy format)
        """
        # Handle different cached result formats
        if hasattr(cached_result, 'request_id') and hasattr(cached_result, 'finished'):
            # Standard RequestOutput or CustomRequestOutput format
            if not cached_result.finished:
                # Create a new RequestOutput with finished=True
                finished_result = RequestOutput(
                    request_id=cached_result.request_id,
                    prompt=getattr(cached_result, 'prompt', ""),
                    prompt_token_ids=getattr(cached_result, 'prompt_token_ids', []),
                    prompt_logprobs=getattr(cached_result, 'prompt_logprobs', None),
                    outputs=getattr(cached_result, 'outputs', []),
                    finished=True,  # 确保标记为完成
                    metrics=getattr(cached_result, 'metrics', None),
                    lora_request=getattr(cached_result, 'lora_request', None),
                    encoder_prompt=getattr(cached_result, 'encoder_prompt', None),
                    encoder_prompt_token_ids=getattr(cached_result, 'encoder_prompt_token_ids', None),
                    num_cached_tokens=getattr(cached_result, 'num_cached_tokens', 0),
                    kv_transfer_params=getattr(cached_result, 'kv_transfer_params', None)
                )
                self.put(finished_result)
            else:
                self.put(cached_result)
        else:
            # Legacy format: create a minimal RequestOutput
            # This handles cases where cached_result is just text or token IDs
            if isinstance(cached_result, str):
                text = cached_result
                token_ids = []
            elif isinstance(cached_result, list):
                # Assume it's token IDs
                text = " ".join(map(str, cached_result))
                token_ids = cached_result
            else:
                # Unknown format, convert to string
                text = str(cached_result)
                token_ids = []
            
            # Create a minimal RequestOutput for legacy formats
            from vllm.outputs import CompletionOutput
            completion_output = CompletionOutput(
                index=0,
                text=text,
                token_ids=token_ids,
                cumulative_logprob=0.0,
                logprobs=None,
                finish_reason="stop",
                stop_reason=None
            )
            
            # Try to get request_id from cached_result if it has one
            request_id = 'cached'
            if hasattr(cached_result, 'request_id'):
                request_id = cached_result.request_id
            
            legacy_result = RequestOutput(
                request_id=request_id,
                prompt="",
                prompt_token_ids=[],
                prompt_logprobs=None,
                outputs=[completion_output],
                finished=True,
                metrics=None,
                lora_request=None
            )
            self.put(legacy_result)