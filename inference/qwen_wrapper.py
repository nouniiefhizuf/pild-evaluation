"""Qwen2.5-72B local inference wrapper via vLLM."""

from typing import Optional
from .base import BaseModelWrapper


class QwenWrapper(BaseModelWrapper):
    """Wrapper for Qwen2.5-72B-Instruct via vLLM."""

    def __init__(self, model_path: str = "Qwen/Qwen2.5-72B-Instruct",
                 tensor_parallel_size: int = 4,
                 temperature: float = 0.0, max_tokens: int = 2048,
                 **kwargs):
        super().__init__(model_path, temperature, max_tokens,
                        tensor_parallel_size=tensor_parallel_size, **kwargs)

        try:
            from vllm import LLM, SamplingParams
            self.llm = LLM(
                model=model_path,
                tensor_parallel_size=tensor_parallel_size,
                dtype="auto"
            )
            self.sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens
            )
        except ImportError:
            raise ImportError("vLLM required for local inference.")

    def generate(self, prompt: str, system_message: Optional[str] = None) -> str:
        """Generate code using Qwen2.5-72B."""
        if system_message:
            full_prompt = f"<|im_start|>system
{system_message}<|im_end|>
"
            full_prompt += f"<|im_start|>user
{prompt}<|im_end|>
"
            full_prompt += "<|im_start|>assistant
"
        else:
            full_prompt = f"<|im_start|>user
{prompt}<|im_end|>
"
            full_prompt += "<|im_start|>assistant
"

        outputs = self.llm.generate(full_prompt, self.sampling_params)
        return outputs[0].outputs[0].text
