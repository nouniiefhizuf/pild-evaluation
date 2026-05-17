"""Llama-3.3-70B local inference wrapper via vLLM."""

from typing import Optional
from .base import BaseModelWrapper


class LlamaWrapper(BaseModelWrapper):
    """Wrapper for Llama-3.3-70B-Instruct via vLLM."""

    def __init__(self, model_path: str = "meta-llama/Llama-3.3-70B-Instruct",
                 tensor_parallel_size: int = 4,
                 temperature: float = 0.0, max_tokens: int = 2048,
                 repetition_penalty: float = 1.0, **kwargs):
        super().__init__(model_path, temperature, max_tokens,
                        tensor_parallel_size=tensor_parallel_size,
                        repetition_penalty=repetition_penalty, **kwargs)

        try:
            from vllm import LLM, SamplingParams
            self.llm = LLM(
                model=model_path,
                tensor_parallel_size=tensor_parallel_size,
                dtype="auto"
            )
            self.sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
                repetition_penalty=repetition_penalty
            )
        except ImportError:
            raise ImportError("vLLM required for local inference. Install: pip install vllm")

    def generate(self, prompt: str, system_message: Optional[str] = None) -> str:
        """Generate code using Llama-3.3-70B."""
        if system_message:
            full_prompt = f"<|system|>
{system_message}
<|user|>
{prompt}
<|assistant|>
"
        else:
            full_prompt = f"<|user|>
{prompt}
<|assistant|>
"

        outputs = self.llm.generate(full_prompt, self.sampling_params)
        return outputs[0].outputs[0].text
