"""Base class for model wrappers."""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import time
import json


class BaseModelWrapper(ABC):
    """Abstract base for all LLM inference wrappers."""

    def __init__(self, model_name: str, temperature: float = 0.0, 
                 max_tokens: int = 2048, **kwargs):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.metadata = {
            "model": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }

    @abstractmethod
    def generate(self, prompt: str, system_message: Optional[str] = None) -> str:
        """Generate code from prompt. Must be implemented by subclasses."""
        pass

    def generate_with_retry(self, prompt: str, max_retries: int = 3,
                           system_message: Optional[str] = None) -> str:
        """Generate with exponential backoff retry."""
        for attempt in range(max_retries):
            try:
                return self.generate(prompt, system_message)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait = 2 ** attempt
                time.sleep(wait)
        return ""

    def extract_code(self, response: str) -> str:
        """Extract Python code from model response."""
        # Try to extract code blocks
        import re

        # Match ```python ... ``` blocks
        pattern = r"```python
(.*?)
```"
        matches = re.findall(pattern, response, re.DOTALL)

        if matches:
            return matches[-1]  # Return last code block

        # Try generic code blocks
        pattern = r"```
(.*?)
```"
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return matches[-1]

        # If no code blocks, return full response (may be raw code)
        return response.strip()

    def log_inference(self, prompt: str, response: str, 
                      example_id: str, prompt_level: str,
                      output_dir: str = "results"):
        """Log inference details for reproducibility."""
        import os
        from pathlib import Path
        import hashlib

        log_dir = Path(output_dir) / self.model_name.replace("/", "_")
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create unique filename
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
        filename = f"{example_id}_{prompt_level}_{prompt_hash}.json"

        log_entry = {
            "example_id": example_id,
            "prompt_level": prompt_level,
            "model": self.model_name,
            "metadata": self.metadata,
            "prompt": prompt,
            "raw_response": response,
            "extracted_code": self.extract_code(response),
            "timestamp": time.time()
        }

        with open(log_dir / filename, "w") as f:
            json.dump(log_entry, f, indent=2)

        return log_dir / filename
