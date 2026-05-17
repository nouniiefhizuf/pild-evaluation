"""DeepSeek-V3 API wrapper."""

import os
from typing import Optional
import requests
from .base import BaseModelWrapper


class DeepSeekWrapper(BaseModelWrapper):
    """Wrapper for DeepSeek-V3 API."""

    def __init__(self, api_key: Optional[str] = None,
                 temperature: float = 0.0, max_tokens: int = 2048,
                 **kwargs):
        super().__init__("deepseek-v3", temperature, max_tokens, **kwargs)

        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DeepSeek API key required. Set DEEPSEEK_API_KEY env var.")

        self.base_url = "https://api.deepseek.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def generate(self, prompt: str, system_message: Optional[str] = None) -> str:
        """Generate code using DeepSeek-V3."""
        messages = []

        if system_message:
            messages.append({"role": "system", "content": system_message})
        else:
            messages.append({
                "role": "system",
                "content": "You are an expert physics simulation programmer."
            })

        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": "deepseek-v3",
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload
        )
        response.raise_for_status()

        return response.json()["choices"][0]["message"]["content"]
