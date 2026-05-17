"""GPT-4o API wrapper."""

import os
from typing import Optional
import openai
from .base import BaseModelWrapper


class GPT4oWrapper(BaseModelWrapper):
    """Wrapper for OpenAI GPT-4o API."""

    def __init__(self, api_key: Optional[str] = None, 
                 temperature: float = 0.0, max_tokens: int = 2048,
                 top_p: float = 1.0, **kwargs):
        super().__init__("gpt-4o", temperature, max_tokens, 
                        top_p=top_p, **kwargs)

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var.")

        self.client = openai.OpenAI(api_key=self.api_key)
        self.top_p = top_p

    def generate(self, prompt: str, system_message: Optional[str] = None) -> str:
        """Generate code using GPT-4o."""
        messages = []

        if system_message:
            messages.append({"role": "system", "content": system_message})
        else:
            messages.append({
                "role": "system", 
                "content": "You are an expert physics simulation programmer. "
                          "Write clean, correct Python code using NumPy and SciPy."
            })

        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p
        )

        return response.choices[0].message.content
