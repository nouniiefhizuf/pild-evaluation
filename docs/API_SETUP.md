# API Setup Guide

## OpenAI (GPT-4o)

1. Get API key from [OpenAI Platform](https://platform.openai.com/)
2. Set environment variable:
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

## DeepSeek (DeepSeek-V3)

1. Register at [DeepSeek Platform](https://platform.deepseek.com/)
2. Set environment variable:
   ```bash
   export DEEPSEEK_API_KEY="sk-..."
   ```

## Local Models (Llama, Qwen)

Requires 4× NVIDIA A100 80GB GPUs with vLLM installed:

```bash
# Install vLLM
pip install vllm

# Download models (via HuggingFace)
huggingface-cli download meta-llama/Llama-3.3-70B-Instruct
huggingface-cli download Qwen/Qwen2.5-72B-Instruct
```

## Environment File

Create `.env` file in project root:

```bash
OPENAI_API_KEY=sk-your-key-here
DEEPSEEK_API_KEY=sk-your-key-here
HF_TOKEN=hf-your-token-here  # For downloading gated models
```
