"""Model alias table: the single source for supported-model sugar.

Every command that takes --model accepts either an alias below or ANY raw
Hugging Face id (aliases are convenience, not a gate — arbitrary ids get
their weight size from the Hub's safetensors metadata at plan time). The
lineup mirrors the common open-weights fine-tuning roster as of 2026-07:
Qwen 3/3.5/3.6, Llama 3.x, gpt-oss, Kimi K2.x, DeepSeek, Nemotron 3,
plus the two aliases this project started with.

MODEL_WEIGHT_GB is the bf16 footprint (2 bytes/param, MoE = total params,
which is what the frozen base occupies under fsdp) used for offline island
sizing and disk provisioning. Every weight key must be an alias (tests
enforce it), but an alias MAY omit its weight: planning then falls through
to the Hub safetensors-metadata path, which is the honest choice when a
model's parameter count is unconfirmed — never guess a size into this
table.

The base is always trained in bf16. Some frontier checkpoints are published
only in a native low-precision format (DeepSeek fp8, gpt-oss mxfp4); those
are inference artifacts whose forward kernels have no backward, so training
materializes bf16 weights and the bf16 footprint above is the right sizing.
"""

from __future__ import annotations

MODEL_ALIASES = {
    # existing project aliases
    "gemma4": "google/gemma-4-12B-it",
    "deepseek4flash": "deepseek-ai/DeepSeek-V4-Flash",
    # Qwen dense + MoE
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen35-4b": "Qwen/Qwen3.5-4B",
    "qwen35-9b": "Qwen/Qwen3.5-9B",
    "qwen35-9b-base": "Qwen/Qwen3.5-9B-Base",
    "qwen35-35b-a3b": "Qwen/Qwen3.5-35B-A3B-Base",
    "qwen35-397b-a17b": "Qwen/Qwen3.5-397B-A17B",
    "qwen36-27b": "Qwen/Qwen3.6-27B",
    "qwen36-35b-a3b": "Qwen/Qwen3.6-35B-A3B",
    # Llama
    "llama32-1b": "meta-llama/Llama-3.2-1B",
    "llama32-3b": "meta-llama/Llama-3.2-3B",
    "llama31-8b": "meta-llama/Llama-3.1-8B",
    "llama31-8b-it": "meta-llama/Llama-3.1-8B-Instruct",
    "llama31-70b": "meta-llama/Llama-3.1-70B",
    "llama33-70b-it": "meta-llama/Llama-3.3-70B-Instruct",
    # gpt-oss
    "gptoss-20b": "openai/gpt-oss-20b",
    "gptoss-120b": "openai/gpt-oss-120b",
    # Kimi
    "kimi-k2-thinking": "moonshotai/Kimi-K2-Thinking",
    "kimi-k25": "moonshotai/Kimi-K2.5",
    "kimi-k26": "moonshotai/Kimi-K2.6",
    # DeepSeek
    "deepseek31": "deepseek-ai/DeepSeek-V3.1",
    "deepseek-r1": "deepseek-ai/DeepSeek-R1",
    "deepseek4pro": "deepseek-ai/DeepSeek-V4-Pro",
    # NVIDIA Nemotron 3
    "nemotron3-nano": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "nemotron3-super": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
    "nemotron3-ultra": "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
    # GLM (z.ai)
    "glm45-air": "zai-org/GLM-4.5-Air",
    "glm46": "zai-org/GLM-4.6",
    "glm52": "zai-org/GLM-5.2",
    # Llama 4 MoE
    "llama4-scout": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "llama4-maverick": "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    # coding / misc frontier
    "qwen3-coder-480b": "Qwen/Qwen3-Coder-480B-A35B-Instruct",
    "minimax-m2": "MiniMaxAI/MiniMax-M2",
    "minimax-m3": "MiniMaxAI/MiniMax-M3",  # size via Hub metadata
    "kimi-k27-code": "moonshotai/Kimi-K2.7-Code",  # size via Hub metadata
    "mistral-small3": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    # Ornith (DeepReinforce agentic-coding family, MIT)
    "ornith-9b": "deepreinforce-ai/Ornith-1.0-9B",
    "ornith-31b": "deepreinforce-ai/Ornith-1.0-31B",
    "ornith-35b": "deepreinforce-ai/Ornith-1.0-35B",
    "ornith-397b": "deepreinforce-ai/Ornith-1.0-397B-FP8",  # fp8 checkpoint; bf16 at load
    # Liquid AI LFM2.5 (edge/on-device)
    "lfm25-230m": "LiquidAI/LFM2.5-230M",
    "lfm25-1b": "LiquidAI/LFM2.5-1B",
    "lfm25-8b-a1b": "LiquidAI/LFM2.5-8B-A1B",
    # VibeThinker (small reasoning)
    "vibethinker-15b": "WeiboAI/VibeThinker-1.5B",
    "vibethinker-3b": "WeiboAI/VibeThinker-3B",
}

MODEL_WEIGHT_GB = {
    "gemma4": 66,
    "deepseek4flash": 568,
    "qwen3-8b": 17,
    "qwen35-4b": 8,
    "qwen35-9b": 18,
    "qwen35-9b-base": 18,
    "qwen35-35b-a3b": 70,
    "qwen35-397b-a17b": 794,
    "qwen36-27b": 54,
    "qwen36-35b-a3b": 70,
    "llama32-1b": 3,
    "llama32-3b": 7,
    "llama31-8b": 16,
    "llama31-8b-it": 16,
    "llama31-70b": 141,
    "llama33-70b-it": 141,
    "gptoss-20b": 42,
    "gptoss-120b": 234,
    "kimi-k2-thinking": 2060,
    "kimi-k25": 2060,
    "kimi-k26": 2060,
    "deepseek31": 1343,
    "deepseek-r1": 1343,
    "deepseek4pro": 3200,  # 1.6T-total / 49B-active MoE
    "nemotron3-nano": 61,
    "nemotron3-super": 240,
    "nemotron3-ultra": 1100,
    "glm45-air": 212,
    "glm46": 714,
    "glm52": 1488,  # 744B MoE
    "llama4-scout": 218,  # 109B total / 17B active
    "llama4-maverick": 800,  # 400B total / 17B active
    "qwen3-coder-480b": 960,
    "minimax-m2": 460,  # 230B total / 10B active
    "mistral-small3": 48,
    "ornith-9b": 19,
    "ornith-31b": 62,
    "ornith-35b": 70,  # MoE total
    "ornith-397b": 794,  # 397B MoE; fp8 checkpoint, bf16 footprint
    "lfm25-230m": 0.5,
    "lfm25-1b": 2,
    "lfm25-8b-a1b": 17,
    "vibethinker-15b": 3,
    "vibethinker-3b": 6,
}


def resolve(model: str) -> str:
    """Alias -> HF id; raw HF ids pass through unchanged."""
    return MODEL_ALIASES.get(model, model)
