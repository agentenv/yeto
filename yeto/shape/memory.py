"""Will the model fit, and how many nodes does an island need?

FSDP shards weights across every GPU in an island, so fitting is a
function of (weight bytes, tuning mode, GPUs in the island) — not of any
single GPU. These are coarse envelope checks: the goal is to reject
shapes that cannot possibly work before money is spent, not to predict
allocator behavior to the megabyte.
"""

from __future__ import annotations

import math
from typing import Any

# Mirrors yeto/learner.py MODEL_ALIASES. Copied (2 entries) rather than
# imported: learner.py pulls in torch at module level, which planning code
# must never pay for. Keep in sync.
MODEL_ALIASES = {
    "gemma4": "google/gemma-4-12B-it",
    "deepseek4flash": "deepseek-ai/DeepSeek-V4-Flash",
}

# Per-GPU shard multiplier over bf16 weight bytes. lora: only the frozen
# bf16 base is sharded (adapters are negligible and replicated). full:
# learner.py's fsdp-full keeps fp32 originals + fp32 optimizer state
# (see the dtype comment there) — fp32 master + grad + Adam m,v is
# ~16 bytes/param vs the 2 bytes/param the bf16 weight figure measures.
_TUNING_FACTOR = {"lora": 1.0, "full": 8.0}


def _fetch_hub_weights(model_id: str) -> float:
    """Sum the .safetensors shard sizes on the Hub — the exact bytes the
    learner will load. Factored out so tests can monkeypatch it."""
    from huggingface_hub import HfApi

    info = HfApi().model_info(model_id, files_metadata=True)
    total = sum(
        f.size
        for f in info.siblings
        if f.rfilename.endswith(".safetensors") and f.size is not None
    )
    return float(math.ceil(total / 1e9))


def model_weights_gb(
    model: str, override: float | None = None, cache: Any = None
) -> float:
    """Weight size in GB: explicit override > launcher's known-model table
    (accepts alias or resolved HF id) > Hugging Face Hub metadata query.

    `cache` is any object with `.get_or(key, fetch)`; the Hub answer for a
    model id never changes mid-project, so caching it avoids a network
    round-trip per planning run.
    """
    if override is not None:
        return float(override)

    from yeto import launcher  # sky inside launcher is lazy; safe to import

    model_id = MODEL_ALIASES.get(model, model)
    if model in launcher.MODEL_WEIGHT_GB:
        return float(launcher.MODEL_WEIGHT_GB[model])
    for alias, hf_id in MODEL_ALIASES.items():
        if hf_id == model_id and alias in launcher.MODEL_WEIGHT_GB:
            return float(launcher.MODEL_WEIGHT_GB[alias])

    def fetch() -> float:
        return _fetch_hub_weights(model_id)

    try:
        gb = (
            cache.get_or(f"hf-weights:{model_id}", fetch)
            if cache is not None
            else fetch()
        )
    except Exception as exc:
        raise ValueError(
            f"could not determine weight size for {model_id!r} from the "
            f"Hugging Face Hub ({exc}); pass --weights-gb explicitly"
        ) from exc
    if gb <= 0:
        raise ValueError(
            f"Hugging Face Hub lists no .safetensors weights for "
            f"{model_id!r}; pass --weights-gb explicitly"
        )
    return float(gb)


def fits(
    weights_gb: float,
    tuning: str,
    gpu_mem_gb: int,
    total_gpus: int,
    seq_len: int = 2048,
) -> bool:
    """Does the FSDP-sharded footprint fit on each GPU of the island?

    shard = weights / total_gpus, scaled by the tuning-mode factor (see
    _TUNING_FACTOR). Overhead: 2 GB CUDA context/fragmentation plus a
    ~6 GB-at-2048 activation estimate (micro-batch 1, scales linearly
    with sequence length). We only claim 92% of the card — allocator
    fragmentation and transient all-gather buffers eat the rest.
    """
    try:
        factor = _TUNING_FACTOR[tuning]
    except KeyError:
        raise ValueError(f"unknown tuning mode {tuning!r} (expected 'lora' or 'full')")
    shard_gb = weights_gb / total_gpus * factor
    overhead_gb = 2.0 + 6.0 * (seq_len / 2048)
    return shard_gb + overhead_gb <= 0.92 * gpu_mem_gb


def min_nodes(
    weights_gb: float,
    tuning: str,
    gpu_mem_gb: int,
    gpus_per_node: int,
    seq_len: int = 2048,
    max_nodes: int = 8,
) -> int | None:
    """Smallest island (in nodes) that fits the model; None if even
    max_nodes does not.

    Callers should use exactly this value, never a larger island:
    single-node (or minimal) islands are strongly preferred — spot
    placement odds fall sharply with simultaneous multi-node asks, and a
    preemption of any node kills the whole island, so blast radius grows
    with island size. Want more compute? Add islands, not nodes.
    """
    for n in range(1, max_nodes + 1):
        if fits(weights_gb, tuning, gpu_mem_gb, n * gpus_per_node, seq_len):
            return n
    return None
