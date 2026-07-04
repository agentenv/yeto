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

# Single-source alias table; yeto/models.py has no heavy dependencies, so
# planning code can import it directly (learner/launcher re-export it).
from ..models import MODEL_ALIASES

# Per-GPU shard multiplier over bf16 weight bytes. lora: only the frozen
# base is sharded (adapters are negligible and replicated). full:
# learner.py's fsdp-full keeps fp32 originals + fp32 optimizer state
# (see the dtype comment there) — fp32 master + grad + Adam m,v is
# ~16 bytes/param vs the 2 bytes/param the bf16 weight figure measures.
_TUNING_FACTOR = {"lora": 1.0, "full": 8.0}


def _fetch_hub_param_count(model_id: str) -> int:
    """Total parameter count from the Hub's safetensors metadata.

    `parameter_count` is a per-dtype dict (e.g. {"F8_E4M3": 283e9,
    "BF16": 1e6}); we want the sum regardless of stored dtype. Raises on
    any problem (missing metadata, gated repo, zero params) so the caller
    can fall back. Factored out so tests can monkeypatch it.
    """
    from huggingface_hub import HfApi

    meta = HfApi().get_safetensors_metadata(model_id)
    total = sum(meta.parameter_count.values())
    if total <= 0:
        raise ValueError(
            f"safetensors metadata for {model_id!r} lists no parameters"
        )
    return int(total)


def _fetch_hub_weights(model_id: str) -> float:
    """Sum the .safetensors shard sizes on the Hub — the *stored* bytes,
    uncorrected for dtype. Fallback only: prefer _fetch_hub_param_count,
    which is dtype-independent. Factored out so tests can monkeypatch it."""
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
    """bf16-equivalent weight size in GB — what the training job must fit.

    Precedence: explicit override > launcher's known-model table (accepts
    alias or resolved HF id) > Hugging Face Hub metadata query.

    The learner always materializes the frozen base in bf16 (see
    load_model_and_tokenizer in yeto/learner.py), so the figure we need is
    total_param_count * 2 bytes — *not* the checkpoint's stored size: an
    fp8 checkpoint (common for large MoEs) under-reports the bf16 footprint
    by 2x and an fp32 one over-reports by 2x. The Hub path therefore
    prefers parameter counts from the safetensors metadata; only when that
    is unavailable does it fall back to summing stored .safetensors bytes,
    which is exact for bf16 checkpoints and a dtype-uncorrected estimate
    otherwise.

    `cache` is any object with `.get_or(key, fetch)`; the Hub answer for a
    model id never changes mid-project, so caching it avoids a network
    round-trip per planning run.
    """
    if override is not None:
        return float(override)

    from ..models import MODEL_WEIGHT_GB

    model_id = MODEL_ALIASES.get(model, model)
    if model in MODEL_WEIGHT_GB:
        return float(MODEL_WEIGHT_GB[model])
    for alias, hf_id in MODEL_ALIASES.items():
        if hf_id == model_id and alias in MODEL_WEIGHT_GB:
            return float(MODEL_WEIGHT_GB[alias])

    def fetch() -> float:
        try:
            return _fetch_hub_param_count(model_id) * 2 / 1e9
        except Exception:
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
    _TUNING_FACTOR). The base is always bf16. Overhead: 2 GB CUDA
    context/fragmentation plus a ~6 GB-at-2048 activation estimate
    (micro-batch 1, scales linearly with sequence length). We only claim
    92% of the card — allocator fragmentation and transient all-gather
    buffers eat the rest.
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
