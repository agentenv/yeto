"""Launch-time checks of a Kimi-K3.1 full-parameter (partial-experts) run.

Both run before any cloud resource (``yeto launch`` and ``--dry-run``):

* :func:`check_expert_selection` loads ``--rl-expert-selection`` through the
  same validator the selection tool's ``--validate`` uses and checks it against
  the launch (model, ``--expert-parallel``, the checkpoint's MoE layers when its
  ``config.json`` is readable). Any trainable count that EP divides is
  accepted (64 and 128 of 896 are the usual forms).
* :func:`memory_estimate` is a rough per-GPU budget of the full form (trainable
  experts with Adam state, frozen MXFP4 experts, the non-expert weights,
  recomputed activations, the logits). It never refuses a launch; it returns
  warnings the launcher prints, because the figures are first-order (see the
  ``assumptions`` it reports).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

# bytes per trainable parameter: bf16 weight + fp32 grad + fp32 master/m/v (DP 1: no sharding)
TRAINABLE_BYTES = 2 + 4 + 12
# MXFP4: 4-bit values + one 8-bit scale per 32 values
MXFP4_BYTES = 4.25 / 8
# non-expert parameters of one K3.1 decoder layer (attention + shared expert), rough
NON_EXPERT_PER_LAYER = 0.33e9
WARN_FRACTION = 0.85  # the estimate leaves out fragmentation, NCCL buffers and graphs
LONG_CONTEXT = 65536


def _model_config(model: str) -> dict[str, Any] | None:
    path = Path(os.path.expanduser(str(model)))
    candidates = [path / "config.json"] if path.is_dir() else []
    if not candidates:
        try:
            from ..memory_estimate import _hf_cache_configs

            candidates = [Path(p) for p in _hf_cache_configs(str(model))]
        except Exception:  # noqa: BLE001 - the cache lookup is best effort
            candidates = []
    for candidate in candidates:
        try:
            return json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
    return None


def _text_config(cfg: dict[str, Any]) -> dict[str, Any]:
    text = cfg.get("text_config")
    return text if isinstance(text, dict) else cfg


def check_expert_selection(args) -> dict[str, Any] | None:
    """Validate ``--rl-expert-selection`` against the launch; returns a summary."""
    path = getattr(args, "rl_expert_selection", None)
    if not path:
        return None
    from ..rl.adapters.miles.models.kimi_k31.model import MODEL_TYPE, NUM_EXPERTS, NUM_MOE_LAYERS
    from ..rl.adapters.miles.models.partial_experts.selection import ExpertSelection, SelectionError

    hint = ("generate one with python -m yeto.rl.adapters.miles.models.partial_experts.selection "
            "--model kimi_k31 --num-trainable 64 --ep-size <EP> ... and check it with --validate")
    try:
        selection = ExpertSelection.load_file(os.path.expanduser(str(path)))
    except FileNotFoundError:
        raise ValueError(f"--rl-expert-selection {path}: no such file; {hint}") from None
    except (SelectionError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"--rl-expert-selection {path} is not a valid selection: {exc}; {hint}") from None
    if selection.model != MODEL_TYPE:
        raise ValueError(f"--rl-expert-selection is for model {selection.model!r}, not {MODEL_TYPE!r}")
    ep = int(getattr(args, "expert_parallel", 0) or 0)
    if ep and selection.ep_size != ep:
        raise ValueError(
            f"--rl-expert-selection was built for EP {selection.ep_size} but the launch uses "
            f"--expert-parallel {ep}; re-derive the slot order with the selection tool "
            f"(--input {path} --ep-size {ep})")
    cfg = _model_config(getattr(args, "model", ""))
    warnings: list[str] = []
    if cfg is not None:
        text = _text_config(cfg)
        total = text.get("num_experts", text.get("n_routed_experts"))
        layers = text.get("num_hidden_layers")
        dense = int(text.get("first_k_dense_replace", 1) or 0)
        if total is not None and int(total) != selection.num_experts:
            raise ValueError(
                f"--rl-expert-selection covers {selection.num_experts} experts per layer; the "
                f"checkpoint has {total}")
        if layers is not None:
            try:
                selection.check_layers(range(dense, int(layers)))
            except SelectionError as exc:
                raise ValueError(f"--rl-expert-selection does not match the checkpoint: {exc}") from None
    else:
        if selection.num_experts != NUM_EXPERTS or len(selection.trainable) != NUM_MOE_LAYERS:
            warnings.append(
                f"--rl-expert-selection covers {len(selection.trainable)} MoE layers x "
                f"{selection.num_experts} experts; Kimi-K3.1-Pro has {NUM_MOE_LAYERS} x {NUM_EXPERTS}. "
                "The model's config.json is not readable here, so the layer check runs on the island")
    return {
        "sha256": selection.sha256,
        "num_trainable": selection.num_trainable,
        "num_experts": selection.num_experts,
        "ep_size": selection.ep_size,
        "moe_layers": len(selection.trainable),
        "warnings": warnings,
    }


def memory_estimate(args, *, num_trainable: int, moe_layers: int, gpu: str | None) -> dict[str, Any]:
    """Rough per-GPU memory of the K3.1 full form on the busiest pipeline stage."""
    from ..memory_estimate import gpu_capacity_gib
    from ..rl.adapters.miles.models.kimi_k31 import model as k31

    tp = max(1, int(getattr(args, "tensor_parallel", 1) or 1))
    pp = max(1, int(getattr(args, "pipeline_parallel", 1) or 1))
    ep = max(1, int(getattr(args, "expert_parallel", 1) or 1))
    cp = max(1, int(getattr(args, "context_parallel", 1) or 1))
    seq = int(getattr(args, "seq_len", 0) or 0)
    response = int(getattr(args, "rollout_max_response_len", 0) or 0)
    layers_per_stage = math.ceil(moe_layers / pp)
    experts_per_gpu = layers_per_stage * (num_trainable / ep)
    frozen_per_gpu = layers_per_stage * ((k31.NUM_EXPERTS - num_trainable) / ep)
    gib = 1024 ** 3
    tokens = seq / cp
    # --rl-optimizer-offload-fraction F keeps F of the FP32 master/m/v on the host;
    # --rl-precision-aware-optimizer: bf16 main grads (2 bytes) and bf16 moments on the device
    offload = min(1.0, max(0.0, float(getattr(args, "rl_optimizer_offload_fraction", 0.0) or 0.0)))
    precision_aware = bool(getattr(args, "rl_precision_aware_optimizer", False))
    grad_bytes = 2 if precision_aware else 4
    optim_bytes = (6 if precision_aware else 12) * (1.0 - offload)
    trainable_bytes = 2 + grad_bytes + optim_bytes
    parts = {
        "trainable_experts": experts_per_gpu * k31.EXPERT_PARAMETERS * trainable_bytes / gib,
        "frozen_experts_mxfp4": frozen_per_gpu * k31.EXPERT_PARAMETERS * MXFP4_BYTES / gib,
        # the full form trains the selected routed experts only: the rest stays frozen bf16
        "non_expert_weights": (layers_per_stage * NON_EXPERT_PER_LAYER
                               + 2 * k31.VOCAB_SIZE * k31.HIDDEN_SIZE / pp) * 2 / tp / gib,
        # full recompute: one bf16 hidden state per layer, sequence-parallel over TP, x PP in flight
        "activation_checkpoints": layers_per_stage * tokens * k31.HIDDEN_SIZE * 2 / tp * pp / gib,
        # one MoE layer's working set: the routed tokens of this EP rank
        "moe_working_set": tokens * k31.TOPK / ep
        * (2 * k31.MOE_LATENT_SIZE + 2 * k31.MOE_FFN_HIDDEN_SIZE) * 2 / gib,
        "logits_fp32": tokens * k31.VOCAB_SIZE / tp * 4 / gib,
    }
    total = sum(parts.values())
    capacity = gpu_capacity_gib(gpu) if gpu else None
    warnings = []
    if capacity is not None and total > WARN_FRACTION * capacity:
        warnings.append(
            f"Kimi-K3.1 full form ({num_trainable} trainable experts/layer, seq {seq}, "
            f"TP{tp} PP{pp} EP{ep} CP{cp}) needs about {total:.0f} GiB per GPU, above "
            f"{WARN_FRACTION:.0%} of {gpu}'s {capacity:.0f} GiB; raise --context-parallel or "
            "--pipeline-parallel, lower --seq-len, or train fewer experts (64 per layer)")
    if seq >= LONG_CONTEXT and response and response * 4 < seq:
        warnings.append(
            f"--seq-len {seq} budgets memory for {seq} tokens but --rollout-max-response-len is "
            f"{response}; size the response length from the sequence length (prompt + response "
            "<= seq-len)")
    if seq >= LONG_CONTEXT and cp == 1:
        warnings.append(
            f"--seq-len {seq} with --context-parallel 1 keeps the whole sequence on every TP group; "
            "the activation and logits terms above grow linearly with it")
    host_optimizer = experts_per_gpu * k31.EXPERT_PARAMETERS * (12 * offload) / gib
    return {
        "host_optimizer_gib_per_gpu": round(host_optimizer, 1),
        "estimated": True,
        "rough": True,
        "per_gpu_gib": round(total, 1),
        "capacity_gib": None if capacity is None else round(capacity, 1),
        "parts_gib": {k: round(v, 1) for k, v in parts.items()},
        "assumptions": [
            "busiest pipeline stage holds ceil(moe_layers / PP) MoE layers",
            "trainable experts: bf16 weight + fp32 grad (bf16 with --rl-precision-aware-optimizer) + the "
            "device share of the Adam state (1 - --rl-optimizer-offload-fraction), DP 1",
            "non-expert weights frozen (bf16): the full form trains the selected routed experts only",
            "frozen routed experts stay MXFP4 (4.25 bits)",
            "full activation recompute, sequence parallel over TP, PP micro-batches in flight",
            "no allocator fragmentation, no NCCL buffers, no CUDA graphs",
        ],
        "warnings": warnings,
    }


def check_full_form(args, specs) -> dict[str, Any] | None:
    """Selection check + memory estimate of a K3.1 ``--tuning full`` launch.

    Prints every warning to stderr and returns the record the dry-run plan
    carries (``kimi_k31_full``)."""
    import sys

    if getattr(args, "rl_model_recipe", None) != "kimi_k31" or getattr(args, "tuning", None) != "full":
        return None
    summary = check_expert_selection(args)
    if summary is None:
        return None
    gpu = specs[0].gpu if specs else None
    estimate = memory_estimate(args, num_trainable=summary["num_trainable"],
                               moe_layers=summary["moe_layers"], gpu=gpu)
    record = {"expert_selection": summary, "memory": estimate}
    for message in (*summary["warnings"], *estimate["warnings"]):
        print(f"[launcher] warning: {message}", file=sys.stderr)
    return record


__all__ = ["check_expert_selection", "check_full_form", "memory_estimate"]
