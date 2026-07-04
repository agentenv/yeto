"""peft-convention LoRA for MLX models.

The DiLoCo fragment layout is keyed by trainable-parameter NAMES and packed
in row-major flatten order, so an MLX learner can only share a syncer with
torch learners if its adapters are bit-compatible with peft's: same FQNs,
same shapes, same flatten order, same scale semantics. mlx-lm's own
LoRALinear stores transposed factors (lora_a: (in, r)) under different names,
so we keep our own thin layer instead:

* ``lora_A``: (r, in), kaiming-uniform init (peft's default reduces to
  U(-1/sqrt(in), 1/sqrt(in)) for a (r, in) matrix);
* ``lora_B``: (out, r), zeros;
* forward: ``y = base(x) + (alpha/r) * x A^T B^T`` computed in f32 and cast
  back to the base activation dtype (adapters stay f32, like peft under a
  bf16 base — a bf16 AdamW second moment is too noisy);
* canonical name: MLX tree path ``model.layers.N...q_proj.lora_A`` maps to
  peft's ``base_model.model.model.layers.N...q_proj.lora_A.default.weight``.

Target selection mirrors ``yeto.learner.resolve_lora_targets``: the same
attention-projection regex, the same all-linear-except-lm-head expansion,
the same MoE auto rule — driven by the HF config so both backends make the
same choice for the same model.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from ..learner import _ATTENTION_TARGETS, is_moe_config

log = logging.getLogger("mlx-learner")

_ATTENTION_RE = re.compile(_ATTENTION_TARGETS)


class LoRALinear(nn.Module):
    """A frozen linear plus a trainable peft-shaped low-rank update."""

    def __init__(self, base: nn.Linear, r: int, alpha: int):
        super().__init__()
        out_dims, in_dims = base.weight.shape
        self.base = base
        self.scale = alpha / r
        bound = 1.0 / math.sqrt(in_dims)
        self.lora_A = mx.random.uniform(-bound, bound, (r, in_dims), dtype=mx.float32)
        self.lora_B = mx.zeros((out_dims, r), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        y = self.base(x)
        z = (x.astype(mx.float32) @ self.lora_A.T) @ self.lora_B.T
        return y + (self.scale * z).astype(y.dtype)


@dataclass
class AdapterInfo:
    """Where a canonical adapter tensor lives in the MLX parameter tree."""

    path: str  # tree path, e.g. "model.layers.0.self_attn.q_proj.lora_A"
    shape: tuple[int, ...]
    dtype: object  # mx dtype


def canonical_name(tree_path: str) -> str:
    """MLX tree path of a lora_A/lora_B array -> the peft FQN the torch
    learner would report for the same tensor."""
    return f"base_model.model.{tree_path}.default.weight"


def _named_linears(model) -> list[tuple[str, nn.Linear]]:
    # Plain Linears only: quantized/fused variants have no peft counterpart
    # under the torch learner (which trains from bf16 checkpoints).
    return [
        (path, m)
        for path, m in model.named_modules()
        if type(m) is nn.Linear
    ]


def resolve_targets(model, hf_config, choice: str) -> list[tuple[str, nn.Linear]]:
    """The (path, module) list to adapt, mirroring the torch learner rules."""
    linears = _named_linears(model)
    if choice == "auto":
        choice = "attention" if is_moe_config(hf_config) else "all-linear"
    if choice == "attention":
        return [(p, m) for p, m in linears if _ATTENTION_RE.fullmatch(p)]
    if is_moe_config(hf_config):
        log.warning(
            "--lora-targets all-linear on a MoE model: MLX expert GEMMs are "
            "fused (not nn.Linear) and get no adapters here, while the torch "
            "backend would adapt them — the two backends would disagree on "
            "the fragment layout. Use --lora-targets attention."
        )
    return [(p, m) for p, m in linears if p.rsplit(".", 1)[-1] != "lm_head"]


def attach_lora(model, hf_config, choice: str, r: int, alpha: int) -> dict[str, AdapterInfo]:
    """Freeze the base, wrap the targets, and return the canonical registry
    {peft FQN: AdapterInfo} in the order build_layout expects to see names."""
    model.freeze()
    targets = resolve_targets(model, hf_config, choice)
    if not targets:
        raise ValueError(f"--lora-targets {choice!r} matched no linear modules")
    model.update_modules(
        tree_unflatten([(path, LoRALinear(m, r, alpha)) for path, m in targets])
    )
    registry: dict[str, AdapterInfo] = {}
    for path, arr in tree_flatten(model.trainable_parameters()):
        if not path.endswith(("lora_A", "lora_B")):
            raise RuntimeError(f"unexpected trainable parameter {path!r} after freeze")
        registry[canonical_name(path)] = AdapterInfo(path, tuple(arr.shape), arr.dtype)
    log.info("attached LoRA r=%d alpha=%d to %d linears (%s)", r, alpha, len(targets), choice)
    return registry
