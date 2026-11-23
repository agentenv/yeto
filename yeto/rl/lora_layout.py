"""Canonical LoRA layout before training starts (offline specs).

The decoupled fragment plan, the syncer HELLO and the checkpoint bind the
canonical LoRA layout (``core.canonical_layout_hash``) before the first export,
so every synced LoRA island predicts it on the CPU and its first export must
match it (``core.canonical_state_from_owned_tensors`` refuses a different
layout). For most tensors PEFT is the source: :func:`yeto.rl.export.derive_peft_lora_specs`
wraps the model built on empty weights and reads ``get_peft_model_state_dict``.

Routed experts are the exception. Transformers 5 stores a mixture-of-experts
block as one fused module (``<block>.experts.gate_up_proj [E, 2I, H]`` and
``<block>.experts.down_proj [E, H, I]``), which PEFT's ``target_modules`` cannot
see. PEFT's ``target_parameters`` (``ParamWrapper``) does reach it, but with a
stacked layout (``experts.base_layer.lora_A [r*E, H]``), not the one-adapter-per-
expert layout Megatron-Bridge exports and SGLang serves
(``<block>.experts.<e>.{gate,up,down}_proj.lora_{A,B}.weight``, Miles'
``share_expert_adapters=False``). So :func:`fused_expert_lora_specs` writes the
per-expert names and shapes from the fused parameters, and
:func:`routed_expert_lora_specs` gives every ``nn.Linear`` PEFT's pair directly
(see there why not through ``get_peft_model``). The island's first export
through Bridge checks them, so a model whose checkpoint names differ is refused
there rather than trained under a wrong layout.

Torch, transformers and PEFT are imported lazily.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any

from .core import CanonicalTensorSpec

CANONICAL_PREFIX = "base_model.model."
# One synced island's canonical LoRA, FP32, above which the LoRA fragment stack
# (one learner connection, the main rank assembles the policy) is the wrong tool:
# the main rank holds it on the host every round. Larger trainable states belong
# on the full-parameter stack (each rank pushes its own shard).
MAX_CANONICAL_LORA_BYTES = 8 * 1024**3

_FUSED_GATE_UP = re.compile(r"^(?P<block>.+\.experts)\.gate_up_proj$")
_FUSED_DOWN = re.compile(r"^(?P<block>.+\.experts)\.down_proj$")


class LoraLayoutError(ValueError):
    """The canonical LoRA layout cannot be derived, or is too large to sync."""


def _spec(name: str, shape: tuple[int, int]) -> CanonicalTensorSpec:
    return CanonicalTensorSpec(name, shape, "float32", shape[0] * shape[1])


def _pair(module: str, in_features: int, out_features: int, rank: int) -> list[CanonicalTensorSpec]:
    base = CANONICAL_PREFIX + module
    return [_spec(f"{base}.lora_A.weight", (rank, in_features)),
            _spec(f"{base}.lora_B.weight", (out_features, rank))]


def _other(shape: Sequence[int], hidden: int, name: str) -> int:
    """The non-hidden dimension of a fused ``[E, a, b]`` expert weight."""

    a, b = int(shape[1]), int(shape[2])
    if b == hidden:
        return a
    if a == hidden:
        return b
    raise LoraLayoutError(f"fused expert weight {name} {tuple(shape)} has no hidden-size ({hidden}) dimension")


def fused_expert_lora_specs(named_shapes: Iterable[tuple[str, Sequence[int]]], *, rank: int,
                            hidden_size: int) -> tuple[CanonicalTensorSpec, ...]:
    """Per-expert canonical LoRA specs of every fused expert block.

    ``named_shapes``: ``(name, shape)`` of the model's parameters (a model built
    on empty weights). Each block needs both ``experts.gate_up_proj`` and
    ``experts.down_proj``; expert ``e`` gets ``gate_proj`` / ``up_proj``
    (``H -> I``) and ``down_proj`` (``I -> H``) adapters.
    """

    if rank <= 0:
        raise LoraLayoutError("LoRA rank must be positive")
    gate_up: dict[str, Sequence[int]] = {}
    down: dict[str, Sequence[int]] = {}
    for name, shape in named_shapes:
        for pattern, found in ((_FUSED_GATE_UP, gate_up), (_FUSED_DOWN, down)):
            match = pattern.match(name)
            if match is not None and len(shape) == 3:
                found[match.group("block")] = shape
    if set(gate_up) != set(down):
        raise LoraLayoutError(
            f"fused expert blocks without both gate_up_proj and down_proj: {sorted(set(gate_up) ^ set(down))[:2]}")
    specs: list[CanonicalTensorSpec] = []
    for block in sorted(gate_up):
        fused, out = gate_up[block], down[block]
        experts = int(fused[0])
        doubled = _other(fused, hidden_size, f"{block}.gate_up_proj")
        intermediate = _other(out, hidden_size, f"{block}.down_proj")
        if int(out[0]) != experts or doubled != 2 * intermediate:
            raise LoraLayoutError(
                f"fused experts {block}: gate_up_proj {tuple(fused)} and down_proj {tuple(out)} do not "
                "describe one gated MLP per expert")
        for expert in range(experts):
            prefix = f"{block}.{expert}"
            specs += _pair(f"{prefix}.gate_proj", hidden_size, intermediate, rank)
            specs += _pair(f"{prefix}.up_proj", hidden_size, intermediate, rank)
            specs += _pair(f"{prefix}.down_proj", intermediate, hidden_size, rank)
    return tuple(specs)


def all_linear_modules(model: Any) -> list[str]:
    """PEFT's ``all-linear`` selection for a model with routed experts, minus the
    routers: every ``nn.Linear`` except the output head, and except a ``gate`` /
    ``router`` next to an ``experts`` module (Megatron-Bridge and SGLang adapt no
    router; its load-balancing loss is not trained). Per-expert ``nn.Linear``
    layers (models that keep one module per expert) are included as they are."""

    import torch

    modules = dict(model.named_modules())
    output = getattr(model, "get_output_embeddings", lambda: None)()
    selected = []
    for name, module in modules.items():
        if not isinstance(module, torch.nn.Linear) or module is output:
            continue
        parent, _, leaf = name.rpartition(".")
        if leaf in ("gate", "router") and f"{parent}.experts" in modules:
            continue
        selected.append(name)
    return sorted(selected)


def linear_leaf_names(model: Any) -> list[str]:
    """The distinct leaf names of the model's ``nn.Linear`` layers (error messages)."""

    import torch

    return sorted({name.rsplit(".", 1)[-1] for name, m in model.named_modules() if isinstance(m, torch.nn.Linear)})


def linear_lora_specs(model: Any, modules: Iterable[str], *, rank: int) -> tuple[CanonicalTensorSpec, ...]:
    """PEFT's LoRA pair of each named ``nn.Linear``: ``lora_A [r, in]``, ``lora_B [out, r]``."""

    by_name = dict(model.named_modules())
    specs: list[CanonicalTensorSpec] = []
    for name in modules:
        linear = by_name[name]
        specs += _pair(name, int(linear.in_features), int(linear.out_features), rank)
    return tuple(specs)


def routed_expert_lora_specs(model: Any, *, rank: int, config: Any) -> tuple[CanonicalTensorSpec, ...]:
    """``all-linear`` with routed experts on a model built with empty weights.

    Every adapted ``nn.Linear`` (:func:`all_linear_modules`) gets PEFT's pair,
    and each fused expert block one pair per expert. The pairs are computed here
    rather than through ``get_peft_model``: on Transformers 5, PEFT converts a
    target list to the fused layout through the checkpoint conversion mapping
    (``convert_peft_config_for_transformers``), and on DeepSeek-V3 / GLM-4.5
    layouts that wraps ``mlp.experts`` as a stacked parameter and drops the
    dense and shared-expert MLPs. On models where PEFT is unaffected the two are
    identical (``tests/test_rl_lora_layout.py``).
    """

    linear = all_linear_modules(model)
    specs = list(linear_lora_specs(model, linear, rank=rank))
    if not has_expert_specs(specs):  # no per-expert nn.Linear: the experts are fused
        text = getattr(config, "text_config", None) or config
        experts = fused_expert_lora_specs(((n, tuple(p.shape)) for n, p in model.named_parameters()),
                                          rank=rank, hidden_size=int(text.hidden_size))
        if not experts:
            raise LoraLayoutError(
                "--lora-targets all-linear on this mixture-of-experts model: it has neither per-expert "
                "linear layers nor fused experts.gate_up_proj / experts.down_proj, so its routed experts "
                "have no canonical LoRA names; use --lora-targets attention")
        specs += experts
    return tuple(sorted(specs))


def adapts_routed_experts(targets: str | Sequence[str], config: Any) -> bool:
    """Whether ``--lora-targets`` asks for routed-expert adapters on this model."""

    from ..learner import is_moe_config

    return targets == "all-linear" and is_moe_config(getattr(config, "text_config", None) or config)


def has_expert_specs(specs: Iterable[CanonicalTensorSpec]) -> bool:
    return any(".experts." in spec.name for spec in specs)


def check_canonical_size(specs: Iterable[CanonicalTensorSpec], *,
                         limit: int = MAX_CANONICAL_LORA_BYTES) -> int:
    """FP32 bytes of the canonical LoRA; refuse a layout above ``limit``."""

    size = 4 * sum(spec.numel for spec in specs)
    if size > limit:
        raise LoraLayoutError(
            f"the canonical LoRA is {size / 1024**3:.1f} GiB in FP32, above the "
            f"{limit / 1024**3:.0f} GiB a synced LoRA island assembles on its main rank every "
            "round; lower --lora-r, narrow --lora-targets (attention), or train the experts on "
            "the full-parameter stack (--tuning full)")
    return size


def offline_canonical_specs(profile: Any, *, rank: int, targets: str, model_path: str,
                            revision: str | None = None,
                            trust_remote_code: bool = False) -> tuple[CanonicalTensorSpec, ...]:
    """The canonical LoRA specs of a Miles island, before it starts.

    ``profile``: the model's ``ModelSyncProfile`` (``yeto.rl.model_registry.sync_profile``),
    or ``None`` for a generic checkpoint. Only the generic recipe has an offline
    layout (PEFT on empty weights plus the per-expert routed-expert specs); a
    provider recipe (``layout_source`` ``first-export``) learns it from its first
    export and cannot bind a decoupled fragment plan before start-up.
    """

    if profile is not None:
        source = getattr(profile, "layout_source", "offline")
        if source != "offline":
            raise LoraLayoutError(
                f"recipe {getattr(profile, 'recipe', '?')}: its canonical LoRA layout is learned from the "
                f"first export (layout_source {source}), not derivable before start-up")
    from .export import derive_peft_lora_specs

    return derive_peft_lora_specs(model_path, revision, rank=rank, targets=targets,
                                  trust_remote_code=trust_remote_code, routed_experts=True)


__all__ = ["CANONICAL_PREFIX", "MAX_CANONICAL_LORA_BYTES", "LoraLayoutError", "adapts_routed_experts",
           "all_linear_modules", "check_canonical_size", "fused_expert_lora_specs", "has_expert_specs",
           "linear_leaf_names", "linear_lora_specs", "offline_canonical_specs", "routed_expert_lora_specs"]
