"""Audited logical-expert cloning contract for DeepSeek V4 Flash.

The expanded model has 288 independently stored routed experts per decoder
layer, but it deliberately retains the original 256-way gate.  For each
layer, 32 gate categories have a cloned expert.  A deterministic function of
``(token_id, layer, source_expert)`` moves a selected route to exactly one of
the source/clone pair.  Consequently:

* top-k is still computed over the original 256 categories;
* every token still activates exactly six experts;
* a source and its clone can never consume two top-k slots; and
* exact weight copies are function-equivalent at initialization.

The token-ID split is intentionally independent of hidden-state numerics so
BF16 Megatron training and FP8 SGLang rollout make the same routing decision.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ORIGINAL_EXPERTS = 256
CLONES_PER_LAYER = 32
TOTAL_EXPERTS = ORIGINAL_EXPERTS + CLONES_PER_LAYER
NUM_LAYERS = 43
TOPK = 6

SPLIT_ALGORITHM = "token-id-affine-mod-prime-v1"
SPLIT_MODULUS = 2_147_483_647
SPLIT_THRESHOLD = 1_073_741_824
SPLIT_TOKEN_MULTIPLIER = 1_103_515_245
SPLIT_LAYER_MULTIPLIER = 12_345
SPLIT_SOURCE_MULTIPLIER = 2_654_435_761

_HEX = frozenset("0123456789abcdef")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a SHA256 string")
    normalized = value.lower()
    if len(normalized) != 64 or any(character not in _HEX for character in normalized):
        raise ValueError(f"{name} must be a lowercase SHA256 string")
    return normalized


def _sources(value: Any) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("source_experts_by_layer must be a sequence")
    layers = []
    for layer, raw in enumerate(value):
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise ValueError(f"clone sources for layer {layer} must be a sequence")
        sources = tuple(int(expert) for expert in raw)
        if len(sources) != CLONES_PER_LAYER or len(set(sources)) != len(sources):
            raise ValueError(
                f"layer {layer} must contain {CLONES_PER_LAYER} unique clone sources"
            )
        if any(expert < 0 or expert >= ORIGINAL_EXPERTS for expert in sources):
            raise ValueError(f"layer {layer} contains an invalid source expert")
        layers.append(sources)
    if len(layers) != NUM_LAYERS:
        raise ValueError(f"clone contract must cover exactly {NUM_LAYERS} layers")
    return tuple(layers)


@dataclass(frozen=True)
class ExpertCloneContract:
    source_experts_by_layer: tuple[tuple[int, ...], ...]
    selection_sha256: str
    selection_contract_sha256: str

    @property
    def clone_ids(self) -> tuple[int, ...]:
        return tuple(range(ORIGINAL_EXPERTS, TOTAL_EXPERTS))

    def config_value(self) -> dict[str, Any]:
        return {
            "schema": 1,
            "mode": "logical-token-split",
            "original_experts": ORIGINAL_EXPERTS,
            "total_experts": TOTAL_EXPERTS,
            "clones_per_layer": CLONES_PER_LAYER,
            "clone_ids": list(self.clone_ids),
            "source_experts_by_layer": [
                list(sources) for sources in self.source_experts_by_layer
            ],
            "selection_sha256": self.selection_sha256,
            "selection_contract_sha256": self.selection_contract_sha256,
            "split": {
                "algorithm": SPLIT_ALGORITHM,
                "modulus": SPLIT_MODULUS,
                "threshold": SPLIT_THRESHOLD,
                "token_multiplier": SPLIT_TOKEN_MULTIPLIER,
                "layer_multiplier": SPLIT_LAYER_MULTIPLIER,
                "source_multiplier": SPLIT_SOURCE_MULTIPLIER,
            },
            "mtp": "disabled-unmodified",
        }


def contract_from_selection(
    path: str | Path,
    *,
    expected_selection_sha256: str | None = None,
) -> ExpertCloneContract:
    selection_path = Path(path)
    actual_sha256 = sha256_file(selection_path)
    if expected_selection_sha256 is not None:
        expected = _sha256(expected_selection_sha256, "expected selection SHA256")
        if actual_sha256 != expected:
            raise ValueError(
                "expert selection SHA256 mismatch: "
                f"expected {expected}, got {actual_sha256}"
            )
    value = json.loads(selection_path.read_text(encoding="utf-8"))
    if value.get("schema") != 1:
        raise ValueError("unsupported expert selection schema")
    if value.get("bucket") != "always":
        raise ValueError("expert clone selection must come from the always bucket")
    if (
        value.get("selection_scope")
        != "32 independent source experts per decoder layer"
    ):
        raise ValueError("expert clone selection must be per-layer")
    if value.get("task_count") != 654:
        raise ValueError("expert clone selection must contain 654 audited tasks")
    eligibility = value.get("eligibility")
    if eligibility != {
        "rounds_total": 3,
        "rounds_real": 3,
        "rounds_solved": 3,
        "env_failures": 0,
        "prompt_tier": "l2",
    }:
        raise ValueError("expert clone selection eligibility is not fail-closed always/L2")
    layers = value.get("layers")
    if not isinstance(layers, list) or len(layers) != NUM_LAYERS:
        raise ValueError(f"expert selection must contain exactly {NUM_LAYERS} layers")
    sources = []
    expected_clone_ids = list(range(ORIGINAL_EXPERTS, TOTAL_EXPERTS))
    for layer_id, layer in enumerate(layers):
        if not isinstance(layer, Mapping) or layer.get("layer") != layer_id:
            raise ValueError(f"expert selection layer {layer_id} is malformed")
        selected = layer.get("selected")
        if not isinstance(selected, list) or len(selected) != CLONES_PER_LAYER:
            raise ValueError(
                f"expert selection layer {layer_id} must contain {CLONES_PER_LAYER} rows"
            )
        clone_ids = [int(row["clone_expert_id"]) for row in selected]
        if clone_ids != expected_clone_ids:
            raise ValueError(f"expert selection layer {layer_id} clone IDs are unstable")
        sources.append(tuple(int(row["source_expert_id"]) for row in selected))
    return ExpertCloneContract(
        _sources(sources),
        actual_sha256,
        _sha256(value.get("contract_sha256"), "selection contract SHA256"),
    )


def contract_from_config(config: Any) -> ExpertCloneContract | None:
    raw = _field(config, "yeto_routed_expert_clone")
    routed_experts = int(_field(config, "n_routed_experts", 0) or 0)
    if raw is None:
        if routed_experts == ORIGINAL_EXPERTS:
            return None
        raise ValueError(
            "non-standard DeepSeek V4 expert count has no Yeto clone contract"
        )
    if not isinstance(raw, Mapping):
        raise ValueError("yeto_routed_expert_clone must be an object")
    expected_scalars = {
        "schema": 1,
        "mode": "logical-token-split",
        "original_experts": ORIGINAL_EXPERTS,
        "total_experts": TOTAL_EXPERTS,
        "clones_per_layer": CLONES_PER_LAYER,
        "clone_ids": list(range(ORIGINAL_EXPERTS, TOTAL_EXPERTS)),
        "mtp": "disabled-unmodified",
    }
    for name, expected in expected_scalars.items():
        if raw.get(name) != expected:
            raise ValueError(f"invalid clone contract field {name!r}")
    if routed_experts != TOTAL_EXPERTS:
        raise ValueError("expanded config n_routed_experts must be 288")
    if int(_field(config, "num_hidden_layers", 0) or 0) != NUM_LAYERS:
        raise ValueError("expanded clone contract requires 43 decoder layers")
    if int(_field(config, "num_experts_per_tok", 0) or 0) != TOPK:
        raise ValueError("expanded clone contract requires top-6 routing")
    if int(_field(config, "num_nextn_predict_layers", 0) or 0) != 0:
        raise ValueError("expanded clone checkpoint must disable the unmodified MTP layer")
    split = raw.get("split")
    expected_split = {
        "algorithm": SPLIT_ALGORITHM,
        "modulus": SPLIT_MODULUS,
        "threshold": SPLIT_THRESHOLD,
        "token_multiplier": SPLIT_TOKEN_MULTIPLIER,
        "layer_multiplier": SPLIT_LAYER_MULTIPLIER,
        "source_multiplier": SPLIT_SOURCE_MULTIPLIER,
    }
    if split != expected_split:
        raise ValueError("expanded config has an unsupported route split contract")
    return ExpertCloneContract(
        _sources(raw.get("source_experts_by_layer")),
        _sha256(raw.get("selection_sha256"), "selection SHA256"),
        _sha256(
            raw.get("selection_contract_sha256"),
            "selection contract SHA256",
        ),
    )


def split_bucket(token_id: int, layer_id: int, source_expert: int) -> int:
    if token_id < 0:
        raise ValueError("token ID must be non-negative")
    if layer_id < 0 or layer_id >= NUM_LAYERS:
        raise ValueError("layer ID is outside the 43-layer decoder")
    if source_expert < 0 or source_expert >= ORIGINAL_EXPERTS:
        raise ValueError("source expert ID is outside the original router")
    return (
        token_id * SPLIT_TOKEN_MULTIPLIER
        + (layer_id + 1) * SPLIT_LAYER_MULTIPLIER
        + source_expert * SPLIT_SOURCE_MULTIPLIER
    ) % SPLIT_MODULUS


def use_clone(token_id: int, layer_id: int, source_expert: int) -> bool:
    return split_bucket(token_id, layer_id, source_expert) < SPLIT_THRESHOLD


def expand_routes_torch(
    base_probs: Any,
    base_map: Any,
    input_ids: Any,
    *,
    layer_id: int,
    source_experts: Sequence[int],
) -> tuple[Any, Any]:
    """Move selected 256-way routes into the 288-way source/clone layout."""

    import torch

    sources = tuple(int(expert) for expert in source_experts)
    if len(sources) != CLONES_PER_LAYER or len(set(sources)) != len(sources):
        raise ValueError("route expansion requires 32 unique source experts")
    if base_probs.ndim != 2 or tuple(base_probs.shape) != tuple(base_map.shape):
        raise ValueError("routing probabilities/map must be matching rank-2 tensors")
    if base_probs.shape[1] != ORIGINAL_EXPERTS:
        raise ValueError("route expansion requires a 256-way base router")
    token_ids = input_ids.reshape(-1)
    if token_ids.numel() != base_probs.shape[0]:
        raise ValueError("token IDs do not align with flattened router rows")
    if token_ids.requires_grad:
        raise ValueError("token IDs must not require gradients")
    if torch.any(token_ids < 0).item():
        raise ValueError("token IDs must be non-negative")

    source_ids = torch.tensor(sources, dtype=torch.long, device=base_probs.device)
    token_values = token_ids.to(device=base_probs.device, dtype=torch.int64).unsqueeze(1)
    buckets = torch.remainder(
        token_values * SPLIT_TOKEN_MULTIPLIER
        + (int(layer_id) + 1) * SPLIT_LAYER_MULTIPLIER
        + source_ids.unsqueeze(0) * SPLIT_SOURCE_MULTIPLIER,
        SPLIT_MODULUS,
    )
    selected = base_map.index_select(1, source_ids).bool()
    clone_selected = selected & (buckets < SPLIT_THRESHOLD)
    source_probs = base_probs.index_select(1, source_ids)

    expanded_probs = base_probs.new_zeros((base_probs.shape[0], TOTAL_EXPERTS))
    expanded_probs[:, :ORIGINAL_EXPERTS] = base_probs
    expanded_probs[:, source_ids] = torch.where(
        clone_selected,
        torch.zeros_like(source_probs),
        source_probs,
    )
    expanded_probs[:, ORIGINAL_EXPERTS:TOTAL_EXPERTS] = torch.where(
        clone_selected,
        source_probs,
        torch.zeros_like(source_probs),
    )

    expanded_map = torch.zeros(
        (base_map.shape[0], TOTAL_EXPERTS),
        dtype=torch.bool,
        device=base_map.device,
    )
    expanded_map[:, :ORIGINAL_EXPERTS] = base_map.bool()
    expanded_map[:, source_ids] = selected & ~clone_selected
    expanded_map[:, ORIGINAL_EXPERTS:TOTAL_EXPERTS] = clone_selected

    if not torch.equal(expanded_map.sum(dim=1), base_map.bool().sum(dim=1)):
        raise RuntimeError("clone split changed the number of active experts")
    return expanded_probs, expanded_map


def remap_topk_ids_torch(
    base_topk_ids: Any,
    input_ids: Any,
    *,
    layer_id: int,
    source_experts: Sequence[int],
    source_expert_ids: Any | None = None,
) -> Any:
    """Apply the same clone split directly to SGLang's compact top-k IDs."""

    import torch

    sources = tuple(int(expert) for expert in source_experts)
    if len(sources) != CLONES_PER_LAYER or len(set(sources)) != len(sources):
        raise ValueError("top-k remap requires 32 unique source experts")
    if base_topk_ids.ndim != 2 or base_topk_ids.shape[1] != TOPK:
        raise ValueError("top-k IDs must have shape [tokens, 6]")
    token_ids = input_ids.reshape(-1)
    if token_ids.numel() != base_topk_ids.shape[0]:
        raise ValueError("token IDs do not align with SGLang top-k rows")
    if token_ids.requires_grad:
        raise ValueError("token IDs must not require gradients")

    # SGLang invokes this function while recording its decode CUDA graph.
    # Data-dependent host reads such as ``Tensor.item()`` and ``torch.equal``
    # invalidate an active capture.  The same invariants are checked on eager
    # calls (including SGLang's graph warmup); during capture, keep only the
    # metadata checks above and the device-side remap below.
    capturing_cuda_graph = bool(
        base_topk_ids.is_cuda and torch.cuda.is_current_stream_capturing()
    )
    if not capturing_cuda_graph:
        if torch.any(token_ids < 0).item():
            raise ValueError("token IDs must be non-negative")
        if torch.any(base_topk_ids >= ORIGINAL_EXPERTS).item():
            raise ValueError("base top-k emitted an expert outside the 256-way gate")

    if source_expert_ids is None:
        if capturing_cuda_graph:
            raise RuntimeError(
                "CUDA-graph top-k remap requires preallocated source expert IDs"
            )
        source_ids = torch.tensor(
            sources,
            dtype=base_topk_ids.dtype,
            device=base_topk_ids.device,
        )
    else:
        source_ids = source_expert_ids
        if tuple(source_ids.shape) != (CLONES_PER_LAYER,):
            raise ValueError("preallocated source expert IDs must have shape [32]")
        if source_ids.device != base_topk_ids.device:
            raise ValueError("preallocated source expert IDs are on the wrong device")
        if source_ids.dtype != base_topk_ids.dtype:
            raise ValueError("preallocated source expert IDs have the wrong dtype")
        if source_ids.requires_grad:
            raise ValueError("preallocated source expert IDs must not require gradients")
    matches = base_topk_ids.unsqueeze(-1) == source_ids.view(1, 1, -1)
    is_cloned_source = matches.any(dim=-1)
    source_rank = matches.to(dtype=torch.int64).argmax(dim=-1)

    token_values = token_ids.to(
        device=base_topk_ids.device,
        dtype=torch.int64,
    ).unsqueeze(1)
    buckets = torch.remainder(
        token_values * SPLIT_TOKEN_MULTIPLIER
        + (int(layer_id) + 1) * SPLIT_LAYER_MULTIPLIER
        + source_ids.to(dtype=torch.int64).unsqueeze(0) * SPLIT_SOURCE_MULTIPLIER,
        SPLIT_MODULUS,
    )
    clone_for_source = buckets < SPLIT_THRESHOLD
    choose_clone = is_cloned_source & clone_for_source.gather(1, source_rank)
    clone_ids = source_rank.to(dtype=base_topk_ids.dtype) + ORIGINAL_EXPERTS
    remapped = torch.where(choose_clone, clone_ids, base_topk_ids)

    if not capturing_cuda_graph:
        if torch.any(remapped >= TOTAL_EXPERTS).item():
            raise RuntimeError("clone split produced an invalid logical expert ID")
        if not torch.equal(
            (remapped >= 0).sum(dim=1),
            (base_topk_ids >= 0).sum(dim=1),
        ):
            raise RuntimeError("clone split changed the number of valid top-k IDs")
    return remapped
