"""Runtime-only Miles patches for attention LoRA plus clone-expert full tuning."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import json
import os
import re
import sys
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace
from typing import Any

from .deepseek_v4_expert_clone import (
    CLONES_PER_EXPERT_RANK,
    CLONES_PER_LAYER,
    NUM_LAYERS,
    ORIGINAL_EXPERTS,
    TRAINING_EXPERTS_PER_RANK,
    logical_to_training_expert_id,
    training_to_logical_expert_name,
)

_EXPERT_WEIGHT = re.compile(
    r"^(?:base_model\.model\.)?model\.layers\.(?P<layer>\d+)\.mlp\."
    r"experts\.(?P<expert>\d+)\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\.weight$"
)

# Leave one 32 MiB tensor's headroom below Ray's 1 GiB transport boundary for
# mapping metadata and serializer framing.
_RAY_STATE_CHUNK_BYTES = 31 * (1 << 25)
_MAX_APPLY_PROGRESS_MARKERS = 16


def _emit_apply_progress(
    policy_version: int,
    phase: str,
    **progress: int,
) -> None:
    marker = {
        "event": "rl_policy_apply_progress",
        "phase": phase,
        "policy_version": int(policy_version),
        **{name: int(value) for name, value in progress.items()},
    }
    print(
        "[yeto-rl-policy-apply-progress] "
        + json.dumps(marker, sort_keys=True, separators=(",", ":")),
        flush=True,
    )


def _apply_progress_due(completed: int, total: int) -> bool:
    if completed <= 0 or completed > total or total <= 0:
        return False
    stride = max(
        1,
        (total + _MAX_APPLY_PROGRESS_MARKERS - 1)
        // _MAX_APPLY_PROGRESS_MARKERS,
    )
    return completed == total or completed % stride == 0


@dataclass(frozen=True)
class _TrainableStateFragment:
    """One canonical rank's disjoint share of an exported trainable state."""

    source_rank: int
    policy_version: int
    expected_names: tuple[str, ...]
    tensors: Mapping[str, Any]
    is_metrics_source: bool = False
    train_rollout_kl: float | None = None
    ess_ratio: float | None = None
    pg_clipfrac: float | None = None
    train_seconds: float | None = None


@dataclass
class _ChunkedExportFragment:
    """One rank's export metadata plus bounded Ray object references."""

    source_rank: int
    policy_version: int
    expected_names: tuple[str, ...]
    chunk_tensor_names: tuple[tuple[str, ...], ...]
    chunk_tensor_bytes: tuple[int, ...]
    chunk_refs: list[Any]
    is_metrics_source: bool = False
    train_rollout_kl: float | None = None
    ess_ratio: float | None = None
    pg_clipfrac: float | None = None
    train_seconds: float | None = None


@dataclass(frozen=True)
class _OwnedTrainableState:
    """Canonical exported state whose fresh CPU tensor storage is transferable."""

    policy_version: int
    layout_hash: str
    tensors: Mapping[str, Any]
    train_rollout_kl: float | None = None
    ess_ratio: float | None = None
    pg_clipfrac: float | None = None
    train_seconds: float | None = None
    _yeto_owned_tensors: str = "canonical-v1"


@dataclass
class _ChunkedPolicyExport:
    """Validated owner-sharded export whose payload remains in Ray objects."""

    policy_version: int
    expected_names: tuple[str, ...]
    fragments: tuple[_ChunkedExportFragment, ...]
    train_rollout_kl: float | None = None
    ess_ratio: float | None = None
    pg_clipfrac: float | None = None
    train_seconds: float | None = None
    _remaining_names: set[str] = field(init=False, repr=False)
    _yeto_chunked_export: str = "owner-sharded-v1"

    def __post_init__(self) -> None:
        self._remaining_names = set(self.expected_names)

    def take_tensors(self, expected_specs, *, resolve_ref=None):
        return _take_chunked_export_tensors(
            self,
            expected_specs,
            resolve_ref=resolve_ref,
        )

    def finish(self) -> None:
        if self._remaining_names:
            raise RuntimeError("chunked hybrid export was not completely consumed")
        if any(
            reference is not None
            for fragment in self.fragments
            for reference in fragment.chunk_refs
        ):
            raise RuntimeError("chunked hybrid export retained a Ray object reference")

    def discard(self) -> None:
        for fragment in self.fragments:
            fragment.chunk_refs[:] = [None] * len(fragment.chunk_refs)
        self._remaining_names.clear()


@dataclass(frozen=True)
class _ChunkedStateManifest:
    """Small Ray message whose payload remains in bounded object-store objects."""

    policy_version: int
    layout_hash: str
    tensor_names: tuple[str, ...]
    chunk_tensor_names: tuple[tuple[str, ...], ...]
    chunk_tensor_bytes: tuple[int, ...]
    train_rollout_kl: float | None = None
    ess_ratio: float | None = None
    pg_clipfrac: float | None = None
    train_seconds: float | None = None


@dataclass
class _ChunkApplyContext:
    manifest: _ChunkedStateManifest
    reset_optimizer: bool
    next_chunk: int = 0


def _expert_count() -> int:
    raw = os.environ.get("YETO_DSV4_EXPERT_FULL_COUNT")
    try:
        count = int(raw or "")
    except ValueError as exc:
        raise RuntimeError("YETO_DSV4_EXPERT_FULL_COUNT must be an integer") from exc
    if not 1 <= count <= CLONES_PER_LAYER:
        raise RuntimeError(
            f"YETO_DSV4_EXPERT_FULL_COUNT must be in [1, {CLONES_PER_LAYER}]"
        )
    return count


def selected_expert_hf_name(name: str, *, expert_count: int) -> bool:
    match = _EXPERT_WEIGHT.fullmatch(name)
    return bool(
        match
        and 0 <= int(match.group("layer")) < NUM_LAYERS
        and ORIGINAL_EXPERTS
        <= int(match.group("expert"))
        < ORIGINAL_EXPERTS + expert_count
    )


def _mapping_hf_names(task: Any) -> tuple[str, ...]:
    if task is None:
        return ()
    value = getattr(getattr(task, "mapping", None), "hf_param", None)
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping) and all(isinstance(item, str) for item in value.values()):
        return tuple(value.values())
    return ()


def _logical_mapping_hf_names(task: Any) -> tuple[str, ...]:
    return tuple(
        training_to_logical_expert_name(name)
        for name in _mapping_hf_names(task)
    )


def filter_selected_expert_tasks(tasks, *, expert_count: int) -> list[Any]:
    selected = []
    for task in tasks:
        names = _logical_mapping_hf_names(task)
        if names and all(
            selected_expert_hf_name(name, expert_count=expert_count)
            for name in names
        ):
            selected.append(task)
    return selected


def filter_collective_expert_tasks(tasks, *, expert_count: int) -> list[Any]:
    """Select identical local expert offsets on every EP rank."""

    selected_offsets = {
        logical_to_training_expert_id(expert) % TRAINING_EXPERTS_PER_RANK
        for expert in range(ORIGINAL_EXPERTS, ORIGINAL_EXPERTS + expert_count)
    }
    selected = []
    for task in tasks:
        matches = tuple(
            _EXPERT_WEIGHT.fullmatch(name) for name in _mapping_hf_names(task)
        )
        if matches and all(
            match is not None
            and int(match.group("expert")) % TRAINING_EXPERTS_PER_RANK
            in selected_offsets
            for match in matches
        ):
            selected.append(task)
    return selected


def _validate_hybrid_names(tensors: Mapping[str, Any], expert_count: int) -> None:
    expected_experts = {
        "base_model.model.model.layers."
        f"{layer}.mlp.experts.{expert}.{projection}.weight"
        for layer in range(NUM_LAYERS)
        for expert in range(ORIGINAL_EXPERTS, ORIGINAL_EXPERTS + expert_count)
        for projection in ("gate_proj", "up_proj", "down_proj")
    }
    actual_experts = {name for name in tensors if _EXPERT_WEIGHT.fullmatch(name)}
    outside = sorted(actual_experts - expected_experts)
    if outside:
        raise ValueError(
            "expert tensor is outside the selected clone policy: "
            f"{outside[0]!r}"
        )
    missing = sorted(expected_experts - actual_experts)
    if missing:
        raise ValueError(f"hybrid policy is missing expert tensor {missing[0]!r}")
    lora = [
        name
        for name in tensors
        if name.endswith((".lora_A.weight", ".lora_B.weight"))
    ]
    if not lora:
        raise ValueError("hybrid policy contains no attention LoRA tensors")
    allowed = expected_experts.union(lora)
    extra = sorted(set(tensors) - allowed)
    if extra:
        raise ValueError(f"hybrid policy contains unsupported tensor {extra[0]!r}")


def make_hybrid_trainable_state(
    module: ModuleType,
    policy_version: int,
    tensors: Mapping[str, Any],
    *,
    train_rollout_kl: float | None = None,
    ess_ratio: float | None = None,
    pg_clipfrac: float | None = None,
    train_seconds: float | None = None,
):
    import torch

    if policy_version < 0:
        raise ValueError("policy version must be non-negative")
    if not tensors:
        raise ValueError("trainable state is empty")
    _validate_hybrid_names(tensors, _expert_count())
    canonical = {}
    for name, tensor in sorted(tensors.items()):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"hybrid policy tensor {name!r} is not a tensor")
        value = tensor.detach() if tensor.requires_grad else tensor
        if value.device.type != "cpu" or value.dtype != torch.float32 or not value.is_contiguous():
            value = value.to(device="cpu", dtype=torch.float32).contiguous()
        if not torch.isfinite(value).all().item():
            raise ValueError(f"{name!r} contains NaN or Inf")
        canonical[name] = value
    return module.TrainableState(
        policy_version,
        module._layout_hash(canonical),
        canonical,
        train_rollout_kl,
        ess_ratio,
        pg_clipfrac,
        train_seconds,
    )


def _state_metrics(state) -> dict[str, float | None]:
    return {
        "train_rollout_kl": getattr(state, "train_rollout_kl", None),
        "ess_ratio": getattr(state, "ess_ratio", None),
        "pg_clipfrac": getattr(state, "pg_clipfrac", None),
        "train_seconds": getattr(state, "train_seconds", None),
    }


def _merge_export_fragments(module: ModuleType, fragments):
    indexed_fragments = [
        (rank, fragment)
        for rank, fragment in enumerate(fragments)
        if fragment is not None
    ]
    fragments = [fragment for _rank, fragment in indexed_fragments]
    if not fragments:
        raise RuntimeError("no Megatron rank exported a hybrid state fragment")
    if any(not isinstance(fragment, _TrainableStateFragment) for fragment in fragments):
        raise TypeError("Megatron rank returned an invalid hybrid state fragment")
    if any(rank != fragment.source_rank for rank, fragment in indexed_fragments):
        raise RuntimeError("hybrid state fragment source rank does not match result rank")

    ranks = [fragment.source_rank for fragment in fragments]
    if len(set(ranks)) != len(ranks):
        raise RuntimeError("duplicate Megatron rank in hybrid state fragments")
    roots = [fragment for fragment in fragments if fragment.source_rank == 0]
    if len(roots) != 1:
        raise RuntimeError("hybrid state fragments must contain rank zero exactly once")
    root = roots[0]
    expected_names = root.expected_names
    if len(set(expected_names)) != len(expected_names):
        raise RuntimeError("hybrid state fragment contains duplicate expected names")
    if any(fragment.policy_version != root.policy_version for fragment in fragments):
        raise RuntimeError("Megatron ranks disagree on hybrid policy version")
    if any(fragment.expected_names != expected_names for fragment in fragments):
        raise RuntimeError("Megatron ranks disagree on hybrid tensor layout")
    metrics_sources = [
        fragment for fragment in fragments if fragment.is_metrics_source
    ]
    if len(metrics_sources) != 1:
        raise RuntimeError(
            "hybrid state fragments must contain one Megatron metrics source"
        )
    metrics_source = metrics_sources[0]
    if any(
        any(value is not None for value in _state_metrics(fragment).values())
        for fragment in fragments
        if not fragment.is_metrics_source
    ):
        raise RuntimeError("non-main rank returned hybrid training metrics")

    tensors = {}
    for fragment in sorted(fragments, key=lambda item: item.source_rank):
        overlap = set(tensors).intersection(fragment.tensors)
        if overlap:
            raise RuntimeError(
                "hybrid state fragments contain duplicate tensor "
                f"{min(overlap)!r}"
            )
        tensors.update(fragment.tensors)
    actual_names = set(tensors)
    expected_set = set(expected_names)
    missing = sorted(expected_set - actual_names)
    extra = sorted(actual_names - expected_set)
    if missing or extra:
        raise RuntimeError(
            "hybrid state fragments do not exactly cover the policy: "
            f"missing={missing[:1]}, extra={extra[:1]}"
        )
    return make_hybrid_trainable_state(
        module,
        root.policy_version,
        tensors,
        **_state_metrics(metrics_source),
    )


def _tensor_bytes(name: str, tensor: Any) -> int:
    size = getattr(tensor, "numel", None)
    element_size = getattr(tensor, "element_size", None)
    if size is None or element_size is None:
        raise TypeError(f"hybrid policy tensor {name!r} is not a tensor")
    return int(size()) * int(element_size())


def _chunk_export_fragment_for_ray(
    fragment: _TrainableStateFragment,
    *,
    ray_module=None,
    max_chunk_bytes: int = _RAY_STATE_CHUNK_BYTES,
    tensor_groups: tuple[tuple[str, ...], ...] | None = None,
) -> _ChunkedExportFragment:
    """Move a rank-local export into bounded Ray objects before returning it."""

    if not isinstance(fragment, _TrainableStateFragment):
        raise TypeError("Megatron rank produced an invalid hybrid state fragment")
    if not isinstance(fragment.tensors, dict):
        raise TypeError("hybrid export fragment does not own mutable tensor storage")
    if ray_module is None:
        import ray as ray_module

    if tensor_groups is None:
        groups = (tuple(fragment.tensors),)
    else:
        flattened = tuple(name for group in tensor_groups for name in group)
        if len(set(flattened)) != len(flattened) or set(flattened) != set(
            fragment.expected_names
        ):
            raise RuntimeError(
                "chunked hybrid export groups do not exactly cover the policy"
            )
        groups = tensor_groups
    chunk_names = []
    chunk_sizes = []
    for group in groups:
        local = {
            name: fragment.tensors[name]
            for name in group
            if name in fragment.tensors
        }
        if not local:
            continue
        _names, local_chunks, local_sizes = _plan_state_chunks(
            SimpleNamespace(tensors=local),
            max_chunk_bytes=max_chunk_bytes,
        )
        chunk_names.extend(local_chunks)
        chunk_sizes.extend(local_sizes)
    chunk_names = tuple(chunk_names)
    chunk_sizes = tuple(chunk_sizes)
    chunk_refs = []
    for names_in_chunk in chunk_names:
        chunk = {name: fragment.tensors[name] for name in names_in_chunk}
        chunk_refs.append(ray_module.put(chunk))
        for name in names_in_chunk:
            fragment.tensors.pop(name)
        del chunk
    if fragment.tensors:
        raise RuntimeError("chunked hybrid export did not consume its tensor storage")
    return _ChunkedExportFragment(
        source_rank=fragment.source_rank,
        policy_version=fragment.policy_version,
        expected_names=fragment.expected_names,
        chunk_tensor_names=chunk_names,
        chunk_tensor_bytes=chunk_sizes,
        chunk_refs=chunk_refs,
        is_metrics_source=fragment.is_metrics_source,
        **_state_metrics(fragment),
    )


def _validate_chunked_export_metadata(fragments):
    indexed_fragments = [
        (rank, fragment)
        for rank, fragment in enumerate(fragments)
        if fragment is not None
    ]
    fragments = [fragment for _rank, fragment in indexed_fragments]
    if not fragments:
        raise RuntimeError("no Megatron rank exported a hybrid state fragment")
    if any(not isinstance(fragment, _ChunkedExportFragment) for fragment in fragments):
        raise TypeError("Megatron rank returned an invalid chunked export fragment")
    if any(rank != fragment.source_rank for rank, fragment in indexed_fragments):
        raise RuntimeError(
            "hybrid state fragment source rank does not match result rank"
        )
    ranks = [fragment.source_rank for fragment in fragments]
    if len(set(ranks)) != len(ranks):
        raise RuntimeError("duplicate Megatron rank in hybrid state fragments")
    roots = [fragment for fragment in fragments if fragment.source_rank == 0]
    if len(roots) != 1:
        raise RuntimeError("hybrid state fragments must contain rank zero exactly once")
    root = roots[0]
    expected_names = root.expected_names
    if len(set(expected_names)) != len(expected_names):
        raise RuntimeError("hybrid state fragment contains duplicate expected names")
    if any(fragment.policy_version != root.policy_version for fragment in fragments):
        raise RuntimeError("Megatron ranks disagree on hybrid policy version")
    if any(fragment.expected_names != expected_names for fragment in fragments):
        raise RuntimeError("Megatron ranks disagree on hybrid tensor layout")
    metrics_sources = [fragment for fragment in fragments if fragment.is_metrics_source]
    if len(metrics_sources) != 1:
        raise RuntimeError(
            "hybrid state fragments must contain one Megatron metrics source"
        )
    metrics_source = metrics_sources[0]
    if any(
        any(value is not None for value in _state_metrics(fragment).values())
        for fragment in fragments
        if not fragment.is_metrics_source
    ):
        raise RuntimeError("non-main rank returned hybrid training metrics")

    exported_names = []
    for fragment in fragments:
        if not (
            fragment.chunk_tensor_names
            and len(fragment.chunk_tensor_names)
            == len(fragment.chunk_tensor_bytes)
            == len(fragment.chunk_refs)
        ):
            raise RuntimeError("chunked hybrid export manifest is inconsistent")
        names = tuple(
            name
            for names_in_chunk in fragment.chunk_tensor_names
            for name in names_in_chunk
        )
        if (
            not names
            or any(
                not names_in_chunk
                or names_in_chunk != tuple(sorted(names_in_chunk))
                for names_in_chunk in fragment.chunk_tensor_names
            )
            or len(set(names)) != len(names)
        ):
            raise RuntimeError(
                "chunked hybrid export does not follow canonical tensor order"
            )
        if any(
            size <= 0 or size > _RAY_STATE_CHUNK_BYTES
            for size in fragment.chunk_tensor_bytes
        ):
            raise RuntimeError("chunked hybrid export exceeds the transport cap")
        exported_names.extend(names)
    if len(set(exported_names)) != len(exported_names):
        raise RuntimeError("hybrid state fragments contain a duplicate tensor")
    actual_names = set(exported_names)
    expected_set = set(expected_names)
    missing = sorted(expected_set - actual_names)
    extra = sorted(actual_names - expected_set)
    if missing or extra:
        raise RuntimeError(
            "hybrid state fragments do not exactly cover the policy: "
            f"missing={missing[:1]}, extra={extra[:1]}"
        )
    return sorted(fragments, key=lambda item: item.source_rank), root, metrics_source


def _prepare_chunked_policy_export(fragments) -> _ChunkedPolicyExport:
    fragments, root, metrics_source = _validate_chunked_export_metadata(fragments)
    return _ChunkedPolicyExport(
        policy_version=root.policy_version,
        expected_names=root.expected_names,
        fragments=tuple(fragments),
        **_state_metrics(metrics_source),
    )


def _take_chunked_export_tensors(
    export: _ChunkedPolicyExport,
    expected_specs,
    *,
    resolve_ref=None,
):
    """Resolve and relinquish exactly one pre-aligned logical tensor group."""

    import torch

    if resolve_ref is None:
        import ray

        resolve_ref = ray.get
    specs = tuple(expected_specs)
    names = tuple(spec.name for spec in specs)
    if not names or len(set(names)) != len(names):
        raise RuntimeError("chunked export request has invalid tensor specs")
    requested = set(names)
    if not requested <= export._remaining_names:
        raise RuntimeError("chunked export tensor group was already consumed or unknown")
    spec_by_name = {spec.name: spec for spec in specs}
    tensors = {}
    for fragment in export.fragments:
        for chunk_index, (chunk_names, expected_bytes, reference) in enumerate(
            zip(
                fragment.chunk_tensor_names,
                fragment.chunk_tensor_bytes,
                fragment.chunk_refs,
                strict=True,
            )
        ):
            if reference is None or not requested.intersection(chunk_names):
                continue
            if not set(chunk_names) <= requested:
                raise RuntimeError(
                    "chunked hybrid export was not aligned to the requested group"
                )
            try:
                chunk = resolve_ref(reference)
                if not isinstance(chunk, Mapping) or tuple(chunk) != chunk_names:
                    raise RuntimeError(
                        "hybrid export chunk does not match its tensor manifest"
                    )
                for name, tensor in chunk.items():
                    spec = spec_by_name[name]
                    if not isinstance(tensor, torch.Tensor):
                        raise TypeError(f"hybrid policy tensor {name!r} is not a tensor")
                    value = tensor.detach() if tensor.requires_grad else tensor
                    if (
                        value.device.type != "cpu"
                        or value.dtype != torch.float32
                        or not value.is_contiguous()
                    ):
                        value = value.to(
                            device="cpu",
                            dtype=torch.float32,
                        ).contiguous()
                    if tuple(value.shape) != spec.shape or value.numel() != spec.numel:
                        raise RuntimeError(
                            f"hybrid policy tensor {name!r} changed shape"
                        )
                    if not torch.isfinite(value).all().item():
                        raise ValueError(f"{name!r} contains NaN or Inf")
                    tensors[name] = value
                actual_bytes = sum(
                    _tensor_bytes(name, tensor) for name, tensor in tensors.items()
                    if name in chunk_names
                )
                if actual_bytes != expected_bytes:
                    raise RuntimeError(
                        "hybrid export chunk violates its size manifest"
                    )
                del chunk
            finally:
                fragment.chunk_refs[chunk_index] = None
    if set(tensors) != requested:
        missing = sorted(requested - set(tensors))
        raise RuntimeError(
            f"chunked hybrid export did not resolve requested tensors: {missing[:1]}"
        )
    export._remaining_names.difference_update(requested)
    return {name: tensors[name] for name in names}


def _merge_chunked_export_fragments(
    module: ModuleType,
    fragments,
    *,
    resolve_ref=None,
):
    """Resolve a distributed export one bounded chunk at a time."""

    import torch

    if resolve_ref is None:
        import ray

        resolve_ref = ray.get
    fragments, root, metrics_source = _validate_chunked_export_metadata(fragments)
    tensors = {}
    for fragment in fragments:
        for chunk_index, (names, expected_bytes, reference) in enumerate(
            zip(
                fragment.chunk_tensor_names,
                fragment.chunk_tensor_bytes,
                fragment.chunk_refs,
                strict=True,
            )
        ):
            try:
                chunk = resolve_ref(reference)
                if not isinstance(chunk, Mapping) or not chunk:
                    raise RuntimeError("hybrid export chunk is empty or invalid")
                if tuple(chunk) != names:
                    raise RuntimeError(
                        "hybrid export chunk does not match canonical tensor order"
                    )
                canonical = {}
                for name, tensor in chunk.items():
                    if not isinstance(tensor, torch.Tensor):
                        raise TypeError(f"hybrid policy tensor {name!r} is not a tensor")
                    value = tensor.detach() if tensor.requires_grad else tensor
                    if (
                        value.device.type != "cpu"
                        or value.dtype != torch.float32
                        or not value.is_contiguous()
                    ):
                        value = value.to(device="cpu", dtype=torch.float32).contiguous()
                    if not torch.isfinite(value).all().item():
                        raise ValueError(f"{name!r} contains NaN or Inf")
                    canonical[name] = value
                actual_bytes = sum(
                    _tensor_bytes(name, tensor) for name, tensor in canonical.items()
                )
                if actual_bytes != expected_bytes:
                    raise RuntimeError("hybrid export chunk violates its size manifest")
                tensors.update(canonical)
                del canonical, chunk
            finally:
                # Drop the producer-owned object as soon as its canonical
                # tensors have been retained.  If Ray deserialized by copy,
                # this releases that transport copy before the next chunk.
                fragment.chunk_refs[chunk_index] = None
    tensors = dict(sorted(tensors.items()))
    _validate_hybrid_names(tensors, _expert_count())
    layout_hash = module._layout_hash(tensors)
    return _OwnedTrainableState(
        policy_version=root.policy_version,
        layout_hash=layout_hash,
        tensors=tensors,
        **_state_metrics(metrics_source),
    )


def _chunk_state_for_ray(
    state,
    *,
    ray_module=None,
    max_chunk_bytes: int = _RAY_STATE_CHUNK_BYTES,
) -> tuple[_ChunkedStateManifest, tuple[Any, ...]]:
    if max_chunk_bytes <= 0:
        raise ValueError("Ray state chunk size must be positive")
    if ray_module is None:
        import ray as ray_module

    names, chunk_names, chunk_sizes = _plan_state_chunks(
        state,
        max_chunk_bytes=max_chunk_bytes,
    )
    chunk_refs = tuple(
        ray_module.put({name: state.tensors[name] for name in names_in_chunk})
        for names_in_chunk in chunk_names
    )
    manifest = _ChunkedStateManifest(
        policy_version=state.policy_version,
        layout_hash=state.layout_hash,
        tensor_names=names,
        chunk_tensor_names=chunk_names,
        chunk_tensor_bytes=chunk_sizes,
        **_state_metrics(state),
    )
    return manifest, chunk_refs


def _plan_state_chunks(state, *, max_chunk_bytes: int):
    if max_chunk_bytes <= 0:
        raise ValueError("Ray state chunk size must be positive")
    names = tuple(sorted(state.tensors))
    if not names:
        raise ValueError("cannot chunk an empty trainable state")
    chunks = []
    sizes = []
    chunk = []
    chunk_bytes = 0
    for name in names:
        tensor_bytes = _tensor_bytes(name, state.tensors[name])
        if tensor_bytes > max_chunk_bytes:
            raise ValueError(
                f"hybrid policy tensor {name!r} exceeds the Ray chunk cap"
            )
        if chunk and chunk_bytes + tensor_bytes > max_chunk_bytes:
            chunks.append(tuple(chunk))
            sizes.append(chunk_bytes)
            chunk = []
            chunk_bytes = 0
        chunk.append(name)
        chunk_bytes += tensor_bytes
    if chunk:
        chunks.append(tuple(chunk))
        sizes.append(chunk_bytes)
    return names, tuple(chunks), tuple(sizes)


def _chunk_manifest(state, *, max_chunk_bytes: int = _RAY_STATE_CHUNK_BYTES):
    """Describe canonical chunks without allocating or pinning Ray objects."""

    names, chunk_names, chunk_sizes = _plan_state_chunks(
        state,
        max_chunk_bytes=max_chunk_bytes,
    )
    return _ChunkedStateManifest(
        policy_version=state.policy_version,
        layout_hash=state.layout_hash,
        tensor_names=names,
        chunk_tensor_names=chunk_names,
        chunk_tensor_bytes=chunk_sizes,
        **_state_metrics(state),
    )


def _state_from_chunk_manifest(
    module: ModuleType,
    manifest: _ChunkedStateManifest,
    *,
    chunks,
):
    if not isinstance(manifest, _ChunkedStateManifest):
        raise TypeError("rank zero did not receive a chunked hybrid state manifest")
    chunks = tuple(chunks)
    if not (
        len(chunks)
        == len(manifest.chunk_tensor_names)
        == len(manifest.chunk_tensor_bytes)
    ):
        raise RuntimeError("chunked hybrid state manifest is inconsistent")
    if not chunks:
        raise RuntimeError("chunked hybrid state manifest is empty")
    if len(set(manifest.tensor_names)) != len(manifest.tensor_names):
        raise RuntimeError("chunked hybrid state manifest has duplicate tensor names")
    tensors = {}
    for index, (chunk, expected_names, expected_bytes) in enumerate(
        zip(
            chunks,
            manifest.chunk_tensor_names,
            manifest.chunk_tensor_bytes,
            strict=True,
        )
    ):
        if not isinstance(chunk, Mapping) or not chunk:
            raise RuntimeError(f"hybrid state chunk {index} is empty or invalid")
        actual_bytes = sum(_tensor_bytes(name, tensor) for name, tensor in chunk.items())
        if actual_bytes != expected_bytes or actual_bytes > _RAY_STATE_CHUNK_BYTES:
            raise RuntimeError(f"hybrid state chunk {index} violates its size manifest")
        if set(chunk) != set(expected_names) or len(chunk) != len(expected_names):
            raise RuntimeError(
                f"hybrid state chunk {index} does not match its name manifest"
            )
        overlap = set(tensors).intersection(chunk)
        if overlap:
            raise RuntimeError(
                f"hybrid state chunks contain duplicate tensor {min(overlap)!r}"
            )
        tensors.update(chunk)
    if set(tensors) != set(manifest.tensor_names):
        missing = sorted(set(manifest.tensor_names) - set(tensors))
        extra = sorted(set(tensors) - set(manifest.tensor_names))
        raise RuntimeError(
            "hybrid state chunks do not exactly cover their manifest: "
            f"missing={missing[:1]}, extra={extra[:1]}"
        )
    layout_hash = module._layout_hash(tensors)
    if layout_hash != manifest.layout_hash:
        raise RuntimeError("chunked hybrid trainable-state layout hash mismatch")
    return module.TrainableState(
        manifest.policy_version,
        layout_hash,
        tensors,
        manifest.train_rollout_kl,
        manifest.ess_ratio,
        manifest.pg_clipfrac,
        manifest.train_seconds,
    )


def install_on_lora_utils(module: ModuleType) -> None:
    if getattr(module, "_yeto_expert_full_installed", False):
        return
    original = module.create_lora_instance

    def create_lora_instance(args):
        from .deepseek_v4_expert_full import wrap_attention_lora_with_expert_full

        lora = original(args)
        return wrap_attention_lora_with_expert_full(
            lora,
            configure_kwargs={"expert_count": _expert_count()},
        )

    module.create_lora_instance = create_lora_instance
    module._yeto_expert_full_installed = True


def _expert_lr() -> float:
    try:
        value = float(os.environ.get("YETO_DSV4_EXPERT_FULL_LR", ""))
    except ValueError as exc:
        raise RuntimeError("YETO_DSV4_EXPERT_FULL_LR must be a float") from exc
    if value <= 0:
        raise RuntimeError("YETO_DSV4_EXPERT_FULL_LR must be positive")
    return value


def install_on_arguments(module: ModuleType) -> None:
    if getattr(module, "_yeto_expert_full_installed", False):
        return
    original = module.set_default_megatron_args

    def set_default_megatron_args(args):
        args = original(args)
        if (args.optimizer or "adam").lower() != "adam":
            raise RuntimeError("expert-full RL requires the Adam optimizer")
        args.use_distributed_optimizer = True
        return args

    module.set_default_megatron_args = set_default_megatron_args
    module._yeto_expert_full_installed = True


def install_on_model(module: ModuleType) -> None:
    if getattr(module, "_yeto_expert_full_installed", False):
        return
    original = module.get_megatron_optimizer

    def get_megatron_optimizer(
        *,
        config,
        model_chunks,
        config_overrides=None,
        use_gloo_process_groups=True,
        **kwargs,
    ):
        from megatron.core.optimizer import get_standard_config_overrides
        from megatron.core.optimizer.optimizer_config import ParamKey

        overrides = (
            get_standard_config_overrides(config)
            if config_overrides is None
            else dict(config_overrides)
        )
        overrides[ParamKey(attr="_yeto_expert_full")] = {
            "max_lr": _expert_lr(),
            "min_lr": _expert_lr(),
        }
        return original(
            config=config,
            model_chunks=model_chunks,
            config_overrides=overrides,
            use_gloo_process_groups=use_gloo_process_groups,
            **kwargs,
        )

    module.get_megatron_optimizer = get_megatron_optimizer
    module._yeto_expert_full_installed = True


def _canonical_name(name: str) -> str:
    return name if name.startswith("base_model.model.") else "base_model.model." + name


def _model_chunks(models: Any) -> tuple[Any, ...]:
    return tuple(models) if isinstance(models, (list, tuple)) else (models,)


def _optimizer_master_view(parameter):
    import torch

    main = getattr(parameter, "main_param", None)
    if (
        main is None
        or main.dtype != torch.float32
        or main.numel() != parameter.numel()
    ):
        raise RuntimeError("trainable parameter has no complete FP32 optimizer master")
    return main.view(parameter.shape)


@contextmanager
def _attention_masters_as_model_parameters(sides):
    """Expose FP32 attention masters while Bridge builds apply transforms."""

    originals = []
    for side in sides.values():
        parameter = side.param_weight
        if parameter is None:
            continue
        originals.append((parameter, parameter.data))
        parameter.data = _optimizer_master_view(parameter)
    try:
        yield
    finally:
        for parameter, original in originals:
            parameter.data = original


def _expected_specs(actor) -> tuple[Any, ...]:
    specs = tuple(getattr(actor.args, "yeto_rl_expected_specs", ()))
    if not specs:
        raise RuntimeError("expert-full actor has no canonical policy specs")
    return specs


def _attention_specs(actor) -> tuple[Any, ...]:
    return tuple(
        spec
        for spec in _expected_specs(actor)
        if spec.name.endswith((".lora_A.weight", ".lora_B.weight"))
    )


def _expert_specs(actor) -> tuple[Any, ...]:
    return tuple(
        spec
        for spec in _expected_specs(actor)
        if selected_expert_hf_name(spec.name, expert_count=_expert_count())
    )


def _actor_bridge(actor):
    cached = getattr(actor, "_yeto_expert_full_bridge", None)
    if cached is not None:
        return cached
    # Conversion tasks must use the HuggingFace key layout that was loaded into
    # the trainer.  Quantized rollout checkpoints can expose an engine-specific
    # layout that omits the canonical expert names required by Megatron-Bridge.
    source = getattr(actor.args, "ref_load", None) or getattr(
        actor.args, "hf_checkpoint", None
    )
    if not source:
        raise RuntimeError(
            "expert-full actor has no HuggingFace checkpoint for Bridge tasks"
        )
    from megatron.bridge import AutoBridge

    cached = AutoBridge.from_hf_pretrained(
        source,
        trust_remote_code=bool(actor.args.yeto_rl_trust_remote_code),
    )
    actor._yeto_expert_full_bridge = cached
    return cached


def _attention_sides(actor) -> dict[str, Any]:
    cached = getattr(actor, "_yeto_expert_full_attention_sides", None)
    if cached is not None:
        return cached
    model_bridge = getattr(_actor_bridge(actor), "_model_bridge", None)
    build = getattr(model_bridge, "build_adapter_conversion_tasks", None)
    if build is None:
        raise RuntimeError("Megatron-Bridge lacks adapter conversion tasks")
    sides = {}
    for tasks in build(actor.model).values():
        for task in tasks:
            for side in (task.linear_in_task, task.linear_out_task):
                name = _canonical_name(str(side.mapping.hf_param))
                if not name.endswith((".lora_A.weight", ".lora_B.weight")):
                    continue
                if name in sides:
                    raise RuntimeError(f"duplicate attention LoRA mapping {name!r}")
                sides[name] = side
    expected = {spec.name for spec in _attention_specs(actor)}
    if set(sides) != expected:
        missing = sorted(expected - set(sides))
        extra = sorted(set(sides) - expected)
        raise RuntimeError(
            f"attention LoRA mapping mismatch: missing={missing[:4]}, extra={extra[:4]}"
        )
    actor._yeto_expert_full_attention_sides = sides
    return sides


def _expert_views(actor) -> dict[str, Any]:
    cached = getattr(actor, "_yeto_expert_full_views", None)
    if cached is not None:
        return cached
    views = {}
    expert_parameters = {}
    for chunk in _model_chunks(actor.model):
        for parameter in chunk.parameters():
            if not getattr(parameter, "_yeto_expert_full", False):
                continue
            expert_parameters[id(parameter)] = parameter

    expected = {spec.name for spec in _expert_specs(actor)}
    mapped_parameters = set()
    tasks = filter_selected_expert_tasks(
        _actor_bridge(actor).get_conversion_tasks(actor.model),
        expert_count=_expert_count(),
    )
    for task in tasks:
        parameter = getattr(task, "param_weight", None)
        if parameter is None:
            continue
        if id(parameter) not in expert_parameters:
            raise RuntimeError(
                "expert-full conversion task does not own a local trainable parameter"
            )
        names = tuple(
            _canonical_name(name) for name in _logical_mapping_hf_names(task)
        )
        projections = {}
        for name in names:
            match = _EXPERT_WEIGHT.fullmatch(name)
            if match is None or name not in expected:
                raise RuntimeError(
                    f"expert-full conversion task is outside the policy: {name!r}"
                )
            if int(match.group("expert")) != int(parameter._yeto_expert_id):
                raise RuntimeError(
                    f"expert-full conversion task has the wrong owner: {name!r}"
                )
            projection = match.group("projection")
            if projection in projections or name in views:
                raise RuntimeError(
                    f"duplicate expert-full conversion mapping: {name!r}"
                )
            projections[projection] = name
        master = _optimizer_master_view(parameter)
        branch = str(parameter._yeto_expert_branch)
        if branch == "linear_fc1" and set(projections) == {"gate_proj", "up_proj"}:
            gate, up = master.chunk(2, dim=0)
            views[projections["gate_proj"]] = gate
            views[projections["up_proj"]] = up
        elif branch == "linear_fc2" and set(projections) == {"down_proj"}:
            views[projections["down_proj"]] = master
        else:
            raise RuntimeError(
                f"expert-full conversion task does not match {branch!r}: {names!r}"
            )
        mapped_parameters.add(id(parameter))
    if mapped_parameters != set(expert_parameters):
        raise RuntimeError("expert-full conversion tasks do not cover local parameters")

    attention_parameters = {
        id(side.param_weight)
        for side in _attention_sides(actor).values()
        if side.param_weight is not None
    }
    trainable = {
        id(parameter)
        for chunk in _model_chunks(actor.model)
        for parameter in chunk.parameters()
        if parameter.requires_grad
    }
    if trainable != set(expert_parameters) | attention_parameters:
        raise RuntimeError("hybrid policy does not cover every trainable parameter")
    actor._yeto_expert_full_views = views
    return views


def _frozen_expert_versions(actor) -> dict[int, int]:
    return {
        id(parameter): int(parameter._version)
        for chunk in _model_chunks(actor.model)
        for parameter in chunk.parameters()
        if getattr(parameter, "_yeto_expert_full_configured", False)
        and not getattr(parameter, "_yeto_expert_full", False)
    }


def _assert_frozen_experts_unchanged(actor) -> None:
    current = _frozen_expert_versions(actor)
    previous = getattr(actor, "_yeto_frozen_expert_versions", None)
    if previous is not None and current != previous:
        raise RuntimeError("a frozen original or unselected expert was modified")
    actor._yeto_frozen_expert_versions = current


def _export_attention(actor, *, retain: bool) -> dict[str, Any]:
    import torch
    import torch.distributed as dist

    specs = _attention_specs(actor)
    sides = _attention_sides(actor)
    local = {}
    with torch.no_grad():
        for spec in specs:
            side = sides[spec.name]
            parameter = (
                None
                if side.param_weight is None
                else _optimizer_master_view(side.param_weight)
            )
            exported = {
                _canonical_name(str(name)): value
                for name, value in side.mapping.megatron_to_hf(
                    parameter,
                    side.megatron_module,
                ).items()
            }
            if set(exported) != {spec.name}:
                raise RuntimeError(
                    f"attention LoRA export mapping mismatch for {spec.name!r}"
                )
            # The mapping call already broadcasts across PP and gathers TP shards.
            # A None parameter means the task is remote, not that its mapped value
            # is absent; retain rank 0's canonical copy plus physical-owner copies
            # used by the world-level replica validation below.
            if side.param_weight is not None or dist.get_rank() == 0:
                local[spec.name] = exported[spec.name].detach().to(
                    device="cpu"
                ).contiguous()

    local_meta = {
        name: (tuple(value.shape), str(value.dtype))
        for name, value in local.items()
    }
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_meta)
    tensors = {}
    for spec in specs:
        owners = [rank for rank, meta in enumerate(gathered) if spec.name in meta]
        if not owners:
            raise RuntimeError(f"no pipeline stage owns attention tensor {spec.name!r}")
        shapes = {gathered[rank][spec.name][0] for rank in owners}
        dtypes = {gathered[rank][spec.name][1] for rank in owners}
        if shapes != {tuple(spec.shape)} or len(dtypes) != 1:
            raise RuntimeError(
                f"pipeline-stage owners disagree on attention tensor {spec.name!r}"
            )
        source = min(owners)
        sample = next(iter(local.values()), None)
        device = (
            sample.device
            if sample is not None and sample.device.type != "cpu"
            else torch.device("cuda", torch.cuda.current_device())
        )
        dtype = getattr(torch, next(iter(dtypes)).removeprefix("torch."))
        if dist.get_rank() == source:
            value = local[spec.name].to(device=device, dtype=dtype).contiguous()
        else:
            value = torch.empty(spec.shape, device=device, dtype=dtype)
        dist.broadcast(value, src=source)
        if dist.get_rank() in owners and dist.get_rank() != source:
            replica = local[spec.name].to(device=device, dtype=dtype).contiguous()
            if not torch.equal(replica, value):
                raise RuntimeError(
                    f"pipeline-stage replicas disagree on attention tensor {spec.name!r}"
                )
            del replica
        if retain:
            tensors[spec.name] = value.to(
                device="cpu", dtype=torch.float32
            ).contiguous()
        del value
    return tensors


def _export_experts(
    actor,
    *,
    retain: bool,
    canonical_sources_only: bool = False,
) -> dict[str, Any]:
    import torch
    import torch.distributed as dist

    local = _expert_views(actor)
    local_meta = {
        name: (tuple(value.shape), str(value.dtype)) for name, value in local.items()
    }
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_meta)
    tensors = {}
    for spec in _expert_specs(actor):
        owners = [rank for rank, meta in enumerate(gathered) if spec.name in meta]
        if not owners:
            raise RuntimeError(f"no EP rank owns expert tensor {spec.name!r}")
        shapes = {gathered[rank][spec.name][0] for rank in owners}
        dtypes = {gathered[rank][spec.name][1] for rank in owners}
        if shapes != {tuple(spec.shape)} or len(dtypes) != 1:
            raise RuntimeError(f"EP owners disagree on expert tensor {spec.name!r}")
        source = min(owners)
        if dist.get_rank() == source:
            value = local[spec.name].detach().contiguous()
        else:
            sample = next(iter(local.values()), None)
            device = (
                sample.device
                if sample is not None
                else torch.device("cuda", torch.cuda.current_device())
            )
            dtype = getattr(torch, next(iter(dtypes)).removeprefix("torch."))
            value = torch.empty(spec.shape, device=device, dtype=dtype)
        dist.broadcast(value, src=source)
        if (
            dist.get_rank() in owners
            and dist.get_rank() != source
            and not torch.equal(local[spec.name], value)
        ):
            raise RuntimeError(
                f"DP replicas disagree on expert tensor {spec.name!r}"
            )
        if retain and (not canonical_sources_only or dist.get_rank() == source):
            tensors[spec.name] = value.to(
                device="cpu", dtype=torch.float32
            ).contiguous()
        del value
    return tensors


def export_hybrid_trainable_state(module: ModuleType, actor, *, policy_version: int):
    import torch.distributed as dist

    _assert_frozen_experts_unchanged(actor)
    rank = dist.get_rank()
    tensors = _export_attention(actor, retain=rank == 0)
    tensors.update(
        _export_experts(
            actor,
            retain=True,
            canonical_sources_only=True,
        )
    )
    if not tensors:
        return None
    is_metrics_source = bool(
        getattr(actor, "_is_first_replica_megatron_main_rank", False)
    )
    metrics = {}
    if is_metrics_source and getattr(actor.args, "external_policy_sync_path", None) is not None:
        metrics = getattr(actor.args, "_external_train_metrics", {})
    return _TrainableStateFragment(
        source_rank=rank,
        policy_version=policy_version,
        expected_names=tuple(spec.name for spec in _expected_specs(actor)),
        tensors=tensors,
        is_metrics_source=is_metrics_source,
        train_rollout_kl=metrics.get("train/train_rollout_kl"),
        ess_ratio=metrics.get("train/ess_ratio"),
        pg_clipfrac=metrics.get("train/pg_clipfrac"),
        train_seconds=(
            getattr(actor.args, "_external_train_seconds", None)
            if is_metrics_source
            else None
        ),
    )


def _broadcast_policy_tensor(spec, state):
    import torch
    import torch.distributed as dist

    if dist.get_rank() == 0:
        if state is None or spec.name not in state.tensors:
            raise RuntimeError(f"global hybrid policy is missing {spec.name!r}")
        value = state.tensors[spec.name].to(
            device=torch.device("cuda", torch.cuda.current_device()),
            dtype=torch.float32,
        ).contiguous()
    else:
        value = torch.empty(
            spec.shape,
            device=torch.device("cuda", torch.cuda.current_device()),
            dtype=torch.float32,
        )
    dist.broadcast(value, src=0)
    return value


def _optimizer_children(optimizer) -> list[Any]:
    return list(getattr(optimizer, "chained_optimizers", (optimizer,)))


def _copy_optimizer_masters_to_model(optimizer) -> None:
    for child in _optimizer_children(optimizer):
        copy = getattr(child, "_copy_main_params_to_model_params", None)
        if copy is None:
            raise RuntimeError("Megatron optimizer lacks main-to-model copy")
        copy()


def _apply_policy_specs(actor, state, specs) -> None:
    import torch

    sides = _attention_sides(actor)
    views = _expert_views(actor)
    with _attention_masters_as_model_parameters(sides):
        for spec in specs:
            value = _broadcast_policy_tensor(spec, state)
            if spec.name.endswith((".lora_A.weight", ".lora_B.weight")):
                side = sides[spec.name]
                if side.param_weight is not None:
                    mapped = side.mapping.hf_to_megatron(
                        value,
                        side.megatron_module,
                    )
                    if mapped is None or mapped.numel() != side.param_weight.numel():
                        raise RuntimeError(
                            f"attention LoRA shape mismatch for {spec.name!r}"
                        )
                    with torch.no_grad():
                        _optimizer_master_view(side.param_weight).copy_(
                            mapped.reshape(side.param_weight.shape)
                        )
                    del mapped
            elif selected_expert_hf_name(
                spec.name,
                expert_count=_expert_count(),
            ):
                target = views.get(spec.name)
                if target is not None:
                    if target.shape != value.shape:
                        raise RuntimeError(
                            f"expert shape mismatch for {spec.name!r}"
                        )
                    with torch.no_grad():
                        target.copy_(value)
            else:
                raise RuntimeError(
                    f"hybrid apply received unsupported tensor {spec.name!r}"
                )
            del value


def _finish_hybrid_apply(
    module: ModuleType,
    actor,
    *,
    policy_version: int,
    reset_optimizer: bool,
) -> int:
    import torch
    import torch.distributed as dist

    module._align_scheduler(actor, policy_version)
    with torch.no_grad():
        _copy_optimizer_masters_to_model(actor.optimizer)
    if dist.is_initialized():
        dist.barrier()
    if reset_optimizer:
        for child in _optimizer_children(actor.optimizer):
            inner = getattr(child, "optimizer", child)
            inner.state.clear()
    actor.weights_backuper.backup("actor")
    _assert_frozen_experts_unchanged(actor)
    if dist.is_initialized():
        dist.barrier()
    actor._external_policy_version = policy_version
    return len(_expected_specs(actor)) if reset_optimizer else 0


def _validate_chunk_manifest(module: ModuleType, actor, manifest) -> None:
    import torch

    if not isinstance(manifest, _ChunkedStateManifest):
        raise TypeError("chunked hybrid apply requires a state manifest")
    if manifest.policy_version < 0:
        raise ValueError("policy version must be non-negative")
    if not (
        manifest.chunk_tensor_names
        and len(manifest.chunk_tensor_names) == len(manifest.chunk_tensor_bytes)
    ):
        raise RuntimeError("chunked hybrid state manifest is inconsistent")
    expected_specs = tuple(sorted(_expected_specs(actor), key=lambda spec: spec.name))
    expected_names = tuple(spec.name for spec in expected_specs)
    if manifest.tensor_names != expected_names:
        raise RuntimeError("chunked hybrid state does not match actor tensor names")
    flattened = tuple(
        name for names in manifest.chunk_tensor_names for name in names
    )
    if flattened != expected_names or len(set(flattened)) != len(flattened):
        raise RuntimeError(
            "chunked hybrid state chunks do not follow canonical tensor order"
        )
    if any(
        size <= 0 or size > _RAY_STATE_CHUNK_BYTES
        for size in manifest.chunk_tensor_bytes
    ):
        raise RuntimeError("chunked hybrid state exceeds the transport cap")
    expected_layout = module._layout_hash(
        {
            spec.name: torch.empty(spec.shape, dtype=torch.float32, device="meta")
            for spec in expected_specs
        }
    )
    if manifest.layout_hash != expected_layout:
        raise RuntimeError("chunked hybrid state layout does not match the actor")


def begin_chunked_hybrid_apply(
    module: ModuleType,
    actor,
    manifest,
    *,
    reset_optimizer: bool,
) -> int:
    if getattr(actor, "_yeto_chunk_apply", None) is not None:
        raise RuntimeError("a chunked hybrid apply is already active")
    _validate_chunk_manifest(module, actor, manifest)
    actor._yeto_chunk_apply = _ChunkApplyContext(
        manifest=manifest,
        reset_optimizer=reset_optimizer,
    )
    return len(manifest.chunk_tensor_names)


def _validated_chunk_tensors(
    chunk,
    *,
    expected_names: tuple[str, ...],
    expected_shapes: Mapping[str, tuple[int, ...]],
    expected_bytes: int,
) -> dict[str, Any]:
    import torch

    if not isinstance(chunk, Mapping) or not chunk:
        raise RuntimeError("hybrid state chunk is empty or invalid")
    if tuple(chunk) != expected_names:
        raise RuntimeError("hybrid state chunk does not match canonical tensor order")
    canonical = {}
    for name, tensor in chunk.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"hybrid policy tensor {name!r} is not a tensor")
        if tuple(tensor.shape) != tuple(expected_shapes[name]):
            raise RuntimeError(f"hybrid policy tensor {name!r} has the wrong shape")
        value = tensor.detach() if tensor.requires_grad else tensor
        if (
            value.device.type != "cpu"
            or value.dtype != torch.float32
            or not value.is_contiguous()
        ):
            value = value.to(device="cpu", dtype=torch.float32).contiguous()
        if not torch.isfinite(value).all().item():
            raise ValueError(f"{name!r} contains NaN or Inf")
        canonical[name] = value
    actual_bytes = sum(
        _tensor_bytes(name, tensor) for name, tensor in canonical.items()
    )
    if actual_bytes != expected_bytes or actual_bytes > _RAY_STATE_CHUNK_BYTES:
        raise RuntimeError("hybrid state chunk violates its size manifest")
    return canonical


def apply_chunked_hybrid_state(
    actor,
    chunk_index: int,
    chunk,
) -> int:
    import torch.distributed as dist

    context = getattr(actor, "_yeto_chunk_apply", None)
    if context is None:
        raise RuntimeError("no chunked hybrid apply is active")
    if chunk_index != context.next_chunk:
        raise RuntimeError(
            "hybrid state chunk is duplicate or out of order: "
            f"expected {context.next_chunk}, got {chunk_index}"
        )
    manifest = context.manifest
    expected_names = manifest.chunk_tensor_names[chunk_index]
    expected_bytes = manifest.chunk_tensor_bytes[chunk_index]
    specs_by_name = {spec.name: spec for spec in _expected_specs(actor)}
    status = [None]
    tensors = None
    if dist.get_rank() == 0:
        try:
            tensors = _validated_chunk_tensors(
                chunk,
                expected_names=expected_names,
                expected_shapes={
                    name: tuple(specs_by_name[name].shape)
                    for name in expected_names
                },
                expected_bytes=expected_bytes,
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            status[0] = f"{type(exc).__name__}: {exc}"
    elif chunk is not None:
        raise RuntimeError("nonzero rank received a hybrid state chunk")
    dist.broadcast_object_list(status, src=0)
    if status[0] is not None:
        raise RuntimeError(f"rank-zero hybrid chunk validation failed: {status[0]}")
    specs = tuple(specs_by_name[name] for name in expected_names)
    state = SimpleNamespace(tensors=tensors) if dist.get_rank() == 0 else None
    _apply_policy_specs(actor, state, specs)
    context.next_chunk += 1
    return len(specs)


def finish_chunked_hybrid_apply(module: ModuleType, actor) -> int:
    context = getattr(actor, "_yeto_chunk_apply", None)
    if context is None:
        raise RuntimeError("no chunked hybrid apply is active")
    manifest = context.manifest
    if context.next_chunk != len(manifest.chunk_tensor_names):
        raise RuntimeError("chunked hybrid apply is missing state chunks")
    result = _finish_hybrid_apply(
        module,
        actor,
        policy_version=manifest.policy_version,
        reset_optimizer=context.reset_optimizer,
    )
    del actor._yeto_chunk_apply
    return result


def apply_hybrid_trainable_state(
    module: ModuleType,
    actor,
    state_or_header,
    *,
    reset_optimizer: bool,
) -> int:
    import torch
    import torch.distributed as dist

    header = [None]
    if dist.get_rank() == 0:
        state = state_or_header
        incoming = make_hybrid_trainable_state(
            module,
            state.policy_version,
            state.tensors,
        )
        if incoming.layout_hash != state.layout_hash:
            raise RuntimeError("hybrid trainable-state layout hash mismatch")
        header[0] = (state.policy_version, state.layout_hash)
    else:
        state = None
    dist.broadcast_object_list(header, src=0)
    policy_version, layout_hash = header[0]
    expected_layout = module._layout_hash(
        {
            spec.name: torch.empty(spec.shape, dtype=torch.float32, device="meta")
            for spec in _expected_specs(actor)
        }
    )
    if layout_hash != expected_layout:
        raise RuntimeError("global hybrid policy layout does not match the actor")

    _apply_policy_specs(actor, state, _expected_specs(actor))
    return _finish_hybrid_apply(
        module,
        actor,
        policy_version=policy_version,
        reset_optimizer=reset_optimizer,
    )


def install_on_trainable_state(module: ModuleType) -> None:
    if getattr(module, "_yeto_expert_full_installed", False):
        return

    def make_trainable_state(policy_version, tensors, **kwargs):
        return make_hybrid_trainable_state(
            module,
            policy_version,
            tensors,
            **kwargs,
        )

    def export_trainable_state(actor, *, policy_version):
        return export_hybrid_trainable_state(
            module,
            actor,
            policy_version=policy_version,
        )

    def apply_trainable_state(actor, state, *, reset_optimizer):
        return apply_hybrid_trainable_state(
            module,
            actor,
            state,
            reset_optimizer=reset_optimizer,
        )

    module.make_trainable_state = make_trainable_state
    module.export_trainable_state = export_trainable_state
    module.apply_trainable_state = apply_trainable_state
    module._yeto_expert_full_installed = True


def install_on_actor(module: ModuleType) -> None:
    if getattr(module, "_yeto_expert_full_installed", False):
        return
    original_switch_model = getattr(
        module.MegatronTrainRayActor,
        "_switch_model",
        None,
    )

    def _switch_model(self, target_tag):
        result = original_switch_model(self, target_tag)
        if target_tag == "actor":
            self._yeto_frozen_expert_versions = _frozen_expert_versions(self)
        return result

    def export_trainable_state(self):
        fragment = module.export_external_trainable_state(
            self,
            policy_version=getattr(self, "_external_policy_version", 0),
        )
        if fragment is None:
            return None
        return _chunk_export_fragment_for_ray(fragment)

    def export_trainable_state_chunks(self, tensor_groups):
        fragment = module.export_external_trainable_state(
            self,
            policy_version=getattr(self, "_external_policy_version", 0),
        )
        if fragment is None:
            return None
        return _chunk_export_fragment_for_ray(
            fragment,
            tensor_groups=tuple(tuple(group) for group in tensor_groups),
        )

    def apply_trainable_state(self, state, *, reset_optimizer):
        reset_count = module.apply_external_trainable_state(
            self,
            state,
            reset_optimizer=reset_optimizer,
        )
        return reset_count

    def begin_chunked_trainable_state(self, manifest, *, reset_optimizer):
        from miles.backends.megatron_utils import trainable_state as state_module

        return begin_chunked_hybrid_apply(
            state_module,
            self,
            manifest,
            reset_optimizer=reset_optimizer,
        )

    def apply_trainable_state_chunk(self, chunk_index, chunk):
        return apply_chunked_hybrid_state(self, chunk_index, chunk)

    def finish_chunked_trainable_state(self):
        from miles.backends.megatron_utils import trainable_state as state_module

        return finish_chunked_hybrid_apply(state_module, self)

    if original_switch_model is not None:
        module.MegatronTrainRayActor._switch_model = _switch_model
    module.MegatronTrainRayActor.export_trainable_state = export_trainable_state
    module.MegatronTrainRayActor.export_trainable_state_chunks = (
        export_trainable_state_chunks
    )
    module.MegatronTrainRayActor.apply_trainable_state = apply_trainable_state
    module.MegatronTrainRayActor.begin_chunked_trainable_state = (
        begin_chunked_trainable_state
    )
    module.MegatronTrainRayActor.apply_trainable_state_chunk = (
        apply_trainable_state_chunk
    )
    module.MegatronTrainRayActor.finish_chunked_trainable_state = (
        finish_chunked_trainable_state
    )
    module._yeto_expert_full_installed = True


def install_on_actor_group(module: ModuleType) -> None:
    if getattr(module, "_yeto_expert_full_installed", False):
        return

    async def export_trainable_state(self):
        from miles.backends.megatron_utils import trainable_state as state_module

        fragments = await self._broadcast("export_trainable_state")
        return _merge_chunked_export_fragments(state_module, fragments)

    async def export_trainable_state_chunks(self, tensor_groups):
        fragments = await self._broadcast(
            "export_trainable_state_chunks",
            tensor_groups,
        )
        return _prepare_chunked_policy_export(fragments)

    async def apply_trainable_state(self, state, *, reset_optimizer):
        import asyncio

        import ray

        manifest = _chunk_manifest(state)
        begin_results = await asyncio.gather(
            *(
                actor.begin_chunked_trainable_state.remote(
                    manifest,
                    reset_optimizer=reset_optimizer,
                )
                for actor in self._actor_handles
            )
        )
        if not begin_results or any(
            result != len(manifest.chunk_tensor_names) for result in begin_results
        ):
            raise RuntimeError("Megatron ranks disagree on hybrid chunk manifest")
        total_chunks = len(manifest.chunk_tensor_names)
        for chunk_index, chunk_names in enumerate(manifest.chunk_tensor_names):
            # Wire-decoded tensors share one model-sized flat storage.  Compact
            # only this bounded chunk before Ray serializes its backing storage.
            chunk = {name: state.tensors[name].clone() for name in chunk_names}
            chunk_ref = ray.put(chunk)
            try:
                chunk_results = await asyncio.gather(
                    *(
                        actor.apply_trainable_state_chunk.remote(
                            chunk_index,
                            chunk_ref if rank == 0 else None,
                        )
                        for rank, actor in enumerate(self._actor_handles)
                    )
                )
            finally:
                del chunk_ref, chunk
            expected_count = len(manifest.chunk_tensor_names[chunk_index])
            if any(result != expected_count for result in chunk_results):
                raise RuntimeError(
                    f"Megatron ranks disagree after hybrid chunk {chunk_index}"
                )
            completed_chunks = chunk_index + 1
            if _apply_progress_due(completed_chunks, total_chunks):
                _emit_apply_progress(
                    manifest.policy_version,
                    "chunk_progress",
                    completed_chunks=completed_chunks,
                    total_chunks=total_chunks,
                )
        results = await asyncio.gather(
            *(
                actor.finish_chunked_trainable_state.remote()
                for actor in self._actor_handles
            )
        )
        if not results or any(result != results[0] for result in results[1:]):
            raise RuntimeError("Megatron ranks disagree after applying hybrid state")
        _emit_apply_progress(
            manifest.policy_version,
            "chunks_finished",
            total_chunks=total_chunks,
        )
        return results[0]

    module.RayTrainGroup.export_trainable_state = export_trainable_state
    module.RayTrainGroup.export_trainable_state_chunks = (
        export_trainable_state_chunks
    )
    module.RayTrainGroup.apply_trainable_state = apply_trainable_state
    module._yeto_expert_full_installed = True


def _weight_iterator_task_bridge(iterator):
    cached = getattr(iterator, "_yeto_expert_full_task_bridge", None)
    if cached is not None:
        return cached
    source = getattr(iterator.args, "ref_load", None) or getattr(
        iterator.args, "hf_checkpoint", None
    )
    if not source:
        raise RuntimeError(
            "expert-full weight iterator has no HuggingFace checkpoint for "
            "Bridge tasks"
        )
    from megatron.bridge import AutoBridge

    cached = AutoBridge.from_hf_pretrained(
        source,
        trust_remote_code=bool(iterator.args.yeto_rl_trust_remote_code),
    )
    iterator._yeto_expert_full_task_bridge = cached
    return cached


def install_on_weight_iterator(module: ModuleType) -> None:
    if getattr(module, "_yeto_expert_full_installed", False):
        return
    original = module.HfWeightIteratorBridge.get_hf_weight_chunks

    def get_hf_weight_chunks(self, megatron_local_weights, weight_type="base"):
        if weight_type != "base":
            yield from original(self, megatron_local_weights, weight_type=weight_type)
            return
        renamed = {
            module.strip_param_name_prefix(name): value
            for name, value in megatron_local_weights.items()
        }
        with module.megatron_bridge_utils.patch_megatron_model(self.model):
            expert_count = _expert_count()
            task_bridge = _weight_iterator_task_bridge(self)
            tasks = filter_collective_expert_tasks(
                task_bridge.get_conversion_tasks(self.model),
                expert_count=expert_count,
            )
            expected = NUM_LAYERS * min(expert_count, CLONES_PER_EXPERT_RANK) * 2
            if len(tasks) != expected:
                raise RuntimeError(
                    f"expert-full base sync found {len(tasks)} conversion tasks, "
                    f"expected {expected}"
                )
            tasks = module._process_conversion_tasks(tasks, renamed)
            named_weights = task_bridge.export_hf_weights(
                self.model,
                cpu=False,
                conversion_tasks=tasks,
                merge_adapter_weights=False,
            )
            named_weights = (
                item
                for item in named_weights
                if selected_expert_hf_name(item[0], expert_count=expert_count)
            )
            named_weights = self._postprocess_and_quantize(named_weights, weight_type)
            named_weights = (
                (hf_name, weight, megatron_name)
                for hf_name, weight, megatron_name in named_weights
                if not module.is_lora_weight_name(hf_name)
            )
            groups = module.get_atomic_update_groups(self.args, self.model_name)
            units = module._stream_atomic_units(named_weights, groups)
            yield from module._chunk_atomic_units_by_size(
                units,
                chunk_size=self.args.update_weight_buffer_size,
            )

    module.HfWeightIteratorBridge.get_hf_weight_chunks = get_hf_weight_chunks
    module._yeto_expert_full_installed = True


def install_on_update_weight(module: ModuleType) -> None:
    if getattr(module, "_yeto_expert_full_installed", False):
        return
    module.lora_base_cpu_backup_enabled = lambda _args: False
    module._yeto_expert_full_installed = True


def install() -> None:
    """Patch imported Miles modules or arm lazy process-wide import hooks."""

    targets = (
        ("miles.backends.megatron_utils.lora_utils", install_on_lora_utils),
        ("miles.backends.megatron_utils.arguments", install_on_arguments),
        ("miles.backends.megatron_utils.model", install_on_model),
        ("miles.backends.megatron_utils.trainable_state", install_on_trainable_state),
        ("miles.backends.megatron_utils.actor", install_on_actor),
        ("miles.ray.actor_group", install_on_actor_group),
        (
            "miles.backends.megatron_utils.update_weight.hf_weight_iterator_bridge",
            install_on_weight_iterator,
        ),
        (
            "miles.backends.megatron_utils.update_weight.update_weight_from_tensor",
            install_on_update_weight,
        ),
    )
    for fullname, installer in targets:
        _install_or_defer(fullname, installer)


class _Loader(importlib.abc.Loader):
    def __init__(self, wrapped, installer) -> None:
        self.wrapped = wrapped
        self.installer = installer

    def create_module(self, spec):
        create = getattr(self.wrapped, "create_module", None)
        return None if create is None else create(spec)

    def exec_module(self, module) -> None:
        self.wrapped.exec_module(module)
        self.installer(module)


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self, fullname: str, installer) -> None:
        self.fullname = fullname
        self.installer = installer

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.fullname:
            return None
        try:
            sys.meta_path.remove(self)
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        spec.loader = _Loader(spec.loader, self.installer)
        return spec


_FINDERS: list[_Finder] = []


def _install_or_defer(fullname: str, installer) -> None:
    loaded = sys.modules.get(fullname)
    if loaded is not None:
        installer(loaded)
        return
    finder = _Finder(fullname, installer)
    _FINDERS.append(finder)
    sys.meta_path.insert(0, finder)
