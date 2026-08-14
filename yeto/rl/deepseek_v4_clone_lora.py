"""Clone-only routed-expert LoRA for expanded DeepSeek V4 checkpoints.

Megatron's grouped-expert adapter stores one packed parameter per side with
shape ``[local_experts, ...]``.  The expanded checkpoint deliberately keeps
experts 0..255 frozen and permits only cloned experts 256..287 to learn.  This
module zeros the original-expert adapter slices and masks their gradients while
leaving attention adapters untouched.

The packed parameters remain present on every EP rank.  That preserves the
standard Megatron-Bridge/SGLang per-expert LoRA conversion contract; inactive
expert tensors export as exact zeros instead of requiring a sparse, custom
checkpoint format.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import re
import sys
from dataclasses import dataclass
from typing import Any

from .deepseek_v4_expert_clone import (
    EXPERT_PARALLEL_SIZE,
    ORIGINAL_EXPERTS,
    ORIGINAL_EXPERTS_PER_RANK,
    TOTAL_EXPERTS,
    TRAINING_EXPERTS_PER_RANK,
    training_to_logical_expert_id,
)


_ROUTED_ADAPTER = re.compile(
    r"(?:^|\.)decoder\.layers\.(?P<layer>\d+)\.mlp\.experts\."
    r"(?P<branch>linear_fc[12])$"
)
_MILES_TRAINABLE_STATE = "miles.backends.megatron_utils.trainable_state"
_MILES_ACTOR = "miles.backends.megatron_utils.actor"
_MILES_ACTOR_GROUP = "miles.ray.actor_group"
_MILES_TRAINABLE_STATE_FINDER = None


@dataclass(frozen=True)
class CloneLoraRecord:
    module_name: str
    base_linear_name: str
    layer: int
    branch: str
    expert_parallel_rank: int
    local_expert_ids: tuple[int, ...]
    trainable_clone_ids: tuple[int, ...]


def _model_chunks(models: Any) -> tuple[Any, ...]:
    if isinstance(models, (list, tuple)):
        return tuple(models)
    return (models,)


def _parallel_coordinates(
    *,
    expert_parallel_rank: int | None,
    expert_parallel_size: int | None,
) -> tuple[int, int]:
    if expert_parallel_rank is None or expert_parallel_size is None:
        from megatron.core import parallel_state

        if expert_parallel_rank is None:
            expert_parallel_rank = int(
                parallel_state.get_expert_model_parallel_rank()
            )
        if expert_parallel_size is None:
            expert_parallel_size = int(
                parallel_state.get_expert_model_parallel_world_size()
            )
    rank = int(expert_parallel_rank)
    size = int(expert_parallel_size)
    if size <= 0 or not 0 <= rank < size:
        raise ValueError(
            f"invalid expert-parallel coordinates rank={rank}, size={size}"
        )
    if size != EXPERT_PARALLEL_SIZE:
        raise ValueError(
            f"balanced E288 training requires EP size {EXPERT_PARALLEL_SIZE}, got {size}"
        )
    return rank, size


def _mask_gradient(mask):
    def hook(gradient):
        return gradient * mask.to(dtype=gradient.dtype)

    return hook


def configure_clone_only_grouped_lora(
    models: Any,
    *,
    expert_parallel_rank: int | None = None,
    expert_parallel_size: int | None = None,
    adapter_type: type | tuple[type, ...] | None = None,
) -> tuple[CloneLoraRecord, ...]:
    """Freeze packed LoRA slices for original experts and keep clones live.

    ``adapter_type`` and explicit EP coordinates are dependency-injection hooks
    used by CPU unit tests.  Production callers leave them unset.
    """

    import torch

    if adapter_type is None:
        from megatron.bridge.peft.utils import GroupedExpertLinearAdapter

        adapter_type = GroupedExpertLinearAdapter
    ep_rank, ep_size = _parallel_coordinates(
        expert_parallel_rank=expert_parallel_rank,
        expert_parallel_size=expert_parallel_size,
    )
    local_count = TOTAL_EXPERTS // ep_size
    local_start = ep_rank * local_count
    local_ids = tuple(
        training_to_logical_expert_id(expert)
        for expert in range(local_start, local_start + local_count)
    )
    active = torch.tensor(
        [expert >= ORIGINAL_EXPERTS for expert in local_ids],
        dtype=torch.bool,
    )

    records: list[CloneLoraRecord] = []
    seen_modules: set[int] = set()
    for chunk in _model_chunks(models):
        for module_name, module in chunk.named_modules():
            if not isinstance(module, adapter_type) or id(module) in seen_modules:
                continue
            seen_modules.add(id(module))
            base_name = str(getattr(module, "base_linear_name", ""))
            match = _ROUTED_ADAPTER.search(base_name)
            if match is None:
                continue
            if getattr(module, "_yeto_clone_only_lora_installed", False):
                raise RuntimeError(
                    f"clone-only LoRA was installed twice on {base_name!r}"
                )
            if int(getattr(module, "num_local_experts", -1)) != local_count:
                raise RuntimeError(
                    f"{base_name} has {getattr(module, 'num_local_experts', None)} "
                    f"local experts, expected {local_count} for E288/EP{ep_size}"
                )
            weights = (
                getattr(getattr(module, "linear_in", None), "weight", None),
                getattr(getattr(module, "linear_out", None), "weight", None),
            )
            if any(weight is None for weight in weights):
                raise RuntimeError(f"{base_name} has no packed LoRA A/B weights")
            mask = active.to(device=weights[0].device).reshape(local_count, 1, 1)
            for weight in weights:
                if weight.ndim != 3 or weight.shape[0] != local_count:
                    raise RuntimeError(
                        f"{base_name} has an unsupported packed LoRA shape "
                        f"{tuple(weight.shape)}"
                    )
                if weight.device != mask.device:
                    raise RuntimeError(f"{base_name} LoRA sides are on different devices")
                with torch.no_grad():
                    weight.mul_(mask.to(dtype=weight.dtype))
                weight.register_hook(_mask_gradient(mask))

            module.register_buffer(
                "_yeto_clone_active_mask",
                active.to(device=weights[0].device),
                persistent=False,
            )
            module._yeto_clone_only_lora_installed = True
            trainable_ids = tuple(
                expert for expert in local_ids if expert >= ORIGINAL_EXPERTS
            )
            records.append(
                CloneLoraRecord(
                    module_name=module_name,
                    base_linear_name=base_name,
                    layer=int(match.group("layer")),
                    branch=match.group("branch"),
                    expert_parallel_rank=ep_rank,
                    local_expert_ids=local_ids,
                    trainable_clone_ids=trainable_ids,
                )
            )

    if not records:
        raise RuntimeError(
            "clone-only LoRA requested but no routed grouped-expert adapters were found"
        )
    branches: dict[tuple[str, int], set[str]] = {}
    for record in records:
        # Module names are chunk-local under PP; pair branches within each
        # decoder prefix rather than assuming layer indices are globally unique.
        prefix = record.base_linear_name.rsplit(".", 1)[0]
        branches.setdefault((prefix, record.layer), set()).add(record.branch)
    incomplete = {
        key: value
        for key, value in branches.items()
        if value != {"linear_fc1", "linear_fc2"}
    }
    if incomplete:
        raise RuntimeError(f"clone-only LoRA has incomplete expert branches: {incomplete}")
    return tuple(sorted(records, key=lambda item: item.base_linear_name))


def _require_miles_packed_layout(
    parameter,
    *,
    expert_parallel_rank: int,
    expert_parallel_size: int,
) -> int:
    rank, size = int(expert_parallel_rank), int(expert_parallel_size)
    if size != EXPERT_PARALLEL_SIZE or not 0 <= rank < size:
        raise RuntimeError(
            "balanced clone-only policy requires "
            f"EP{EXPERT_PARALLEL_SIZE}, got rank={rank}, size={size}"
        )
    local_count = int(parameter.shape[0])
    if local_count != TRAINING_EXPERTS_PER_RANK:
        raise RuntimeError(
            "balanced clone-only policy requires "
            f"{TRAINING_EXPERTS_PER_RANK} packed experts per rank, got {local_count}"
        )
    return rank * local_count


def _sparse_expert_updates(
    module,
    name: str,
    side: Any,
    tensors,
    *,
    expert_parallel_rank: int,
    expert_parallel_size: int,
):
    """Prepare each EP8 rank's four logical clone writes for pinned Miles."""

    import torch

    match = module._EXPERT_LORA.fullmatch(name)
    if match is None:
        raise RuntimeError(f"cannot pack non-expert LoRA side {name!r}")
    parameter = side.param_weight
    if parameter is None or parameter.ndim != 3:
        raise RuntimeError(f"packed expert side {name!r} is not a rank-3 parameter")
    start = _require_miles_packed_layout(
        parameter,
        expert_parallel_rank=expert_parallel_rank,
        expert_parallel_size=expert_parallel_size,
    )
    projection = match.group("projection")
    adapter_side = match.group("side")
    if projection == "up_proj":
        raise RuntimeError("packed fc1 representative unexpectedly uses up_proj")

    updates = []
    for local_index in range(
        ORIGINAL_EXPERTS_PER_RANK,
        TRAINING_EXPERTS_PER_RANK,
    ):
        expert = training_to_logical_expert_id(start + local_index)
        if projection == "down_proj":
            value = tensors[
                module._expert_name(match, expert, "down_proj", adapter_side)
            ]
        elif adapter_side == "A":
            gate = tensors[module._expert_name(match, expert, "gate_proj", "A")]
            up = tensors[module._expert_name(match, expert, "up_proj", "A")]
            if not torch.equal(gate, up):
                raise RuntimeError(
                    "fused fc1 gate/up LoRA A tensors differ for layer "
                    f"{match.group('layer')} expert {expert}"
                )
            value = gate
        else:
            gate = tensors[module._expert_name(match, expert, "gate_proj", "B")]
            up = tensors[module._expert_name(match, expert, "up_proj", "B")]
            value = torch.cat((gate, up), dim=0)
        if tuple(value.shape) != tuple(parameter.shape[1:]):
            raise RuntimeError(
                f"packed expert LoRA slice mismatch for {name!r}: "
                f"got {tuple(value.shape)}, expected {tuple(parameter.shape[1:])}"
            )
        updates.append((local_index, value))

    return (
        parameter.main_param.view(parameter.shape),
        ORIGINAL_EXPERTS_PER_RANK,
        tuple(updates),
    )


def _assert_original_packed_masters_zero(
    module,
    sides,
    *,
    expert_parallel_rank: int,
) -> None:
    import torch

    found = 0
    for name, side in sides:
        match = module._EXPERT_LORA.fullmatch(name)
        parameter = side.param_weight
        if match is None or parameter is None:
            continue
        found += 1
        _require_miles_packed_layout(
            parameter,
            expert_parallel_rank=expert_parallel_rank,
            expert_parallel_size=EXPERT_PARALLEL_SIZE,
        )
        master = parameter.main_param.view(parameter.shape)
        if torch.count_nonzero(master[:ORIGINAL_EXPERTS_PER_RANK]).item():
            raise RuntimeError(
                f"original packed expert LoRA master is nonzero for {name!r}"
            )
    if not found:
        raise RuntimeError("clone-only actor has no packed expert LoRA sides")


def install_on_trainable_state(module) -> None:
    """Patch the two pinned Miles helpers that encode EP expert ownership."""

    if getattr(module, "_yeto_balanced_expert_layout_installed", False):
        return

    def sparse_expert_updates(name, side, tensors, **coordinates):
        return _sparse_expert_updates(
            module,
            name,
            side,
            tensors,
            **coordinates,
        )

    def assert_original_packed_masters_zero(sides, **coordinates):
        return _assert_original_packed_masters_zero(
            module,
            sides,
            **coordinates,
        )

    module._sparse_expert_updates = sparse_expert_updates
    module._assert_original_packed_masters_zero = (
        assert_original_packed_masters_zero
    )
    module._yeto_balanced_expert_layout_installed = True


def install_on_streaming_actor(module) -> None:
    """Export clone-only policy tensors into bounded Ray objects."""

    if getattr(module, "_yeto_clone_stream_export_installed", False):
        return

    def export_trainable_state_chunks(self, tensor_groups):
        import torch.distributed as dist

        from .deepseek_v4_expert_full_runtime import (
            _TrainableStateFragment,
            _chunk_export_fragment_for_ray,
        )

        groups = tuple(tuple(group) for group in tensor_groups)
        expected_names = tuple(name for group in groups for name in group)
        state = module.export_external_trainable_state(
            self,
            policy_version=getattr(self, "_external_policy_version", 0),
        )
        if state is None:
            return None
        if state.policy_version != getattr(self, "_external_policy_version", 0):
            raise RuntimeError("clone-only export changed policy version")
        tensors = state.tensors
        if not isinstance(tensors, dict):
            tensors = dict(tensors)
        fragment = _TrainableStateFragment(
            source_rank=dist.get_rank() if dist.is_initialized() else 0,
            policy_version=state.policy_version,
            expected_names=expected_names,
            tensors=tensors,
            is_metrics_source=True,
            train_rollout_kl=getattr(state, "train_rollout_kl", None),
            ess_ratio=getattr(state, "ess_ratio", None),
            pg_clipfrac=getattr(state, "pg_clipfrac", None),
            train_seconds=getattr(state, "train_seconds", None),
        )
        return _chunk_export_fragment_for_ray(
            fragment,
            tensor_groups=groups,
        )

    module.MegatronTrainRayActor.export_trainable_state_chunks = (
        export_trainable_state_chunks
    )
    module._yeto_clone_stream_export_installed = True


def install_on_streaming_actor_group(module) -> None:
    """Expose the bounded clone-only export through Miles' v1 train group."""

    if getattr(module, "_yeto_clone_stream_export_installed", False):
        return

    async def export_trainable_state_chunks(self, tensor_groups):
        from .deepseek_v4_expert_full_runtime import (
            _prepare_chunked_policy_export,
        )

        fragments = await self._broadcast(
            "export_trainable_state_chunks",
            tensor_groups,
        )
        return _prepare_chunked_policy_export(fragments)

    module.RayTrainGroup.export_trainable_state_chunks = (
        export_trainable_state_chunks
    )
    module._yeto_clone_stream_export_installed = True


class _MilesTrainableStateLoader(importlib.abc.Loader):
    def __init__(self, wrapped) -> None:
        self.wrapped = wrapped

    def create_module(self, spec):
        create = getattr(self.wrapped, "create_module", None)
        return None if create is None else create(spec)

    def exec_module(self, module) -> None:
        self.wrapped.exec_module(module)
        install_on_trainable_state(module)


class _MilesTrainableStateFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _MILES_TRAINABLE_STATE:
            return None
        try:
            sys.meta_path.remove(self)
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        spec.loader = _MilesTrainableStateLoader(spec.loader)
        return spec


def install() -> None:
    """Patch clone-only Miles state/export paths or arm lazy import hooks."""

    global _MILES_TRAINABLE_STATE_FINDER
    loaded = sys.modules.get(_MILES_TRAINABLE_STATE)
    if loaded is not None:
        install_on_trainable_state(loaded)
    elif _MILES_TRAINABLE_STATE_FINDER is None:
        _MILES_TRAINABLE_STATE_FINDER = _MilesTrainableStateFinder()
        sys.meta_path.insert(0, _MILES_TRAINABLE_STATE_FINDER)

    from .deepseek_v4_expert_full_runtime import _install_or_defer

    _install_or_defer(_MILES_ACTOR, install_on_streaming_actor)
    _install_or_defer(_MILES_ACTOR_GROUP, install_on_streaming_actor_group)


def assert_original_expert_lora_zero(models: Any) -> None:
    """Fail if any masked original-expert adapter slice has drifted from zero."""

    import torch

    found = 0
    for chunk in _model_chunks(models):
        for module in chunk.modules():
            mask = getattr(module, "_yeto_clone_active_mask", None)
            if mask is None:
                continue
            found += 1
            inactive = ~mask.bool()
            for side in (module.linear_in.weight, module.linear_out.weight):
                if inactive.any() and torch.count_nonzero(side[inactive]).item() != 0:
                    raise RuntimeError(
                        f"original-expert LoRA drifted on {module.base_linear_name}"
                    )
    if not found:
        raise RuntimeError("no clone-only expert LoRA masks are installed")


class _CloneOnlyLoraProxy:
    def __init__(self, inner: Any, configure_kwargs: dict[str, Any] | None = None) -> None:
        self._inner = inner
        self._configure_kwargs = dict(configure_kwargs or {})
        self.clone_records: tuple[CloneLoraRecord, ...] = ()

    def __call__(self, models: Any, *args: Any, **kwargs: Any) -> Any:
        transformed = self._inner(models, *args, **kwargs)
        self.clone_records = configure_clone_only_grouped_lora(
            transformed,
            **self._configure_kwargs,
        )
        return transformed

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def wrap_clone_only_lora(
    lora: Any,
    *,
    configure_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Wrap a Megatron LoRA transform with the E288 clone-slice contract."""

    if isinstance(lora, _CloneOnlyLoraProxy):
        raise RuntimeError("clone-only LoRA wrapper was installed twice")
    return _CloneOnlyLoraProxy(lora, configure_kwargs)
