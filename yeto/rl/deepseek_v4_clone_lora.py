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

import re
from dataclasses import dataclass
from typing import Any

from .deepseek_v4_expert_clone import ORIGINAL_EXPERTS, TOTAL_EXPERTS


_ROUTED_ADAPTER = re.compile(
    r"(?:^|\.)decoder\.layers\.(?P<layer>\d+)\.mlp\.experts\."
    r"(?P<branch>linear_fc[12])$"
)


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
    if TOTAL_EXPERTS % size:
        raise ValueError(
            f"{TOTAL_EXPERTS} routed experts are not divisible by EP size {size}"
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
    local_ids = tuple(range(local_start, local_start + local_count))
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
