"""Clone-expert full tuning with attention-only LoRA for DeepSeek V4."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .deepseek_v4_expert_clone import (
    CLONES_PER_LAYER,
    NUM_LAYERS,
    ORIGINAL_EXPERTS,
    TOTAL_EXPERTS,
    contract_from_config,
)


EXPERT_PARAMETER_COUNT = NUM_LAYERS * 3 * 4096 * 2048

_INDIVIDUAL_EXPERT = re.compile(
    r"(?:^|\.)decoder\.layers\.(?P<layer>\d+)\.mlp\.experts\."
    r"(?P<branch>linear_fc[12])\.weight(?P<local_expert>\d+)$"
)


@dataclass(frozen=True)
class ExpertFullRecord:
    parameter_name: str
    layer: int
    branch: str
    expert_parallel_rank: int
    local_expert_ids: tuple[int, ...]
    trainable_clone_ids: tuple[int, ...]


def expert_full_specs(
    config: Any,
    *,
    expert_count: int,
    expected_selection_sha256: str,
    expected_selection_contract_sha256: str,
):
    """Build a prefix of the clone policy from an attested E288 config."""

    from .core import CanonicalTensorSpec

    if not 1 <= expert_count <= CLONES_PER_LAYER:
        raise ValueError(
            f"expert_count must be between 1 and {CLONES_PER_LAYER}"
        )
    contract = contract_from_config(config)
    if contract is None:
        raise ValueError("expert-full policy requires an expanded E288 clone contract")
    if contract.selection_sha256 != expected_selection_sha256:
        raise ValueError(
            "expert selection SHA256 mismatch: "
            f"expected {expected_selection_sha256}, got {contract.selection_sha256}"
        )
    if contract.selection_contract_sha256 != expected_selection_contract_sha256:
        raise ValueError(
            "expert selection contract SHA256 mismatch: "
            f"expected {expected_selection_contract_sha256}, "
            f"got {contract.selection_contract_sha256}"
        )
    model_type = str(getattr(config, "model_type", ""))
    architectures = tuple(getattr(config, "architectures", None) or ())
    if model_type != "deepseek_v4" or "DeepseekV4ForCausalLM" not in architectures:
        raise ValueError("expert-full policy requires the pinned DeepSeek V4 architecture")
    hidden_size = int(getattr(config, "hidden_size", 0) or 0)
    expert_size = int(getattr(config, "moe_intermediate_size", 0) or 0)
    if (hidden_size, expert_size) != (4096, 2048):
        raise ValueError(
            "expert-full policy requires hidden_size=4096 and "
            "moe_intermediate_size=2048"
        )

    shapes = {
        "gate_proj": (expert_size, hidden_size),
        "up_proj": (expert_size, hidden_size),
        "down_proj": (hidden_size, expert_size),
    }
    specs = []
    for layer in range(NUM_LAYERS):
        for expert in range(ORIGINAL_EXPERTS, ORIGINAL_EXPERTS + expert_count):
            for projection, shape in shapes.items():
                specs.append(
                    CanonicalTensorSpec(
                        "base_model.model.model.layers."
                        f"{layer}.mlp.experts.{expert}.{projection}.weight",
                        shape,
                        "float32",
                        shape[0] * shape[1],
                    )
                )
    result = tuple(sorted(specs))
    if (
        len(result) != NUM_LAYERS * expert_count * 3
        or sum(spec.numel for spec in result)
        != EXPERT_PARAMETER_COUNT * expert_count
    ):
        raise RuntimeError("expert-full canonical policy geometry changed")
    return result


def _model_chunks(models: Any) -> tuple[Any, ...]:
    return tuple(models) if isinstance(models, (list, tuple)) else (models,)


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
    rank, size = int(expert_parallel_rank), int(expert_parallel_size)
    if size <= 0 or not 0 <= rank < size or TOTAL_EXPERTS % size:
        raise ValueError(f"invalid expert-parallel coordinates rank={rank}, size={size}")
    return rank, size


def configure_clone_expert_full(
    models: Any,
    *,
    expert_count: int,
    expert_parallel_rank: int | None = None,
    expert_parallel_size: int | None = None,
) -> tuple[ExpertFullRecord, ...]:
    """Enable only the requested clone experts on their owning EP rank."""

    if not 1 <= expert_count <= CLONES_PER_LAYER:
        raise ValueError(
            f"expert_count must be between 1 and {CLONES_PER_LAYER}"
        )

    rank, size = _parallel_coordinates(
        expert_parallel_rank=expert_parallel_rank,
        expert_parallel_size=expert_parallel_size,
    )
    local_count = TOTAL_EXPERTS // size
    local_start = rank * local_count
    local_ids = tuple(range(local_start, local_start + local_count))
    selected_end = ORIGINAL_EXPERTS + expert_count
    trainable_ids = tuple(
        expert
        for expert in local_ids
        if ORIGINAL_EXPERTS <= expert < selected_end
    )
    records = []
    for chunk in _model_chunks(models):
        for name, parameter in chunk.named_parameters():
            match = _INDIVIDUAL_EXPERT.search(name)
            if match is None:
                continue
            local_expert = int(match.group("local_expert"))
            if not 0 <= local_expert < local_count or parameter.ndim != 2:
                raise RuntimeError(
                    f"{name} is not one of {local_count} individual expert weights"
                )
            parameter_ids = (local_start + local_expert,)
            parameter_trainable_ids = tuple(
                expert for expert in parameter_ids if expert in trainable_ids
            )
            if getattr(parameter, "_yeto_expert_full_configured", False):
                raise RuntimeError(f"expert-full parameter was configured twice: {name}")
            parameter._yeto_expert_full_configured = True
            parameter._yeto_expert_layer = int(match.group("layer"))
            parameter._yeto_expert_branch = match.group("branch")
            parameter.requires_grad_(bool(parameter_trainable_ids))
            if parameter_trainable_ids:
                parameter._yeto_expert_full = True
                parameter._yeto_expert_id = parameter_trainable_ids[0]
            records.append(
                ExpertFullRecord(
                    name,
                    int(match.group("layer")),
                    match.group("branch"),
                    rank,
                    parameter_ids,
                    parameter_trainable_ids,
                )
            )

    local_layers = {record.layer for record in records}
    if not local_layers:
        raise RuntimeError("expert-full policy found no individual expert weights")
    expected = len(local_layers) * 2 * local_count
    if len(records) != expected:
        raise RuntimeError(
            "expert-full policy found "
            f"{len(records)} individual expert weights, expected {expected}"
        )
    keys = {
        (record.layer, record.branch, record.local_expert_ids[0])
        for record in records
    }
    required = {
        (layer, branch, expert)
        for layer in local_layers
        for branch in ("linear_fc1", "linear_fc2")
        for expert in local_ids
    }
    if keys != required:
        raise RuntimeError("expert-full policy has incomplete individual expert branches")
    return tuple(sorted(records, key=lambda record: record.parameter_name))


class _AttentionLoraExpertFullProxy:
    def __init__(self, inner: Any, configure_kwargs: dict[str, Any] | None) -> None:
        self._inner = inner
        self._configure_kwargs = dict(configure_kwargs or {})
        self.expert_records: tuple[ExpertFullRecord, ...] = ()

    def __call__(self, models: Any, *args: Any, **kwargs: Any) -> Any:
        return self._inner(models, *args, **kwargs)

    def set_params_to_save(self, models: Any) -> None:
        self._inner.set_params_to_save(models)
        self.expert_records = configure_clone_expert_full(
            models,
            **self._configure_kwargs,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def wrap_attention_lora_with_expert_full(
    lora: Any,
    *,
    configure_kwargs: dict[str, Any] | None = None,
) -> Any:
    if isinstance(lora, _AttentionLoraExpertFullProxy):
        raise RuntimeError("expert-full LoRA wrapper was installed twice")
    return _AttentionLoraExpertFullProxy(lora, configure_kwargs)
