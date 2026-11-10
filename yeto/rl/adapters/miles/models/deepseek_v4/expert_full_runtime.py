"""Runtime helpers for attention LoRA plus clone-expert full tuning on ports Miles.

:mod:`.ports_runtime` installs these through ports Miles module hooks. The
model-independent pieces (import hooks, optimizer-master validation, export
task filtering) come from :mod:`..partial_experts`; this module keeps the
DeepSeek V4 clone geometry and its environment-driven settings.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from types import ModuleType
from typing import Any

from yeto.rl.adapters.miles.ext_args import ext_attr

from yeto.rl.adapters.miles.models.deepseek_v4.expert_clone import (
    CLONES_PER_EXPERT_RANK,
    CLONES_PER_LAYER,
    NUM_LAYERS,
    ORIGINAL_EXPERTS,
    TRAINING_EXPERTS_PER_RANK,
    logical_to_training_expert_id,
)
from yeto.rl.adapters.miles.models.deepseek_v4.expert_full import TRAINABLE_PARAM_ATTR
from yeto.rl.adapters.miles.models.partial_experts import export as _export
from yeto.rl.adapters.miles.models.partial_experts import optimizer as _optimizer
from yeto.rl.adapters.miles.models.partial_experts.config import (
    CLONE_PREFIX,
    PartialExpertModelConfig,
    register_model,
)
from yeto.rl.adapters.miles.models.partial_experts.hooks import (  # noqa: F401  (re-exported)
    _FINDERS,
    _Finder,
    _Loader,
    install_or_defer,
)

LABEL = "expert-full"

# The DeepSeek V4 Flash contract: a prefix of the 32 clones per layer trains;
# Adam keeps replicated FP32 masters (streamed by the full-parameter
# transport), so the distributed optimizer stays refused.
MODEL_CONFIG = register_model(PartialExpertModelConfig(
    model_type="deepseek_v4",
    num_routed_experts=ORIGINAL_EXPERTS,
    num_moe_layers=NUM_LAYERS,
    trainable_strategy=CLONE_PREFIX,
    ep_size_default=8,
    allow_distributed_optimizer=False,
    requires_complete_master=True,
    trainable_param_attr=TRAINABLE_PARAM_ATTR,
    needs_engine_patch=True,
    num_trainable_per_layer=None,
    trainable_param_fragment=".mlp.experts.",
    routed_experts_dtype="uint16",
    notes=("clone experts 256..287 per layer; count from YETO_DSV4_EXPERT_FULL_COUNT",),
))

_EXPERT_WEIGHT = re.compile(
    r"^(?:base_model\.model\.)?model\.layers\.(?P<layer>\d+)\.mlp\."
    r"experts\.(?P<expert>\d+)\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\.weight$"
)


def configured_expert_count() -> int:
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


_mapping_hf_names = _export.mapping_hf_names


def filter_collective_expert_tasks(tasks, *, expert_count: int) -> list[Any]:
    """Select identical local expert offsets on every EP rank."""

    selected_offsets = {
        logical_to_training_expert_id(expert) % TRAINING_EXPERTS_PER_RANK
        for expert in range(ORIGINAL_EXPERTS, ORIGINAL_EXPERTS + expert_count)
    }

    def accept(name: str) -> bool:
        match = _EXPERT_WEIGHT.fullmatch(name)
        return (
            match is not None
            and int(match.group("expert")) % TRAINING_EXPERTS_PER_RANK in selected_offsets
        )

    return _export.filter_tasks(tasks, accept)


def _expert_lr() -> float:
    try:
        value = float(os.environ.get("YETO_DSV4_EXPERT_FULL_LR", ""))
    except ValueError as exc:
        raise RuntimeError("YETO_DSV4_EXPERT_FULL_LR must be a float") from exc
    if value <= 0:
        raise RuntimeError("YETO_DSV4_EXPERT_FULL_LR must be positive")
    return value


def wrap_expert_full_optimizer(original):
    """``get_megatron_optimizer`` with the expert LR group and complete FP32 masters.

    Miles calls Megatron's factory with keyword arguments from its model module.
    """

    return _optimizer.wrap_megatron_optimizer(
        original,
        label=LABEL,
        allow_distributed_optimizer=MODEL_CONFIG.allow_distributed_optimizer,
        param_attr=MODEL_CONFIG.trainable_param_attr,
        lr=_expert_lr,
        # late-bound: tests replace this module's validator
        validate=lambda chunks, optimizer: _validate_complete_optimizer_masters(chunks, optimizer),
    )


def _validate_complete_optimizer_masters(model_chunks, optimizer) -> None:
    """Require every trainable model parameter's replicated optimizer master."""

    _optimizer.validate_optimizer_masters(
        model_chunks,
        optimizer,
        require_complete=MODEL_CONFIG.requires_complete_master,
        label=LABEL,
    )


_optimizer_children = _optimizer.optimizer_children


def weight_iterator_task_bridge(iterator):
    cached = getattr(iterator, "_yeto_expert_full_task_bridge", None)
    if cached is not None:
        return cached
    source = getattr(iterator.args, "ref_load", None) or getattr(
        iterator.args, "hf_checkpoint", None
    )
    if not source:
        raise RuntimeError(
            "expert-full weight iterator has no HuggingFace checkpoint for Bridge tasks"
        )
    from megatron.bridge import AutoBridge

    cached = AutoBridge.from_hf_pretrained(
        source,
        trust_remote_code=bool(ext_attr(iterator.args, "rl_trust_remote_code")),
    )
    iterator._yeto_expert_full_task_bridge = cached
    return cached


def expected_expert_task_count(expert_count: int) -> int:
    """Selected clone-expert conversion tasks: per layer, fc1 and fc2 per local offset."""

    return NUM_LAYERS * min(expert_count, CLONES_PER_EXPERT_RANK) * 2


def export_selected_expert_weights(
    module: ModuleType,
    iterator,
    weights,
    *,
    export_kwargs: Mapping[str, Any] | None = None,
    materialize: bool = True,
):
    """``(hf_name, weight, megatron_names)`` of the selected clone experts only.

    ``module`` is the ports Miles Bridge weight iterator module.  Every EP rank exports the same local expert
    offsets so the Bridge export collectives align; a non-sending rank
    (``materialize=False``) still runs them and yields nothing.  The task
    Bridge reads the trainer checkpoint, never a quantized rollout one.
    """

    expert_count = configured_expert_count()
    task_bridge = weight_iterator_task_bridge(iterator)
    renamed = {module.strip_param_name_prefix(name): value for name, value in weights.items()}
    with module.megatron_bridge_utils.patch_megatron_model(iterator.model):
        tasks = filter_collective_expert_tasks(
            task_bridge.get_conversion_tasks(iterator.model), expert_count=expert_count
        )
        expected = expected_expert_task_count(expert_count)
        if len(tasks) != expected:
            raise RuntimeError(
                f"expert-full publish found {len(tasks)} conversion tasks, expected {expected}"
            )
        named_weights = task_bridge.export_hf_weights(
            iterator.model,
            cpu=False,
            conversion_tasks=module._process_conversion_tasks(tasks, renamed),
            merge_adapter_weights=False,
            **dict(export_kwargs or {}),
        )
        if not materialize:
            for _ in named_weights:
                pass
            return
        selected = _export.select_named_weights(
            named_weights, lambda name: selected_expert_hf_name(name, expert_count=expert_count)
        )
        for item in iterator._postprocess_and_quantize(selected, "base"):
            if not module.is_lora_weight_name(item[0]):
                yield item


__all__ = [
    "MODEL_CONFIG",
    "configured_expert_count",
    "expected_expert_task_count",
    "export_selected_expert_weights",
    "filter_collective_expert_tasks",
    "install_or_defer",
    "selected_expert_hf_name",
    "weight_iterator_task_bridge",
    "wrap_expert_full_optimizer",
]
