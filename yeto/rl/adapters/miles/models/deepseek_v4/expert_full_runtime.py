"""Runtime helpers for attention LoRA plus clone-expert full tuning on ports Miles.

:mod:`.ports_runtime` installs these through ports Miles module hooks.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import re
import sys
from collections.abc import Mapping
from types import ModuleType
from typing import Any

from yeto.rl.adapters.miles.models.deepseek_v4.expert_clone import (
    CLONES_PER_EXPERT_RANK,
    CLONES_PER_LAYER,
    NUM_LAYERS,
    ORIGINAL_EXPERTS,
    TRAINING_EXPERTS_PER_RANK,
    logical_to_training_expert_id,
)

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


def _mapping_hf_names(task: Any) -> tuple[str, ...]:
    if task is None:
        return ()
    value = getattr(getattr(task, "mapping", None), "hf_param", None)
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping) and all(
        isinstance(item, str) for item in value.values()
    ):
        return tuple(value.values())
    return ()


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

        if config.use_distributed_optimizer:
            raise RuntimeError(
                "expert-full RL requires replicated FP32 optimizer masters"
            )
        overrides = (
            get_standard_config_overrides(config)
            if config_overrides is None
            else dict(config_overrides)
        )
        overrides[ParamKey(attr="_yeto_expert_full")] = {
            "max_lr": _expert_lr(),
            "min_lr": _expert_lr(),
        }
        optimizer = original(
            config=config,
            model_chunks=model_chunks,
            config_overrides=overrides,
            use_gloo_process_groups=use_gloo_process_groups,
            **kwargs,
        )
        _validate_complete_optimizer_masters(model_chunks, optimizer)
        return optimizer

    return get_megatron_optimizer


def _model_chunks(models: Any) -> tuple[Any, ...]:
    return tuple(models) if isinstance(models, (list, tuple)) else (models,)


def _validate_complete_optimizer_masters(model_chunks, optimizer) -> None:
    """Require every trainable model parameter's replicated optimizer master."""

    import torch

    optimizer_parameters = set()
    for child in _optimizer_children(optimizer):
        inner = getattr(child, "optimizer", None)
        param_groups = getattr(inner, "param_groups", None)
        if param_groups is None:
            param_groups = getattr(child, "param_groups", None)
        if param_groups is None:
            if getattr(child, "is_stub_optimizer", False):
                continue
            raise RuntimeError("expert-full Adam child has no parameter groups")
        for group in param_groups:
            if not isinstance(group, Mapping) or "params" not in group:
                raise RuntimeError("expert-full Adam parameter group is invalid")
            optimizer_parameters.update(id(parameter) for parameter in group["params"])

    trainable = []
    seen = set()
    for chunk in _model_chunks(model_chunks):
        for parameter in chunk.parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                trainable.append(parameter)
                seen.add(id(parameter))
    if not trainable:
        raise RuntimeError("expert-full optimizer has no trainable parameters")
    for parameter in trainable:
        main = getattr(parameter, "main_param", None)
        if (
            getattr(parameter, "main_param_sharded", False)
            or main is None
            or main.dtype != torch.float32
            or main.numel() != parameter.numel()
            or id(main) not in optimizer_parameters
        ):
            raise RuntimeError(
                "expert-full trainable parameter lacks a complete, "
                "optimizer-owned FP32 master"
            )


def _optimizer_children(optimizer) -> list[Any]:
    return list(getattr(optimizer, "chained_optimizers", (optimizer,)))


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
        trust_remote_code=bool(iterator.args.yeto_rl_trust_remote_code),
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
        selected = (
            item for item in named_weights
            if selected_expert_hf_name(item[0], expert_count=expert_count)
        )
        for item in iterator._postprocess_and_quantize(selected, "base"):
            if not module.is_lora_weight_name(item[0]):
                yield item


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


def install_or_defer(fullname: str, installer) -> None:
    loaded = sys.modules.get(fullname)
    if loaded is not None:
        installer(loaded)
        return
    finder = _Finder(fullname, installer)
    _FINDERS.append(finder)
    sys.meta_path.insert(0, finder)
