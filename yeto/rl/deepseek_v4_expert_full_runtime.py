"""Runtime-only Miles patches for attention LoRA plus clone-expert full tuning."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import re
import sys
from collections.abc import Mapping
from types import ModuleType
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
            raise ValueError(f"hybrid policy tensor {name!r} is not a tensor")
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
    from megatron.bridge import AutoBridge

    cached = AutoBridge.from_hf_pretrained(
        actor.args.hf_checkpoint,
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
        branch = str(parameter._yeto_expert_branch)
        if branch == "linear_fc1" and set(projections) == {"gate_proj", "up_proj"}:
            gate, up = parameter.chunk(2, dim=0)
            views[projections["gate_proj"]] = gate
            views[projections["up_proj"]] = up
        elif branch == "linear_fc2" and set(projections) == {"down_proj"}:
            views[projections["down_proj"]] = parameter
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


def _assert_frozen_experts_unchanged(actor) -> None:
    current = {
        id(parameter): int(parameter._version)
        for chunk in _model_chunks(actor.model)
        for parameter in chunk.parameters()
        if getattr(parameter, "_yeto_expert_full_configured", False)
        and not getattr(parameter, "_yeto_expert_full", False)
    }
    previous = getattr(actor, "_yeto_frozen_expert_versions", None)
    if previous is not None and current != previous:
        raise RuntimeError("a frozen original or unselected expert was modified")
    actor._yeto_frozen_expert_versions = current


def _export_attention(actor, *, retain: bool) -> dict[str, Any]:
    import torch

    expected = {spec.name for spec in _attention_specs(actor)}
    tensors = {}
    seen = set()
    for item in _actor_bridge(actor).export_adapter_weights(
        actor.model,
        cpu=True,
        show_progress=False,
    ):
        name, value = item[0], item[1]
        name = _canonical_name(name)
        if name not in expected:
            raise RuntimeError(f"unexpected adapter tensor in hybrid export: {name!r}")
        if name in seen:
            raise RuntimeError(f"duplicate adapter tensor in hybrid export: {name!r}")
        seen.add(name)
        if retain:
            tensors[name] = value.detach().to(dtype=torch.float32).contiguous()
    if seen != expected:
        raise RuntimeError("hybrid export did not produce every attention LoRA tensor")
    return tensors


def _export_experts(actor, *, retain: bool) -> dict[str, Any]:
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
        if dist.get_rank() in owners and dist.get_rank() != source:
            if not torch.equal(local[spec.name], value):
                raise RuntimeError(
                    f"DP replicas disagree on expert tensor {spec.name!r}"
                )
        if retain:
            tensors[spec.name] = value.to(
                device="cpu", dtype=torch.float32
            ).contiguous()
        del value
    return tensors


def export_hybrid_trainable_state(module: ModuleType, actor, *, policy_version: int):
    import torch.distributed as dist

    _assert_frozen_experts_unchanged(actor)
    retain = dist.get_rank() == 0
    tensors = _export_attention(actor, retain=retain)
    tensors.update(_export_experts(actor, retain=retain))
    if not retain:
        return None
    metrics = (
        getattr(actor.args, "_external_train_metrics", {})
        if getattr(actor.args, "external_policy_sync_path", None) is not None
        else {}
    )
    return make_hybrid_trainable_state(
        module,
        policy_version,
        tensors,
        train_rollout_kl=metrics.get("train/train_rollout_kl"),
        ess_ratio=metrics.get("train/ess_ratio"),
        pg_clipfrac=metrics.get("train/pg_clipfrac"),
        train_seconds=getattr(actor.args, "_external_train_seconds", None),
    )


def _broadcast_policy_tensor(spec, state):
    import torch
    import torch.distributed as dist

    if dist.get_rank() == 0:
        if state is None or spec.name not in state.tensors:
            raise RuntimeError(f"global hybrid policy is missing {spec.name!r}")
        value = state.tensors[spec.name].to(
            device=torch.device("cuda", torch.cuda.current_device()),
            dtype=torch.bfloat16,
        ).contiguous()
    else:
        value = torch.empty(
            spec.shape,
            device=torch.device("cuda", torch.cuda.current_device()),
            dtype=torch.bfloat16,
        )
    dist.broadcast(value, src=0)
    return value


def _optimizer_children(optimizer) -> list[Any]:
    return list(getattr(optimizer, "chained_optimizers", (optimizer,)))


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

    sides = _attention_sides(actor)
    for spec in _attention_specs(actor):
        value = _broadcast_policy_tensor(spec, state)
        side = sides[spec.name]
        if side.param_weight is not None:
            mapped = side.mapping.hf_to_megatron(value, side.megatron_module)
            if mapped is None or mapped.numel() != side.param_weight.numel():
                raise RuntimeError(f"attention LoRA shape mismatch for {spec.name!r}")
            side.param_weight.copy_(mapped.reshape(side.param_weight.shape))
            del mapped
        del value

    views = _expert_views(actor)
    for spec in _expert_specs(actor):
        value = _broadcast_policy_tensor(spec, state)
        target = views.get(spec.name)
        if target is not None:
            if target.shape != value.shape:
                raise RuntimeError(f"expert shape mismatch for {spec.name!r}")
            target.copy_(value)
        del value

    module._align_scheduler(actor, policy_version)
    actor.optimizer.reload_model_params()
    if dist.is_initialized():
        dist.barrier()
    if reset_optimizer:
        for child in _optimizer_children(actor.optimizer):
            inner = getattr(child, "optimizer", child)
            inner.state.clear()
    actor.weights_backuper.backup("actor")
    _assert_frozen_experts_unchanged(actor)
    actor._external_policy_version = policy_version
    return len(_expected_specs(actor)) if reset_optimizer else 0


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

    def export_trainable_state(self):
        state = module.export_external_trainable_state(
            self,
            policy_version=getattr(self, "_external_policy_version", 0),
        )
        return state

    def apply_trainable_state(self, state, *, reset_optimizer):
        reset_count = module.apply_external_trainable_state(
            self,
            state,
            reset_optimizer=reset_optimizer,
        )
        return reset_count

    module.MegatronTrainRayActor.export_trainable_state = export_trainable_state
    module.MegatronTrainRayActor.apply_trainable_state = apply_trainable_state
    module._yeto_expert_full_installed = True


def install_on_actor_group(module: ModuleType) -> None:
    if getattr(module, "_yeto_expert_full_installed", False):
        return

    async def apply_trainable_state(self, state, *, reset_optimizer):
        import asyncio

        refs = [
            actor.apply_trainable_state.remote(
                state if rank == 0 else None,
                reset_optimizer=reset_optimizer,
            )
            for rank, actor in enumerate(self._actor_handles)
        ]
        results = await asyncio.gather(*refs)
        if not results or any(result != results[0] for result in results[1:]):
            raise RuntimeError("Megatron ranks disagree after applying hybrid state")
        return results[0]

    module.RayTrainGroup.apply_trainable_state = apply_trainable_state
    module._yeto_expert_full_installed = True


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
            tasks = filter_collective_expert_tasks(
                self._bridge.get_conversion_tasks(self.model),
                expert_count=expert_count,
            )
            expected = NUM_LAYERS * min(expert_count, CLONES_PER_EXPERT_RANK) * 2
            if len(tasks) != expected:
                raise RuntimeError(
                    f"expert-full base sync found {len(tasks)} conversion tasks, "
                    f"expected {expected}"
                )
            tasks = module._process_conversion_tasks(tasks, renamed)
            named_weights = self._bridge.export_hf_weights(
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
