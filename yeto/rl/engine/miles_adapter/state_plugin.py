"""Trainable-state export/apply executed *inside* every Megatron rank (task 3.4, D4).

Loaded by upstream's generic entry point (``michaellchung/miles`` ``yeto/ports``):
``TrainGroup.run_plugin(fn_path, kwargs)`` -> ``train_actor.run_plugin`` ->
``load_function(fn_path)(actor, **kwargs)`` on every rank of every cell (the
ports path guarantees a single cell). Every function here is collective: all
ranks must enter it.

Ported from the agentenv fork's ``megatron_backends/.../trainable_state.py``
onto upstream's structure (Bridge-based LoRA under ``megatron_utils/lora/``;
``actor.model`` / ``actor.optimizer`` / ``actor.opt_param_scheduler`` /
``actor.weights_backuper``), with the #64 gradient-flow invariant:

* never assign ``Parameter.data`` across dtypes. To expose an FP32 master as
  the module parameter (Bridge conversion reads module parameters), a *new*
  ``Parameter`` sharing the master storage with ``requires_grad=False`` is
  placed temporarily in ``module._parameters[name]`` and the original
  Parameter object is restored afterwards (:func:`masters_as_module_parameters`);
* after apply, every adapter Parameter is still the registered object with
  ``requires_grad=True`` (:func:`assert_grad_flow_intact`), so the next step's
  grad accumulation hooks still fire.

torch / megatron / miles are imported lazily; importing this module is cheap.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

CANONICAL_PREFIX = "base_model.model."
OPTIMIZER_MODES = ("preserve", "reset")

_PLUGIN_MODULE = "yeto.rl.engine.miles_adapter.state_plugin"
EXPORT_STATE = f"{_PLUGIN_MODULE}.export_state"
APPLY_STATE = f"{_PLUGIN_MODULE}.apply_state"
GRAD_NORM = f"{_PLUGIN_MODULE}.grad_norm"
APPLIED_LRS = f"{_PLUGIN_MODULE}.applied_lrs"
STEP_LOSSES = f"{_PLUGIN_MODULE}.step_losses"
# rl-algo-critic-family 3.2: run in the critic's processes before its train.
CRITIC_RECORDERS = f"{_PLUGIN_MODULE}.install_critic_recorders"
EXPLAINED_VARIANCE_KEY = "explained_variance"


class StatePluginError(RuntimeError):
    pass


@dataclass(frozen=True)
class AdapterBinding:
    """One trainable adapter tensor: canonical PEFT name <-> Megatron Parameter."""

    name: str
    parameter: Any  # torch.nn.Parameter registered in the model
    to_hf: Callable[[Any], Any]  # megatron tensor -> canonical HF tensor
    from_hf: Callable[[Any], Any]  # canonical HF tensor -> megatron-shaped tensor


# --------------------------------------------------------------------------
# Gradient-flow-safe master exposure (#64 / D4)
# --------------------------------------------------------------------------


def master_of(parameter: Any) -> Any:
    """The FP32 optimizer master viewed in the parameter's shape."""

    import torch

    main = getattr(parameter, "main_param", None)
    if main is None:
        if parameter.dtype != torch.float32:
            raise StatePluginError("low-precision adapter parameter has no FP32 optimizer master")
        return parameter.detach()
    if main.dtype != torch.float32 or main.numel() != parameter.numel():
        raise StatePluginError(
            "adapter parameter has no complete FP32 optimizer master "
            "(sharded distributed-optimizer masters are not supported)"
        )
    return main.view(parameter.shape)


def has_complete_master(parameter: Any) -> bool:
    """FP32 itself, or a ``main_param`` covering the whole parameter.

    Megatron's DistributedOptimizer sets ``main_param`` to this rank's FP32
    *shard* (``main_param_sharded=True``) on owned ranges and leaves it unset
    elsewhere; both are DP-sharded masters, not complete ones.
    """

    import torch

    if parameter.dtype == torch.float32:
        return True
    main = getattr(parameter, "main_param", None)
    return (
        main is not None
        and not getattr(parameter, "main_param_sharded", False)
        and main.numel() == parameter.numel()
    )


# --------------------------------------------------------------------------
# DistributedOptimizer: DP-sharded FP32 masters (rl-infra-spec 2.4, decision (a))
# --------------------------------------------------------------------------
# Under Megatron's DistributedOptimizer with DP>1 a bf16 adapter parameter has
# no ``main_param``: each DP rank's optimizer owns the FP32 main copy of a
# sub-range ``gbuf_ranges[..]["param_map"][param]["param"]`` of the flattened
# parameter (same layout fork-M5 v2 reads). Export gathers the full master by
# filling each rank's range into zeros and summing over the DP group (every
# element is owned by exactly one rank); apply writes each rank's range from
# the full target and sets the bf16 model copy on every rank.


def _optimizer_leaves(optimizer: Any) -> list[Any]:
    chained = getattr(optimizer, "chained_optimizers", None)
    if chained is not None:
        return [leaf for member in chained for leaf in _optimizer_leaves(member)]
    return [optimizer]


def _is_distributed(leaf: Any) -> bool:
    return hasattr(leaf, "gbuf_ranges") and hasattr(leaf, "model_param_group_index_map")


def distributed_ranges(optimizer: Any) -> tuple[bool, dict[int, tuple[Any, Any, int, int]]]:
    """(has a DistributedOptimizer, {id(model_param): (leaf, param, start, end)} owned here)."""
    found, owned = False, {}
    for leaf in _optimizer_leaves(optimizer) if optimizer is not None else ():
        if not _is_distributed(leaf):
            continue
        found = True
        if getattr(getattr(leaf, "config", None), "use_precision_aware_optimizer_no_fp8_or_ds_fp8", False):
            raise StatePluginError("sharded masters under the precision-aware optimizer are not supported")
        for gbuf_range_maps in leaf.gbuf_ranges:
            for per_bucket in gbuf_range_maps.values():
                for bucket_range_map in per_bucket:
                    for param, range_map in bucket_range_map["param_map"].items():
                        r = range_map["param"]
                        if r.end > r.start:
                            if id(param) in owned:
                                raise StatePluginError("a parameter range appears twice on one rank")
                            owned[id(param)] = (leaf, param, int(r.start), int(r.end))
    return found, owned


def _leaf_members(leaf: Any) -> set[int]:
    """ids of every model parameter this distributed leaf shards (owned here or not).

    ``gbuf_ranges``/``model_param_group_index_map`` list only the locally owned
    ranges; the grad buffers (``leaf.buffers[..].param_index_map``) list all.
    """

    members: set[int] = set()
    for buffer in getattr(leaf, "buffers", None) or ():
        members.update(id(p) for p in getattr(buffer, "param_index_map", {}) or {})
    for gbuf_range_maps in getattr(leaf, "gbuf_ranges", None) or ():
        for per_bucket in gbuf_range_maps.values():
            for bucket_range_map in per_bucket:
                members.update(id(p) for p in bucket_range_map["param_map"])
    return members


def _leaf_for(optimizer: Any, param: Any, owned: Mapping[int, tuple[Any, Any, int, int]]) -> Any:
    """The distributed leaf (hence DP group) that shards ``param``.

    ChainedOptimizer: dense params belong to the dense leaf (dense DP group),
    expert params to the expert leaf (expert DP group).
    """

    if id(param) in owned:
        return owned[id(param)][0]
    leaves = [leaf for leaf in _optimizer_leaves(optimizer) if _is_distributed(leaf)]
    hits = [leaf for leaf in leaves if id(param) in _leaf_members(leaf)]
    if len(hits) == 1:
        return hits[0]
    if not hits and len(leaves) == 1:
        return leaves[0]
    raise StatePluginError("cannot tell which distributed optimizer shards an adapter parameter")


def _dp_all_reduce_sum(leaf: Any, flat: Any) -> None:
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return
    dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=getattr(leaf, "data_parallel_group", None))


def needs_gather(optimizer: Any, parameters: Sequence[Any]) -> bool:
    """Whether some low-precision parameter's master lives only in DistributedOptimizer shards."""

    if optimizer is None or not any(_is_distributed(leaf) for leaf in _optimizer_leaves(optimizer)):
        return False
    return any(not has_complete_master(p) for p in parameters)


def full_masters(optimizer: Any, parameters: Sequence[Any], *, all_reduce_sum=None, reduce=None) -> list[Any]:
    """FP32 masters in each parameter's shape; DP-sharded ones are gathered.

    Collective over each owning leaf's DP group when any parameter is
    DP-sharded (every rank must call it with the same parameters in the same
    order). ``reduce(flat, leaf)`` / ``all_reduce_sum(flat)`` replace the
    all-reduce (tests).
    """
    import torch

    has_dist, owned = distributed_ranges(optimizer)
    out: list[Any] = []
    pending: dict[int, tuple[Any, list[tuple[int, Any]]]] = {}
    for i, param in enumerate(parameters):
        if has_complete_master(param):
            out.append(master_of(param))
            continue
        if not has_dist:
            out.append(master_of(param))  # raises: no master anywhere
            continue
        full = torch.zeros(param.numel(), dtype=torch.float32, device=param.device)
        if id(param) in owned:
            _, _, start, end = owned[id(param)]
            leaf = owned[id(param)][0]
            shard = leaf._get_main_param_and_optimizer_states(param)["param"]
            if shard.dtype != torch.float32 or shard.numel() != end - start:
                raise StatePluginError("distributed-optimizer main shard does not match its range")
            full[start:end] = shard.detach().reshape(-1)
        leaf = _leaf_for(optimizer, param, owned)
        out.append(full)
        pending.setdefault(id(leaf), (leaf, []))[1].append((i, full))
    # Leaf order is the same on every rank of a DP group, so the per-group
    # collectives are issued in the same order.
    order = {id(leaf): k for k, leaf in enumerate(_optimizer_leaves(optimizer))} if pending else {}
    for _, (leaf, items) in sorted(pending.items(), key=lambda kv: order.get(kv[0], 0)):
        flat = torch.cat([f for _, f in items])
        if reduce is not None:
            reduce(flat, leaf)
        elif all_reduce_sum is not None:
            all_reduce_sum(flat)
        else:
            _dp_all_reduce_sum(leaf, flat)
        offset = 0
        for i, full in items:
            n = full.numel()
            out[i] = flat[offset : offset + n].view(parameters[i].shape)
            offset += n
    return out


def write_masters(optimizer: Any, parameters: Sequence[Any], targets: Sequence[Any]) -> bool:
    """Write full FP32 targets into the masters; returns whether model copies were set here.

    For DP-sharded masters each rank writes its own range and sets the bf16
    model copy directly (in place; ``Parameter.data`` is not reassigned).
    """
    import torch

    has_dist, owned = distributed_ranges(optimizer)
    wrote_model = False
    for param, target in zip(parameters, targets, strict=True):
        if has_complete_master(param):
            master_of(param).copy_(target)
            continue
        if not has_dist:
            master_of(param)  # raises
        if id(param) in owned:
            leaf, _, start, end = owned[id(param)]
            shard = leaf._get_main_param_and_optimizer_states(param)["param"]
            shard.data.copy_(target.reshape(-1)[start:end].reshape(shard.shape))
        param.data.copy_(target.to(dtype=param.dtype))
        wrote_model = True
    return wrote_model


def parameter_owners(modules: Iterable[Any], parameters: Iterable[Any]) -> dict[int, tuple[Any, str]]:
    wanted = {id(p) for p in parameters}
    owners: dict[int, tuple[Any, str]] = {}
    for root in modules:
        for module in root.modules():
            for attr, param in module._parameters.items():
                if param is not None and id(param) in wanted:
                    owners.setdefault(id(param), (module, attr))
    missing = wanted - set(owners)
    if missing:
        raise StatePluginError(f"{len(missing)} adapter parameters are not registered in the model")
    return owners


@contextmanager
def masters_as_module_parameters(modules: Sequence[Any], parameters: Sequence[Any], masters: Sequence[Any] | None = None):
    """Temporarily register FP32 masters as the module parameters.

    A fresh ``Parameter`` that shares the master's storage and has
    ``requires_grad=False`` replaces the entry in ``module._parameters``; the
    original Parameter object (with its grad-accumulation hooks and
    ``main_grad``) is untouched and restored on exit. ``Parameter.data`` is
    never reassigned. ``masters`` (e.g. gathered DistributedOptimizer shards,
    see ``full_masters``) overrides ``master_of``.
    """

    import torch

    owners = parameter_owners(modules, parameters)
    swapped: list[tuple[Any, str, Any]] = []
    try:
        if masters is None:
            masters = [master_of(p) for p in parameters]
        for param, master in zip(parameters, masters, strict=True):
            if master.data_ptr() == param.data_ptr() and master.dtype == param.dtype:
                continue  # already FP32 and self-mastered
            module, attr = owners[id(param)]
            module._parameters[attr] = torch.nn.Parameter(master, requires_grad=False)
            swapped.append((module, attr, param))
        yield
    finally:
        for module, attr, param in reversed(swapped):
            module._parameters[attr] = param


def assert_grad_flow_intact(modules: Sequence[Any], parameters: Sequence[Any]) -> None:
    owners = parameter_owners(modules, parameters)
    for param in parameters:
        module, attr = owners[id(param)]
        if module._parameters[attr] is not param:
            raise StatePluginError(f"adapter parameter {attr!r} was replaced")
        if not param.requires_grad:
            raise StatePluginError(f"adapter parameter {attr!r} no longer requires grad")


# --------------------------------------------------------------------------
# Binding resolution
# --------------------------------------------------------------------------


def _canonical(name: str) -> str:
    return name if name.startswith(CANONICAL_PREFIX) else CANONICAL_PREFIX + name


def adapter_bindings(actor: Any) -> tuple[AdapterBinding, ...]:
    """Resolve (and cache) adapter bindings via Megatron-Bridge conversion tasks.

    Tests (and alternative backends) may preset ``actor._yeto_adapter_bindings``.
    """

    cached = getattr(actor, "_yeto_adapter_bindings", None)
    if cached is not None:
        return tuple(cached)

    # Upstream's cached bridge: Miles owns the remote-code decision for its checkpoint.
    from miles.backends.megatron_utils.hf_export import _get_hf_bridge

    bridge = _get_hf_bridge(actor.args.hf_checkpoint)
    model_bridge = getattr(bridge, "_model_bridge", None)
    build_tasks = getattr(model_bridge, "build_adapter_conversion_tasks", None)
    if build_tasks is None:
        raise StatePluginError("Megatron-Bridge lacks adapter conversion tasks")
    bindings = []
    tasks_by_base = build_tasks(actor.model)
    for base_name in sorted(tasks_by_base):
        for task in sorted(tasks_by_base[base_name], key=lambda t: t.adapter_key or ""):
            for side in (task.linear_in_task, task.linear_out_task):
                if side.param_weight is None:
                    continue
                # names only: the model copy has the master's shape
                converted = side.mapping.megatron_to_hf(side.param_weight, side.megatron_module)
                if len(converted) != 1:
                    raise StatePluginError(f"ambiguous LoRA mapping for {side.param_name!r}")
                name = _canonical(next(iter(converted)))

                def to_hf(tensor, _side=side):
                    return next(iter(_side.mapping.megatron_to_hf(tensor, _side.megatron_module).values()))

                def from_hf(tensor, _side=side):
                    return _side.mapping.hf_to_megatron(tensor, _side.megatron_module)

                bindings.append(AdapterBinding(name, side.param_weight, to_hf, from_hf))
    names = [b.name for b in bindings]
    if not names or len(names) != len(set(names)):
        raise StatePluginError("Megatron produced an empty or duplicate LoRA mapping")
    trainable = {id(p) for chunk in actor.model for p in chunk.parameters() if p.requires_grad}
    if {id(b.parameter) for b in bindings} != trainable:
        raise StatePluginError("adapter conversion does not cover every trainable parameter")
    actor._yeto_adapter_bindings = tuple(sorted(bindings, key=lambda b: b.name))
    return actor._yeto_adapter_bindings


def _model_parallel(actor: Any) -> bool:
    args = actor.args
    return any(
        int(getattr(args, name, 1) or 1) > 1
        for name in (
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "expert_model_parallel_size",
        )
    )


def _is_main_rank(actor: Any) -> bool:
    flag = getattr(actor, "_is_first_replica_megatron_main_rank", None)
    if flag is not None:
        return bool(flag)
    try:
        import torch.distributed as dist
    except ImportError:  # pragma: no cover
        return True
    return not dist.is_initialized() or dist.get_rank() == 0


def _barrier() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


@contextmanager
def trainer_resident(actor: Any):
    """Run a state plugin on a resident trainer, restoring its residency.

    Upstream ``--offload-train`` actors start asleep (``sleep()`` at the end of
    init) and ``update_weights`` works while asleep, so export/apply can be
    reached with the Megatron memory paused and the process groups destroyed.
    Wake the actor for the plugin and put it back to sleep afterwards.
    """

    asleep = bool(getattr(actor, "_asleep", False)) and bool(
        getattr(getattr(actor, "args", None), "offload_train", False)
    )
    if asleep:
        actor.wake_up()
    try:
        yield
    finally:
        if asleep:
            actor.sleep()


# --------------------------------------------------------------------------
# Plugins (signature: fn(actor, **kwargs))
# --------------------------------------------------------------------------


def export_state(actor: Any, *, policy_version: int) -> dict[str, Any] | None:
    """Canonical FP32 PEFT tensors on the first-replica main rank, else ``None``."""

    install_grad_norm_recorder()
    with trainer_resident(actor):
        return _export_state(actor, policy_version=policy_version)


def is_native_flash_next(actor: Any) -> bool:
    """Qwen3.8-Flash-Next (qwen4_exp) trains through Miles' native LoRA plugin.

    The pinned Megatron-Bridge has no qwen4_exp bridge, so ``adapter_bindings``
    cannot resolve its adapters; the learner marks the run explicitly
    (``yeto_rl_native_lora_export``) and this module dispatches on that.
    """

    return getattr(getattr(actor, "args", None), "yeto_rl_native_lora_export", None) == "qwen3_8_next"


def _flash_next_exporter() -> Callable[[Any], Iterable[Any]]:
    from miles_plugins.models.qwen3_8_next.lora import export_qwen3_8_next_lora_hf_chunks

    return export_qwen3_8_next_lora_hf_chunks


_FN_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
# Tensors per layer emitted by Miles c35702e ``export_qwen3_8_next_lora_hf_chunks``:
# GDN attention 5 projections x A/B, QSA q/k/v/o x A/B, shared expert
# gate/up/down x A/B, routed experts gate_up/down x A/B (every layer is MoE).
_FN_ATTN_TENSORS = {"linear_attn": 10, "self_attn": 8}
_FN_MLP_TENSORS = {"mlp.shared_expert": 6, "mlp.experts": 4}


def _fn_layer(name: str) -> int:
    match = _FN_LAYER_RE.search(name)
    if match is None:
        raise StatePluginError(f"Flash-Next LoRA tensor {name!r} has no layer index")
    return int(match.group(1))


def merge_pp_stage_exports(stages: Sequence[Mapping[str, Any]], *, num_layers: int | None = None) -> dict[str, Any]:
    """Merge per-PP-stage Flash-Next exports (stage order) into one full-model dict.

    Miles names adapters with Megatron's global ``layer_number - 1``, so names
    are already global: nothing is renumbered.  Refuses duplicate names across
    stages, overlapping / out-of-order stage layer ranges, missing layers and
    layers whose tensor count is not GDN/QSA attention + shared + routed experts.
    """

    merged: dict[str, Any] = {}
    previous_max = -1
    for index, stage in enumerate(stages):
        if not stage:
            raise StatePluginError(f"Flash-Next PP stage {index} exported no LoRA tensors")
        layers = {_fn_layer(name) for name in stage}
        if min(layers) <= previous_max:
            raise StatePluginError(
                f"Flash-Next PP stage {index} layers {sorted(layers)} overlap or precede earlier stages "
                f"(max {previous_max}); names must carry global layer indices")
        previous_max = max(layers)
        for name, value in stage.items():
            if name in merged:
                raise StatePluginError(f"duplicate Flash-Next LoRA tensor {name!r} across PP stages")
            merged[name] = value
    per_layer: dict[int, list[str]] = {}
    for name in merged:
        per_layer.setdefault(_fn_layer(name), []).append(name)
    expected_layers = range(num_layers) if num_layers else range(max(per_layer) + 1)
    missing = sorted(set(expected_layers) - set(per_layer))
    extra = sorted(set(per_layer) - set(expected_layers))
    if missing or extra:
        raise StatePluginError(f"Flash-Next LoRA export missing layers {missing} / unexpected layers {extra}")
    for layer, names in sorted(per_layer.items()):
        attention = [kind for kind in _FN_ATTN_TENSORS if any(f".{kind}." in n for n in names)]
        if len(attention) != 1:
            raise StatePluginError(f"Flash-Next layer {layer} has attention kinds {attention}")
        want = _FN_ATTN_TENSORS[attention[0]] + sum(_FN_MLP_TENSORS.values())
        if len(names) != want:
            raise StatePluginError(f"Flash-Next layer {layer} exported {len(names)} tensors, expected {want}")
    return merged


def _pp_gather_default(local: dict[str, Any] | None, is_main: bool) -> list[dict[str, Any] | None] | None:
    """Gather each PP stage's export to the main rank over the PP group.

    Only the PP group that contains the main rank carries tensors (one tiny
    flag all-gather per group decides); others send ``None``.  Returns the
    stage-ordered list on the main rank, ``None`` elsewhere.  Without PP (or
    without torch.distributed) the local export is the whole model.
    """

    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return [local] if is_main else None
    from megatron.core import mpu

    if mpu.get_pipeline_model_parallel_world_size() <= 1:
        return [local] if is_main else None
    group = mpu.get_pipeline_model_parallel_group()
    size = dist.get_world_size(group)
    flags: list[Any] = [None] * size
    dist.all_gather_object(flags, (bool(is_main), dist.get_rank()), group=group)
    mains = [rank for flag, rank in flags if flag]
    if not mains:
        return None
    if len(mains) != 1:
        raise StatePluginError(f"multiple Flash-Next main ranks in one PP group: {mains}")
    gathered: list[Any] | None = [None] * size if is_main else None
    dist.gather_object(local, gathered, dst=mains[0], group=group)
    return gathered  # PP group ranks are in stage order


def _export_flash_next(actor: Any, *, policy_version: int, exporter=None, pp_gather=None) -> dict[str, Any] | None:
    """Fingerprint export for the no-sync Flash-Next island (S11 try24 fix).

    Uses Miles' own collective HF export (every rank must call it: TP/EP
    gathers inside one PP stage). Names are Miles' SGLang adapter names
    (``model.language_model.layers.N...lora_{A,B}.weight`` with global N, q/k/v
    share one A, expert tensors padded to ``--lora-rank``) under the canonical
    prefix; values are the bf16 model copies upcast to fp32.  Miles only covers
    the local PP stage, so every stage's export is gathered over the PP group
    to the main rank (last stage) and merged into the full model.  The layout is
    learned from the first export (MilesPolicyState with
    ``expected_layout_hash=None``) and pinned after.  The state is never applied
    back (``apply_state`` refuses Flash-Next).
    """

    import torch

    exporter = exporter or _flash_next_exporter()
    pp_gather = pp_gather or _pp_gather_default
    is_main = _is_main_rank(actor)
    # Miles' TP/EP gathers leave stage-complete tensors on every rank; keep
    # them only on the main rank's PP peers (same TP/DP coordinates).
    retain = is_main or _tp_ep_leader(actor)
    tensors: dict[str, Any] = {}
    with torch.no_grad():
        for chunk in exporter(actor.model):
            for raw_name, value in chunk:
                name = _canonical(raw_name)
                if name in tensors:
                    raise StatePluginError(f"duplicate Flash-Next LoRA tensor {name!r}")
                tensors[name] = (
                    value.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
                    if retain
                    else None
                )
    if not tensors:
        raise StatePluginError("Flash-Next native LoRA export produced no tensors")
    stages = pp_gather(tensors if retain else None, is_main)
    if not is_main:
        return None
    if not stages or any(stage is None for stage in stages):
        raise StatePluginError("Flash-Next PP gather is missing a stage export")
    num_layers = getattr(getattr(actor, "args", None), "num_layers", None)
    merged = merge_pp_stage_exports(stages, num_layers=int(num_layers) if num_layers else None)
    print(
        "[rl] Flash-Next native LoRA export: per-PP-stage tensors "
        f"{[len(stage) for stage in stages]}, merged {len(merged)}",
        flush=True,
    )
    for name, value in merged.items():
        if not torch.isfinite(value).all().item():
            raise StatePluginError(f"{name!r} contains NaN or Inf")
    return {"policy_version": int(policy_version), "tensors": dict(sorted(merged.items()))}


def _tp_ep_leader(actor: Any) -> bool:
    """Miles' main-rank test minus the last-stage clause: the main rank's PP peers."""

    try:
        import torch.distributed as dist
        from megatron.core import mpu
    except ImportError:  # pragma: no cover
        return False
    if not (dist.is_available() and dist.is_initialized()):
        return False
    return (
        mpu.get_tensor_model_parallel_rank() == 0
        and mpu.get_data_parallel_rank(with_context_parallel=True) == 0
    )


def _export_state(actor: Any, *, policy_version: int) -> dict[str, Any] | None:

    import torch

    if is_native_flash_next(actor):
        return _export_flash_next(actor, policy_version=policy_version)
    bindings = adapter_bindings(actor)
    params = [b.parameter for b in bindings]
    tensors: dict[str, Any] = {}
    with torch.no_grad():
        if _model_parallel(actor):
            tensors = _collective_export(actor, bindings)
        else:
            masters = full_masters(getattr(actor, "optimizer", None), params)
            for b, master in zip(bindings, masters, strict=True):
                tensors[b.name] = b.to_hf(master)
        if not _is_main_rank(actor):
            return None
        out = {}
        for name, value in sorted(tensors.items()):
            value = value.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
            if not torch.isfinite(value).all().item():
                raise StatePluginError(f"{name!r} contains NaN or Inf")
            out[name] = value
    assert_grad_flow_intact(actor.model, params)
    return {"policy_version": int(policy_version), "tensors": out}


def _collective_export(actor: Any, bindings: Sequence[AdapterBinding]) -> dict[str, Any]:
    """TP/PP/EP: Bridge adapter export is collective; masters exposed per D4."""

    import torch
    from miles.backends.megatron_utils.hf_export import _get_hf_bridge
    from miles.utils import megatron_bridge_utils

    bridge = _get_hf_bridge(actor.args.hf_checkpoint)
    retain = _is_main_rank(actor)
    expected = {b.name for b in bindings}
    tensors: dict[str, Any] = {}
    names: set[str] = set()
    params = [b.parameter for b in bindings]
    opt = getattr(actor, "optimizer", None)
    # DistributedOptimizer (e.g. EP>1 with dense DP>1): no main_param on the
    # parameter; gather the full FP32 master within each owning leaf's DP group.
    masters = full_masters(opt, params) if needs_gather(opt, params) else None
    with masters_as_module_parameters(actor.model, params, masters):
        with megatron_bridge_utils.patch_megatron_model(actor.model):
            for item in bridge.export_adapter_weights(actor.model, cpu=False, show_progress=False):
                name = _canonical(item[0])
                names.add(name)
                if retain:
                    value = item[1].detach().to(device="cpu", dtype=torch.float32).contiguous()
                    prev = tensors.get(name)
                    if prev is not None and not torch.equal(prev, value):
                        raise StatePluginError(f"conflicting collective LoRA tensor {name!r}")
                    tensors[name] = value
    # Local bindings only cover this rank's stage; the exported name set is global.
    if retain and not expected <= names:
        raise StatePluginError(f"collective LoRA export misses local tensors: {sorted(expected - names)[:4]}")
    return tensors


def _optimizer_children(optimizer: Any) -> list[Any]:
    return list(getattr(optimizer, "chained_optimizers", None) or (optimizer,))


def _copy_masters_to_model(actor: Any, bindings: Sequence[AdapterBinding]) -> None:
    if all(getattr(b.parameter, "main_param", None) is None for b in bindings):
        return  # FP32 params are their own masters
    for child in _optimizer_children(actor.optimizer):
        copy = getattr(child, "_copy_main_params_to_model_params", None)
        if copy is None:
            raise StatePluginError("Megatron optimizer lacks main-to-model copy")
        copy()


def reset_optimizer_moments(optimizer: Any) -> None:
    """Zero moments in place; the optimizer object and its param groups survive."""

    try:
        from miles.backends.megatron_utils.optimizer_state_reset import reset_optimizer_states
    except ImportError:
        reset_optimizer_states = None
    if reset_optimizer_states is not None:
        reset_optimizer_states(optimizer)
        return
    import torch

    def leaves(opt):
        for attr in ("chained_optimizers", "sub_optimizers"):
            if hasattr(opt, attr):
                for child in getattr(opt, attr):
                    yield from leaves(child)
                return
        inner = getattr(opt, "optimizer", None)
        yield from (leaves(inner) if inner is not None else (opt,))

    for leaf in leaves(optimizer):
        for state in leaf.state.values():
            for key, value in state.items():
                if key == "step":
                    if isinstance(value, torch.Tensor):
                        value.zero_()
                    else:
                        state[key] = 0
                elif key in ("exp_avg", "exp_avg_sq", "momentum_buffer", "moment2_buffer"):
                    value.zero_()


def align_scheduler(actor: Any, local_step: int) -> int:
    """Scheduler progress := ``local_step`` optimizer steps (in samples, as Megatron counts)."""

    scheduler = getattr(actor, "opt_param_scheduler", None)
    batch = int(actor.args.global_batch_size)
    target = int(local_step) * batch
    if scheduler is None or batch <= 0 or scheduler.num_steps % batch:
        raise StatePluginError("Megatron scheduler progress is not an integral optimizer step")
    if scheduler.num_steps > target:
        raise StatePluginError(
            f"Megatron scheduler ({scheduler.num_steps // batch} steps) is ahead of local step {local_step}"
        )
    if scheduler.num_steps < target:
        scheduler.step(increment=target - scheduler.num_steps)
    return target


def apply_state(
    actor: Any,
    *,
    tensors: Mapping[str, Any],
    policy_version: int,
    local_step: int,
    optimizer: str,
) -> dict[str, Any]:
    """Write canonical tensors into FP32 masters, then the model copies.

    Returns an identical summary on every rank.
    """

    install_grad_norm_recorder()
    with trainer_resident(actor):
        return _apply_state(
            actor,
            tensors=tensors,
            policy_version=policy_version,
            local_step=local_step,
            optimizer=optimizer,
        )


def _apply_state(
    actor: Any,
    *,
    tensors: Mapping[str, Any],
    policy_version: int,
    local_step: int,
    optimizer: str,
) -> dict[str, Any]:

    import torch

    if optimizer not in OPTIMIZER_MODES:
        raise StatePluginError(f"optimizer mode must be one of {OPTIMIZER_MODES}")
    if is_native_flash_next(actor):
        # The no-sync Flash-Next island never applies a state; its export is a
        # lossy (bf16, tied q/k/v A, padded) fingerprint, not an apply contract.
        raise StatePluginError("Flash-Next native LoRA has no apply_state contract (no-sync only)")
    bindings = adapter_bindings(actor)
    params = [b.parameter for b in bindings]
    local = {b.name for b in bindings}
    incoming = set(tensors)
    # Under PP a rank holds a stage; every local tensor must be supplied and
    # (single-stage) nothing unknown may be supplied.
    missing = sorted(local - incoming)
    extra = sorted(incoming - local) if not _model_parallel(actor) else []
    if missing or extra:
        raise StatePluginError(f"global LoRA mapping mismatch: missing={missing[:4]}, extra={extra[:4]}")
    with torch.no_grad():
        targets = {}
        for b in bindings:
            value = tensors[b.name]
            if value.dtype != torch.float32 or not torch.isfinite(value).all().item():
                raise StatePluginError(f"{b.name!r} is not finite FP32")
            target = b.from_hf(value.to(device=b.parameter.device))
            if target.numel() != b.parameter.numel():
                raise StatePluginError(f"global LoRA shape mismatch for {b.name!r}")
            targets[b.name] = target.reshape(b.parameter.shape)
        scheduler_samples = align_scheduler(actor, local_step)
        opt = getattr(actor, "optimizer", None)
        sharded = write_masters(opt, params, [targets[b.name] for b in bindings])
        if not sharded:
            _copy_masters_to_model(actor, bindings)
    _barrier()
    if optimizer == "reset":
        reset_optimizer_moments(actor.optimizer)
    backuper = getattr(actor, "weights_backuper", None)
    if backuper is not None:  # colocated update_weights reads the CPU backup
        backuper.backup("actor")
    assert_grad_flow_intact(actor.model, params)
    return {
        "policy_version": int(policy_version),
        "tensors": len(tensors),
        "optimizer": optimizer,
        "scheduler_samples": scheduler_samples,
    }


# Megatron's per-step grad norms, recorded in this rank's process. Upstream
# ``train`` does not return them, and once the step is done the optimizer's
# ``get_grad_norm`` recomputes over already-consumed buffers (observed 0.0 on
# GPU while Miles logged a non-zero ``train/grad_norm``).
_STEP_GRAD_NORMS: list[float] = []
# LR each optimizer step applies, read on entry to ``train_one_step`` (before
# ``optimizer.step()`` and the scheduler step that follows it); upstream logs
# only the post-step LR (fix-decoupled-lr-schedule D4).
_STEP_APPLIED_LRS: list[float] = []
# Per optimizer step: ``pg_clipfrac`` from the loss dict ``train_one_step``
# returns (last pipeline stage only) and the step's loss token count. Upstream
# returns only micro-batch-averaged metrics, so the token count is None.
_STEP_LOSSES: list[dict[str, float | None]] = []
_CLIPFRAC_KEYS = ("pg_clipfrac", "train/pg_clipfrac")
_RECORDER_INSTALLED = False


def install_grad_norm_recorder() -> bool:
    """Wrap upstream ``train_one_step`` to record the grad norm it returns.

    Idempotent; called from every state plugin, and the driver always exports
    the trainable state before the first training step.
    """

    global _RECORDER_INSTALLED
    if _RECORDER_INSTALLED:
        return True
    try:
        from miles.backends.megatron_utils import model as megatron_model
    except ImportError:
        return False
    original = getattr(megatron_model, "train_one_step", None)
    if original is None:
        return False

    def train_one_step(*args: Any, **kwargs: Any):
        _record_applied_lr(original, args, kwargs)
        _arm_grad_audit(original, args, kwargs)
        result = original(*args, **kwargs)
        try:
            norm = result[1]
            _STEP_GRAD_NORMS.append(float(norm.item() if hasattr(norm, "item") else norm))
        except (TypeError, IndexError, ValueError):
            pass
        _record_step_losses(result)
        return result

    megatron_model.train_one_step = train_one_step
    _RECORDER_INSTALLED = True
    return True


_VALUE_METRICS_INSTALLED = False


def explained_variance(returns: Any, values: Any, mask: Any = None) -> float | None:
    """``1 - Var(returns - values) / Var(returns)`` over the (masked) tokens.

    ``values`` are the critic's pre-update predictions (``batch["values"]``),
    ``returns`` the GAE returns Miles computed (``batch["returns"]``); yeto
    computes no return or advantage itself. None when Var(returns) is 0 or
    there are fewer than two tokens.
    """

    import torch

    returns = returns.detach().float().flatten()
    values = values.detach().float().flatten()
    if mask is not None:
        keep = mask.detach().flatten().bool()
        returns, values = returns[keep], values[keep]
    if returns.numel() < 2:
        return None
    variance = torch.var(returns, unbiased=False)
    if float(variance) == 0.0:
        return None
    return float(1.0 - torch.var(returns - values, unbiased=False) / variance)


def install_value_metrics_recorder() -> bool:
    """Wrap upstream ``value_loss_function`` to add explained variance (3.2).

    The metric joins the loss dict Miles reports (next to ``value_loss`` and
    ``value_clipfrac``), so it reaches ``train_one_step``'s result and the
    step-loss records like every other loss-dict scalar. Per micro-batch;
    how Miles reduces loss-dict scalars across micro-batches applies to it
    unchanged (GPU G1 checks the value is finite).
    """

    global _VALUE_METRICS_INSTALLED
    if _VALUE_METRICS_INSTALLED:
        return True
    try:
        from miles.backends.training_utils.loss_hub import losses
    except ImportError:
        return False
    original = getattr(losses, "value_loss_function", None)
    if original is None:
        return False

    def value_loss_function(args, batch, logits, sum_of_sample_mean):
        loss, reported = original(args, batch, logits, sum_of_sample_mean)
        try:
            import torch

            masks = batch.get("loss_masks")
            ev = explained_variance(
                torch.cat(batch["returns"], dim=0),
                torch.cat(batch["values"], dim=0),
                torch.cat(masks, dim=0) if masks else None,
            )
            if ev is not None:
                reported = {**reported, EXPLAINED_VARIANCE_KEY: torch.tensor(ev)}
        except (KeyError, RuntimeError, TypeError, ValueError):
            pass
        return loss, reported

    losses.value_loss_function = value_loss_function
    _VALUE_METRICS_INSTALLED = True
    return True


CRITIC_STATE_SUMMARY = f"{_PLUGIN_MODULE}.critic_state_summary"


def critic_state_summary(actor: Any) -> dict[str, Any]:
    """Plugin (critic process): this rank's trainable critic parameter specs and
    the content hash of their values (rl-algo-critic-family 4.1/4.4)."""

    import torch

    from yeto.rl.critic_state import critic_weights_sha256

    tensors: dict[str, Any] = {}
    specs = []
    for index, chunk in enumerate(actor.model):
        for name, parameter in chunk.named_parameters():
            if not parameter.requires_grad:
                continue
            key = f"{index}:{name}"
            specs.append((key, list(parameter.shape), str(parameter.dtype)))
            tensors[key] = parameter.detach()
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    return {"rank": rank, "specs": specs, "weights_sha256": critic_weights_sha256(tensors)}


def install_critic_recorders(actor: Any) -> bool:
    """Plugin: the per-step recorders in a critic process (idempotent)."""

    del actor
    return install_grad_norm_recorder() and install_value_metrics_recorder()


def _record_applied_lr(original: Any, args: tuple, kwargs: dict) -> float | None:
    import inspect

    from yeto.rl.applied_lr import optimizer_lr

    optimizer = inspect.signature(original).bind_partial(*args, **kwargs).arguments.get("optimizer")
    if optimizer is None:
        return None
    lr = optimizer_lr(optimizer)
    _STEP_APPLIED_LRS.append(lr)
    return lr


def _record_step_losses(result: Any) -> None:
    try:
        losses = result[0]
    except (TypeError, IndexError):
        return
    if not isinstance(losses, dict) or not losses:
        return  # not the last pipeline stage
    clipfrac = None
    for key in _CLIPFRAC_KEYS:
        if losses.get(key) is not None:
            raw = losses[key]
            clipfrac = float(raw.item() if hasattr(raw, "item") else raw)
            break
    scalars = {}
    for key, raw in losses.items():
        try:
            scalars[str(key)] = float(raw.item() if hasattr(raw, "item") else raw)
        except (TypeError, ValueError):
            continue
    _STEP_LOSSES.append({"pg_clipfrac": clipfrac, "loss_tokens": None, "metrics": scalars})


def step_losses(actor: Any) -> list[dict[str, float | None]]:
    """Per-step ``pg_clipfrac`` / loss token records since the last call (then cleared)."""

    del actor
    values = list(_STEP_LOSSES)
    _STEP_LOSSES.clear()
    return values


def applied_lrs(actor: Any) -> list[float]:
    """LRs applied by the optimizer steps since the last call (then cleared)."""

    del actor
    values = list(_STEP_APPLIED_LRS)
    _STEP_APPLIED_LRS.clear()
    return values


def _arm_grad_audit(original: Any, args: tuple, kwargs: dict) -> bool:
    """``YETO_RL_AUDIT_GRADS=1``: capture this step's pre-clip LoRA gradients (D12).

    Same capture point and semantics as the legacy hook (``yeto.rl.grad_audit``):
    entry of ``optimizer.step()``, after DP reduction, before clipping.
    """

    from types import SimpleNamespace

    from yeto.rl import grad_audit

    if not grad_audit.enabled():
        return False
    import inspect

    bound = inspect.signature(original).bind_partial(*args, **kwargs).arguments
    miles_args, model = bound.get("args"), bound.get("model")
    holder = SimpleNamespace(args=miles_args, model=model)

    def bindings():
        cached = _GRAD_BINDINGS.get(id(model))
        if cached is None:
            cached = tuple(
                grad_audit.GradBinding(grad_audit.canonical_grad_name(b.name), b.parameter, b.to_hf)
                for b in adapter_bindings(holder)
            )
            _GRAD_BINDINGS[id(model)] = cached
        return cached

    return grad_audit.arm(
        miles_args,
        bound.get("rollout_id", 0),
        bound.get("step_id", 0),
        bound.get("optimizer"),
        bindings,
        engine="ports",
    )


_GRAD_BINDINGS: dict[int, Any] = {}


def grad_norm(actor: Any) -> float:
    """Norm of the last step's gradients (Megatron's own reduction when available).

    Must run after ``train`` and before ``offload``. Clipping may have scaled
    the value; the driver's invariant only depends on it being zero or not.
    """

    import math

    import torch

    if _STEP_GRAD_NORMS:
        norms = list(_STEP_GRAD_NORMS)
        _STEP_GRAD_NORMS.clear()
        return max(norms)
    getter = getattr(actor.optimizer, "get_grad_norm", None)
    if callable(getter):
        value = getter()
        return float(value.item() if isinstance(value, torch.Tensor) else value)
    total = 0.0
    for chunk in actor.model:
        for p in chunk.parameters():
            if not p.requires_grad:
                continue
            g = getattr(p, "main_grad", None)
            if g is None:
                g = p.grad
            if g is not None:
                total += float(g.detach().float().pow(2).sum().item())
    return math.sqrt(total)
