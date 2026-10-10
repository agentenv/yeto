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

import os
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

CANONICAL_PREFIX = "base_model.model."
OPTIMIZER_MODES = ("preserve", "reset")

_PLUGIN_MODULE = "yeto.rl.adapters.miles.state_plugin"
EXPORT_STATE = f"{_PLUGIN_MODULE}.export_state"
# rl-publish-fastpath: export + hash on the trainer, return only the digests.
EXPORT_DIGEST = f"{_PLUGIN_MODULE}.export_digest"
WEIGHTS_VERSION = f"{_PLUGIN_MODULE}.weights_version"

# rl-publish-fastpath D3: version of the weights this trainer process holds.
# Kept in each trainer process (module state), bumped on every path that writes
# trainable weights: each optimizer step (wrapped train_one_step), apply_state,
# cut restore (cut_plugin).  The process id makes a rebuilt / restarted trainer
# (counter back to 0) never look like the old one.
_WEIGHTS_PROCESS_ID = __import__("uuid").uuid4().hex
_WEIGHTS_VERSION = 0


def bump_weights_version() -> int:
    global _WEIGHTS_VERSION
    _WEIGHTS_VERSION += 1
    return _WEIGHTS_VERSION


def current_weights_version() -> dict[str, Any]:
    return {"process_id": _WEIGHTS_PROCESS_ID, "version": _WEIGHTS_VERSION}


def weights_version(actor: Any) -> dict[str, Any] | None:
    """Plugin: the main rank's weights version (other ranks None, as export_state)."""

    install_grad_norm_recorder()
    return current_weights_version() if _is_main_rank(actor) else None
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


def _single_rank_dp(leaf: Any) -> bool:
    """True only when torch.distributed is up and the leaf's DP group has one rank
    (no collective needed). Not initialised: False, the caller keeps the
    ``_dp_all_reduce_sum`` path (a no-op there, patched in tests)."""
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return False
    try:
        return int(dist.get_world_size(group=getattr(leaf, "data_parallel_group", None))) == 1
    except (RuntimeError, ValueError, TypeError, AttributeError):
        return False  # unknown group: keep the collective path


def full_masters(optimizer: Any, parameters: Sequence[Any], *, all_reduce_sum=None, reduce=None,
                 device: Any = None) -> list[Any]:
    """FP32 masters in each parameter's shape; DP-sharded ones are gathered.

    Collective over each owning leaf's DP group when any parameter is
    DP-sharded (every rank must call it with the same parameters in the same
    order). ``reduce(flat, leaf)`` / ``all_reduce_sum(flat)`` replace the
    all-reduce (tests).

    One flat FP32 buffer per leaf holds the gathered masters (each returned
    master is a view of it); no second concatenated copy is made
    (s19-compaction-g3-20261010a: the critic export ran out of GPU memory on
    that copy). ``device``: where the gather buffers live (default: each
    parameter's device). A leaf whose DP group has one rank needs no
    collective, so a CPU buffer is allowed there; a CPU buffer with a
    multi-rank group is refused (NCCL reduces GPU tensors only).
    """
    import torch

    has_dist, owned = distributed_ranges(optimizer)
    out: list[Any] = [None] * len(parameters)
    pending: dict[int, tuple[Any, list[int]]] = {}
    for i, param in enumerate(parameters):
        if has_complete_master(param):
            out[i] = master_of(param)
            continue
        if not has_dist:
            out[i] = master_of(param)  # raises: no master anywhere
            continue
        leaf = _leaf_for(optimizer, param, owned)
        pending.setdefault(id(leaf), (leaf, []))[1].append(i)
    # Leaf order is the same on every rank of a DP group, so the per-group
    # collectives are issued in the same order.
    order = {id(leaf): k for k, leaf in enumerate(_optimizer_leaves(optimizer))} if pending else {}
    for _, (leaf, items) in sorted(pending.items(), key=lambda kv: order.get(kv[0], 0)):
        hooked = reduce is not None or all_reduce_sum is not None
        where = device if device is not None else parameters[items[0]].device
        if (not hooked and torch.device(where).type == "cpu"
                and parameters[items[0]].device.type != "cpu" and not _single_rank_dp(leaf)):
            raise StatePluginError("CPU gather buffer with a multi-rank DP group")
        flat = torch.zeros(sum(parameters[i].numel() for i in items), dtype=torch.float32, device=where)
        offset = 0
        for i in items:
            param = parameters[i]
            n = param.numel()
            full = flat[offset : offset + n]
            if id(param) in owned:
                _, _, start, end = owned[id(param)]
                shard = owned[id(param)][0]._get_main_param_and_optimizer_states(param)["param"]
                if shard.dtype != torch.float32 or shard.numel() != end - start:
                    raise StatePluginError("distributed-optimizer main shard does not match its range")
                full[start:end].copy_(shard.detach().reshape(-1))
            out[i] = full.view(param.shape)
            offset += n
        if reduce is not None:
            reduce(flat, leaf)
        elif all_reduce_sum is not None:
            all_reduce_sum(flat)
        elif not _single_rank_dp(leaf):
            _dp_all_reduce_sum(leaf, flat)
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


def export_digest(
    actor: Any, *, policy_version: int, base_model_revision: str, lora_config_hash: str,
) -> dict[str, Any] | None:
    """rl-publish-fastpath: :func:`export_state` on the trainer, but return only the
    publication digests (policy tensor hash, payload hash/bytes, specs) instead of
    the tensors.  The hash definitions are those of ``yeto.rl.core.policy_tensor_hash``
    and ``publish.payload_digest`` (see ``policy_digest``); only where they run moves."""

    from yeto.rl.engine.policy_digest import digest_canonical_tensors, layout_hash_of

    import time

    started = time.monotonic()
    exported = export_state(actor, policy_version=policy_version)
    export_seconds = round(time.monotonic() - started, 3)
    if exported is None:
        return None
    tensors = exported["tensors"]
    specs = [(name, tuple(int(d) for d in value.shape)) for name, value in sorted(tensors.items())]
    digest = digest_canonical_tensors(
        tensors,
        base_model_revision=base_model_revision,
        lora_config_hash=lora_config_hash,
        layout_hash=layout_hash_of(specs),
    )
    return {"policy_version": int(policy_version), "digest": digest.to_wire(),
            "export_seconds": export_seconds, "weights": current_weights_version()}


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


def _torch_group(group: Any) -> Any:
    """The registered torch ProcessGroup behind a Miles ``ReloadableProcessGroup``."""

    inner = group.__dict__.get("group") if hasattr(group, "__dict__") else None
    return inner if inner is not None else group


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
    # Miles wraps groups in ReloadableProcessGroup; torch's gather_object resolves
    # ``dst`` through the group registry, which only knows the inner group.
    group = _torch_group(mpu.get_pipeline_model_parallel_group())
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
    bump_weights_version()  # rl-publish-fastpath: written (or partly) from here on
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
        _EV_STATS.clear()
        _POLICY_STATS.clear()
        try:
            result = original(*args, **kwargs)
        finally:
            bump_weights_version()  # rl-publish-fastpath: any step may have written weights
        try:
            norm = result[1]
            _STEP_GRAD_NORMS.append(float(norm.item() if hasattr(norm, "item") else norm))
        except (TypeError, IndexError, ValueError):
            pass
        _record_step_losses(result)
        return result

    megatron_model.train_one_step = train_one_step
    _RECORDER_INSTALLED = True
    # Opt-in until verified on GPU (hot path: per-micro-batch tensor reductions).
    if os.environ.get("YETO_POLICY_METRICS") == "1":
        install_policy_metrics_recorder()
    return True


# rl-algo-supplement follow-up (S19 #13, 4.7 / 5.2): per optimizer step, sufficient
# statistics of the advantages Miles trains on and of the PPO ratio against the clip
# bounds.  Observation only: the wrapped functions' return values are passed through
# untouched.  Per rank (no data-parallel / CP reduction).  Reset on entry to
# train_one_step; written into the step-loss record by ``_record_step_losses``.
_POLICY_STATS: dict[str, float] = {}
_POLICY_METRICS_INSTALLED = False
POLICY_STAT_KEYS = (
    "yeto/adv_tokens", "yeto/adv_token_mean", "yeto/adv_token_var",
    "yeto/adv_samples", "yeto/adv_sample_mean", "yeto/adv_sample_var",
    "yeto/ratio_tokens", "yeto/ratio_pos_adv_tokens", "yeto/ratio_neg_adv_tokens",
    "yeto/ratio_above_high_pos_adv", "yeto/ratio_below_low_neg_adv",
    "yeto/ratio_min", "yeto/ratio_max", "yeto/clip_eps_low", "yeto/clip_eps_high",
    "yeto/clipfrac_recomputed",
)


def _add(key: str, value: float) -> None:
    _POLICY_STATS[key] = _POLICY_STATS.get(key, 0.0) + float(value)


def accumulate_advantage_stats(advantages: Any, loss_masks: Any = None) -> None:
    """Token- and sample-level sums of the (loss-masked) advantages of one micro-batch.

    ``advantages`` / ``loss_masks``: per-sample 1-D tensors (Miles ``batch`` layout).
    """
    for i, adv in enumerate(advantages):
        adv = adv.detach().double().flatten()
        if loss_masks is not None and i < len(loss_masks) and loss_masks[i] is not None:
            mask = loss_masks[i].detach().flatten().bool()
            if mask.numel() == adv.numel():
                adv = adv[mask]
        if adv.numel() == 0:
            continue
        _add("_tok_n", adv.numel())
        _add("_tok_s", float(adv.sum()))
        _add("_tok_s2", float((adv * adv).sum()))
        m = float(adv.mean())
        _add("_smp_n", 1)
        _add("_smp_s", m)
        _add("_smp_s2", m * m)


def accumulate_ratio_stats(ppo_kl: Any, advantages: Any, eps_clip: float, eps_clip_high: float) -> None:
    """Counts behind ``pg_clipfrac`` for the standard PPO clip (tokens with A != 0 only:
    Miles zeroes ppo_kl and A on masked tokens, so those never count)."""
    import torch

    with torch.no_grad():
        kl = ppo_kl.detach().double().flatten()
        adv = advantages.detach().double().flatten()
        ratio = torch.exp(-kl)
        pos, neg = adv > 0, adv < 0
        active = pos | neg
        _add("_r_n", int(active.sum()))
        _add("_r_pos", int(pos.sum()))
        _add("_r_neg", int(neg.sum()))
        _add("_r_hi", int((pos & (ratio > 1 + eps_clip_high)).sum()))
        _add("_r_lo", int((neg & (ratio < 1 - eps_clip)).sum()))
        if bool(active.any()):
            r = ratio[active]
            lo, hi = float(r.min()), float(r.max())
            _POLICY_STATS["_r_min"] = min(_POLICY_STATS.get("_r_min", lo), lo)
            _POLICY_STATS["_r_max"] = max(_POLICY_STATS.get("_r_max", hi), hi)
        _POLICY_STATS["_eps_lo"] = float(eps_clip)
        _POLICY_STATS["_eps_hi"] = float(eps_clip_high)


def step_policy_stats(stats: dict[str, float] | None = None) -> dict[str, float]:
    """The step's ``yeto/*`` advantage and ratio metrics (population variances);
    keys without data are omitted, never filled in."""
    st = _POLICY_STATS if stats is None else stats
    out: dict[str, float] = {}
    n = st.get("_tok_n", 0.0)
    if n > 0:
        mean = st["_tok_s"] / n
        out.update({"yeto/adv_tokens": n, "yeto/adv_token_mean": mean,
                    "yeto/adv_token_var": max(st["_tok_s2"] / n - mean * mean, 0.0)})
    k = st.get("_smp_n", 0.0)
    if k > 0:
        mean = st["_smp_s"] / k
        out.update({"yeto/adv_samples": k, "yeto/adv_sample_mean": mean,
                    "yeto/adv_sample_var": max(st["_smp_s2"] / k - mean * mean, 0.0)})
    if "_eps_lo" in st:
        rn = st.get("_r_n", 0.0)
        out.update({"yeto/ratio_tokens": rn, "yeto/ratio_pos_adv_tokens": st.get("_r_pos", 0.0),
                    "yeto/ratio_neg_adv_tokens": st.get("_r_neg", 0.0),
                    "yeto/ratio_above_high_pos_adv": st.get("_r_hi", 0.0),
                    "yeto/ratio_below_low_neg_adv": st.get("_r_lo", 0.0),
                    "yeto/clip_eps_low": st["_eps_lo"], "yeto/clip_eps_high": st["_eps_hi"]})
        if "_r_min" in st:
            out.update({"yeto/ratio_min": st["_r_min"], "yeto/ratio_max": st["_r_max"]})
        if rn > 0:
            out["yeto/clipfrac_recomputed"] = (st.get("_r_hi", 0.0) + st.get("_r_lo", 0.0)) / rn
    return out


def install_policy_metrics_recorder() -> bool:
    """Wrap Miles ``policy_loss_function`` (advantage stats) and the
    ``compute_policy_loss`` it calls (ratio vs clip counts); idempotent."""

    global _POLICY_METRICS_INSTALLED
    if _POLICY_METRICS_INSTALLED:
        return True
    try:
        from miles.backends.training_utils.loss_hub import losses
    except ImportError:
        return False
    original_fn = getattr(losses, "policy_loss_function", None)
    original_cpl = getattr(losses, "compute_policy_loss", None)
    if original_fn is None:
        return False

    def policy_loss_function(args, batch, logits, sum_of_sample_mean):
        try:
            accumulate_advantage_stats(batch["advantages"], batch.get("loss_masks"))
        except (KeyError, RuntimeError, TypeError, ValueError, AttributeError):
            pass
        return original_fn(args, batch, logits, sum_of_sample_mean)

    losses.policy_loss_function = policy_loss_function
    if original_cpl is not None:
        def compute_policy_loss(ppo_kl, advantages, eps_clip, eps_clip_high, *rest, **kw):
            result = original_cpl(ppo_kl, advantages, eps_clip, eps_clip_high, *rest, **kw)
            try:
                accumulate_ratio_stats(ppo_kl, advantages, eps_clip, eps_clip_high)
            except (RuntimeError, TypeError, ValueError):
                pass
            return result

        losses.compute_policy_loss = compute_policy_loss
    _POLICY_METRICS_INSTALLED = True
    return True


_VALUE_METRICS_INSTALLED = False
# Explained-variance sufficient statistics over every micro-batch of the
# current optimizer step (critic process): n, sum G, sum G^2, sum R, sum R^2
# with R = G - v.  Per-micro-batch EV is undefined at the default
# --micro-batch-size 1 under PPO with gamma = lambd = 1 and no KL reward: one
# sample's returns are one constant, Var(G) = 0 (s13-g1-modal-20261007b, every
# round's explained_variance missing).  Reset on entry to train_one_step.
_EV_STATS: list[float] = []


def _accumulate_ev_stats(returns: Any, values: Any, mask: Any = None) -> None:
    returns = returns.detach().double().flatten()
    values = values.detach().double().flatten()
    if mask is not None:
        keep = mask.detach().flatten().bool()
        returns, values = returns[keep], values[keep]
    residual = returns - values
    add = [float(returns.numel()), float(returns.sum()), float((returns * returns).sum()),
           float(residual.sum()), float((residual * residual).sum())]
    if not _EV_STATS:
        _EV_STATS.extend([0.0] * 5)
    for i, value in enumerate(add):
        _EV_STATS[i] += value


# Var(G) at or below this fraction of mean(G^2) (floor 1) is rounding noise,
# not return spread. Miles' returns are advantages + values in fp32, so a
# step whose rewards are all 0 leaves |G| ~ 1e-8 instead of exactly 0; EV then
# divides by ~1e-16 (s14-forkg1-sao-20261007a round 5: EV -3.3e12).
_EV_RELATIVE_VARIANCE_FLOOR = 1e-8


def _degenerate_return_variance(var_g: float, mean_g2: float) -> bool:
    return not var_g > _EV_RELATIVE_VARIANCE_FLOOR * max(1.0, mean_g2)


def step_explained_variance(stats: list[float] | None = None) -> float | None:
    """EV over all tokens of the step's micro-batches (population variances);
    None without >= 2 tokens or when Var(G) is 0."""

    n, sg, sg2, sr, sr2 = (stats if stats is not None else _EV_STATS) or [0.0] * 5
    if n < 2:
        return None
    var_g = sg2 / n - (sg / n) ** 2
    if _degenerate_return_variance(var_g, sg2 / n):
        return None
    var_r = max(sr2 / n - (sr / n) ** 2, 0.0)
    return 1.0 - var_r / var_g


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
    if _degenerate_return_variance(float(variance), float((returns.double() ** 2).mean())):
        return None
    return float(1.0 - torch.var(returns - values, unbiased=False) / variance)


def install_value_metrics_recorder() -> bool:
    """Wrap upstream ``value_loss_function`` to collect explained variance (3.2).

    Each micro-batch adds its (masked) returns / old values to the step's
    sufficient statistics; ``_record_step_losses`` writes the step-level
    ``explained_variance`` into the step-loss record next to ``value_loss``.
    Per-rank (no data-parallel reduction); the loss dict is left unchanged.
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
            _accumulate_ev_stats(
                torch.cat(batch["returns"], dim=0),
                torch.cat(batch["values"], dim=0),
                torch.cat(masks, dim=0) if masks else None,
            )
        except (KeyError, RuntimeError, TypeError, ValueError):
            pass
        return loss, reported

    losses.value_loss_function = value_loss_function
    _VALUE_METRICS_INSTALLED = True
    return True


CRITIC_STATE_SUMMARY = f"{_PLUGIN_MODULE}.critic_state_summary"


def critic_state_summary(actor: Any) -> dict[str, Any]:
    """Plugin (critic process): this rank's trainable critic parameter specs and
    the content hash of their values (rl-algo-critic-family 4.1/4.4).

    The colocated critic is asleep (memory paused) between its train steps;
    reading paused parameters fails with ``CUDA error: invalid argument``
    (s13-g1-modal-20261007a), so wake it like the other plugins."""

    with trainer_resident(actor):
        return _critic_state_summary(actor)


def _critic_state_summary(actor: Any) -> dict[str, Any]:
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
    return {"rank": _rank(), "specs": specs, "weights_sha256": critic_weights_sha256(tensors)}


EXPORT_CRITIC_TENSORS = f"{_PLUGIN_MODULE}.export_critic_tensors"
IMPORT_CRITIC_TENSORS = f"{_PLUGIN_MODULE}.import_critic_tensors"
SAVE_CRITIC_CUT = f"{_PLUGIN_MODULE}.save_critic_cut"
RESTORE_CRITIC_CUT = f"{_PLUGIN_MODULE}.restore_critic_cut"


def _critic_parameters(actor: Any) -> dict[str, Any]:
    """``index:name`` -> trainable critic parameter of this rank (same keys as
    critic_state_summary)."""

    out = {}
    for index, chunk in enumerate(actor.model):
        for name, parameter in chunk.named_parameters():
            if parameter.requires_grad:
                out[f"{index}:{name}"] = parameter
    return out


def _rank() -> int:
    import torch

    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


# rl-algo-critic-family (user decision 2026-10-07): the critic's two-island
# state is its FP32 optimizer masters, not the (bf16) model parameters.
# Export reads the masters (Megatron DistributedOptimizer: this rank's main
# shard, gathered over the owning DP group by ``full_masters``; complete
# ``main_param``; or the FP32 parameter itself when it is its own master).
# Write-back writes the masters (``write_masters``) and regenerates every
# low-precision model parameter from them (the cast the optimizer's
# main->model copy does after a step), so the next optimizer step continues
# from the average instead of overwriting it with the stale masters. Hashes
# (syncer channel, write-back check, round-cut) are over the FP32 masters.
# A low-precision parameter without any FP32 master is refused.
#
# Critic optimizer state (moments, step counts, LR scheduler) is kept across
# strict-avg rounds -- only the weights are replaced; the actor's optimizer is
# reset every round. This is the first-version choice (design D4).


def _critic_masters(actor: Any, params: Mapping[str, Any]) -> dict[str, Any]:
    """``key -> FP32 master`` (parameter shape) for the sorted critic keys.

    Collective over the DP group when masters are DistributedOptimizer
    shards: every rank calls it with the same keys in the same order.
    """

    keys = sorted(params)
    optimizer = getattr(actor, "optimizer", None)
    leaves = [leaf for leaf in _optimizer_leaves(optimizer) if _is_distributed(leaf)] if optimizer is not None else []
    # Single-rank DP groups need no collective: gather on the CPU, so the
    # export/import does not need a full FP32 critic copy on the GPU
    # (s19-compaction-g3-20261010a: OOM with the actor still resident).
    device = "cpu" if leaves and all(_single_rank_dp(leaf) for leaf in leaves) else None
    masters = full_masters(optimizer, [params[k] for k in keys], device=device)
    return dict(zip(keys, masters, strict=True))


def _write_critic_masters(actor: Any, params: Mapping[str, Any], targets: Mapping[str, Any]) -> None:
    """Write FP32 ``targets`` into the critic masters, then set each
    low-precision model parameter from its master."""

    import torch

    keys = sorted(params)
    plist = [params[k] for k in keys]
    # Targets stay where they are (CPU from the syncer); copy_ moves each one.
    tlist = [targets[k].to(dtype=torch.float32) for k in keys]
    write_masters(getattr(actor, "optimizer", None), plist, tlist)
    for param, target in zip(plist, tlist, strict=True):
        if param.dtype != torch.float32:
            # complete-master params: write_masters only set the master;
            # sharded ones were already set (idempotent cast).
            param.data.copy_(target.to(dtype=param.dtype))


def _masters_cpu(masters: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    return {k: m.detach().to("cpu", torch.float32).contiguous().clone() for k, m in masters.items()}


def _export_critic_tensors(actor: Any) -> dict[str, Any]:
    """Plugin (critic process, 4.2.2): this rank's full-parameter critic FP32
    masters (CPU copies) with their content hash, for the critic syncer channel."""

    import torch

    from yeto.rl.critic_state import critic_weights_sha256

    with torch.no_grad():
        tensors = _masters_cpu(_critic_masters(actor, _critic_parameters(actor)))
    return {"rank": _rank(), "tensors": tensors, "weights_sha256": critic_weights_sha256(tensors)}


def _import_critic_tensors(actor: Any, *, by_rank: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Plugin (critic process, 4.2.2): ``by_rank[rank] = {"tensors", "sha256"}``;
    write the averaged critic tensors of this rank into the FP32 masters,
    regenerate the model parameters from them, then re-hash the masters
    against ``sha256`` and check every model parameter equals its master cast.

    A name/shape mismatch is refused before any write; a hash mismatch after the
    write is returned as a refusal (the caller treats the critic as dirty and
    rolls back to the last committed round)."""

    import torch

    from yeto.rl.critic_state import critic_weights_sha256

    rank = _rank()
    if rank not in by_rank:
        return {"refused": f"no critic tensors for rank {rank}", "rank": rank}
    tensors, expect_sha256 = by_rank[rank]["tensors"], by_rank[rank]["sha256"]
    params = _critic_parameters(actor)
    if set(params) != set(tensors):
        return {"refused": f"critic tensor names differ: missing {sorted(set(params) - set(tensors))[:4]}, "
                           f"unexpected {sorted(set(tensors) - set(params))[:4]}"}
    for key, value in tensors.items():
        if tuple(value.shape) != tuple(params[key].shape):
            return {"refused": f"critic tensor {key} shape {tuple(value.shape)} != {tuple(params[key].shape)}"}
    if critic_weights_sha256(tensors) != expect_sha256:
        return {"refused": "incoming critic tensors differ from their announced hash"}
    try:
        with torch.no_grad():
            _write_critic_masters(actor, params, tensors)
            masters = _critic_masters(actor, params)
            written = _masters_cpu(masters)
            stale = [k for k, p in params.items() if p.dtype != torch.float32
                     and not torch.equal(p.detach(), masters[k].to(device=p.device, dtype=p.dtype).view(p.shape))]
    except StatePluginError as error:
        return {"refused": f"critic masters: {error}", "rank": rank}
    got = critic_weights_sha256(written)
    if got != expect_sha256:
        return {"refused": f"written critic hash {got[:12]} != expected {expect_sha256[:12]}",
                "rank": rank, "weights_sha256": got}
    if stale:
        return {"refused": f"critic model parameters not regenerated from masters: {stale[:4]}",
                "rank": rank, "weights_sha256": got}
    return {"rank": rank, "weights_sha256": got}


def _save_critic_cut(actor: Any, *, directory: str, round_id: int) -> dict[str, Any]:
    """Plugin (critic process, 4.3): this rank's critic FP32 masters + optimizer +
    LR-scheduler state into ``directory/rank-<r>/`` via CriticCheckpointStore
    (manifest committed last). The weights hash is over the FP32 masters, the
    same as the critic syncer channel."""

    import torch

    from yeto.rl.critic_state import CriticCheckpointStore

    rank = _rank()
    with torch.no_grad():
        weights = _masters_cpu(_critic_masters(actor, _critic_parameters(actor)))
    scheduler = getattr(actor, "opt_param_scheduler", None)
    manifest = CriticCheckpointStore(os.path.join(directory, f"rank-{rank}")).save(
        round_id=int(round_id), weights=weights,
        optimizer={"optimizer": actor.optimizer.state_dict(),
                   "scheduler": scheduler.state_dict() if scheduler is not None else None},
    )
    return {"rank": rank, **manifest}


def _restore_critic_cut(actor: Any, *, directory: str, actor_round: int,
                       critic_round: int) -> dict[str, Any]:
    """Plugin (critic process, 4.3): load this rank's critic round (refuses a
    critic round != actor round and a manifest mismatch), load the optimizer
    state, then write the saved FP32 masters (after the optimizer load, so they
    win) and regenerate the model parameters; re-hash the masters."""

    import torch

    from yeto.rl.critic_state import CriticCheckpointStore, CriticStateError, critic_weights_sha256

    rank = _rank()
    try:
        weights, state = CriticCheckpointStore(os.path.join(directory, f"rank-{rank}")).restore(
            actor_round=int(actor_round), critic_round=int(critic_round))
    except (CriticStateError, OSError, KeyError, ValueError) as error:
        return {"refused": f"{type(error).__name__}: {error}", "rank": rank}
    params = _critic_parameters(actor)
    if set(params) != set(weights):
        return {"refused": "critic checkpoint tensor names differ from the running critic", "rank": rank}
    actor.optimizer.load_state_dict(state["optimizer"])
    scheduler = getattr(actor, "opt_param_scheduler", None)
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    try:
        with torch.no_grad():
            _write_critic_masters(actor, params, weights)
            written = _masters_cpu(_critic_masters(actor, params))
    except StatePluginError as error:
        return {"refused": f"critic masters: {error}", "rank": rank}
    return {"rank": rank, "weights_sha256": critic_weights_sha256(written)}


def export_critic_tensors(actor: Any) -> dict[str, Any]:
    with trainer_resident(actor):
        return _export_critic_tensors(actor)


def import_critic_tensors(actor: Any, **kwargs: Any) -> dict[str, Any]:
    with trainer_resident(actor):
        return _import_critic_tensors(actor, **kwargs)


def save_critic_cut(actor: Any, **kwargs: Any) -> dict[str, Any]:
    with trainer_resident(actor):
        return _save_critic_cut(actor, **kwargs)


def restore_critic_cut(actor: Any, **kwargs: Any) -> dict[str, Any]:
    with trainer_resident(actor):
        return _restore_critic_cut(actor, **kwargs)


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
    ev = step_explained_variance()
    if ev is not None:
        scalars[EXPLAINED_VARIANCE_KEY] = ev
    scalars.update(step_policy_stats())
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
