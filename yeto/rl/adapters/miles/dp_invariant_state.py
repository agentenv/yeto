"""DP-invariant LoRA training state, vendored from the ports-v1 Miles fork.

Source: michaellchung/miles 28bc6cc24 ``miles/backends/megatron_utils/lora/dp_invariant_state.py``
(fork M5, fb04d6ff .. ed5f64fd7), copied verbatim below this note. The k31
Miles lineage does not carry M5 (upstream's adapter library replaced the
LoRA checkpoint format), but the cut plugin's name-keyed optimizer state
export/restore and RNG capture (``cut_plugin.MilesCutBackend``) are built on
these helpers for LoRA and full-parameter cuts alike, so yeto keeps its own
copy. Everything here is pure torch/CPU; Megatron optimizers are reached by
duck typing (the private HybridDeviceOptimizer / DistributedOptimizer methods
it touches are probed by ``runtime_manifest`` under ``Cut.dp_invariant_state``).

DP-invariant LoRA training state: optimizer state keyed by parameter name, plus RNG capture/restore.

The default LoRA checkpoint stores ``optimizer.state_dict()`` per global rank, whose ``state`` is keyed by the
optimizer's positional parameter index. That index depends on how the optimizer was built on that rank, so a
checkpoint written at one DP size cannot be loaded at another. Here the state is re-keyed by parameter name and
written once per (tp, pp, dp) coordinate together with the range of the parameter that rank owns; a loader
gathers every DP shard of its own (tp, pp) coordinate into full tensors and slices out the ranges it owns now, so
the DP size may change while TP/PP stay fixed. The fp32 main copies of bf16 training (Float16Optimizer) and the
DP-sharded main params/state of Megatron's DistributedOptimizer are covered (see ``optimizer_slots``).

Everything in this module is pure torch/CPU; Megatron optimizers are accessed by duck typing.
"""

from __future__ import annotations

import copy
import random
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

NAMED_STATE_FORMAT = "lora_named_optimizer_v2"

# What a load does with RNG. "exact": restore this rank's saved RNG and refuse a DP change or a missing coordinate.
# "keep_on_dp_change": restore when the DP size is unchanged, otherwise keep the fresh RNG of this process.
RNG_POLICIES = ("exact", "keep_on_dp_change")


class DpInvariantStateError(RuntimeError):
    """A configuration or checkpoint the DP-invariant LoRA state cannot handle."""


_NAMED_FILE_RE = re.compile(r"^training_state_named_tp(\d+)_pp(\d+)_dp(\d+)\.pt$")


def named_training_state_filename(*, tp_rank: int, pp_rank: int, dp_rank: int) -> str:
    return f"training_state_named_tp{tp_rank}_pp{pp_rank}_dp{dp_rank}.pt"


def dp_invariant_adapter_filename(*, tp_rank: int, pp_rank: int) -> str:
    return f"adapter_dp_invariant_tp{tp_rank}_pp{pp_rank}.pt"


def list_named_training_state_files(checkpoint_dir: Path, *, tp_rank: int, pp_rank: int) -> dict[int, Path]:
    """dp_rank -> file for every DP shard written under this (tp, pp) coordinate."""
    found: dict[int, Path] = {}
    for path in Path(checkpoint_dir).iterdir():
        if (m := _NAMED_FILE_RE.match(path.name)) and (int(m[1]), int(m[2])) == (tp_rank, pp_rank):
            found[int(m[3])] = path
    return dict(sorted(found.items()))


# ---------------------------------------------------------------------------
# optimizer state <-> parameter names (with DP-shard ranges)
# ---------------------------------------------------------------------------
#
# Every optimizer is reduced to "slots": one per model parameter this rank updates, holding the range
# [start, end) of the flattened model parameter that this rank's optimizer owns, plus accessors for the fp32 main
# copy ("param") and the per-element optimizer state (exp_avg, ...). Three optimizer layouts are supported:
#
# * a plain torch optimizer (or Megatron FP32Optimizer): main param == model param, full range;
# * Megatron Float16OptimizerWithFloat16Params (bf16/fp16 without DistOpt): the torch optimizer holds fp32 main
#   copies; they are matched to model params through ``float16_groups``/``fp32_from_float16_groups``; full range;
# * Megatron DistributedOptimizer: each DP rank owns a contiguous slice of every bucket, i.e. a sub-range of some
#   parameters (``gbuf_ranges[..]["param_map"][model_param]["param"]``); main shard + state come from
#   ``_get_main_param_and_optimizer_states``.
# A DistributedOptimizer with --optimizer-cpu-offload (Megatron requires --use-precision-aware-optimizer with it) runs a
# HybridDeviceOptimizer (HDO) whose fp32 masters ("master_param") and moments are standalone tensors, mostly on the
# host; it is covered when every optimizer dtype is fp32 (nothing scaled), see ``_hybrid_offload``.
# ChainedOptimizer is unwrapped into its members. Saving writes each rank's slots keyed by parameter name with their
# ranges; loading merges every DP shard of one (tp, pp) coordinate into full flat tensors (gather) and slices out
# the ranges the new rank owns (reshard). No collective is needed: the checkpoint directory is the gather point.


class _Slot:
    __slots__ = ("name", "model_param", "start", "end", "get", "set", "group", "leaf")

    def __init__(self, name, model_param, start, end, get, set_, group, leaf=None):
        self.name, self.model_param, self.start, self.end = name, model_param, start, end
        self.get, self.set, self.group, self.leaf = get, set_, group, leaf


def _optimizer_leaves(optimizer: Any) -> list[Any]:
    chained = getattr(optimizer, "chained_optimizers", None)
    if chained is not None:
        return [leaf for member in chained for leaf in _optimizer_leaves(member)]
    if getattr(optimizer, "is_stub_optimizer", False):
        # Megatron's placeholder for a parameter group with nothing to train (e.g. the
        # dense group of a partial-experts run, every dense weight frozen): no slots
        return []
    return [optimizer]


def _is_distributed(leaf: Any) -> bool:
    return hasattr(leaf, "gbuf_ranges") and hasattr(leaf, "model_param_group_index_map")


def _torch_slots(torch_optimizer: torch.optim.Optimizer, main_to_model: Mapping[int, torch.nn.Parameter], name_of):
    slots = []
    for group in torch_optimizer.param_groups:
        for main in group["params"]:
            model_param = main_to_model.get(id(main), main)
            if id(model_param) not in name_of:
                raise NotImplementedError(
                    f"optimizer holds a tensor of shape {tuple(main.shape)} that maps to no model parameter"
                )

            def get(main=main):
                tensors = {"param": main}
                tensors.update(
                    {k: v for k, v in torch_optimizer.state.get(main, {}).items() if isinstance(v, torch.Tensor)}
                )
                return tensors

            def set_(tensors, main=main):
                state = torch_optimizer.state[main]
                for key, value in tensors.items():
                    if key == "param":
                        main.data.copy_(value.reshape(main.shape))
                    elif value.dim() == 0:
                        state[key] = value.clone()
                    else:
                        state[key] = value.reshape(main.shape).to(device=main.device).clone()

            slots.append(_Slot(name_of[id(model_param)], model_param, 0, model_param.numel(), get, set_, group))
    return slots


# HybridDeviceOptimizer construction settings Megatron passes as optimizer defaults; torch copies them into every
# param group, but they are wiring (classes, flags), not hyperparameters of the update.
_HDO_WIRING_KEYS = frozenset(
    {
        "offload_fraction",
        "cpu_optimizer_cls",
        "gpu_optimizer_cls",
        "param_update_in_fp32",
        "pin_cpu_grads",
        "pin_cpu_params",
        "overlap_cpu_optimizer_d2h_h2d",
    }
)


def _hybrid_offload(leaf: Any) -> bool:
    """True when ``leaf`` is a DistributedOptimizer over a HybridDeviceOptimizer (``--optimizer-cpu-offload``)."""
    try:
        from megatron.core.optimizer.cpu_offloading import HybridDeviceOptimizer
    except ImportError:  # no Megatron: no HDO can exist
        return False
    return isinstance(getattr(leaf, "optimizer", None), HybridDeviceOptimizer)


def _check_precision(leaf: Any) -> None:
    """The precision-aware path is supported only as Megatron's CPU-offload carrier with fp32 state.

    Under ``--optimizer-cpu-offload`` the HDO keeps standalone masters and moments, read and written unscaled by
    ``_get/_set_main_param_and_optimizer_states``; on GPU the precision-aware path keeps scaled or low-precision state
    (e.g. int16 master remainders inside TE FusedAdam), which this format does not represent.
    """
    config = getattr(leaf, "config", None)
    if not getattr(config, "use_precision_aware_optimizer_no_fp8_or_ds_fp8", False):
        return
    if not _hybrid_offload(leaf):
        raise NotImplementedError("DP-invariant LoRA state does not support the precision-aware optimizer on GPU")
    dtypes = {
        name: getattr(config, name, torch.float32) for name in ("main_params_dtype", "exp_avg_dtype", "exp_avg_sq_dtype")
    }
    low = sorted(name for name, dtype in dtypes.items() if dtype != torch.float32)
    if low:
        raise NotImplementedError(f"DP-invariant state with --optimizer-cpu-offload needs fp32 {low}")


def _distributed_slots(leaf: Any, name_of) -> list[_Slot]:
    _check_precision(leaf)
    hybrid = _hybrid_offload(leaf)
    slots = []
    for gbuf_range_maps in leaf.gbuf_ranges:
        for per_bucket in gbuf_range_maps.values():
            for bucket_range_map in per_bucket:
                for model_param, range_map in bucket_range_map["param_map"].items():
                    if id(model_param) not in name_of:
                        raise NotImplementedError(
                            f"distributed optimizer holds a parameter of shape {tuple(model_param.shape)} that is not "
                            f"a named model parameter"
                        )
                    group_index, group_order = leaf.model_param_group_index_map[model_param]

                    def get(model_param=model_param, group_index=group_index, group_order=group_order):
                        if hybrid:
                            sharded = leaf.optimizer.param_groups[group_index]["params"][group_order]
                            return _hdo_stepped_state(leaf.optimizer, sharded)
                        return _read_distributed_state(leaf, model_param)

                    def set_(tensors, model_param=model_param):
                        current = _read_distributed_state(leaf, model_param)
                        missing = current.keys() - tensors.keys()
                        if missing:
                            raise DpInvariantStateError(
                                f"saved state of {name_of[id(model_param)]!r} lacks {sorted(missing)}"
                            )
                        # Megatron copies only the keys the target already holds, so an extra saved key would be
                        # dropped without a word (e.g. exp_avg onto a state left empty by an earlier read)
                        extra = tensors.keys() - current.keys()
                        if extra:
                            raise DpInvariantStateError(
                                f"optimizer state of {name_of[id(model_param)]!r} has no {sorted(extra)} to load the "
                                f"saved values into: the checkpoint was most likely written by a different optimizer "
                                f"implementation (e.g. torch Adam vs TE FusedAdam, which keep 'step' differently); "
                                f"resume with the optimizer that saved it"
                            )
                        leaf._set_main_param_and_optimizer_states(model_param, tensors)

                    param_range = range_map["param"]
                    slots.append(
                        _Slot(
                            name_of[id(model_param)],
                            model_param,
                            param_range.start,
                            param_range.end,
                            get,
                            set_,
                            leaf.optimizer.param_groups[group_index],
                            leaf,
                        )
                    )
    return slots


def _main_param_of(leaf: Any, model_param) -> Any:
    group_index, group_order = leaf.model_param_group_index_map[model_param]
    return leaf.optimizer.param_groups[group_index]["params"][group_order]


def _read_distributed_state(leaf: Any, model_param) -> dict[str, torch.Tensor]:
    """``_get_main_param_and_optimizer_states`` without its side effect: the read goes through the optimizer's
    ``state`` defaultdict, which would leave an empty entry for a parameter that has no state yet."""
    state = leaf.optimizer.state
    main = _main_param_of(leaf, model_param)
    had_entry = main in state
    try:
        return leaf._get_main_param_and_optimizer_states(model_param)
    finally:
        if not had_entry and main in state and not state[main]:
            del state[main]


def _distributed_needs_state_init(leaf: Any, slots: Sequence[_Slot], merged: Mapping[str, Mapping[str, Any]]) -> bool:
    """Whether some slot of ``leaf`` lacks a saved optimizer-state key (a fresh or lazily read optimizer)."""
    for slot in slots:
        if slot.leaf is not leaf:
            continue
        saved = set(merged[slot.name]["tensors"]) | set(merged[slot.name]["scalars"])
        if not saved <= set(slot.get()):
            return True
    return False


def optimizer_slots(optimizer: Any, named_params: Iterable[tuple[str, torch.nn.Parameter]]) -> list[_Slot]:
    """The parameter slots this rank's optimizer updates (see the section comment)."""
    name_of = {}
    for name, param in named_params:
        assert id(param) not in name_of, f"parameter {name!r} is registered twice"
        name_of[id(param)] = name
    slots = []
    for leaf in _optimizer_leaves(optimizer):
        if isinstance(leaf, torch.optim.Optimizer):
            slots += _torch_slots(leaf, {}, name_of)
        elif _is_distributed(leaf):
            slots += _distributed_slots(leaf, name_of)
        elif hasattr(leaf, "float16_groups") and hasattr(leaf, "fp32_from_float16_groups"):
            main_to_model = {
                id(main): model
                for model_group, main_group in zip(leaf.float16_groups, leaf.fp32_from_float16_groups, strict=True)
                for model, main in zip(model_group, main_group, strict=True)
            }
            slots += _torch_slots(leaf.optimizer, main_to_model, name_of)
        elif isinstance(getattr(leaf, "optimizer", None), torch.optim.Optimizer):
            slots += _torch_slots(leaf.optimizer, {}, name_of)
        else:
            raise NotImplementedError(f"DP-invariant LoRA state does not support {type(leaf).__name__}")
    names = [slot.name for slot in slots]
    duplicated = sorted({n for n in names if names.count(n) > 1})
    assert not duplicated, f"parameters {duplicated} appear in more than one optimizer slot"
    return slots


def _hyper(group: Mapping[str, Any]) -> dict[str, Any]:
    return {k: copy.deepcopy(v) for k, v in group.items() if k != "params" and k not in _HDO_WIRING_KEYS}


def export_named_optimizer_state(optimizer: Any, named_params) -> dict[str, Any]:
    """This rank's optimizer state keyed by parameter name, each entry tagged with its range in the parameter."""
    entries = {}
    for slot in optimizer_slots(optimizer, named_params):
        tensors, scalars = {}, {}
        for key, value in slot.get().items():
            value = value.detach().cpu()
            if value.dim() == 0:
                scalars[key] = value.clone()
                continue
            flat = value.reshape(-1).clone()
            if flat.numel() != slot.end - slot.start:
                raise DpInvariantStateError(
                    f"{slot.name!r}: state {key!r} has {flat.numel()} elements but the owned range is "
                    f"[{slot.start}, {slot.end})"
                )
            tensors[key] = flat
        entries[slot.name] = {
            "numel": slot.model_param.numel(),
            "shape": tuple(slot.model_param.shape),
            "start": slot.start,
            "end": slot.end,
            "tensors": tensors,
            "scalars": scalars,
            "hyper": _hyper(slot.group),
        }
    return {"format": NAMED_STATE_FORMAT, "entries": entries}


def merge_named_optimizer_states(shards: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Gather the DP shards of one (tp, pp) coordinate into full flat tensors per parameter name.

    Ranges written by more than one rank (replicated optimizers) must agree exactly; every parameter must be
    covered completely, otherwise a shard file is missing.
    """
    assert shards, "no shards to merge"
    merged: dict[str, dict[str, Any]] = {}
    covered: dict[str, torch.Tensor] = {}
    for shard in shards:
        if shard.get("format") != NAMED_STATE_FORMAT:
            raise DpInvariantStateError(f"not a {NAMED_STATE_FORMAT} optimizer state: {shard.get('format')!r}")
        for name, entry in shard["entries"].items():
            start, end = entry["start"], entry["end"]
            if name not in merged:
                merged[name] = {
                    "numel": entry["numel"],
                    "shape": tuple(entry["shape"]),
                    "tensors": {
                        k: torch.zeros(entry["numel"], dtype=v.dtype) for k, v in entry["tensors"].items()
                    },
                    "scalars": dict(entry["scalars"]),
                    "hyper": entry["hyper"],
                }
                covered[name] = torch.zeros(entry["numel"], dtype=torch.bool)
            target = merged[name]
            if (target["numel"], target["shape"]) != (entry["numel"], tuple(entry["shape"])):
                raise DpInvariantStateError(f"DP shards disagree on the shape of {name!r}")
            if target["tensors"].keys() != entry["tensors"].keys():
                raise DpInvariantStateError(f"DP shards disagree on the state keys of {name!r}")
            if not _equal(target["scalars"], entry["scalars"]):
                raise DpInvariantStateError(f"DP shards disagree on the scalar state (e.g. step) of {name!r}")
            if not _equal(target["hyper"], entry["hyper"]):
                raise DpInvariantStateError(f"DP shards disagree on the hyperparameters of {name!r}")
            overlap = covered[name][start:end]
            for key, piece in entry["tensors"].items():
                existing = target["tensors"][key][start:end]
                if overlap.any() and not torch.equal(existing[overlap], piece[overlap]):
                    raise DpInvariantStateError(f"DP shards disagree on the optimizer state {key!r} of {name!r}")
                existing.copy_(piece)
            covered[name][start:end] = True
    incomplete = sorted(name for name, mask in covered.items() if not bool(mask.all()))
    if incomplete:
        raise DpInvariantStateError(
            f"optimizer state of {incomplete} is not fully covered; a DP shard file is missing"
        )
    return merged


def check_named_optimizer_state(optimizer: Any, named_params, merged: Mapping[str, Mapping[str, Any]]) -> None:
    """Validate that ``merged`` fits this rank's optimizer without writing anything (names, sizes, groups)."""
    slots = optimizer_slots(optimizer, named_params)
    missing = sorted(slot.name for slot in slots if slot.name not in merged)
    if missing:
        raise KeyError(f"named optimizer state lacks parameters {missing}")
    for slot in slots:
        if merged[slot.name]["numel"] != slot.model_param.numel():
            raise DpInvariantStateError(
                f"{slot.name!r} has {slot.model_param.numel()} elements, the checkpoint {merged[slot.name]['numel']}"
            )
    for names in _names_by_group(slots).values():
        hypers = [merged[n]["hyper"] for n in names[1]]
        if any(not _equal(h, hypers[0]) for h in hypers):
            raise DpInvariantStateError(f"parameters {names[1]} come from groups with different hyperparameters")


def _names_by_group(slots) -> dict[int, tuple[dict, list[str]]]:
    groups: dict[int, tuple[dict, list[str]]] = {}
    for slot in slots:
        groups.setdefault(id(slot.group), (slot.group, []))[1].append(slot.name)
    return groups


def load_named_optimizer_state(optimizer: Any, named_params, merged: Mapping[str, Mapping[str, Any]]) -> None:
    """Scatter the merged (gathered) state onto the ranges this rank's optimizer owns.

    Everything is validated (``check_named_optimizer_state``) before the optimizer is touched. A failure after
    that (e.g. ``DpInvariantStateError`` from a key mismatch) leaves the optimizer state invalid: the dummy Adam
    step may already have run and some slots may already hold loaded values. Rebuild the optimizer before retrying.
    """
    named_params = list(named_params)
    check_named_optimizer_state(optimizer, named_params, merged)
    slots = optimizer_slots(optimizer, named_params)
    leaves = _optimizer_leaves(optimizer)
    for leaf in leaves:
        # A freshly built DistOpt has no state tensors yet (Megatron's own loaders initialize them first). Decide per
        # parameter, not on the whole ``state``: an earlier read may have left empty entries that hide a missing init.
        if _is_distributed(leaf) and _distributed_needs_state_init(leaf, slots, merged):
            _init_states_with_dummy_values(leaf)
    for slot in slots:
        full = merged[slot.name]
        tensors = {k: v[slot.start : slot.end] for k, v in full["tensors"].items()}
        tensors.update({k: v.clone() for k, v in full["scalars"].items()})
        slot.set(tensors)
    for group, names in _names_by_group(slots).values():
        group.update(copy.deepcopy(merged[names[0]]["hyper"]))
    for leaf in leaves:
        if _is_distributed(leaf) and _hybrid_offload(leaf):
            _finish_hybrid_load(leaf, slots, merged)


def _init_states_with_dummy_values(leaf: Any) -> None:
    """Megatron's dummy step, without its side effect on the model under CPU offload.

    The HDO steps its host copies and copies them back into the parameters it was built over -- under the
    precision-aware path the bf16 model shards -- so a zero-grad AdamW step would decay model weights the caller may
    already have restored. Those parameters are put back after the step; the masters and moments are overwritten by
    the load that follows.
    """
    if not _hybrid_offload(leaf):
        leaf._init_optimizer_states_with_dummy_values()
        return
    params = [p for group in leaf.optimizer.param_groups for p in group["params"]]
    saved = [p.detach().clone() for p in params]
    leaf._init_optimizer_states_with_dummy_values()
    with torch.no_grad():
        for param, value in zip(params, saved, strict=True):
            param.copy_(value)


def _finish_hybrid_load(leaf: Any, slots: Sequence[_Slot], merged: Mapping[str, Mapping[str, Any]]) -> None:
    """Complete a load into a HybridDeviceOptimizer the way Megatron's own loaders do.

    ``_set_main_param_and_optimizer_states`` skips the HDO "step" (Megatron restores it from param_groups in its
    checkpoint path), so the saved per-parameter step is written here; ``_sync_hdo_state_to_sub_optimizers`` then
    hands the state to the CPU/GPU sub-optimizers and copies every "master_param" into the fp32 tensor the
    sub-optimizer actually steps.
    """
    name_of = {id(slot.model_param): slot.name for slot in slots}
    hdo = leaf.optimizer
    for model_param, (group_index, group_order) in leaf.model_param_group_index_map.items():
        if id(model_param) not in name_of:
            continue
        sharded = hdo.param_groups[group_index]["params"][group_order]
        step = merged[name_of[id(model_param)]]["scalars"].get("step")
        if step is not None:
            hdo.state[sharded]["step"] = step.clone()
    hdo._sync_hdo_state_to_sub_optimizers()
    _bind_masters_to_stepped_tensors(hdo)


def _hdo_inner_param(hdo: Any, param: torch.Tensor) -> torch.Tensor:
    """The tensor the HDO's sub-optimizer steps for ``param``: the fp32 copy of a low-precision parameter,
    the host copy of an offloaded fp32 parameter, else the parameter itself."""
    inner = getattr(hdo, "param_to_inner_param", None)
    if inner is not None:
        return inner.get(param, param)
    return hdo.param_to_fp32_param.get(param, param)


def _hdo_stepped_state(hdo: Any, param: torch.Tensor) -> dict[str, torch.Tensor]:
    """The state the HDO's sub-optimizer actually steps for ``param``, keyed like
    ``_get_main_param_and_optimizer_states`` ("param" plus the tensor state, e.g. "step", "exp_avg").

    The export reads these tensors, not the HDO's ``state["master_param"]``: the two are the same tensor only while
    the HDO is consistent, and a cut digest over the HDO copy cannot see a sub-optimizer that steps something else
    (a restore that left it on its build-time host copy, GPU T-G3 2026-10-20). A fresh HDO has no state yet: only
    "param".
    """
    inner = _hdo_inner_param(hdo, param)
    for sub in hdo.sub_optimizers:
        state = sub.state.get(inner)
        if state:
            tensors = {k: v for k, v in state.items() if isinstance(v, torch.Tensor) and k != "master_param"}
            tensors["param"] = inner
            return tensors
    return {"param": inner}


def _bind_masters_to_stepped_tensors(hdo: Any) -> None:
    """Make every loaded "master_param" the tensor the sub-optimizer steps, holding the loaded value.

    ``_set_main_param_and_optimizer_states`` replaces ``state["master_param"]`` with the loaded tensor, and
    ``_update_fp32_params_by_new_state`` copies it only into the fp32 copies of low-precision parameters. A native
    fp32 parameter (e.g. KDA ``A_log``/``dt_bias``, mHC ``alpha_*``/``bias``, kept in fp32) has no such copy: its
    sub-optimizer steps the host copy the HDO took when it was built, so without this the first step after a
    restore would continue from that stale snapshot and write it back into the model (GPU T-G3, 2026-10-20).
    """
    with torch.no_grad():
        for param, state in hdo.state.items():
            master = state.get("master_param")
            if not isinstance(master, torch.Tensor):
                continue
            inner = _hdo_inner_param(hdo, param)
            if inner is master:
                continue
            inner.copy_(master)
            state["master_param"] = inner


def _equal(a: Any, b: Any) -> bool:
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    return a == b


# ---------------------------------------------------------------------------
# RNG
# ---------------------------------------------------------------------------


def capture_rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
    }
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
        state["megatron_cuda_tracker"] = _megatron_tracker_states()
    return state


def _megatron_tracker_states() -> dict | None:
    """Megatron's model-parallel CUDA RNG tracker (dropout etc.); None when Megatron is absent. GPU-only path."""
    try:
        from megatron.core import tensor_parallel
    except ImportError:
        return None
    return tensor_parallel.get_cuda_rng_tracker().get_states()


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if (cuda_state := state.get("torch_cuda")) is not None:
        assert torch.cuda.is_available(), "the checkpoint holds CUDA RNG state but CUDA is unavailable"
        assert len(cuda_state) == torch.cuda.device_count(), (
            f"the checkpoint holds CUDA RNG state for {len(cuda_state)} devices, this process sees "
            f"{torch.cuda.device_count()}"
        )
        torch.cuda.set_rng_state_all(cuda_state)
    if (tracker_states := state.get("megatron_cuda_tracker")) is not None:
        from megatron.core import tensor_parallel

        tensor_parallel.get_cuda_rng_tracker().set_states(tracker_states)
