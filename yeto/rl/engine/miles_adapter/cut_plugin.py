"""ReconfigurationCut shard save/restore executed inside every Megatron rank (rl-infra-spec 4.2).

Loaded through ``TrainGroup.run_plugin`` like :mod:`.state_plugin`; every
function is collective-safe (no collective inside; every rank writes/reads
only its own shard). Separate from Miles' default checkpoint path: the ports
engine keeps ``--no-save-optim/--no-load-optim/--no-save-rng/--no-load-rng``
and never sets ``--save``/``--load``; a cut is written and read only here.

One shard per rank holds (4.1 audit, ``cut-audit.md``):

* LoRA adapter model copies (bf16 or fp32), keyed by parameter name;
* the optimizer state keyed by parameter name, including the FP32 main copy
  (``param``), the moments and the step/hyper-parameters -- produced by the
  fork-M5 helpers ``miles.backends.megatron_utils.lora.dp_invariant_state``
  (Float16Optimizer, DistributedOptimizer range shards and plain fp32 are
  covered there);
* the LR scheduler ``state_dict`` (``num_steps`` in samples);
* Python / NumPy / Torch CPU / CUDA / Megatron CUDA-tracker RNG
  (``capture_rng_state``; restored exactly -- same-shape only);
* Megatron global counters (``iteration``, ``consumed_train_samples``) when present.

Same-shape only (E2): the shard coordinate (tp, pp, dp, dp_size) must match
on restore. Refused like fork-M5: fp16 (loss-scaler state not saved),
precision-aware optimizer, partial DistOpt instances, CP>1, EP>1, and a
trainer without LoRA adapters. torch / miles are imported lazily.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from typing import Any

from .state_plugin import (
    _STEP_APPLIED_LRS,
    _STEP_GRAD_NORMS,
    _STEP_LOSSES,
    StatePluginError,
    install_grad_norm_recorder,
    trainer_resident,
)

_MODULE = "yeto.rl.engine.miles_adapter.cut_plugin"
SAVE_CUT_SHARD = f"{_MODULE}.save_cut_shard"
RESTORE_CUT_SHARD = f"{_MODULE}.restore_cut_shard"
SHARD_SCHEMA = "yeto.cut_shard/v1"


class CutPluginError(StatePluginError):
    pass


# --------------------------------------------------------------------------
# Backend (Miles fork-M5 helpers); tests preset ``actor._yeto_cut_backend``.
# --------------------------------------------------------------------------


class MilesCutBackend:
    def coord(self) -> dict[str, int]:
        import torch.distributed as dist
        from miles.backends.training_utils.parallel import get_parallel_state

        ps = get_parallel_state()
        return {
            "global_rank": dist.get_rank() if dist.is_initialized() else 0,
            "tp": ps.tp.rank, "pp": ps.pp.rank, "dp": ps.intra_dp.rank, "dp_size": ps.intra_dp.size,
            "tp_size": ps.tp.size, "pp_size": ps.pp.size,
            "cp_size": ps.cp.size, "ep_size": ps.ep.size,
        }

    def named_parameters(self, model: Any) -> list[tuple[str, Any]]:
        # Same naming as fork-M5 ``_named_parameters`` (virtual-pipeline chunks get a prefix).
        if len(model) == 1:
            return list(model[0].named_parameters())
        return [(f"chunk{i}.{n}", p) for i, chunk in enumerate(model) for n, p in chunk.named_parameters()]

    def is_adapter(self, name: str) -> bool:
        from miles.backends.megatron_utils.lora.utils import _is_adapter_param_name

        return _is_adapter_param_name(name)

    def _dps(self):
        from miles.backends.megatron_utils.lora import dp_invariant_state

        return dp_invariant_state

    def export_optimizer(self, optimizer: Any, named: list) -> Any:
        return self._dps().export_named_optimizer_state(optimizer, named)

    def check_optimizer(self, optimizer: Any, named: list, states: list) -> Any:
        """Merge the DP shards of one (tp, pp) (DistOpt ranges) and validate against ``optimizer``."""
        dps = self._dps()
        merged = dps.merge_named_optimizer_states(states)
        dps.check_named_optimizer_state(optimizer, named, merged)
        return merged

    def load_optimizer(self, optimizer: Any, named: list, merged: Any) -> None:
        self._dps().load_named_optimizer_state(optimizer, named, merged)

    def capture_rng(self) -> Any:
        return self._dps().capture_rng_state()

    def restore_rng(self, state: Any) -> None:
        self._dps().restore_rng_state(state)

    def megatron_counters(self) -> dict[str, int]:
        try:
            from megatron.training.global_vars import get_args
        except ImportError:
            return {}
        margs = get_args()
        return {k: int(getattr(margs, k)) for k in ("iteration", "consumed_train_samples")
                if getattr(margs, k, None) is not None}

    def set_megatron_counters(self, counters: Mapping[str, int]) -> None:
        if not counters:
            return
        from megatron.training.global_vars import get_args

        margs = get_args()
        for key, value in counters.items():
            setattr(margs, key, int(value))


def _backend(actor: Any) -> Any:
    return getattr(actor, "_yeto_cut_backend", None) or MilesCutBackend()


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------


def config_problems(args: Any, coord: Mapping[str, int] | None = None) -> list[str]:
    """Configurations whose trainer state a cut cannot carry (mirrors fork-M5's parse-time check)."""
    out = []
    if getattr(args, "fp16", False):
        out.append("--fp16 keeps a dynamic loss-scaler state that the cut does not carry")
    if getattr(args, "use_precision_aware_optimizer", False):
        out.append("--use-precision-aware-optimizer stores scaled/low-precision state (not supported)")
    if int(getattr(args, "num_distributed_optimizer_instances", 1) or 1) > 1:
        out.append("--num-distributed-optimizer-instances > 1 (partial DP optimizer groups)")
    cp = int((coord or {}).get("cp_size") or getattr(args, "context_parallel_size", 1) or 1)
    ep = int((coord or {}).get("ep_size") or getattr(args, "expert_model_parallel_size", 1) or 1)
    tp = int(getattr(args, "tensor_model_parallel_size", 1) or 1)
    pp = int(getattr(args, "pipeline_model_parallel_size", 1) or 1)
    if (tp > 1 or pp > 1) and getattr(args, "use_distributed_optimizer", False):
        # The post-restore republish (state_plugin._collective_export) cannot
        # expose DP-sharded masters under TP/PP; refuse before saving.
        out.append("TP/PP>1 together with DistributedOptimizer-sharded masters is not supported")
    if cp > 1:
        out.append("CP>1 is not supported")
    if ep > 1:
        out.append("EP>1 is not supported")
    return out


def _require(actor: Any, backend: Any) -> dict[str, int]:
    coord = backend.coord()
    problems = config_problems(actor.args, coord)
    if getattr(actor, "optimizer", None) is None:
        problems.append("the trainer has no optimizer")
    if getattr(actor, "opt_param_scheduler", None) is None:
        problems.append("the trainer has no LR scheduler")
    if not problems and (coord.get("tp_size", 1) > 1 or coord.get("pp_size", 1) > 1):
        from .state_plugin import distributed_ranges

        if distributed_ranges(actor.optimizer)[0]:
            problems.append("TP/PP>1 together with DistributedOptimizer-sharded masters is not supported")
    if problems:
        raise CutPluginError("ReconfigurationCut unsupported: " + "; ".join(problems))
    return coord


def _adapters(actor: Any, backend: Any) -> list[tuple[str, Any]]:
    named = [(n, p) for n, p in backend.named_parameters(actor.model) if backend.is_adapter(n)]
    if not named:
        raise CutPluginError("no LoRA adapter parameters: only LoRA trainers are covered by the cut (4.1)")
    trainable = {id(p) for _, p in backend.named_parameters(actor.model) if p.requires_grad}
    if {id(p) for _, p in named} != trainable:
        raise CutPluginError("trainable parameters other than LoRA adapters exist; the cut would miss them")
    return named


# --------------------------------------------------------------------------
# Deterministic digest of a nested state (tensors by raw bytes)
# --------------------------------------------------------------------------


def state_digest(value: Any) -> str:
    digest = hashlib.sha256()
    _feed(digest, value)
    return digest.hexdigest()


def _feed(digest: Any, value: Any) -> None:
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        shape = tuple(value.shape)
        t = value.detach().to("cpu").contiguous().reshape(-1)
        digest.update(f"T{t.dtype}{shape}".encode())
        digest.update(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    elif isinstance(value, np.ndarray):
        digest.update(f"N{value.dtype}{value.shape}".encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    elif isinstance(value, Mapping):
        digest.update(b"{")
        for key in sorted(value, key=repr):
            digest.update(repr(key).encode())
            _feed(digest, value[key])
        digest.update(b"}")
    elif isinstance(value, (list, tuple)):
        digest.update(b"[" if isinstance(value, list) else b"(")
        for item in value:
            _feed(digest, item)
        digest.update(b"]")
    else:
        digest.update(repr(value).encode())


_NDARRAY = "__yeto_ndarray__"


def to_safe(value: Any) -> Any:
    """Encode NumPy arrays/scalars (e.g. ``np.random.get_state()``) for ``torch.load(weights_only=True)``."""
    import numpy as np
    import torch

    if isinstance(value, np.ndarray):
        raw = np.ascontiguousarray(value)
        return {_NDARRAY: torch.frombuffer(bytearray(raw.tobytes()), dtype=torch.uint8),
                "dtype": raw.dtype.str, "shape": list(raw.shape)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {k: to_safe(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(to_safe(v) for v in value)
    if isinstance(value, list):
        return [to_safe(v) for v in value]
    return value


def from_safe(value: Any) -> Any:
    import numpy as np

    if isinstance(value, Mapping):
        if _NDARRAY in value:
            data = value[_NDARRAY].numpy().tobytes()
            return np.frombuffer(data, dtype=np.dtype(value["dtype"])).reshape(value["shape"]).copy()
        return {k: from_safe(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(from_safe(v) for v in value)
    if isinstance(value, list):
        return [from_safe(v) for v in value]
    return value


def _sha256(path: str) -> str:
    from yeto.rl.engine.cut import sha256_file

    return sha256_file(path)


def _snapshot(actor: Any, backend: Any, named: list) -> dict[str, Any]:
    return {
        "adapter": {n: p.detach().to("cpu").clone() for n, p in named},
        "optimizer_named": backend.export_optimizer(actor.optimizer, named),
        "scheduler": actor.opt_param_scheduler.state_dict(),
        "megatron_counters": backend.megatron_counters(),
    }


# --------------------------------------------------------------------------
# Plugins
# --------------------------------------------------------------------------


def shard_name(coord: Mapping[str, int]) -> str:
    return f"trainer_tp{coord['tp']}_pp{coord['pp']}_dp{coord['dp']}.pt"


def save_cut_shard(actor: Any, *, directory: str, cut_id: str) -> dict[str, Any]:
    """Write this rank's shard (tmp + fsync + rename) and return its summary."""
    with trainer_resident(actor):
        return _save(actor, directory=directory, cut_id=cut_id)


def _save(actor: Any, *, directory: str, cut_id: str) -> dict[str, Any]:
    import torch

    backend = _backend(actor)
    coord = _require(actor, backend)
    if _STEP_GRAD_NORMS or _STEP_APPLIED_LRS or _STEP_LOSSES:
        raise CutPluginError("per-step records not drained: the cut is not at an optimizer-step boundary")
    named = _adapters(actor, backend)
    with torch.no_grad():
        snap = _snapshot(actor, backend, named)
        rng = backend.capture_rng()
    shard = {"schema": SHARD_SCHEMA, "cut_id": cut_id, "coord": dict(coord), **snap, "rng": rng}
    os.makedirs(directory, exist_ok=True)
    name = shard_name(coord)
    final = os.path.join(directory, name)
    if os.path.exists(final):
        raise CutPluginError(f"cut shard {final} already exists (cuts are immutable)")
    tmp = final + ".tmp"
    with open(tmp, "wb") as fh:
        torch.save(to_safe(shard), fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, final)
    return {
        "path": name,
        "sha256": _sha256(final),
        "bytes": os.path.getsize(final),
        "coord": dict(coord),
        "scheduler_samples": int(actor.opt_param_scheduler.num_steps),
        "has_optimizer_state": snap["optimizer_named"] is not None,
        "has_rng": rng is not None,
        "state_digest": state_digest(snap),
        "rng_digest": state_digest(rng),
        "adapter_tensors": len(named),
        "adapter_names": sorted(n for n, _ in named),
        "optimizer_names": sorted((snap["optimizer_named"] or {}).get("entries", {})),
    }


def _load_verified(directory: str, entry: Mapping[str, Any], cut_id: str) -> dict[str, Any]:
    import torch

    path = os.path.join(directory, entry["path"])
    if not os.path.isfile(path) or os.path.getsize(path) != int(entry["bytes"]):
        raise CutPluginError(f"cut shard {path} is missing or truncated")
    if _sha256(path) != entry["sha256"]:
        raise CutPluginError(f"cut shard {path} checksum mismatch")
    # weights_only: a cut shard is data, never code (test_provenance).
    shard = from_safe(torch.load(path, map_location="cpu", weights_only=True))
    if shard.get("schema") != SHARD_SCHEMA or shard.get("cut_id") != cut_id:
        raise CutPluginError(f"{path} is not a shard of cut {cut_id!r}")
    return shard


def _peer_entries(files: list[Mapping[str, Any]], coord: Mapping[str, int]) -> list[Mapping[str, Any]]:
    """Other DP ranks' shards of this rank's (tp, pp)."""
    own = shard_name(coord)
    return [
        f for f in files
        if f["path"] != own
        and (f.get("coord") or {}).get("tp") == coord["tp"]
        and (f.get("coord") or {}).get("pp") == coord["pp"]
    ]


def _scheduler_steps(state: Mapping[str, Any]) -> int:
    return int(state["num_iters"] if "num_iters" in state else state["num_steps"])


def check_scheduler(scheduler: Any, saved: Mapping[str, Any]) -> None:
    """Pre-check before Megatron ``OptimizerParamScheduler.load_state_dict``.

    Megatron's load ends with ``step(increment=num_steps)``, i.e. it ADDS the
    saved progress: restore_cut is only for a freshly built trainer whose
    scheduler is at 0. Hyper-parameters are compared strictly (the
    ``_check_and_set`` values: max/min lr, warmup/decay steps and style,
    weight-decay schedule), independent of ``override``/``use_checkpoint``
    flags -- a cut restores the same run.
    """
    if int(scheduler.num_steps) != 0:
        raise CutPluginError(
            f"LR scheduler already at {scheduler.num_steps}: restore_cut only restores into a freshly built trainer"
        )
    current = scheduler.state_dict()
    skip = {"num_steps", "num_iters"}
    diff = sorted(k for k in set(current) | set(saved) if k not in skip and current.get(k) != saved.get(k))
    if diff:
        raise CutPluginError(f"LR scheduler hyper-parameters differ from the cut: {diff}")


def rank_coords(actor: Any) -> dict[str, int]:
    """This rank's actual parallel coordinate (read back after a rebuild)."""
    return _backend(actor).coord()


RANK_COORDS = f"{_MODULE}.rank_coords"


def restore_cut_shard(actor: Any, *, directory: str, files: list[Mapping[str, Any]], cut_id: str) -> dict[str, Any]:
    """Verify and load this rank's shard; return the post-restore summary (identical digests expected)."""
    install_grad_norm_recorder()  # a rebuilt trainer process has no recorder yet
    with trainer_resident(actor):
        return _restore(actor, directory=directory, files=files, cut_id=cut_id)


def _restore(actor: Any, *, directory: str, files: list[Mapping[str, Any]], cut_id: str) -> dict[str, Any]:
    import torch

    backend = _backend(actor)
    coord = _require(actor, backend)
    name = shard_name(coord)
    entry = next((f for f in files if f["path"] == name), None)
    if entry is None:
        raise CutPluginError(f"cut {cut_id!r} has no shard for {name} (layout changed?)")
    shard = _load_verified(directory, entry, cut_id)
    saved = {k: v for k, v in shard["coord"].items() if k != "global_rank"}
    now = {k: v for k, v in coord.items() if k != "global_rank"}
    if saved != now:
        raise CutPluginError(f"shard coordinate {saved} != this rank {now} (same-shape restore only)")
    named = _adapters(actor, backend)
    adapter = shard["adapter"]
    names = {n for n, _ in named}
    if set(adapter) != names:
        raise CutPluginError(
            f"adapter names differ: missing={sorted(names - set(adapter))[:4]} extra={sorted(set(adapter) - names)[:4]}"
        )
    for n, p in named:
        if tuple(adapter[n].shape) != tuple(p.shape) or adapter[n].dtype != p.dtype:
            raise CutPluginError(f"adapter {n!r} shape/dtype differs from the rebuilt trainer")
    if shard["optimizer_named"] is None or shard["rng"] is None:
        raise CutPluginError("cut shard lacks optimizer state or RNG")
    # A DistributedOptimizer rank owns only a range of each parameter; the
    # fork-M5 merge needs every DP shard of this (tp, pp) (shared cut dir).
    states = [shard["optimizer_named"]]
    for peer in _peer_entries(files, coord):
        states.append(_load_verified(directory, peer, cut_id)["optimizer_named"])
    # Validate against the rebuilt optimizer and scheduler before any write.
    merged = backend.check_optimizer(actor.optimizer, named, states)
    check_scheduler(actor.opt_param_scheduler, shard["scheduler"])
    with torch.no_grad():
        for n, p in named:
            p.data.copy_(adapter[n].to(device=p.device))
        backend.load_optimizer(actor.optimizer, named, merged)
    actor.opt_param_scheduler.load_state_dict(shard["scheduler"])
    if int(actor.opt_param_scheduler.num_steps) != _scheduler_steps(shard["scheduler"]):
        raise CutPluginError("LR scheduler progress after restore differs from the cut")
    backend.set_megatron_counters(shard.get("megatron_counters") or {})
    backuper = getattr(actor, "weights_backuper", None)
    if backuper is not None:  # colocated update_weights reads the CPU backup
        backuper.backup("actor")
    for n, p in named:
        if not p.requires_grad:
            raise CutPluginError(f"adapter parameter {n!r} no longer requires grad")
    backend.restore_rng(shard["rng"])  # last: nothing below may consume randomness
    with torch.no_grad():
        snap = _snapshot(actor, backend, named)
    rng_now = backend.capture_rng()
    summary = {
        "path": name,
        "coord": dict(coord),
        "scheduler_samples": int(actor.opt_param_scheduler.num_steps),
        "state_digest": state_digest(snap),
        "rng_digest": state_digest(rng_now),
    }
    return summary


# --------------------------------------------------------------------------
# DP resharding restore (rl-infra-spec 4.6 spike; E3)
# --------------------------------------------------------------------------

RESTORE_RESHARDED_SHARD = f"{_MODULE}.restore_resharded_shard"


def _same_tp_pp(files: list[Mapping[str, Any]], coord: Mapping[str, int]) -> list[Mapping[str, Any]]:
    return [f for f in files if (f.get("coord") or {}).get("tp") == coord["tp"]
            and (f.get("coord") or {}).get("pp") == coord["pp"]]


def _slice_check(export: Mapping[str, Any], merged: Mapping[str, Any]) -> list[str]:
    """After a load, this rank's re-exported ranges must equal the merged (gathered) state."""
    import torch

    out = []
    for name, entry in (export or {}).get("entries", {}).items():
        full = merged.get(name)
        if full is None:
            out.append(f"{name}: not in the cut")
            continue
        start, end = int(entry["start"]), int(entry["end"])
        for key, piece in entry["tensors"].items():
            want = full["tensors"][key][start:end]
            if not torch.equal(piece.reshape(-1).to(want.dtype), want):
                out.append(f"{name}.{key}[{start}:{end}] differs from the cut")
        for key, value in entry["scalars"].items():
            if not torch.equal(torch.as_tensor(value), torch.as_tensor(full["scalars"][key])):
                out.append(f"{name}.{key} (scalar) differs from the cut")
    return out


def restore_resharded_shard(
    actor: Any, *, directory: str, files: list[Mapping[str, Any]], cut_id: str,
    source_dp: int, rng_policy: str = "keep_on_dp_change",
) -> dict[str, Any]:
    """Load a cut written at another DP size into this freshly built rank.

    Every DP shard of this rank's (tp, pp) is read and verified; the adapter
    model copy, the LR scheduler and the Megatron counters are DP-replicated
    and must be identical in all of them. The named optimizer state (FP32
    main, moments, step, hyper-parameters) is merged over the source DP ranks
    (fork-M5 ``merge_named_optimizer_states``) and sliced to the range this
    rank owns now. RNG: 'keep_on_dp_change' keeps the fresh RNG of this
    process when DP changed (the caller records the seed mapping); with an
    unchanged DP the saved RNG of the same coordinate is restored exactly.
    Everything is validated before the first write.
    """
    install_grad_norm_recorder()
    with trainer_resident(actor):
        return _restore_resharded(actor, directory=directory, files=files, cut_id=cut_id,
                                  source_dp=int(source_dp), rng_policy=rng_policy)


def _restore_resharded(actor, *, directory, files, cut_id, source_dp, rng_policy):
    import torch

    backend = _backend(actor)
    coord = _require(actor, backend)
    group = _same_tp_pp(files, coord)
    if len(group) != source_dp:
        raise CutPluginError(f"cut has {len(group)} DP shards for tp{coord['tp']}/pp{coord['pp']}, expected {source_dp}")
    shards = [_load_verified(directory, entry, cut_id) for entry in group]
    for shard in shards:
        saved = shard["coord"]
        for key in ("tp", "pp", "tp_size", "pp_size", "cp_size", "ep_size"):
            if int(saved.get(key, 1)) != int(coord.get(key, 1)):
                raise CutPluginError(f"shard {key}={saved.get(key)} != this rank {coord.get(key)} (only DP may change)")
        if int(saved["dp_size"]) != source_dp:
            raise CutPluginError(f"shard dp_size {saved['dp_size']} != source DP {source_dp}")
        if shard["optimizer_named"] is None or shard["rng"] is None:
            raise CutPluginError("cut shard lacks optimizer state or RNG")
    first = shards[0]
    for shard in shards[1:]:
        for key in ("adapter", "scheduler", "megatron_counters"):
            if state_digest(shard[key]) != state_digest(first[key]):
                raise CutPluginError(f"DP shards disagree on the replicated {key}")
    dp_changes = int(coord["dp_size"]) != source_dp
    if dp_changes and rng_policy != "keep_on_dp_change":
        raise CutPluginError(f"RNG policy {rng_policy!r} cannot change DP {source_dp} -> {coord['dp_size']}")
    own_rng = None
    if not dp_changes:
        own = next((s for s in shards if int(s["coord"]["dp"]) == int(coord["dp"])), None)
        if own is None:
            raise CutPluginError("no shard for this DP rank although DP is unchanged")
        own_rng = own["rng"]
    named = _adapters(actor, backend)
    adapter = first["adapter"]
    names = {n for n, _ in named}
    if set(adapter) != names:
        raise CutPluginError(f"adapter names differ from the cut: {sorted(names ^ set(adapter))[:4]}")
    for n, p in named:
        if tuple(adapter[n].shape) != tuple(p.shape) or adapter[n].dtype != p.dtype:
            raise CutPluginError(f"adapter {n!r} shape/dtype differs from the rebuilt trainer")
    merged = backend.check_optimizer(actor.optimizer, named, [s["optimizer_named"] for s in shards])
    check_scheduler(actor.opt_param_scheduler, first["scheduler"])
    # ---- writes ----
    with torch.no_grad():
        for n, p in named:
            p.data.copy_(adapter[n].to(device=p.device))
        backend.load_optimizer(actor.optimizer, named, merged)
    actor.opt_param_scheduler.load_state_dict(first["scheduler"])
    if int(actor.opt_param_scheduler.num_steps) != _scheduler_steps(first["scheduler"]):
        raise CutPluginError("LR scheduler progress after restore differs from the cut")
    backend.set_megatron_counters(first.get("megatron_counters") or {})
    backuper = getattr(actor, "weights_backuper", None)
    if backuper is not None:
        backuper.backup("actor")
    if own_rng is not None:
        backend.restore_rng(own_rng)
    with torch.no_grad():
        export = backend.export_optimizer(actor.optimizer, named)
        adapters_now = {n: p.detach().to("cpu").clone() for n, p in named}
    problems = _slice_check(export, merged)
    if state_digest(adapters_now) != state_digest(adapter):
        problems.append("adapter after restore differs from the cut")
    if problems:
        raise CutPluginError("resharded restore mismatch: " + "; ".join(problems[:4]))
    return {
        "coord": dict(coord),
        "source_dp": source_dp,
        "scheduler_samples": int(actor.opt_param_scheduler.num_steps),
        # identical on every rank of a (tp, pp): the gathered full state
        "full_state_digest": state_digest({"adapter": adapter, "optimizer": merged,
                                           "scheduler": first["scheduler"]}),
        "rng": "restored" if own_rng is not None else "fresh",
        "rng_digest": state_digest(backend.capture_rng()),
        "optimizer_names": sorted(export.get("entries", {})),
        "adapter_names": sorted(names),
    }

