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
from contextlib import contextmanager
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
# Opt-in: read the optimizer back right after the load and report where a mismatch arises.
RESTORE_DIAGNOSTICS_ENV = "YETO_RL_CUT_RESTORE_DIAGNOSTICS"


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
        if _UNSAFE_STATE_READS[0]:  # diagnostic only: verify the fork fix without the yeto guard
            return self._dps().export_named_optimizer_state(optimizer, named)
        with side_effect_free_state(optimizer):
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


def _state_dicts(optimizer: Any) -> list[Any]:
    """The per-parameter ``state`` mappings under an optimizer (ChainedOptimizer members,
    Megatron wrappers' inner torch optimizer)."""
    out, seen = [], set()
    stack = [optimizer]
    while stack:
        obj = stack.pop()
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        stack.extend(getattr(obj, "chained_optimizers", None) or ())
        stack.append(getattr(obj, "optimizer", None))
        state = getattr(obj, "state", None)
        if isinstance(state, Mapping):
            out.append(state)
    return out


@contextmanager
def side_effect_free_state(optimizer: Any):
    """Reading optimizer state must not create state entries.

    Megatron's ``_get_main_param_and_optimizer_states`` indexes the torch
    optimizer's ``state`` defaultdict, which creates an empty entry per main
    param on a freshly built optimizer; fork-M5's loader then sees a non-empty
    state, skips ``_init_optimizer_states_with_dummy_values`` and its setter
    copies only keys that exist -> exp_avg/exp_avg_sq silently not restored
    (GPU C1 diagnostic 2, 2026-09-30). Entries created during the read that are
    still empty are removed afterwards.
    """
    before = [(state, set(state.keys())) for state in _state_dicts(optimizer)]
    try:
        yield
    finally:
        for state, keys in before:
            for key in [k for k in state.keys() if k not in keys and not state[k]]:
                del state[key]


# Per rank process; False = default (side-effect-free reads). Only the E2 diagnostic
# sub-run turns it on (plugin below), to check the fork-M5 fix on a real DistOpt.
_UNSAFE_STATE_READS = [False]
SET_UNSAFE_STATE_READS = "yeto.rl.engine.miles_adapter.cut_plugin.set_unsafe_state_reads"


def set_unsafe_state_reads(actor: Any, *, enabled: bool) -> dict[str, Any]:
    """Diagnostic switch (E2 plan-v6 C1 sub-run): read optimizer state WITHOUT the
    side-effect-free guard in this rank process."""
    del actor
    _UNSAFE_STATE_READS[0] = bool(enabled)
    return {"unsafe_state_reads": _UNSAFE_STATE_READS[0]}


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


def component_digests(value: Any, *, depth: int = 5, prefix: str = "") -> dict[str, str]:
    """Digest per nested component (mapping keys up to ``depth``) -- names what differs
    when an overall state digest does not match (GPU C1 attempt 4)."""
    if depth <= 0 or not isinstance(value, Mapping):
        return {prefix or ".": state_digest(value)}
    out: dict[str, str] = {}
    for key in sorted(value, key=repr):
        out.update(component_digests(value[key], depth=depth - 1, prefix=f"{prefix}/{key}"))
    return out


def state_diff(saved: Any, now: Any, *, path: str = "", out: list | None = None, limit: int = 400) -> list:
    """Leaf-level differences between two snapshots: (path, kind, detail) -- dtype/shape
    or max |a-b| for tensors, repr for others. Used when a restore does not reproduce the cut."""
    import torch

    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(saved, Mapping) and isinstance(now, Mapping):
        for key in sorted(set(saved) | set(now), key=repr):
            if key not in saved or key not in now:
                out.append((f"{path}/{key}", "missing", "saved" if key not in saved else "restored"))
                continue
            state_diff(saved[key], now[key], path=f"{path}/{key}", out=out, limit=limit)
        return out
    if isinstance(saved, torch.Tensor) and isinstance(now, torch.Tensor):
        a, b = saved.detach().cpu(), now.detach().cpu()
        if a.dtype != b.dtype or a.shape != b.shape:
            out.append((path, "meta", f"{a.dtype}{tuple(a.shape)} vs {b.dtype}{tuple(b.shape)}"))
        elif not torch.equal(a, b):
            d = (a.double() - b.double()).abs()
            out.append((path, "value", f"max={d.max().item():.3e} n={int((d > 0).sum())}/{d.numel()}"))
        return out
    if isinstance(saved, (list, tuple)) and isinstance(now, (list, tuple)) and len(saved) == len(now):
        for i, (a, b) in enumerate(zip(saved, now)):
            state_diff(a, b, path=f"{path}[{i}]", out=out, limit=limit)
        return out
    if state_digest(saved) != state_digest(now):
        out.append((path, "other", f"{saved!r:.80} vs {now!r:.80}"))
    return out


def _entry_tensors(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Tensors of one optimizer entry: top-level ones and one level of groups (fork-M5
    ``tensors``/``scalars``; other formats' ``state``), ``hyper`` excluded."""
    import torch

    out = {}
    for key, value in entry.items():
        if isinstance(value, torch.Tensor):
            out[key] = value
        elif isinstance(value, Mapping) and key != "hyper":
            out.update({f"{key}:{k}": v for k, v in value.items() if isinstance(v, torch.Tensor)})
    return out


def optimizer_diff(a: Mapping[str, Any] | None, b: Mapping[str, Any] | None) -> dict[str, Any]:
    """Per optimizer-state key (param / exp_avg / exp_avg_sq / scalars / hyper) between two
    name-keyed optimizer states (fork-M5 format): entries that differ, max |a-b|, max
    relative diff, dtype pairs, and range/shape mismatches (DistOpt [start, end))."""
    import torch

    out: dict[str, Any] = {"entries": 0, "keys": {}, "ranges": 0, "missing": []}
    ea = (a or {}).get("entries", {})
    eb = (b or {}).get("entries", {})
    out["missing"] = sorted(set(ea) ^ set(eb))[:10]
    for name in sorted(set(ea) & set(eb)):
        x, y = ea[name], eb[name]
        out["entries"] += 1
        if any(x.get(k) != y.get(k) for k in ("start", "end", "numel")) or tuple(x.get("shape", ())) != tuple(
                y.get("shape", ())):
            out["ranges"] += 1
        fx, fy = _entry_tensors(x), _entry_tensors(y)
        for key in sorted(set(fx) | set(fy)):
            stat = out["keys"].setdefault(key, {"differ": 0, "max_abs": 0.0, "max_rel": 0.0,
                                                "dtypes": set()})
            u, v = fx.get(key), fy.get(key)
            if u is None or v is None:
                stat["differ"] += 1
                stat["dtypes"].add("missing")
                continue
            u, v = u.detach().cpu(), v.detach().cpu()
            stat["dtypes"].add(f"{u.dtype}->{v.dtype}")
            if u.shape != v.shape:
                stat["differ"] += 1
                continue
            if not torch.equal(u, v):
                stat["differ"] += 1
                d = (u.double() - v.double()).abs()
                stat["max_abs"] = max(stat["max_abs"], float(d.max()))
                denom = u.double().abs().clamp_min(1e-30)
                stat["max_rel"] = max(stat["max_rel"], float((d / denom).max()))
        if x.get("hyper") != y.get("hyper"):
            out["keys"].setdefault("hyper", {"differ": 0})["differ"] += 1
    for stat in out["keys"].values():
        if "dtypes" in stat:
            stat["dtypes"] = sorted(stat["dtypes"])
    return out


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


REFUSED = "refused"  # key of a "nothing written" result (reason text)
REFUSAL_KIND = "refusal_kind"  # "refused" (deliberate) | "failed_before_write"


def save_cut_shard(actor: Any, *, directory: str, cut_id: str) -> dict[str, Any]:
    """Write this rank's shard (tmp + fsync + rename) and return its summary.

    A refusal (unsupported configuration, not at a step boundary, ...) is
    RETURNED as ``{"refused": reason}`` instead of raised: Miles marks a cell
    errored on any ``run_plugin`` exception (GPU run 2026-09-30, C1), which
    would turn a clean "no cut" into a dead trainer.
    """
    try:
        with trainer_resident(actor):
            return _save(actor, directory=directory, cut_id=cut_id)
    except CutPluginError as error:
        return {REFUSED: str(error), REFUSAL_KIND: "refused"}


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
    from . import cut_injection

    kill = cut_injection.is_rank(coord, cut_injection.save_kill_rank())
    with open(tmp, "wb") as fh:
        if kill:  # test-only (G-4.5): die with a half-written temp file
            import io

            buffer = io.BytesIO()
            torch.save(to_safe(shard), buffer)
            data = buffer.getvalue()
            fh.write(data[: len(data) // 2])
            fh.flush()
            os.fsync(fh.fileno())
            cut_injection.kill_now(f"save_cut shard {name} half written")
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
        "components": component_digests({"state": snap, "rng": rng}),
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
    _inject_restore_sleep(actor)
    progress = {"written": False}
    try:
        with trainer_resident(actor):
            return _restore(actor, directory=directory, files=files, cut_id=cut_id, progress=progress)
    except Exception as error:  # noqa: BLE001
        if progress["written"]:
            raise  # state partly written: the trainer is unusable (RECOVERY_REQUIRED)
        # Nothing written: the trainer is untouched; return instead of raising so
        # Miles does not mark the cell errored (see save_cut_shard). A deliberate
        # refusal (CutPluginError) and any other failure before the first write
        # are told apart by REFUSAL_KIND; the trainer side raises CutError for both.
        kind = "refused" if isinstance(error, CutPluginError) else "failed_before_write"
        return {REFUSED: f"{type(error).__name__}: {error}", REFUSAL_KIND: kind}


def _inject_restore_sleep(actor: Any) -> None:
    """Test-only (G-4.5): one rank sleeps before the restore."""
    from . import cut_injection

    target = cut_injection.restore_sleep()
    if target is not None and cut_injection.is_rank(_backend(actor).coord(), target[0]):
        import time

        time.sleep(target[1])


def _inject_restore_kill(coord: Mapping[str, int]) -> None:
    """Test-only (G-4.5): die after the adapter write, before the optimizer."""
    from . import cut_injection

    if cut_injection.is_rank(coord, cut_injection.restore_kill_rank()):
        cut_injection.kill_now("restore_cut after the adapter write")


def _restore(actor: Any, *, directory: str, files: list[Mapping[str, Any]], cut_id: str,
             progress: dict[str, bool] | None = None) -> dict[str, Any]:
    progress = progress if progress is not None else {"written": False}
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
    progress["written"] = True
    with torch.no_grad():
        for n, p in named:
            p.data.copy_(adapter[n].to(device=p.device))
        _inject_restore_kill(coord)
        backend.load_optimizer(actor.optimizer, named, merged)
        # Opt-in diagnostics (a full optimizer export costs time and host memory on a
        # large model): the state read right after the load, to split "the load wrote
        # something else" from "changed afterwards" (GPU C1 diagnostic 2).
        after_load = (backend.export_optimizer(actor.optimizer, named)
                      if os.environ.get(RESTORE_DIAGNOSTICS_ENV) == "1" else None)
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
        "components": component_digests({"state": snap, "rng": rng_now}),
    }
    saved_snap = {k: shard.get(k) for k in snap}
    if state_digest(saved_snap) != summary["state_digest"]:
        # cut vs final re-export: free (both already in memory); the after-load split
        # only with the opt-in diagnostics
        summary["optimizer_cut_vs_reexport"] = optimizer_diff(shard.get("optimizer_named"), snap["optimizer_named"])
        if after_load is not None:
            summary["optimizer_cut_vs_after_load"] = optimizer_diff(shard.get("optimizer_named"), after_load)
            summary["optimizer_after_load_vs_reexport"] = optimizer_diff(after_load, snap["optimizer_named"])
        diffs = state_diff(saved_snap, snap)
        kinds: dict[str, int] = {}
        for p_, kind, _ in diffs:
            leaf = p_.rsplit("/", 1)[-1]
            kinds[f"{kind}:{leaf}"] = kinds.get(f"{kind}:{leaf}", 0) + 1
        summary["diff"] = {"count": len(diffs), "by_leaf": kinds, "first": diffs[:30]}
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



TRAIN_PARALLEL_CONFIG = f"{_MODULE}.train_parallel_config"


def train_parallel_config(actor: Any) -> dict[str, Any]:
    """The config the rank advertised to the rollout side (Megatron actor ``train_parallel_config``)."""
    return dict(getattr(actor, "train_parallel_config", None) or {})


# --------------------------------------------------------------------------
# E2 GPU harness read-outs (plan-v3): no writes, no randomness consumed.
# --------------------------------------------------------------------------

# plan-v3 §0 (= launcher --rl-deterministic-trainer, entry.DETERMINISM_ENV).
DETERMINISM_ENV = {
    "NCCL_ALGO": "Ring",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "NVIDIA_TF32_OVERRIDE": "0",
}
# Recorded, not required: Megatron --deterministic-mode applies its own default.
DETERMINISM_INFO_ENV = ("NVTE_ALLOW_NONDETERMINISTIC_ALGO",)
STATE_SUMMARY = f"{_MODULE}.state_summary"
RANK_DETERMINISM = f"{_MODULE}.rank_determinism"


def rank_determinism(actor: Any) -> dict[str, Any]:
    """What this rank actually runs with (plan-v3 §0: all must hold, else environment-blocked)."""
    env = {k: os.environ.get(k) for k in DETERMINISM_ENV}
    return {
        "info_env": {k: os.environ.get(k) for k in DETERMINISM_INFO_ENV},
        "coord": _backend(actor).coord(),
        "env": env,
        "env_ok": env == DETERMINISM_ENV,
        "deterministic_mode": bool(getattr(actor.args, "deterministic_mode", False)),
        "lora_dropout": getattr(actor.args, "lora_dropout", None),
    }


def state_summary(actor: Any) -> dict[str, Any]:
    """Digests of this rank's full trainer state for the bitwise arm comparison.

    ``tensors``: sha256 per adapter tensor and per optimizer entry/state key
    (diagnostics when the overall digest differs); ``moments_nonzero``: number
    of optimizer entries whose ``exp_avg`` has a non-zero element;
    ``bf16_master_mismatch``: adapter model copies that differ from their FP32
    master cast to the model dtype over the owned range (plan-v3 §3 L2).
    """
    with trainer_resident(actor):
        return _state_summary(actor)


def _state_summary(actor: Any) -> dict[str, Any]:
    import torch

    backend = _backend(actor)
    named = _adapters(actor, backend)
    with torch.no_grad():
        snap = _snapshot(actor, backend, named)
    rng = backend.capture_rng()
    tensors = {f"adapter/{n}": state_digest(t) for n, t in snap["adapter"].items()}
    entries = (snap["optimizer_named"] or {}).get("entries", {})
    moments_nonzero, mismatch = 0, []
    by_name = dict(named)
    for name, entry in sorted(entries.items()):
        tensors[f"optimizer/{name}"] = state_digest(entry)
        flat = entry.get("tensors") if "tensors" in entry else entry.get("state")
        exp_avg = (flat or {}).get("exp_avg")
        if exp_avg is not None and bool(torch.count_nonzero(exp_avg)):
            moments_nonzero += 1
        master = (flat or {}).get("param", entry.get("param"))
        model = by_name.get(name)
        if master is not None and model is not None and model.dtype != torch.float32:
            start, end = int(entry.get("start", 0)), int(entry.get("end", model.numel()))
            copy = model.detach().reshape(-1)[start:end].to("cpu")
            if not torch.equal(copy, master.reshape(-1).to(model.dtype).to("cpu")):
                mismatch.append(name)
    return {
        "coord": backend.coord(),
        "state_digest": state_digest(snap),
        "rng_digest": state_digest(rng),
        "scheduler_samples": int(actor.opt_param_scheduler.num_steps),
        "tensors": tensors,
        "optimizer_entries": len(entries),
        "moments_nonzero": moments_nonzero,
        "bf16_master_mismatch": mismatch,
    }
