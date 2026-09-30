"""Rank-side read-back plugins for the E3 A8/DEV-GATHER harness (rl-infra-spec 4.6; plan-v3 §2).

Loaded through ``TrainGroup.run_plugin`` like :mod:`.cut_plugin`. Nothing here
changes training: ``install_probe`` wraps three fork functions with recorders
that call the original and return its result unchanged:

* ``training_utils.data.process_rollout_data`` -- the shard this rank really
  received: ``partition`` (sample positions), whether it carries
  ``micro_batch_indices`` / ``num_rollouts`` / ``num_microbatches`` (the
  scheduled split, review M2) and their values;
* ``training_utils.loss.loss_function`` -- ``num_microbatches``,
  ``num_rollouts`` (the normalizer) and ``intra_dp_cp`` size per micro-batch
  (A8 G2);
* ``training_utils.loss.get_loss_function`` -- the UNSCALED per-micro-batch
  loss (mbs=1: per sample) with the batch's ``sample_indices``, as float hex
  (A8 G4 bitwise).

Functions imported by name elsewhere (``from ... import f``) are replaced in
every loaded module that holds the original object. torch/miles are imported
lazily.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from typing import Any

from .cut_plugin import _adapters, _backend, state_digest, to_safe
from .state_plugin import trainer_resident

_MODULE = "yeto.rl.engine.miles_adapter.e3_probe"
INSTALL_PROBE = f"{_MODULE}.install_probe"
DRAIN_PROBE = f"{_MODULE}.drain_probe"
RANK_INFO = f"{_MODULE}.rank_info"
DUMP_STATE = f"{_MODULE}.dump_state"

_RECORDS: list[dict[str, Any]] = []
_INSTALLED: dict[str, Any] = {}


def patch_everywhere(original: Any, replacement: Any, modules: Any = None) -> int:
    """Replace ``original`` by ``replacement`` in every module attribute that is ``original``."""
    count = 0
    for module in list((modules if modules is not None else sys.modules).values()):
        namespace = getattr(module, "__dict__", None)
        if not isinstance(namespace, dict):
            continue
        for name, value in list(namespace.items()):
            if value is original:
                setattr(module, name, replacement)
                count += 1
    return count


def _float_hex(value: Any) -> str:
    return float(value.detach().float().cpu()).hex() if hasattr(value, "detach") else float(value).hex()


def _record_shard(result: Any) -> None:
    shard = result[0] if isinstance(result, tuple) else result
    if not isinstance(shard, dict):
        _RECORDS.append({"kind": "shard", "error": f"unexpected shard type {type(shard).__name__}"})
        return
    _RECORDS.append({
        "kind": "shard",
        "partition": [int(i) for i in shard.get("partition", [])],
        "has_micro_batch_indices": "micro_batch_indices" in shard,
        "has_num_rollouts": "num_rollouts" in shard,
        "micro_batch_indices": shard.get("micro_batch_indices"),
        "num_rollouts": shard.get("num_rollouts"),
        "num_microbatches": shard.get("num_microbatches"),
        "sample_indices": [int(i) for i in shard.get("sample_indices", [])],
    })


def make_wrappers(parallel_size: Callable[[], int]) -> dict[str, Callable[[Callable], Callable]]:
    """The three recorders (separate for CPU tests)."""

    def wrap_process(orig):
        def process_rollout_data(*a, **k):
            result = orig(*a, **k)
            _record_shard(result)
            return result
        process_rollout_data.__wrapped__ = orig
        return process_rollout_data

    def wrap_loss_function(orig):
        def loss_function(args, batch, num_microbatches, logits, *a, **k):
            num_rollouts = k.get("num_rollouts", a[1] if len(a) > 1 else None)
            _RECORDS.append({"kind": "normalizer", "num_microbatches": int(num_microbatches),
                             "num_rollouts": None if num_rollouts is None else int(num_rollouts),
                             "loss_parallel_size": int(parallel_size())})
            return orig(args, batch, num_microbatches, logits, *a, **k)
        loss_function.__wrapped__ = orig
        return loss_function

    def wrap_get_loss_function(orig):
        def get_loss_function(*a, **k):
            func = orig(*a, **k)

            def recorded(args, batch, logits, sum_of_sample_mean, *rest, **kw):
                loss, log = func(args, batch, logits, sum_of_sample_mean, *rest, **kw)
                indices = batch.get("sample_indices") if isinstance(batch, dict) else None
                _RECORDS.append({"kind": "loss", "sample_indices": [int(i) for i in (indices or [])],
                                 "loss_hex": _float_hex(loss)})
                return loss, log
            return recorded
        get_loss_function.__wrapped__ = orig
        return get_loss_function

    return {"process": wrap_process, "loss_function": wrap_loss_function,
            "get_loss_function": wrap_get_loss_function}


def install_probe(actor: Any) -> dict[str, int]:
    """Idempotent per process; returns how many module attributes each wrapper replaced."""
    from .state_plugin import install_grad_norm_recorder

    install_grad_norm_recorder()  # the harness reads GRAD_NORM/APPLIED_LRS/STEP_LOSSES after each step
    if _INSTALLED:
        return {k: 0 for k in _INSTALLED}
    from miles.backends.training_utils import data as data_mod
    from miles.backends.training_utils import loss as loss_mod
    from miles.backends.training_utils.parallel import get_parallel_state

    wrappers = make_wrappers(lambda: get_parallel_state().intra_dp_cp.size)
    counts = {}
    for key, module, name in (("process", data_mod, "process_rollout_data"),
                              ("loss_function", loss_mod, "loss_function")):
        orig = getattr(module, name)
        counts[key] = patch_everywhere(orig, wrappers[key](orig))
        _INSTALLED[key] = orig
    # loss_function looks get_loss_function up in its own module globals.
    orig = loss_mod.get_loss_function
    loss_mod.get_loss_function = wrappers["get_loss_function"](orig)
    _INSTALLED["get_loss_function"] = orig
    counts["get_loss_function"] = 1
    if not counts["process"] or not counts["loss_function"]:
        raise RuntimeError(f"probe did not attach: {counts}")
    return counts


def drain_probe(actor: Any) -> list[dict[str, Any]]:
    out = list(_RECORDS)
    _RECORDS.clear()
    return out


def rank_info(actor: Any) -> dict[str, Any]:
    """Coordinate, seeds actually used (G5), RNG digest, advertised schedule config, dropout args."""
    import torch

    backend = _backend(actor)
    info: dict[str, Any] = {"coord": backend.coord(), "torch_initial_seed": int(torch.initial_seed())}
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        info["cuda_initial_seed"] = int(torch.cuda.initial_seed())
    try:
        from megatron.core import tensor_parallel

        info["megatron_tracker_digest"] = state_digest(tensor_parallel.get_cuda_rng_tracker().get_states())
    except Exception as exc:  # noqa: BLE001 - absent on CPU
        info["megatron_tracker_digest"] = f"unavailable: {type(exc).__name__}"
    info["rng_digest"] = state_digest(backend.capture_rng())
    info["train_parallel_config"] = dict(getattr(actor, "train_parallel_config", None) or {})
    args = actor.args
    info["dropout"] = {k: getattr(args, k, None) for k in ("lora_dropout", "hidden_dropout", "attention_dropout")}
    info["seed"] = getattr(args, "seed", None)
    info["data_parallel_random_init"] = bool(getattr(args, "data_parallel_random_init", False))
    return info


def dump_state(actor: Any, *, directory: str, tag: str) -> dict[str, Any]:
    """Write this rank's adapter, named optimizer state (FP32 main, moments, step) and scheduler."""
    import torch

    backend = _backend(actor)
    with trainer_resident(actor):
        coord = backend.coord()
        named = _adapters(actor, backend)
        with torch.no_grad():
            state = {
                "coord": dict(coord),
                "adapter": {n: p.detach().to("cpu").clone() for n, p in named},
                "optimizer_named": backend.export_optimizer(actor.optimizer, named),
                "scheduler": actor.opt_param_scheduler.state_dict(),
                "megatron_counters": backend.megatron_counters(),
                "weight_version": getattr(getattr(actor, "weight_updater", None), "weight_version", None),
                "rng_digest": state_digest(backend.capture_rng()),
            }
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{tag}_tp{coord['tp']}_pp{coord['pp']}_dp{coord['dp']}.pt")
    torch.save(to_safe(state), path)
    return {"path": path, "coord": dict(coord), "digest": state_digest(state)}
