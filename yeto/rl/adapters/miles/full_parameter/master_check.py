"""Read-only check after a full-parameter apply: the BF16 model equals the
BF16 projection of its FP32 master (SC2 P6 / D4).

Off by default. ``YETO_FULL_PARAMETER_MASTER_CHECK=N`` checks the first ``N``
applies of this island (each pass hashes every trainable parameter on every
rank, so keep ``N`` small). Per rank :func:`rank_master_model_check` returns
the sha256 of the model's BF16 bytes and of the master's BF16 projection, the
number of parameters compared and skipped (a DP-sharded master holds only a
slice), and, under ``--optimizer-cpu-offload``, whether every
HybridDeviceOptimizer CPU copy is pinned and the sha256 of its BF16
projection. :func:`report_master_check` prints one
``[yeto-master-check]`` line per rank. Nothing is written back; a failure is
printed and never stops the island.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from typing import Any

MASTER_CHECK = f"{__name__}.rank_master_model_check"
MASTER_CHECK_ENV = "YETO_FULL_PARAMETER_MASTER_CHECK"
TAG = "[yeto-master-check]"

_DONE = {"applies": 0}


def check_budget(environ: Any = None) -> int:
    raw = (os.environ if environ is None else environ).get(MASTER_CHECK_ENV, "0")
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


def _bf16_bytes(tensor: Any) -> bytes:
    import torch

    value = tensor.detach()
    if value.dtype != torch.bfloat16:
        value = value.to(torch.bfloat16)
    return value.contiguous().view(torch.int16).cpu().numpy().tobytes()


def _hybrid_copies(optimizer: Any) -> dict[int, Any]:
    """id(optimizer-owned tensor) -> HybridDeviceOptimizer inner FP32 copy."""
    copies: dict[int, Any] = {}
    for child in list(getattr(optimizer, "chained_optimizers", None) or (optimizer,)):
        base = getattr(child, "optimizer", child)
        for owned, copy in (getattr(base, "param_to_inner_param", None) or {}).items():
            copies[id(owned)] = copy
    return copies


def master_model_digest(named: Any, copies: dict[int, Any]) -> dict[str, Any]:
    """The check over ``named`` = ``(name, model parameter)`` pairs (see module doc)."""
    model, master, inner = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    compared = skipped = cpu_copies = unpinned = 0
    for name, parameter in named:
        main = getattr(parameter, "main_param", None)
        if main is None and id(parameter) in copies:  # precision-aware: the hybrid copy is the master
            main = copies[id(parameter)]
        if main is None or main.numel() != parameter.numel() or getattr(parameter, "main_param_sharded", False):
            skipped += 1
            continue
        key = name.encode()
        model.update(key)
        model.update(_bf16_bytes(parameter))
        master.update(key)
        master.update(_bf16_bytes(main.view(parameter.shape)))
        copy = copies.get(id(main))
        if copy is not None and copy is not main:
            cpu_copies += 1
            if copy.device.type == "cpu" and not copy.is_pinned():
                unpinned += 1
            inner.update(key)
            inner.update(_bf16_bytes(copy.view(parameter.shape)))
        compared += 1
    out = {"compared": compared, "skipped": skipped, "model_bf16_sha256": model.hexdigest(),
           "master_bf16_sha256": master.hexdigest()}
    out["equal"] = compared > 0 and out["model_bf16_sha256"] == out["master_bf16_sha256"]
    if cpu_copies:
        out.update(cpu_copies=cpu_copies, cpu_copies_pinned=unpinned == 0,
                   cpu_copy_bf16_sha256=inner.hexdigest())
        out["equal"] = out["equal"] and out["cpu_copy_bf16_sha256"] == out["model_bf16_sha256"]
    return out


def distributed_ranges(optimizer: Any) -> list[tuple[str, Any, Any]]:
    """``(key, model range, FP32 master range)`` of every distributed-optimizer
    range on this rank (Megatron pairs ``model_float16_groups`` with the BF16
    views ``shard_float16_groups`` and the FP32 masters
    ``shard_fp32_from_float16_groups``; under the precision-aware optimizer the
    master is the view's HybridDeviceOptimizer copy). Empty without a
    distributed optimizer."""
    out: list[tuple[str, Any, Any]] = []
    for c, child in enumerate(list(getattr(optimizer, "chained_optimizers", None) or (optimizer,))):
        views = getattr(child, "shard_float16_groups", None)
        if views is None:
            continue
        masters = getattr(child, "shard_fp32_from_float16_groups", None)
        inner = getattr(getattr(child, "optimizer", None), "param_to_inner_param", None) or {}
        for g, group in enumerate(views):
            for i, view in enumerate(group):
                if view is None or view.numel() == 0:
                    continue
                master = masters[g][i] if masters is not None else None
                if master is None:
                    master = inner.get(view)
                out.append((f"{c}.{g}.{i}", view, master))
    return out


def distributed_digest(ranges: Any, copies: dict[int, Any]) -> dict[str, Any]:
    """The check over distributed-optimizer ranges (same fields as
    :func:`master_model_digest`; the model side is this rank's slice)."""
    model, master, inner = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    compared = skipped = cpu_copies = unpinned = 0
    for key, view, main in ranges:
        if main is None or view.numel() == 0 or main.numel() != view.numel():
            skipped += 1
            continue
        tag = key.encode()
        model.update(tag)
        model.update(_bf16_bytes(view))
        master.update(tag)
        master.update(_bf16_bytes(main.view(view.shape)))
        copy = copies.get(id(main))
        if copy is None:
            copy = copies.get(id(view))
        if copy is not None and copy is not main:
            cpu_copies += 1
            if copy.device.type == "cpu" and not copy.is_pinned():
                unpinned += 1
            inner.update(tag)
            inner.update(_bf16_bytes(copy.view(view.shape)))
        elif copy is main and main.device.type == "cpu":
            cpu_copies += 1
            if not main.is_pinned():
                unpinned += 1
        compared += 1
    out = {"compared": compared, "skipped": skipped, "model_bf16_sha256": model.hexdigest(),
           "master_bf16_sha256": master.hexdigest(), "distributed": True}
    out["equal"] = compared > 0 and out["model_bf16_sha256"] == out["master_bf16_sha256"]
    if cpu_copies:
        out.update(cpu_copies=cpu_copies, cpu_copies_pinned=unpinned == 0)
        if inner.hexdigest() != hashlib.sha256().hexdigest():
            out["cpu_copy_bf16_sha256"] = inner.hexdigest()
            out["equal"] = out["equal"] and out["cpu_copy_bf16_sha256"] == out["model_bf16_sha256"]
    return out


def rank_master_model_check(actor: Any) -> dict[str, Any]:
    """Trainer-rank plugin (``run_plugin``): the check on this rank's model chunks
    (per parameter), or on its distributed-optimizer ranges when the masters are
    DP-sharded (Megatron's distributed optimizer, also at DP=1 over CP)."""
    copies = _hybrid_copies(actor.optimizer)
    ranges = distributed_ranges(actor.optimizer)
    if ranges:
        return distributed_digest(ranges, copies)
    chunks = actor.model if isinstance(actor.model, (list, tuple)) else [actor.model]
    named = [(f"{index}.{name}", parameter) for index, chunk in enumerate(chunks)
             for name, parameter in chunk.named_parameters() if parameter.requires_grad]
    return master_model_digest(named, copies)


async def report_master_check(group: Any, role: str, *, environ: Any = None) -> None:
    """After an apply: run the check on every rank while the budget lasts."""
    budget = check_budget(environ)
    if _DONE["applies"] >= budget:
        return
    _DONE["applies"] += 1
    run_plugin = getattr(group, "run_plugin", None)
    if run_plugin is None:
        print(f"{TAG} role={role} skipped: the group has no run_plugin", file=sys.stderr, flush=True)
        return
    try:
        results = await run_plugin(MASTER_CHECK, {})
    except Exception as error:  # noqa: BLE001 - diagnostics must not stop the island
        print(f"{TAG} role={role} failed: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return
    for rank, result in enumerate(results or ()):
        print(f"{TAG} role={role} apply={_DONE['applies']} rank={rank} {json.dumps(result, sort_keys=True)}",
              file=sys.stderr, flush=True)
