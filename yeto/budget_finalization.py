"""Exact learner-budget consolidation for replicated torch trainable state."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

from .protocol import DTYPE_Q4, FinalManifest, SyncerClient, bulk_dtype

if TYPE_CHECKING:
    import torch


def validate_consolidation_tape(
    path: str | Path,
    *,
    cutoff_step: int,
    fragments: int,
    learners: int,
    budget_steps: int,
) -> None:
    """Require one full-participation ordinary round per terminal fragment."""
    records = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    terminal = [
        record
        for record in records
        if cutoff_step < int(record["step"]) <= cutoff_step + fragments
    ]
    expected_steps = set(range(cutoff_step + 1, cutoff_step + fragments + 1))
    expected_learners = set(range(learners))
    if len(terminal) != fragments or {
        int(record["step"]) for record in terminal
    } != expected_steps:
        raise RuntimeError("terminal consolidation tape does not cover every step")
    if {int(record["fragment"]) for record in terminal} != set(range(fragments)):
        raise RuntimeError("terminal consolidation tape does not cover every fragment")
    for record in terminal:
        responders = record.get("responders", [])
        if len(responders) != learners or {
            int(item["id"]) for item in responders
        } != expected_learners:
            raise RuntimeError("terminal consolidation did not include every learner")
        if any(int(item["c_steps"]) != budget_steps for item in responders):
            raise RuntimeError("terminal consolidation used the wrong learner budget")


def validate_learner_budget_args(args) -> None:
    target = args.learner_budget_steps
    if target is None:
        return
    if args.syncer == "none":
        raise ValueError("--learner-budget-steps requires a syncer")
    if target <= 0:
        raise ValueError("--learner-budget-steps must be positive")
    if target != args.max_local_steps:
        raise ValueError("--learner-budget-steps must equal --max-local-steps")
    if target > 0xFFFF_FFFF:
        raise ValueError("--learner-budget-steps must fit the protocol c_steps u32")
    if args.tuning != "lora":
        raise ValueError("--learner-budget-steps requires --tuning lora")


def finalize_learner_budget(
    client: SyncerClient | None,
    layout,
    params,
    *,
    rank: int,
    world: int,
    device: torch.device,
    target_steps: int,
    units: int,
) -> FinalManifest:
    """Freeze at ``target_steps``, reconnect, contribute every fragment, then finalize."""
    import torch.distributed as dist

    from .finalization import finalize_torch_island
    from .tensor_io import fragment_flat, pack_tensor, quantize_q4, unpack_fragment

    if target_steps <= 0 or target_steps > 0xFFFF_FFFF:
        raise ValueError("learner budget steps must be in [1, 2^32-1]")
    if units <= 0:
        raise ValueError("learner budget units must be positive")
    if world > 1:
        dist.barrier()

    if rank == 0:
        if client is None:
            raise RuntimeError("learner-budget finalization requires a syncer client")
        generation = client.send_budget_done(target_steps)
        client.wait_for_budget_restart(generation)
        bases: dict[int, tuple[int, torch.Tensor]] = {}
        completed: set[int] = set()

        while len(completed) < layout.num_fragments:
            deadline = time.monotonic() + client.finalization_timeout
            pull = None
            while pull is None or pull.fragment_id not in bases:
                client.check_health()
                for update in client.drain_updates():
                    fid = update.fragment_id
                    if fid in completed:
                        continue
                    base = unpack_fragment(
                        layout.fragments[fid], update.data, bulk_dtype(client.dtype)
                    )
                    bases[fid] = (update.version, base)
                pulls = client.drain_pulls()
                if pulls:
                    if pull is not None or len(pulls) != 1:
                        raise RuntimeError(
                            "budget consolidation received multiple concurrent pulls"
                        )
                    pull = pulls[0]
                    if pull.fragment_id in completed:
                        raise RuntimeError(
                            f"budget consolidation repeated fragment {pull.fragment_id}"
                        )
                if pull is not None and pull.fragment_id in bases:
                    break
                if time.monotonic() >= deadline:
                    missing = "pull" if pull is None else f"fragment {pull.fragment_id} base"
                    raise TimeoutError(
                        f"budget consolidation timed out waiting for {missing}"
                    )
                time.sleep(0.01)

            fid = pull.fragment_id
            base_version, base = bases.pop(fid)
            frozen = fragment_flat(layout.fragments[fid], params).detach().cpu()
            delta = frozen - base
            payload = (
                quantize_q4(delta)
                if client.dtype == DTYPE_Q4
                else pack_tensor(delta, client.dtype)
            )
            client.push_fragment(
                fid,
                pull.global_step,
                pull.round_attempt,
                base_version,
                target_steps,
                target_steps,
                units,
                payload,
            )
            completed.add(fid)

    manifest = finalize_torch_island(
        client,
        layout,
        params,
        rank=rank,
        world=world,
        device=device,
    )
    return manifest
