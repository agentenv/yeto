"""Auto micro-batch sizing for the NAVA learner.

The LM probe (``yeto/autobatch.py``) tiles synthetic token ids, but NAVA's
``AudioVideoDataset`` batches *internally* (the per-GPU micro-batch is fixed when
the dataset is built), so there is nothing to tile. Instead we probe by rebuilding
a one-batch loader at each candidate size and running a real forward / backward /
step, doubling until it OOMs and keeping the largest size that fit.

Because a probe that picks the batch that *just* fits leaves no room for memory
peaks that only appear later in training — allocator fragmentation, variable latent
resolutions across clips, the DiLoCo sync boundary, activation-checkpoint recompute
spikes — the probe is deliberately conservative:

  * **VRAM headroom**: it probes under a memory cap (``_mem_fraction`` of the card,
    default 0.80), so the chosen batch is one that fits in ~80% of VRAM, leaving
    ~20% slack for training-time peaks. The cap is lifted before real training, so
    a batch sized for 80% then runs with the full card underneath it.
  * **steady state**: each candidate runs ``_PROBE_ITERATIONS`` fwd/bwd/step passes,
    so the allocator's fragmented steady state (not just the first clean pass) is
    what has to fit.
  * **low ceiling**: ``_MAX_MICRO_BATCH`` caps how extreme the result can get.

All three are overridable via env (``YETO_NAVA_PROBE_MEM_FRACTION``,
``YETO_NAVA_MAX_MICRO_BATCH``) for a given hardware/config. Runs only when
``--nava-batch-size auto``; an explicit int skips it (unchanged behavior), and CPU
skips it. All ranks in an island probe each size in lockstep and agree via a MIN
all-reduce, so one rank's OOM can't desynchronize the group. Across islands no
consensus is needed — DiLoCo syncs param-shaped LoRA fragments, independent of each
island's batch size.

The probe zeroes gradients before ``opt.step()`` so it materializes AdamW state
(part of the real footprint) without moving the weights.
"""
from __future__ import annotations

import copy
import logging
import os
from contextlib import nullcontext

import torch
import torch.distributed as dist

log = logging.getLogger("nava-learner")


def _max_micro_batch() -> int:
    # AV video activations are large; keep the ceiling low so 'auto' can never
    # pick something extreme even if a probe pass fits with luck.
    return max(1, int(os.environ.get("YETO_NAVA_MAX_MICRO_BATCH", "8")))


def _mem_fraction() -> float:
    # Fraction of VRAM the probe may use; the rest is reserved as headroom for
    # training-time peaks the probe cannot see.
    try:
        f = float(os.environ.get("YETO_NAVA_PROBE_MEM_FRACTION", "0.80"))
    except ValueError:
        f = 0.80
    return min(1.0, max(0.5, f))


_PROBE_ITERATIONS = 2  # second pass sees the allocator's fragmented steady state


def _set_mem_fraction(device, fraction: float) -> None:
    """Best-effort VRAM cap for probing; no-op if the driver won't take it."""
    try:
        idx = device.index if getattr(device, "index", None) is not None else torch.cuda.current_device()
        torch.cuda.set_per_process_memory_fraction(fraction, idx)
    except Exception as e:  # noqa: BLE001 - a cap we can't set just means no headroom
        log.warning("could not set probe memory fraction %.2f: %r", fraction, e)


def _unwrap(maybe_batch):
    # Mirror the training loop's collate unwrapping.
    if isinstance(maybe_batch, list) and len(maybe_batch) == 1:
        return maybe_batch[0]
    return maybe_batch


def _probe_once(build_loader, pipe, params, opt, cfg, micro_batch, global_step, world) -> None:
    trial = copy.deepcopy(cfg)
    trial["batch_size"] = micro_batch
    loader = build_loader(trial)
    model = pipe.model
    # Ranks probe DIFFERENT real batches, so a one-rank OOM must not hang the
    # others on DDP's backward all-reduce; disable grad sync during the probe and
    # rely on the MIN all-reduce below for consensus. (No effect at world==1.)
    sync_guard = model.no_sync() if (world > 1 and hasattr(model, "no_sync")) else nullcontext()
    try:
        done = 0
        for maybe_batch in loader:
            batch = _unwrap(maybe_batch)
            with sync_guard:
                loss, _ = pipe.forward(batch, global_step=global_step)
                loss.backward()
            with torch.no_grad():
                for p in params.values():
                    if p.grad is not None:
                        p.grad.zero_()
            opt.step()
            opt.zero_grad(set_to_none=True)
            done += 1
            if done >= _PROBE_ITERATIONS:  # steady-state fragmentation must fit too
                break
    finally:
        del loader


def resolve_nava_micro_batch(
    args, build_loader, pipe, params, opt, cfg, device, world: int, global_step: int
) -> int:
    """Return the per-GPU micro-batch to train with (probing when 'auto')."""
    if getattr(args, "nava_batch_size", None) != "auto":
        return int(cfg["batch_size"])
    if device.type != "cuda":
        return 1

    ceiling = _max_micro_batch()
    fraction = _mem_fraction()
    _set_mem_fraction(device, fraction)  # reserve headroom while probing
    try:
        best, size = 0, 1
        while size <= ceiling:
            ok = True
            try:
                _probe_once(build_loader, pipe, params, opt, cfg, size, global_step, world)
            except torch.cuda.OutOfMemoryError:
                ok = False
            except Exception as e:  # non-OOM failure: don't guess a size, stop here
                log.warning("NAVA batch probe aborted at size %d: %r; keeping %d", size, e, max(1, best))
                ok = False
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            if world > 1:
                flag = torch.tensor([1.0 if ok else 0.0], device=device)
                dist.all_reduce(flag, op=dist.ReduceOp.MIN)
                ok = flag.item() > 0
            if not ok:
                break
            best = size
            size *= 2
    finally:
        _set_mem_fraction(device, 1.0)  # give real training the whole card
        torch.cuda.empty_cache()

    best = max(1, best)
    log.info(
        "NAVA auto micro-batch: %d (per GPU; probed under %.0f%% VRAM, ceiling %d)",
        best, fraction * 100, ceiling,
    )
    return best
