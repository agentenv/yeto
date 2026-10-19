"""verl / FSDP2 binding of the resume assertions (S25 N7.6; A-TRN report #8).

The checks live in :mod:`yeto.rl.engine.resume_manifest`; this module only maps
a verl FSDP actor worker onto them. verl's ``ActorRolloutRefWorker`` holds the
FSDP2-wrapped actor, its ``torch.optim`` optimizer and LR scheduler as
``actor_module_fsdp``, ``actor_optimizer`` and ``actor_lr_scheduler`` (attribute
names from verl's fsdp worker; not yet checked against a live verl build).

Each rank reads its own FSDP2 shard (``DTensor.to_local``), so the per-rank
state is keyed by rank. Use:

* after the checkpoint is complete: ``write_resume_manifest(dir, worker_manifest(...))``;
* after loading it, before the first step: :func:`require_worker_restored`;
* after the first step: :func:`require_worker_first_step`.

A multi-rank run gathers the per-rank states (e.g. ``all_gather_object``) into
one :class:`ResumeManifest` before writing; every rank then checks its own entry.
"""

from __future__ import annotations

from typing import Any

from yeto.rl.engine.resume_manifest import (
    DEFAULT_SAMPLES,
    OptimizerResumeState,
    ResumeAssertionError,
    ResumeManifest,
    require_first_step,
    require_restored,
    torch_optimizer_state,
)

WORKER_ATTRS = ("actor_module_fsdp", "actor_optimizer", "actor_lr_scheduler")


def rank_label(rank: int) -> str:
    return f"rank{int(rank)}"


def worker_rank_state(worker: Any, *, samples: int = DEFAULT_SAMPLES) -> OptimizerResumeState:
    """This rank's optimizer/scheduler state of a verl FSDP actor worker."""
    missing = [a for a in WORKER_ATTRS if getattr(worker, a, None) is None]
    if missing:
        raise ResumeAssertionError(f"verl worker {type(worker).__name__} lacks {missing}; "
                                   "cannot check the resumed optimizer")
    return torch_optimizer_state(worker.actor_optimizer, worker.actor_module_fsdp.named_parameters(),
                                 worker.actor_lr_scheduler, samples=samples)


def worker_manifest(worker: Any, *, rank: int, global_step: int, data_watermark: int,
                    samples: int = DEFAULT_SAMPLES) -> ResumeManifest:
    """Single-rank manifest (``scheduler_increment`` is 1: verl steps its torch scheduler once per step)."""
    return ResumeManifest(global_step=int(global_step), data_watermark=int(data_watermark), scheduler_increment=1,
                          ranks={rank_label(rank): worker_rank_state(worker, samples=samples)})


def _own(saved: ResumeManifest, rank: int) -> ResumeManifest:
    label = rank_label(rank)
    if label not in saved.ranks:
        raise ResumeAssertionError(f"resume manifest has no state for {label} (ranks: {sorted(saved.ranks)})")
    return ResumeManifest(saved.global_step, saved.data_watermark, saved.scheduler_increment,
                          {label: saved.ranks[label]})


def require_worker_restored(saved: ResumeManifest, worker: Any, *, rank: int, global_step: int,
                            data_watermark: int) -> None:
    require_restored(_own(saved, rank), worker_manifest(worker, rank=rank, global_step=global_step,
                                                        data_watermark=data_watermark))


def require_worker_first_step(saved: ResumeManifest, worker: Any, *, rank: int, global_step: int,
                              data_watermark: int, steps: int = 1) -> None:
    require_first_step(_own(saved, rank), worker_manifest(worker, rank=rank, global_step=global_step,
                                                          data_watermark=data_watermark), steps=steps)
