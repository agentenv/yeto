"""verl / FSDP2 binding of the resume assertions (S25 N7.6; A-TRN report #8).

The checks live in :mod:`yeto.rl.engine.resume_manifest`; this module only maps
a verl FSDP actor worker onto them (:func:`train_handles`). Two worker shapes:

* the engine worker of the pinned verl (``engine_workers.ActorRolloutRefWorker``):
  ``worker.actor.engine`` (a bare ``TrainingWorker``: ``worker.engine``) holds ``module`` / ``optimizer`` / ``lr_scheduler``
  (``workers/engine/fsdp/transformer_impl.py``);
* the legacy fsdp worker: ``actor_module_fsdp`` / ``actor_optimizer`` /
  ``actor_lr_scheduler``.

Attribute names are read from the verl source; not yet checked on a live verl build.
The cut save/restore path that uses these handles is :mod:`.cut`.

Each rank reads its own FSDP2 shard (``DTensor.to_local``), so the per-rank
state is keyed by rank. Use:

* after the checkpoint is complete: ``write_resume_manifest(dir, worker_manifest(...))``;
* after loading it, before the first step: :func:`require_worker_restored`;
* after the first step: :func:`require_worker_first_step`.

A multi-rank run gathers the per-rank states (e.g. ``all_gather_object``) into
one :class:`ResumeManifest` before writing; every rank then checks its own entry.
"""

from __future__ import annotations

from dataclasses import dataclass
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

ENGINE_ATTRS = ("module", "optimizer", "lr_scheduler")
WORKER_ATTRS = ("actor_module_fsdp", "actor_optimizer", "actor_lr_scheduler")


@dataclass(frozen=True)
class TrainHandles:
    """The FSDP2 actor module, its ``torch.optim`` optimizer and LR scheduler on one rank."""

    module: Any
    optimizer: Any
    scheduler: Any


def _engine_of(worker: Any) -> Any:
    inner = getattr(worker, "worker_dict", None)  # ray fused worker: {role: worker}
    if inner:
        worker = next((w for w in inner.values() if hasattr(w, "actor")), worker)
    actor = getattr(worker, "actor", None) or worker  # ActorRolloutRefWorker.actor is a TrainingWorker
    return getattr(actor, "engine", None) or worker


def train_handles(worker: Any) -> TrainHandles:
    """Resolve the train handles of a verl worker (engine or legacy shape, see module doc)."""
    if all(getattr(worker, a, None) is not None for a in WORKER_ATTRS):
        return TrainHandles(*(getattr(worker, a) for a in WORKER_ATTRS))
    engine = _engine_of(worker)
    missing = [a for a in ENGINE_ATTRS if getattr(engine, a, None) is None]
    if missing:
        raise ResumeAssertionError(f"verl worker {type(worker).__name__} lacks {missing} (engine) and "
                                   f"{list(WORKER_ATTRS)} (legacy); cannot check the resumed optimizer")
    return TrainHandles(*(getattr(engine, a) for a in ENGINE_ATTRS))


def rank_label(rank: int) -> str:
    return f"rank{int(rank)}"


def handles_rank_state(handles: TrainHandles, *, samples: int = DEFAULT_SAMPLES) -> OptimizerResumeState:
    return torch_optimizer_state(handles.optimizer, handles.module.named_parameters(), handles.scheduler,
                                 samples=samples)


def worker_rank_state(worker: Any, *, samples: int = DEFAULT_SAMPLES) -> OptimizerResumeState:
    """This rank's optimizer/scheduler state of a verl FSDP actor worker."""
    return handles_rank_state(train_handles(worker), samples=samples)


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
