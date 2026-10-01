"""The learning rate each optimizer step actually applies (fix-decoupled-lr-schedule D4).

Miles steps the LR scheduler *after* ``optimizer.step()`` and logs
``train/lr-pg_0`` after that, so the logged value is always the LR of the
*next* step (0 after the last strict step).  The zero-LR invariant must read
the LR inside the step instead: ``optimizer.param_groups[*]["lr"]`` when the
step is entered, before the scheduler advances.

* ports: ``state_plugin.install_grad_norm_recorder`` wraps upstream
  ``train_one_step`` and calls :func:`optimizer_lr` before the step;
* legacy: ``yeto.rl.learner`` sets the fork's existing
  ``--custom-megatron-before-train-step-hook-path`` to :data:`HOOK_PATH`, a
  combined hook that records the LR and then runs the grad audit hook
  (``yeto.rl.grad_audit.before_train_step``) when that audit is configured, so
  the two never overwrite each other.  The fork's exported ``TrainableState``
  has no room for extra fields, so the hook writes one small JSON record per
  step into ``args.yeto_rl_applied_lr_dir`` (global rank 0 only) and the
  legacy policy-sync bridge consumes them with :func:`collect`.

torch is imported lazily; everything here is CPU-testable.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

APPLIED_LR_DIR_ATTR = "yeto_rl_applied_lr_dir"
HOOK_PATH = "yeto.rl.applied_lr.before_train_step"


class AppliedLrError(RuntimeError):
    pass


def optimizer_lr(optimizer: Any) -> float:
    """LR the next ``optimizer.step()`` applies (max over param groups).

    Megatron's ``ChainedOptimizer`` exposes the union of its children's
    ``param_groups``; each child is also inspected when the property is absent.
    """

    groups: list[Any] = []
    param_groups = getattr(optimizer, "param_groups", None)
    if param_groups is not None:
        groups.extend(param_groups)
    else:
        for child in getattr(optimizer, "chained_optimizers", ()):
            groups.extend(getattr(child, "param_groups", ()))
    values = [float(group["lr"]) for group in groups if "lr" in group]
    if not values:
        raise AppliedLrError("optimizer exposes no param_groups learning rate")
    if not all(math.isfinite(v) for v in values):
        raise AppliedLrError(f"non-finite optimizer learning rate {values}")
    return max(values)


def _record_path(directory: str | Path, rollout_id: int, step_id: int) -> Path:
    return Path(directory) / f"rollout-{int(rollout_id):08d}.step-{int(step_id):03d}.json"


def record(args: Any, rollout_id: int, step_id: int, optimizer: Any, *, is_writer=None) -> float | None:
    """Write this step's applied LR (legacy side channel); returns the LR."""

    if optimizer is None:
        return None
    directory = getattr(args, APPLIED_LR_DIR_ATTR, None)
    if not directory:
        return None
    if is_writer is None:
        from .grad_audit import _is_writer as is_writer
    lr = optimizer_lr(optimizer)
    if not is_writer():
        return lr
    path = _record_path(directory, rollout_id, step_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps({"rollout_id": int(rollout_id), "step_id": int(step_id), "lr": lr}),
        encoding="utf-8",
    )
    os.replace(tmp, path)
    return lr


def collect(directory: str | Path, rollout_id: int, steps: int) -> tuple[float, ...]:
    """Consume the per-step records of ``rollout_id`` (steps ``0..steps-1``)."""

    values = []
    paths = []
    for step_id in range(int(steps)):
        path = _record_path(directory, rollout_id, step_id)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise AppliedLrError(
                f"rollout {rollout_id}: no applied learning rate recorded for "
                f"optimizer step {step_id} in {directory}"
            ) from error
        if payload.get("rollout_id") != int(rollout_id) or payload.get("step_id") != step_id:
            raise AppliedLrError(f"applied learning rate record {path} is for another step")
        values.append(float(payload["lr"]))
        paths.append(path)
    for path in paths:
        path.unlink()
    return tuple(values)


def before_train_step(args, rollout_id, step_id, model, optimizer, opt_param_scheduler) -> None:
    """Legacy Miles before-train-step hook: record the LR, then the grad audit.

    Runs before the forward/backward and ``optimizer.step()``; the scheduler has
    not advanced yet, so ``param_groups[*]["lr"]`` is the LR this step applies.
    """

    record(args, rollout_id, step_id, optimizer)
    from . import grad_audit

    if getattr(args, grad_audit.GRAD_AUDIT_DIR_ATTR, None):
        grad_audit.before_train_step(
            args, rollout_id, step_id, model, optimizer, opt_param_scheduler
        )


def summarize(values: Sequence[float]) -> float | None:
    """The round's applied LR as reported in ``rl_local_round`` (min over steps)."""

    return min(values) if values else None
