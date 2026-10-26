"""The learning rate each optimizer step actually applies (fix-decoupled-lr-schedule D4).

Miles steps the LR scheduler *after* ``optimizer.step()`` and logs
``train/lr-pg_0`` after that, so the logged value is always the LR of the
*next* step (0 after the last strict step).  The zero-LR invariant must read
the LR inside the step instead: ``optimizer.param_groups[*]["lr"]`` when the
step is entered, before the scheduler advances.  The ports state plugin
(``state_plugin.install_grad_norm_recorder``) wraps upstream ``train_one_step``
and calls :func:`optimizer_lr` before the step.
"""

from __future__ import annotations

import math
from typing import Any


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


