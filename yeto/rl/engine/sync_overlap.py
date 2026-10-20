"""Sync/train overlap for the full-parameter sweep (S26 E3 item 3, E2 option).

Round k's outer delta is pushed at boundary k and the island trains round
k+1 immediately on its own weights. At boundary k+1 it waits for the global
result of sweep k (``G_k``) and rebases in place::

    theta <- G_k + (theta_local_{k+1} - base_k)

``base_k`` is the cut round k+1 started from. ``G_k`` becomes the anchor, so
the next push ``theta - G_k`` equals round k+1's progress exactly once: round
k's delta is already inside ``G_k`` and is not counted again.

Rollouts of round k+1 use weights one outer step older than the global
policy, so the run's minimum ``--rl-max-policy-age`` grows by
:data:`EXTRA_POLICY_AGE`.
"""

from __future__ import annotations

import numpy as np

EXTRA_POLICY_AGE = 1


def required_policy_age(minimum: int) -> int:
    """The smallest accepted ``--rl-max-policy-age`` once overlap is on."""
    return int(minimum) + EXTRA_POLICY_AGE


def rebase_values(anchor: np.ndarray, current: np.ndarray, base: np.ndarray) -> np.ndarray:
    """``anchor + (current - base)`` in f32 (the trainer keeps its progress)."""
    if not (anchor.shape == current.shape == base.shape):
        raise ValueError("rebase operands have different shapes")
    for name, value in (("anchor", anchor), ("current", current), ("base", base)):
        if value.dtype != np.float32:
            raise ValueError(f"rebase {name} must be float32")
    out = np.subtract(current, base, dtype=np.float32)
    np.add(anchor, out, out=out)
    if not np.isfinite(out).all():
        raise ValueError("rebased fragment has a non-finite value")
    return out
