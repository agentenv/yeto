"""TEST ONLY: make one training step's loss NaN, to smoke-test the NaN skip (YETO_RL_NAN_SKIP).

``YETO_K31_TEST_NAN_INJECT_ROLLOUT=<rollout_id>`` (unset: off) arms, in the before-train-step hook of
that rollout, a one-shot forward hook on the first pipeline stage's embedding that multiplies its output
by NaN. The next training forward (the first microbatch of that step) then carries NaN through every
stage, so the loss, every gradient and Megatron's grad norm are NaN; the hook disarms itself after that
one forward, so the log-prob forwards of the next rollout are clean. Only ranks holding the embedding
(PP stage 0) inject; the grad norm is reduced over all ranks, so every rank sees the same NaN.

What the smoke checks (preflight 2.3d): the optimizer's get_grad_norm returns NaN, Miles skips
optimizer.step and the weight-version bump for that step, the driver emits ``rl_nan_skip``, and the next
step trains normally. Called from ``first_stage_grad.before_train_step`` when the variable is set.
"""

from __future__ import annotations

import os
from typing import Any

ENV = "YETO_K31_TEST_NAN_INJECT_ROLLOUT"
_HANDLES: list[Any] = []


def target_rollout(environ=os.environ) -> int | None:
    raw = (environ.get(ENV) or "").strip()
    return int(raw) if raw else None


def _embeddings(model: Any) -> list[Any]:
    from .first_stage_grad import _unwrap

    chunks = model if isinstance(model, (list, tuple)) else [model]
    return [e for e in (getattr(_unwrap(c), "embedding", None) for c in chunks) if e is not None]


def arm(model: Any) -> int:
    """Register the one-shot NaN hook on every local embedding; returns how many were armed."""
    embeddings = _embeddings(model)
    for embedding in embeddings:
        def hook(module, _inputs, output, _state={"fired": False}):
            if _state["fired"] or not module.training:
                return output
            _state["fired"] = True
            for handle in list(_HANDLES):
                handle.remove()
                _HANDLES.remove(handle)
            return output * float("nan")

        _HANDLES.append(embedding.register_forward_hook(hook))
    return len(embeddings)


def maybe_arm(rollout_id: int, model: Any, environ=os.environ) -> int:
    target = target_rollout(environ)
    if target is None or int(rollout_id) != target:
        return 0
    armed = arm(model)
    print(f"[yeto] TEST ONLY {ENV}={target}: NaN armed on {armed} embedding(s)", flush=True)
    return armed
