"""Runtime patches around defects of verl's own Megatron engine (S27 G-FT).

Everything else yeto does for the verl Megatron island is wiring on our side:
fifteen blockers (S27 G-FN), eleven (G-FS), six (G-FU) -- every one of them was
closed in yeto, with ``verl`` unchanged.  This module is the third kind, and so
far the only member of it: **a defect inside verl that no configuration of ours
avoids**, worked around at run time instead of by editing the shared
``verl-fork`` checkout (which other runs share) or by copying verl logic into
yeto (which would drift silently when verl changes).

**The defect** (verified by reading verl at the two places named below, and by
run ``ft-2`` on npu1):

* ``verl/trainer/ppo/ray_trainer.py:1311`` reads the inference MFU of the
  old-log-prob pass as ``tu.get(output, "metrics")["mfu"]``, unconditionally;
* verl's Megatron engine leaves ``metrics`` an **empty dict** on its
  forward-only path (``verl/workers/engine/megatron/transformer_impl.py:1400``,
  the ``loss_function is None`` branch), and the worker group's collection turns
  the per-rank empties into a **list**;
* verl's FSDP engine does put real metrics there
  (``verl/workers/engine/fsdp/transformer_impl.py:1654``), so the line works on
  FSDP and cannot work on Megatron.

The value is only reported (``"perf/mfu/actor_infer"`` in verl's metrics dict,
``ray_trainer.py:1598``), so nothing in training depends on it.

**What the patch does, and nothing else**: while
``RayPPOTrainer._compute_old_log_prob`` runs, a read of the ``"metrics"`` field
that does not give a mapping with ``"mfu"`` in it gives one whose ``"mfu"`` is
None.  Reads of other keys, and reads of ``"metrics"`` outside that one call,
are the original function.  So the two other ``tu.get(..., "metrics")`` sites in
the same file (``:1370``, ``:1398``, the update_actor path, where the Megatron
engine *does* fill metrics) are untouched.

**Remove this when upstream fixes it.** :func:`upstream_report` is the text to
send with the report; the fix there is one line (read the MFU with a default, or
have the Megatron engine name the key).

The patch is installed in the island's trainer actor process only, by
:func:`install`, which logs one line saying what it did.  It is idempotent and
single-threaded: the trainer driver calls the train step serially.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from typing import Any

MFU_KEY = "mfu"
METRICS_KEY = "metrics"
_INSTALLED = False


class _MfuOptional(dict):
    """A metrics mapping whose missing ``mfu`` reads as None instead of raising."""

    def __missing__(self, key: str) -> Any:
        if key == MFU_KEY:
            return None
        raise KeyError(key)


def normalize_metrics(value: Any) -> Any:
    """``value`` as a metrics mapping that has an ``mfu`` entry, or ``value`` itself.

    A mapping that already names ``mfu`` is returned unchanged, so a path that
    does report MFU keeps reporting it.  A list (the Megatron engine's per-rank
    empties, collected) is merged in order, which keeps any real entry a rank
    did set."""

    if isinstance(value, Mapping) and MFU_KEY in value:
        return value
    merged = _MfuOptional()
    if isinstance(value, Mapping):
        merged.update(value)
    elif isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, Mapping):
                merged.update(item)
    else:
        return value
    return merged


def install() -> bool:
    """Install the patch in this process; True when it was installed now.

    Returns False when it is already installed, or when the verl in this process
    does not have the two symbols the patch needs (then nothing is changed: an
    upstream that moved on should not be patched blindly)."""

    global _INSTALLED

    if _INSTALLED:
        return False
    try:
        from verl.trainer.ppo import ray_trainer
        from verl.utils import tensordict_utils
    except ImportError:
        return False
    trainer = getattr(ray_trainer, "RayPPOTrainer", None)
    original = getattr(trainer, "_compute_old_log_prob", None)
    if original is None or not hasattr(tensordict_utils, "get"):
        return False

    @functools.wraps(original)
    def _compute_old_log_prob(self, *args, **kwargs):
        plain_get = tensordict_utils.get

        def patched_get(tensordict, key, default=None):
            value = plain_get(tensordict, key, default)
            return normalize_metrics(value) if key == METRICS_KEY else value

        tensordict_utils.get = patched_get
        try:
            return original(self, *args, **kwargs)
        finally:
            tensordict_utils.get = plain_get

    trainer._compute_old_log_prob = _compute_old_log_prob
    _INSTALLED = True
    print("[yeto-verl-patch] RayPPOTrainer._compute_old_log_prob: a missing metrics['mfu'] now "
          "reads as None (verl ray_trainer.py:1311 reads it unconditionally; verl's Megatron "
          "engine leaves metrics empty on its forward-only path, transformer_impl.py:1400). "
          "MFU reporting only; remove when upstream fixes it.", flush=True)
    return True


def upstream_report() -> str:
    """The defect as one paragraph, for an upstream issue."""

    return (
        "verl/trainer/ppo/ray_trainer.py:1311 reads old_log_prob_mfu as "
        "tu.get(output, \"metrics\")[\"mfu\"] without a default. On the Megatron engine the "
        "old-log-prob pass is forward-only, and verl/workers/engine/megatron/transformer_impl.py:1400 "
        "sets metrics = {} in that branch, so the collected field holds no \"mfu\" (and, after the "
        "worker group's collection, is a list): any PPO step on the Megatron engine raises "
        "TypeError: list indices must be integers or slices, not str. The FSDP engine fills metrics "
        "(fsdp/transformer_impl.py:1654), which is why only Megatron is affected. The value is "
        "reported as perf/mfu/actor_infer and nothing in training uses it, so reading it with a "
        "default -- or having the Megatron engine name the key -- is enough."
    )
