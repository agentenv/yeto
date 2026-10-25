"""Full-parameter verl trainers on the fully_async ports (S27 G-EW; VERL-PARITY #14, #11).

The fully_async trainer actor (:mod:`.fully_async_runner`) and the buffer trainer
(:mod:`.buffer_trainer_runner`) hold the same verl FSDP2 actor workers as the
synchronous trainer; only the way the ports reach them differs
(``FullyAsyncIsland.call_workers`` goes through ``yeto_call_workers`` on the
trainer actor). A full-parameter island on these paths therefore uses the pieces
of the synchronous full-parameter island unchanged:

* identity (:func:`full_parameter_hashes`): the model config is the identity
  (``full_parameter_identity``, in the island's LoRA config slot); the layout hash
  is the fragment layout (the plan's for a synced island, else the local state's);
* sync and policy state (:func:`full_parameter_ports`):
  :func:`.full_parameter_stream.full_parameter_policy`, the outer full-parameter
  stream or, without a syncer, the local-only state; both read and write the
  trainer workers through ``island.call_workers`` / ``island.call_workers_shared``;
* the trainer port (``FullyAsyncTrainerGroup.train_step``) records each trained
  round on ``island.full_group``, which the stream binds the round to.

What differs by path stays with the caller: the buffer trainer releases full
weights into the store (``buffer_ports.full_release_weights``); a fully_async
trainer with its own rollouter publishes over verl's checkpoint engine
(VERL-PARITY #11).

Refused here, before the driver starts, because these paths do not wire them:
round cuts (``--rl-resume-store``), HF exports and the data position store.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# plan key -> what the user asked for; not wired on the fully_async paths
UNWIRED_PLAN_KEYS = (
    ("resume", "round cuts (--rl-resume-store)"),
    ("hf_exports", "HF exports (--rl-export-hf-rounds)"),
    ("data_position_store", "--rl-data-position-store"),
    ("publish_selftest", "the LoRA publish self-test"),
)


def is_full_parameter(plan: Mapping[str, Any]) -> bool:
    return plan.get("parameter_mode") == "full"


def full_parameter_hashes(plan: Mapping[str, Any]) -> tuple[str | None, str]:
    """``(layout hash, identity hash)`` for the island before its ports exist: the
    synced island's fragment layout from the plan (None without a syncer: the
    local state plans it) and the model-config identity."""
    from .full_parameter_stream import PLAN_KEY, full_parameter_identity

    _identity_layout, identity = full_parameter_identity(plan["model_config_sha256"])
    section = plan.get(PLAN_KEY)
    return (section["parameter_layout_sha256"] if section else None), identity


def refuse_unwired(plan: Mapping[str, Any]) -> None:
    asked = [label for key, label in UNWIRED_PLAN_KEYS if plan.get(key)]
    if asked:
        raise ValueError("a full-parameter verl island on the fully_async path does not wire "
                         + ", ".join(asked))


def full_parameter_ports(island: Any, plan: Mapping[str, Any], *, backend_identity_sha256: str):
    """``(sync, policy state)`` of a full-parameter fully_async or buffer trainer
    island; sets ``island.full_group`` and, without a syncer, ``island.layout_hash``
    (module doc)."""
    from .full_parameter_stream import full_parameter_policy

    refuse_unwired(plan)
    return full_parameter_policy(island, plan, backend_identity_sha256=backend_identity_sha256)
