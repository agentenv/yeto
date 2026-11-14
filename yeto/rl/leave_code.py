"""The LEAVE code of the island pool, by itself: one byte on the wire, no torch.

:class:`LeaveReason` lives here, not in :mod:`yeto.rl.pool_client`, because the
store-only code paths need it without the pool client's torch import: the sample
buffer chain (``engine.sample_buffer`` -> ``engine.trainer_lifecycle`` ->
``engine.leave_reason``) is import-light on purpose, so a sampler-side or
head-side process (and an Ascend host whose venv has no torch at all) can read
and write the buffer, the dispense ledger and policy releases over the store's
object API. ``pool_client`` re-exports the enum, so
``from yeto.rl.pool_client import LeaveReason`` keeps working.
"""

from __future__ import annotations

from enum import IntEnum


class LeaveReason(IntEnum):
    """Why an island left the pool (one byte of MSG_LEAVE; the syncer's
    ``pool_leave`` record names it, see ``protocol.rs`` LEAVE_REASON_NAMES).
    ``LEASE_EXPIRED`` is assigned by the syncer only."""

    FINISHED = 0
    LEASE_EXPIRED = 1
    STOPPED = 2
    PREEMPTED = 3
    INFRA_FAILED = 4
    TRAINING_FAILED = 5
    CONFIG_MISMATCH = 6
    # The member is not part of this run any more: a newer incarnation
    # took its id ("superseded") or a fresh syncer replaced the run it joined
    # ("other run"). An older syncer names the byte unknown_7.
    FENCED = 7
