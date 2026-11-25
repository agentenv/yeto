"""Learner ids of trainer islands that join a running fleet.

An island launched after the start takes a learner id the syncer admits
(``--rl-sync-max-learners``) and no other island ever had: the syncer treats a
second HELLO with an id already in the pool as that island's re-JOIN and drops
the older connection, so two islands sharing an id would fence each other.

The ids live in the run store's island table (``fleet/trainers/<k>/``, the
directory of :mod:`yeto.launch.island_lifecycle`): an id is taken once its
claim record ``fleet/trainers/<k>/learner.json`` or its state record exists.
:func:`allocate_learner_id` creates the lowest free claim with the store's
exclusive create (If-None-Match), so of two islands joining at once each gets
its own id, and a claim is never deleted, so an id is never handed out again
(an island that left keeps its id; a migrated island relaunches under its own
id without a new claim). Ids ``0..launched-1`` are the launched islands'.
"""

from __future__ import annotations

import json
import socket
import time
from typing import Any

from .island_lifecycle import STATE_DIR

CLAIM_NAME = "learner.json"
SCHEMA = "yeto.fleet.learner-id/v1"


class LearnerIdsExhausted(ValueError):
    """Every id below ``--rl-sync-max-learners`` is taken (a config error: raise the limit)."""


def claim_key(learner_id: int) -> str:
    return f"{STATE_DIR}/{int(learner_id)}/{CLAIM_NAME}"


def taken_ids(objects: Any) -> set[int]:
    """Ids with a claim or a state record in ``fleet/trainers/``."""
    ids = set()
    for key in objects.list(f"{STATE_DIR}/"):
        parts = str(key).split("/")
        if len(parts) >= 4 and parts[-2].isdigit() and parts[-1] in (CLAIM_NAME, "state.json"):
            ids.add(int(parts[-2]))
    return ids


def allocate_learner_id(objects: Any, *, launched: int, max_learners: int, island: str,
                        clock: Any = time.time) -> int:
    """Claim the lowest learner id in ``launched..max_learners-1`` that no island
    ever had and return it. ``island``: who joins (its fleet name), kept in the claim.

    Raises :class:`LearnerIdsExhausted` when none is left, and ValueError for a store
    whose create is not an exclusive create (two islands could take one id there)."""
    from ..rl.engine.object_store import atomic_create

    if not 1 <= int(launched) <= int(max_learners):
        raise ValueError(f"learner ids: need 1 <= launched ({launched}) <= max learners ({max_learners})")
    if not atomic_create(objects):
        raise ValueError("learner ids need a store with an exclusive create (If-None-Match); "
                         "this store's create is check-then-write")
    taken = taken_ids(objects)
    for learner_id in range(int(launched), int(max_learners)):
        if learner_id in taken:
            continue
        body = json.dumps({"schema": SCHEMA, "learner_id": learner_id, "island": str(island),
                           "host": socket.gethostname(), "at": round(clock(), 3)},
                          sort_keys=True).encode()
        if objects.create(claim_key(learner_id), body):
            if not atomic_create(objects):  # the store found out on this create (S3 NotImplemented)
                raise ValueError("learner ids need a store with an exclusive create (If-None-Match)")
            return learner_id
        # another island took it between the listing and the create: try the next id
    raise LearnerIdsExhausted(
        f"no learner id left for island {island!r}: ids 0..{int(max_learners) - 1} are all taken "
        f"(--rl-sync-max-learners {int(max_learners)}; an id is never reused); raise "
        "--rl-sync-max-learners for the next run")
