"""One recovery abstraction: the consistent cut and the one save path.

rl-code-quality-cleanup stage 3. Every save that a later launch may resume from
(or that tells it what was lost) goes through :func:`save`:

* the periodic round cut and the final/stop round cut
  (:class:`~yeto.rl.adapters.miles.round_cut.RoundCutCheckpoint`);
* the reclaim save that :func:`yeto.cloud.preemption.handle_notice` runs, and its
  timed rehearsal (:class:`~yeto.rl.adapters.miles.inflight_save.InFlightSaver`).

Each path describes *what* it saves as a :class:`ConsistentCut` and supplies a
``write`` callable for its payload (trainer shards + state snapshot, or the
in-flight trajectories). :func:`save` embeds the cut record in that payload under
:data:`CUT_KEY`, commits the :class:`~yeto.rl.engine.resume.CheckpointStore` when
the writer did not, and returns one result envelope. So all triggers write the
same cut record and report the same keys; only the payload differs.

Restoring is the store's job (``resume.restore_snapshot`` for the snapshot layout,
``resume.restore_flat_mirror`` for the ``--rl-elastic`` flat layout); the round
cut pointer read back from either carries the same :data:`CUT_KEY` record.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .resume import CheckpointStore

CUT_SCHEMA = "yeto.consistent_cut/v1"
CUT_KEY = "consistent_cut"

# Trigger of each save path: InFlightSaver.save (reclaim, via handle_notice) -> PREEMPT;
# InFlightSaver.rehearse and RoundCutCheckpoint.save(final=False) -> PERIODIC;
# RoundCutCheckpoint.save(final=True) (last round or our own stop) -> MANUAL.
PREEMPT = "preempt"
PERIODIC = "periodic"
MANUAL = "manual"
TRIGGERS = (PREEMPT, PERIODIC, MANUAL)


@dataclass(frozen=True)
class ConsistentCut:
    """The point a run can continue from.

    ``outer_step`` is the outer (syncer / published) version, ``island_steps`` the
    local training step of each island at the cut, ``completed_groups`` the number
    of prompt groups already consumed and ``completed_groups_digest`` their sorted
    sha256 (``resume.consumed_digest``). ``checkpoint_id`` names the payload."""

    checkpoint_id: str
    trigger: str
    outer_step: int | None = None
    island_steps: Mapping[str, int] = field(default_factory=dict)
    completed_groups: int | None = None
    completed_groups_digest: str | None = None

    def __post_init__(self) -> None:
        if self.trigger not in TRIGGERS:
            raise ValueError(f"cut trigger {self.trigger!r} is not one of {TRIGGERS}")
        if not self.checkpoint_id:
            raise ValueError("a consistent cut needs a checkpoint id")

    def to_record(self) -> dict[str, Any]:
        return {"schema": CUT_SCHEMA, "checkpoint_id": self.checkpoint_id, "trigger": self.trigger,
                "outer_step": self.outer_step,
                "island_steps": {str(k): int(v) for k, v in sorted(self.island_steps.items())},
                "completed_groups": self.completed_groups,
                "completed_groups_digest": self.completed_groups_digest}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> ConsistentCut:
        if record.get("schema") != CUT_SCHEMA:
            raise ValueError(f"not a {CUT_SCHEMA} record: schema {record.get('schema')!r}")
        return cls(checkpoint_id=str(record["checkpoint_id"]), trigger=str(record["trigger"]),
                   outer_step=record.get("outer_step"),
                   island_steps={str(k): int(v) for k, v in (record.get("island_steps") or {}).items()},
                   completed_groups=record.get("completed_groups"),
                   completed_groups_digest=record.get("completed_groups_digest"))


Writer = Callable[[dict[str, Any]], Mapping[str, Any]]
"""``write(cut_record) -> result``: writes the payload with ``cut_record`` embedded
under :data:`CUT_KEY`. A result with ``"committed": True`` means the writer already
committed the store (e.g. the snapshot sync); otherwise :func:`save` commits."""


def save(cut: ConsistentCut, *, store: CheckpointStore, write: Writer,
         clock: Callable[[], float] = time.monotonic) -> dict[str, Any]:
    """The one save path. Raises what ``write`` or the commit raise."""
    started = clock()
    record = cut.to_record()
    out = dict(write(record))
    if not out.pop("committed", False):
        store.commit()
    return {**out, CUT_KEY: record, "trigger": cut.trigger, "store_kind": store.kind,
            "cut_total_s": round(clock() - started, 4)}
