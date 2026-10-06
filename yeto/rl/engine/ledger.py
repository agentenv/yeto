"""Group / batch / update ledger (rl-infra-spec 3.6, design D3, alignment A2/F5).

The ledger records, durably and per rollout, what happened to every group the
driver saw, so a retry, a partial group, a publish failure or a restart can
neither train a batch twice nor lose one silently:

``prepared`` (generated, complete, policy-checked) -> ``optimizer_applied``
(the trainer's optimizer step for that batch returned) -> ``outer_recorded``
(the sync boundary for that round returned; the batch is part of the island's
outer progress). Terminal states besides ``outer_recorded``:

* ``filtered`` -- the algorithm intentionally dropped samples/groups this round
  (data cursor advanced, never trained). Recorded with the reason and the
  mechanism; never counted as lost nor as consumed (A2).
* ``superseded`` -- a batch whose optimizer step was applied but whose round
  is not in the authoritative restart cut (the learner restarted from an
  earlier outer state, so that update no longer exists); its groups may be
  generated and trained again. A restart below an ``outer_recorded`` batch is
  refused (commit uncertainty is never replayed automatically, D7-8).
* ``engine_discarded`` -- groups the engine drew (data cursor advanced) and
  threw away for a known engine reason, not an algorithm decision: groups
  still in flight when the batch filled are aborted and dropped with
  ``partial_rollout`` off (cut-audit §3, Miles ``sglang_rollout.py:420-437``).
  Counted, never lost, never ``filtered``, never consumed. (Completed groups
  beyond ``rollout_batch_size`` under over-sampling are ``filtered`` as task
  3.6 names them: an algorithm-configured surplus.)
* ``discarded`` -- a prepared batch that was not trained because the round
  failed before the optimizer step (e.g. a refused train gate); recorded with
  the error so the gap is explicit, never silent.

Non-terminal: ``carried_over`` -- a leftover group the engine keeps for a
later round (F5). It must later be consumed (``prepared`` in a later round) or
become ``filtered``; :meth:`BatchLedger.open_carried_over` lists what is left
and a cut treats it as unconsumed. Under Miles this is a legal boundary, not
a verified path: over-sampling surplus is never returned to the buffer (4.1
audit, ``sglang_rollout.py:505-510``), so with ``--rl-elastic`` metadata the
engine reports ``carried_over = 0`` (``buffer_length == 0`` ->
``RolloutBatchHandle.carried_over = 0``) and every round records
``carried_over_report: carried_over_reported=true, carried_over=0``; without
that metadata it reports None and the ledger writes
``carried_over_reported: false`` instead of guessing. :meth:`carried_over` /
:meth:`filter_carried` have no production caller (unit-tested only); the first
cut version requires ``carried_over == 0`` (``cut.py``).

Storage is the same fsync'd JSONL writer as the reconfiguration journal (one
record per transition) so the ledger survives a learner crash; "in memory it
was once submitted" is never used as the dedup proof (D3).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .journal import Journal

LEDGER_DIR = "ledger"
STATES = ("prepared", "optimizer_applied", "outer_recorded")
TERMINAL = frozenset(
    {"outer_recorded", "filtered", "engine_discarded", "discarded", "superseded"}
)


class LedgerError(RuntimeError):
    """The requested transition would consume twice or skip a state."""


def _batch_hash(group_ids: Iterable[str], sample_ids: Iterable[str]) -> str:
    digest = hashlib.sha256(b"yeto-rl-batch-v1\0")
    for gid in sorted(group_ids):
        digest.update(gid.encode() + b"\0")
    digest.update(b"\1")
    for sid in sorted(sample_ids):
        digest.update(sid.encode() + b"\0")
    return digest.hexdigest()


@dataclass
class _Batch:
    rollout_id: int
    attempt: int
    batch_hash: str
    group_ids: tuple[str, ...]
    policy_token: str
    state: str
    filtered: dict[str, Any] = field(default_factory=dict)
    # rl-infra-spec 4.2 data cursor AFTER this batch drew its prompts (None =
    # the rollout did not report one). Restart point for the data source.
    data_cursor: dict[str, int] | None = None


class BatchLedger:
    def __init__(self, state_dir: str | Path) -> None:
        self._journal = Journal(Path(state_dir).expanduser() / LEDGER_DIR)
        self._batches: dict[int, _Batch] = {}
        self._consumed_groups: dict[str, int] = {}
        self._carried: dict[str, dict[str, Any]] = {}
        self._last_report: dict[str, Any] = {}
        self._engine_discarded: dict[int, int] = {}
        for record in self._journal.records:
            self._replay(record)

    def close(self) -> None:
        self._journal.close()

    # -- replay ----------------------------------------------------------------
    def _replay(self, r: Mapping[str, Any]) -> None:
        kind = r["kind"]
        rid = int(r.get("rollout_id", -1))
        if kind == "prepared":
            cursor = r.get("data_cursor")
            self._batches[rid] = _Batch(
                rid, int(r["attempt"]), r["batch_hash"], tuple(r["group_ids"]),
                r["policy_token"], "prepared",
                data_cursor=({k: int(v) for k, v in cursor.items()} if cursor else None),
            )
            for gid in r["group_ids"]:
                self._carried.pop(gid, None)
        elif kind == "superseded":
            batch = self._batches[rid]
            batch.state = "superseded"
            for gid in batch.group_ids:
                self._consumed_groups.pop(gid, None)
        elif kind in ("optimizer_applied", "outer_recorded", "discarded"):
            self._batches[rid].state = kind
            if kind == "optimizer_applied":
                for gid in self._batches[rid].group_ids:
                    self._consumed_groups[gid] = rid
        elif kind == "filtered":
            if rid in self._batches:
                self._batches[rid].filtered = dict(r.get("detail") or {})
            for gid in r.get("group_ids") or ():
                self._carried.pop(gid, None)
        elif kind == "engine_discarded":
            self._engine_discarded[rid] = int(r["groups"])
        elif kind == "carried_over_report":
            self._last_report = {"carried_over": r.get("carried_over"),
                                 "buffer_length": r.get("buffer_length")}
        elif kind == "carried_over":
            for gid in r["group_ids"]:
                self._carried[gid] = {"rollout_id": rid, "policy_token": r["policy_token"]}

    # -- queries -----------------------------------------------------------------
    def state(self, rollout_id: int) -> str | None:
        batch = self._batches.get(rollout_id)
        return batch.state if batch is not None else None

    def batch(self, rollout_id: int) -> Mapping[str, Any] | None:
        b = self._batches.get(rollout_id)
        if b is None:
            return None
        return {
            "rollout_id": b.rollout_id, "attempt": b.attempt, "batch_hash": b.batch_hash,
            "group_ids": b.group_ids, "policy_token": b.policy_token, "state": b.state,
            "filtered": dict(b.filtered),
            "engine_discarded": self._engine_discarded.get(b.rollout_id, 0),
        }

    def open_carried_over(self) -> dict[str, dict[str, Any]]:
        return dict(self._carried)

    def restart_cursor(self, start_rollout_id: int) -> dict[str, int] | None:
        """The data cursor the rollout data source must resume from when the
        run restarts at ``start_rollout_id`` (after ``rebase``): the cursor
        recorded after rollout ``start_rollout_id - 1`` drew its prompts.

        None when there is nothing to restore (restart at 0) or the record
        carries no cursor (the caller decides whether that is acceptable).
        Miles' group ids are its data source's monotonic ``sample_group_index``
        which a restarted rollout process resets to 0: without seeking the
        source back to this cursor the restarted run re-draws trained groups
        and ``prepare`` refuses them (GPU evidence a4s8-2r2 r6)."""
        if start_rollout_id <= 0:
            return None
        batch = self._batches.get(start_rollout_id - 1)
        if batch is None or batch.state != "outer_recorded" or batch.data_cursor is None:
            return None
        return dict(batch.data_cursor)

    def cut_summary(self) -> dict[str, Any]:
        """``CutContext.ledger`` (4.2): what a cut must treat as unconsumed."""
        ready = [gid for r in self.unconsumed() for gid in self._batches[r].group_ids]
        report = self._last_report
        return {
            "ready_unconsumed": len(ready),
            "ready_unconsumed_group_ids": sorted(ready),
            "carried_over": len(self._carried),
            "carried_over_group_ids": sorted(self._carried),
            "engine_discarded_groups": sum(self._engine_discarded.values()),
            "engine_carried_over": report.get("carried_over"),
            "engine_buffer_length": report.get("buffer_length"),
            "last_rollout_id": max(self._batches) if self._batches else None,
            "last_state": self._batches[max(self._batches)].state if self._batches else None,
        }

    def unconsumed(self) -> list[int]:
        """Prepared batches whose optimizer step is not recorded (cut: unconsumed)."""
        return sorted(r for r, b in self._batches.items() if b.state == "prepared")

    # -- transitions -------------------------------------------------------------
    def prepare(self, batch: Any, *, policy_token: str) -> str:
        """Record a generated batch; refuse a batch whose groups were already trained."""
        group_ids = tuple(g.group_id for g in batch.groups)
        sample_ids = tuple(f"{g.group_id}/{s}" for g in batch.groups for s in g.sample_ids)
        rid = int(batch.rollout_id)
        batch_hash = _batch_hash(group_ids, sample_ids)
        twice = sorted(g for g in group_ids if g in self._consumed_groups)
        if twice:
            raise LedgerError(
                f"rollout {rid}: groups {twice} were already trained in rollout "
                f"{self._consumed_groups[twice[0]]}"
            )
        previous = self._batches.get(rid)
        if previous is not None and previous.state not in ("prepared", "discarded", "superseded"):
            raise LedgerError(
                f"rollout {rid} is already {previous.state}; it is not generated again"
            )
        attempt = 0 if previous is None else previous.attempt + 1
        cursor = getattr(batch, "data_cursor", None)
        self._journal.append(
            "prepared", rollout_id=rid, attempt=attempt, batch_hash=batch_hash,
            group_ids=list(group_ids), policy_token=policy_token,
            samples=len(sample_ids),
            # the data source position after this batch (restart point, 4.2/A6b)
            **({"data_cursor": {k: int(v) for k, v in cursor.items()}} if cursor else {}),
        )
        self._replay(self._journal.records[-1])
        filtered = getattr(batch, "filtered", None)
        filtered_samples = [getattr(g, "filtered_samples", None) for g in batch.groups]
        detail: dict[str, Any] = {}
        if filtered is not None:
            detail["groups"] = int(filtered)
            detail["mechanism"] = "rollout_meta_hook.record_trained_groups"
            detail["reason"] = ("completed groups not trained this round: dynamic-filter drops "
                                "and over-sampling surplus (not returned to the buffer)")
        if any(c is not None for c in filtered_samples):
            detail["samples"] = sum(int(c or 0) for c in filtered_samples)
            detail["sample_mechanism"] = "spec sample filter (rl-algo-grpo-knobs D7)"
        if detail:
            self._journal.append("filtered", rollout_id=rid, attempt=attempt, detail=detail)
            self._replay(self._journal.records[-1])
        aborted = getattr(batch, "aborted_in_flight_groups", None)
        if aborted:
            self._journal.append(
                "engine_discarded", rollout_id=rid, attempt=attempt, groups=int(aborted),
                reason="aborted in flight when the batch filled (partial_rollout off)",
                mechanism="miles generate_rollout abort",
            )
            self._replay(self._journal.records[-1])  # visible to cut_summary() before a reopen
        carried = getattr(batch, "carried_over", None)
        self._journal.append(
            "carried_over_report", rollout_id=rid, attempt=attempt,
            carried_over_reported=carried is not None,
            carried_over=None if carried is None else int(carried),
            buffer_length=getattr(batch, "buffer_length", None),
        )
        self._replay(self._journal.records[-1])
        return batch_hash

    def _advance(self, rollout_id: int, to: str, **fields: Any) -> None:
        batch = self._batches.get(rollout_id)
        if batch is None:
            raise LedgerError(f"rollout {rollout_id} was never prepared")
        order = {"prepared": 0, "optimizer_applied": 1, "outer_recorded": 2}
        if batch.state == to:
            return  # idempotent re-record after a lost acknowledgement
        if batch.state not in order or order[to] != order[batch.state] + 1:
            raise LedgerError(f"rollout {rollout_id}: {batch.state} -> {to} is not allowed")
        self._journal.append(to, rollout_id=rollout_id, attempt=batch.attempt,
                             batch_hash=batch.batch_hash, **fields)
        self._replay(self._journal.records[-1])

    def optimizer_applied(self, rollout_id: int, **fields: Any) -> None:
        self._advance(rollout_id, "optimizer_applied", **fields)

    def outer_recorded(self, rollout_id: int, **fields: Any) -> None:
        self._advance(rollout_id, "outer_recorded", **fields)

    def discard(self, rollout_id: int, *, error: str) -> None:
        batch = self._batches.get(rollout_id)
        if batch is None or batch.state != "prepared":
            return
        self._journal.append("discarded", rollout_id=rollout_id, attempt=batch.attempt, error=error)
        self._replay(self._journal.records[-1])

    def rebase(self, start_rollout_id: int) -> list[int]:
        """Align with the authoritative restart point (``SyncStart.rollout_id``)."""
        ahead = sorted(r for r, b in self._batches.items() if r >= start_rollout_id)
        recorded = [r for r in ahead if self._batches[r].state == "outer_recorded"]
        if recorded:
            raise LedgerError(
                f"restart at rollout {start_rollout_id} is behind outer-recorded rollouts "
                f"{recorded}; refusing to train them again"
            )
        superseded = []
        for rid in ahead:
            batch = self._batches[rid]
            if batch.state == "optimizer_applied":
                self._journal.append("superseded", rollout_id=rid, attempt=batch.attempt,
                                     restart_rollout_id=start_rollout_id)
                self._replay(self._journal.records[-1])
                superseded.append(rid)
            elif batch.state == "prepared":
                self.discard(rid, error=f"restart at rollout {start_rollout_id}")
        # Below the restart point every applied update is part of the
        # authoritative state the outer sync restarted from: record it.
        for rid in sorted(r for r, b in self._batches.items() if r < start_rollout_id):
            if self._batches[rid].state == "optimizer_applied":
                self._advance(rid, "outer_recorded", recovered=True,
                              restart_rollout_id=start_rollout_id)
        return superseded

    def carried_over(self, rollout_id: int, group_ids: Iterable[str], *, policy_token: str) -> None:
        self._journal.append("carried_over", rollout_id=rollout_id,
                             group_ids=sorted(group_ids), policy_token=policy_token)
        self._replay(self._journal.records[-1])

    def filter_carried(self, group_ids: Iterable[str], *, reason: str, mechanism: str) -> None:
        ids = sorted(group_ids)
        unknown = [g for g in ids if g not in self._carried]
        if unknown:
            raise LedgerError(f"groups {unknown} are not carried over")
        self._journal.append("filtered", rollout_id=-1, group_ids=ids,
                             detail={"reason": reason, "mechanism": mechanism})
        self._replay(self._journal.records[-1])
