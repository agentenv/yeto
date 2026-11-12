"""Task dispenser over a shared object store (no syncer, no TCP).

The syncer's dispenser (:mod:`.task_dispense`) needs one TCP endpoint every
sampler reaches. Sampler islands that share nothing but the run's object store
(e.g. NC A-card samplers on separate accounts writing one R2 bucket) take their
tasks from this ledger instead. :class:`StoreDispensedPrompts` is a drop-in for
``DispensedPrompts`` (``__call__``, ``written``, ``cursor``, ``close``).

Slots. The ledger hands out *slots* ``0 .. N-1`` of one global sequence, fixed
by the plan: slot ``i`` is dispenser position ``slot_position(i)`` and so row
``PositionMap.row(position)`` (pass 0 in data order, later passes reshuffled by
``seed``). With one stream a slot is its position. With ``--data-mix`` weights or
task-label streams (``mix`` / ``by_label``) the sequence interleaves the streams
by their weights with the syncer's earliest-deadline rule
(``syncer/src/dispense.rs`` ``pick_fresh``), so the mix holds over the whole run
whichever sampler takes a slot; a sampler with ``labels`` takes only the slots
of its labels (others take the rest).

The only atomic primitive is an exclusive create (S3 ``PutObject`` with
``If-None-Match: *``, measured on R2: 412 ``PreconditionFailed`` for every
loser, one winner per key under 32-way races; ``O_EXCL`` on a local directory).
A store without it (a FUSE bucket mount, an S3 endpoint answering
``NotImplemented``) is refused. Ledger keys under ``<namespace>/``::

    plan.json                  rows, fingerprint, seed, passes, streams, max_redo, lease_s, holdout sha
    claims/<i>/<a>             attempt a of slot i (one winner per attempt; holder, incarnation)
    beats/<i>/<a>              the holder's lease renewal (rewritten; its time is the server's)
    fail/<i>/<a>.<kind>        why attempt a ended without a trained group (infra / expired / release)
    done/<i>.<kind>            the slot is settled (trained / exhausted / dropped)
    drain/<holder>             scale down (the NC pool): {"drain": true, "seq"}: that sampler
                               claims no new task; {"drain": false}: it claims again
    drained/<holder>           its answer: {"seq"} once it holds no claim under that request
    retired/<holder>/<inc>     the controller retired this sampler process (:func:`retire`)

Rules:

* a claim is alive while ``time(beat or claim) + lease > server now``; both
  times come from the store (object ``LastModified`` and the listing's ``Date``
  header), never from a host clock. The holder rewrites its beat every
  ``lease / 3`` from its own thread, so a trial running for hours stays alive (a
  row's ``metadata.dispense_lease_s`` overrides the lease for that task);
* an attempt ends ``infra`` (the group could not be written whole because its
  trials failed on infrastructure), ``expired`` (its holder stopped renewing:
  the sampler died) or ``release`` (fenced, restarted, closing, unwritten: not
  the task's fault). ``infra`` and ``expired`` count toward ``max_redo``; past it
  the slot is settled ``exhausted``. A group the policy got wrong (reward 0) is a
  written group: ``trained``, never dispensed again;
* the next attempt is an exclusive create of ``claims/<i>/<a+1>``; the old holder
  sees it (:meth:`StoreDispensedPrompts.still_holds`) and neither renews nor
  writes the group;
* fencing (``yeto.rl.engine.incarnation_lease``; the NC pool's ``fence_shard``):
  a sampler whose incarnation lease names another incarnation is fenced. It
  stops renewing and writes no group (its own ``fenced`` check); any scanning
  sampler that sees a live claim whose holder's lease names another incarnation
  releases it at once (``release``, reason ``fenced``), without waiting for the
  lease to run out. A restarted sampler (same id, new incarnation) first settles
  ``trained`` the slots its earlier incarnations wrote to the buffer, then
  releases the rest of their claims;
* retire (:func:`retire`, the controller's scale-down / evict): a marker
  ``retired/<holder>/<incarnation>`` plus an immediate ``release`` (reason
  ``retired``) of every open claim of that process. The retired process sees
  the marker in its next listing and takes nothing more (fenced); its
  in-flight groups are refused by :meth:`StoreDispensedPrompts.still_holds`;
  any scanning sampler releases a claim of a retired process it still finds;
* hold-out: every row's task id is checked against the hold-out ids before the
  first claim (a leak refuses the sampler), and the plan records the list's
  digest so every sampler runs with the same one;
* drain (a scale down that keeps the trials in flight): a sampler whose
  ``drain/<holder>`` request is on claims nothing more; its claims in flight
  finish and are written as usual, and once it holds none it writes
  ``drained/<holder>`` with the request's ``seq`` (the pool then fences and
  releases it, so nothing is lost and nothing is given out twice). The request
  key shows up in the scan's listing: no extra request while none is set;
* request budget: at most one scan per ``scan_s`` while nothing could be claimed
  (backing off to ``max_scan_s``). A settled slot never changes again, so a scan
  lists only the slots it has not seen settled (:meth:`StoreLedger.scan`): four
  short range listings (``done/``, ``beats/``, ``claims/``, ``fail/`` from the
  oldest unsettled slot on) plus ``retired/`` and one read of its drain request,
  whatever the size of the ledger; a process's first scan lists ``done/`` whole.
  A lease renewal is one write per held slot; whether another sampler took the
  slot over is read from the scans, and checked by the renewal itself only every
  :data:`SUPERSEDED_CHECK_EVERY`-th time (three existence checks). A beat on a
  superseded attempt changes nothing: only the latest attempt's beat counts, and
  the group write is gated by :meth:`StoreDispensedPrompts.still_holds`.
* a failed read is no verdict: a scan whose listing (or a read on its way)
  fails claims nothing and is repeated at the next scan time
  (``rl_dispense_scan_failed``), the claims in flight stay; a renewal that cannot
  read whether its attempt was superseded still writes its beat; the check
  before a group write that cannot be read answers "unknown"
  (:meth:`StoreDispensedPrompts.hold_verdict`: the group is neither written nor
  released, the sampler asks again). Only a failed write ends a round.

This is the reference implementation of the WorkContract
(:mod:`yeto.rl.engine.work_contract`; conformance cases in
``tests/fixtures/work_contract_cases.json``).

Everything is in the store: a restarted sampler resumes from it, nothing is
given out twice and nothing is lost. ``python -m yeto.rl.engine.store_dispense
status`` prints the per-task table (``--json`` / ``--log`` for the dashboard).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .object_store import is_fatal_store_error
from .task_dispense import (
    DISPENSE_POSITION_KEY,
    DispenseEnded,
    DispenseError,
    DispenseFencedEnd,
    DispenseMismatch,
    PositionMap,
    StreamTable,
    file_fingerprint,
)

logger = logging.getLogger(__name__)

PLAN_VERSION = 2
BACKEND = "store"
DEFAULT_NAMESPACE = "task-dispense"
DEFAULT_LEASE_S = 900.0
DEFAULT_MAX_REDO = 2
DEFAULT_SCAN_S = 30.0
DEFAULT_MAX_SCAN_S = 120.0
CLAIM_STAGGER_SLOTS = 64  # a contended claim spreads over this many next candidates
# A scan lists two unsettled slots in one range when at most this many settled slots
# lie between them: reading past them (about 3 keys a slot, 1000 keys a request) costs
# less than the four requests of a range of its own.
SCAN_MERGE_SLOTS = 1024
SCAN_DIRS = ("beats", "claims", "fail")  # listed per range after done/ (TaskState's other fields)
FULL_SCAN_ENV = "YETO_RL_DISPENSE_FULL_SCAN"  # 1: every scan lists the whole namespace (as before)
SUPERSEDED_CHECK_EVERY = 3  # lease renewals per superseded check of a held slot (about one per lease)
LEASE_KEY = "dispense_lease_s"  # row metadata: this task's lease
ATTEMPT_KEY = "dispense_attempt"  # item metadata: the attempt this sampler holds
COUNTED = ("infra", "expired")  # attempt endings that count toward max_redo
FAIL_KINDS = ("infra", "expired", "release")
DONE_KINDS = ("trained", "exhausted", "dropped")

# events on the sampler's tape (the dashboard reads them; docs in the report)
CLAIM_EVENT = "rl_dispense_claim"
SETTLE_EVENT = "rl_dispense_settle"
REQUEUE_EVENT = "rl_dispense_requeue"
EXPIRED_EVENT = "rl_dispense_expired"
LOST_EVENT = "rl_dispense_lost"
RELEASE_EVENT = "rl_dispense_release"
RECOVER_EVENT = "rl_dispense_recover"
SCAN_FAILED_EVENT = "rl_dispense_scan_failed"  # a scan the store did not answer (and the scan after: recovered)
DRAIN_EVENT = "rl_dispense_drain"
DRAIN_DIR, DRAINED_DIR = "drain", "drained"


def drain_key(namespace: str, holder: str) -> str:
    return f"{namespace.strip('/')}/{DRAIN_DIR}/{holder}"


def drained_key(namespace: str, holder: str) -> str:
    return f"{namespace.strip('/')}/{DRAINED_DIR}/{holder}"


def request_drain(objects: Any, namespace: str, holder: str, *, on: bool = True, seq: str | None = None) -> str:
    """Ask sampler ``holder`` to claim no new task (``on``) or to claim again; returns
    the request's ``seq`` (its ``drained`` answer names it)."""
    seq = str(seq if seq is not None else f"{time.time():.6f}")
    objects.write(drain_key(namespace, holder), json.dumps({"drain": bool(on), "seq": seq}).encode())
    return seq


def drain_answered(objects: Any, namespace: str, holder: str, seq: str) -> bool:
    """Sampler ``holder`` holds no claim under drain request ``seq``."""
    data = objects.read(drained_key(namespace, holder))
    try:
        return bool(data) and str(json.loads(data).get("seq")) == str(seq)
    except ValueError:
        return False


# ------------------------------------------------------------------ objects


class DirObjects:
    """The ledger's object API on a plain local directory (tests, one host)."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root).expanduser()

    def _path(self, key: str) -> Path:
        return self.root / key

    def read(self, key: str) -> bytes | None:
        try:
            return self._path(key).read_bytes()
        except FileNotFoundError:
            return None

    def write(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)

    def create(self, key: str, data: bytes) -> bool:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            return False
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        return True

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def list_meta(self, prefix: str, *, start_after: str = "",
                  end: str = "") -> tuple[list[tuple[str, float]], float]:
        base = self._path(prefix)
        out = []
        if base.exists():
            for path in base.rglob("*"):
                if path.is_file() and not path.name.startswith("."):
                    key = path.relative_to(self.root).as_posix()
                    if key <= start_after or (end and key >= end):
                        continue
                    try:
                        out.append((key, path.stat().st_mtime))
                    except FileNotFoundError:
                        continue
        return out, time.time()


def ledger_objects(store: Any) -> Any:
    """The object API of a store: its ``objects`` (an object store with an
    exclusive create), a bare object API, or a local directory. A bucket mount
    has no atomic exclusive create and is refused."""
    from .resume import BucketMountStore  # noqa: PLC0415

    objects = getattr(store, "objects", None)
    if objects is not None:
        if not callable(getattr(objects, "list_meta", None)) or not callable(getattr(objects, "create", None)):
            raise DispenseError(f"{type(objects).__name__} has no exclusive create / timed listing")
        return objects
    if callable(getattr(store, "list_meta", None)) and callable(getattr(store, "create", None)):
        return store
    if isinstance(store, BucketMountStore):
        raise DispenseError("the store dispenser needs the object store itself (S3 If-None-Match), "
                            "not a bucket mount: its exclusive create is not atomic")
    return DirObjects(getattr(store, "root", store))


# -------------------------------------------------------------------- slots


def interleave(counts: Sequence[int], weights: Sequence[int]) -> list[int]:
    """The stream of each slot: ``sum(counts)`` slots, stream ``s`` taking
    ``counts[s]`` of them in the order of the syncer's ``pick_fresh``
    (Tijdeman's earliest deadline over the weights; the stream furthest behind
    when none is due; exhausted streams skipped)."""
    n_streams = len(counts)
    if n_streams == 1:
        return [0] * int(counts[0])
    total = sum(int(w) for w in weights)
    m = 2 * (n_streams - 1)
    given = [0] * n_streams
    out: list[int] = []
    for n in range(1, sum(counts) + 1):
        due: tuple[int, int] | None = None
        behind: tuple[int, int] | None = None
        for s in range(n_streams):
            if given[s] >= counts[s]:
                continue
            deficit = weights[s] * n - given[s] * total
            if m * deficit >= total:
                a = m * (given[s] + 1) - 1
                if due is None or a * weights[due[0]] < due[1] * weights[s]:
                    due = (s, a)
            if behind is None or deficit > behind[1]:
                behind = (s, deficit)
        s = (due or behind)[0]  # type: ignore[index]
        given[s] += 1
        out.append(s)
    return out


class SlotMap:
    """Slot -> dispenser position -> row (module doc)."""

    def __init__(self, rows: int, *, seed: int, passes: int, table: StreamTable | None = None) -> None:
        from .task_dispense import position_of  # noqa: PLC0415

        self.passes = int(passes)
        self.table = table
        self.positions = PositionMap(rows, seed=seed) if table is None else PositionMap(seed=seed, table=table)
        if table is None:
            self._slots: list[int] | None = None
            self.size = int(rows) * self.passes
        else:
            counts = [len(s.rows) * self.passes for s in table.streams]
            order = interleave(counts, [s.weight for s in table.streams])
            seen = [0] * len(counts)
            self._slots = []
            for s in order:
                self._slots.append(position_of(s, seen[s]))
                seen[s] += 1
            self.size = len(self._slots)

    def position(self, slot: int) -> int:
        return int(slot) if self._slots is None else self._slots[int(slot)]

    def row(self, slot: int) -> int:
        return self.positions.row(self.position(slot))

    def stream(self, slot: int) -> int:
        return self.positions.stream(self.position(slot))

    def decl(self) -> list[list[Any]] | None:
        return None if self.table is None else [list(d) for d in self.table.decl()]


# -------------------------------------------------------------------- state


def _pkey(slot: int) -> str:
    return f"{int(slot):012d}"


def _akey(attempt: int) -> str:
    return f"{int(attempt):04d}"


@dataclass
class TaskState:
    """What the ledger says about one slot."""

    slot: int
    claims: dict[int, float] = field(default_factory=dict)  # attempt -> claim time (server)
    beats: dict[int, float] = field(default_factory=dict)  # attempt -> latest beat time (server)
    fails: dict[int, str] = field(default_factory=dict)  # attempt -> kind (from the key suffix)
    done: str | None = None  # kind

    @property
    def latest(self) -> int | None:
        return max(self.claims) if self.claims else None

    def counted(self) -> int:
        return sum(1 for kind in self.fails.values() if kind in COUNTED)

    def last_seen(self, attempt: int) -> float:
        return max(self.beats.get(attempt, 0.0), self.claims.get(attempt, 0.0))

    def open_attempt(self) -> int | None:
        """The latest attempt when it has not ended and the slot is not settled."""
        a = self.latest
        return None if a is None or self.done is not None or a in self.fails else a


def _member_part(text: str) -> str:
    from urllib.parse import quote  # noqa: PLC0415

    return quote(str(text), safe="")


def parse_retired(keys: Iterable[tuple[str, float]], namespace: str) -> set[tuple[str, str]]:
    """The ``(holder, incarnation)`` pairs of the ledger's retire markers."""
    from urllib.parse import unquote  # noqa: PLC0415

    head = namespace.strip("/") + "/retired/"
    out = set()
    for key, _stamp in keys:
        if key.startswith(head):
            parts = key[len(head):].split("/")
            if len(parts) == 2:
                out.add((unquote(parts[0]), unquote(parts[1])))
    return out


def parse_ledger(keys: Iterable[tuple[str, float]], namespace: str) -> dict[int, TaskState]:
    states: dict[int, TaskState] = {}
    head = namespace.strip("/") + "/"
    for key, stamp in keys:
        if not key.startswith(head):
            continue
        parts = key[len(head):].split("/")
        try:
            if parts[0] == "done" and len(parts) == 2:
                p, kind = parts[1].split(".", 1)
                state = states.setdefault(int(p), TaskState(int(p)))
                # two settlements (a late writer): "trained" wins, the rest are kept in order
                if state.done is None or kind == "trained":
                    state.done = kind
            elif parts[0] in ("claims", "beats") and len(parts) == 3:
                state = states.setdefault(int(parts[1]), TaskState(int(parts[1])))
                attempt = int(parts[2])
                target = state.claims if parts[0] == "claims" else state.beats
                target[attempt] = max(target.get(attempt, 0.0), float(stamp))
            elif parts[0] == "fail" and len(parts) == 3:
                a, kind = parts[2].split(".", 1)
                states.setdefault(int(parts[1]), TaskState(int(parts[1]))).fails[int(a)] = kind
        except ValueError:
            continue
    return states


def lists_ranges(objects: Any) -> bool:
    """``objects.list_meta`` takes ``start_after`` / ``end`` (a range of a prefix)."""
    import inspect  # noqa: PLC0415

    try:
        params = inspect.signature(objects.list_meta).parameters
    except (TypeError, ValueError):
        return False
    return "start_after" in params and "end" in params


def scan_ranges(open_slots: Sequence[int], top: int,
                merge: int | None = None) -> list[tuple[int, int | None]]:
    """The slot ranges ``[lo, hi)`` a scan lists: every slot of ``open_slots`` (the
    slots below ``top`` not seen settled, ascending) and every slot from ``top`` on
    (``hi`` None: no upper bound). Neighbours at most ``merge`` slots apart share a
    range (``merge``: :data:`SCAN_MERGE_SLOTS`)."""
    merge = SCAN_MERGE_SLOTS if merge is None else int(merge)
    out: list[tuple[int, int | None]] = []
    lo = last = None
    for p in open_slots:
        if lo is None:
            lo = last = p
        elif p - last <= merge:
            last = p
        else:
            out.append((lo, last + 1))
            lo = last = p
    if lo is not None and top - last <= merge:
        return [*out, (lo, None)]
    if lo is not None:
        out.append((lo, last + 1))
    return [*out, (int(top), None)]


def list_slots(objects: Any, namespace: str, directory: str,
               ranges: Sequence[tuple[int, int | None]]) -> tuple[list[tuple[str, float]], float]:
    """The keys of ``<namespace>/<directory>/`` of the slots in ``ranges`` and the
    newest server time of the listings. A slot's keys start with its zero-padded
    number (``_pkey``), so a slot range is a key range."""
    head = f"{namespace.strip('/')}/{directory}/"
    keys: list[tuple[str, float]] = []
    now = 0.0
    for lo, hi in ranges:
        got, stamp = objects.list_meta(head, start_after=head + _pkey(lo) if lo > 0 else "",
                                       end="" if hi is None else head + _pkey(hi))
        keys += got
        now = max(now, float(stamp))
    return keys, now


class SettledSlots:
    """The slots a reader has seen settled. A ``done/`` key is never deleted, so
    such a slot needs no further listing. ``floor``: every slot below it is
    settled (their kinds are dropped: memory stays with the unsettled window)."""

    def __init__(self, limit: int | None = None) -> None:
        self.limit = limit
        self.floor = 0
        self.above: dict[int, str] = {}  # settled slots >= floor -> kind
        self.top = -1  # the highest settled slot

    def __contains__(self, slot: object) -> bool:
        return isinstance(slot, int) and (0 <= slot < self.floor or slot in self.above)

    def __len__(self) -> int:
        return self.floor + len(self.above)

    def add(self, slot: int, kind: str) -> None:
        if slot < self.floor or (self.limit is not None and slot >= self.limit):
            return
        self.above[slot] = kind
        self.top = max(self.top, slot)
        while self.floor in self.above:
            del self.above[self.floor]
            self.floor += 1

    def open_below_top(self) -> list[int]:
        return [p for p in range(self.floor, self.top + 1) if p not in self.above]

    def ranges(self) -> list[tuple[int, int | None]]:
        return scan_ranges(self.open_below_top(), self.top + 1)


def scan_slots(objects: Any, namespace: str, settled: SettledSlots, *,
               dirs: Sequence[str] = SCAN_DIRS) -> tuple[list[tuple[str, float]], float]:
    """The ledger keys of every slot not in ``settled`` (and of settled ones that
    share a range with them), as range listings: ``done/`` first (new settlements
    go into ``settled`` and leave the ranges), then ``dirs``. Each slot's keys are
    listed in one fixed order, ``done/`` before ``fail/`` as in a listing of the
    whole namespace."""
    ns = namespace.strip("/")
    keys, now = list_slots(objects, ns, "done", settled.ranges())
    for slot, state in parse_ledger(keys, ns).items():
        if state.done is not None:
            settled.add(slot, state.done)
    for directory in dirs:
        got, stamp = list_slots(objects, ns, directory, settled.ranges())
        keys += got
        now = max(now, stamp)
    return keys, now


# -------------------------------------------------------------------- holdout


def row_task_id(row: Mapping[str, Any], task_key: str | None = "metadata") -> str | None:
    task = row.get(task_key) if task_key else None
    for source in (task, row.get("metadata"), row):
        if isinstance(source, Mapping):
            for name in ("instance_id", "task_id"):
                if source.get(name):
                    return str(source[name])
    return None


def load_holdout(path: str | os.PathLike[str]) -> tuple[set[str], str]:
    """Task ids of a hold-out file (the ``tb2-holdout.json`` schema, a JSON list,
    or one id per line) and the file's SHA-256."""
    raw = Path(path).expanduser().read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    text = raw.decode("utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {line.strip() for line in text.splitlines() if line.strip()}, digest
    if isinstance(data, Mapping):
        from yeto.rl.harness.reward_env.benchmark import holdout_ids  # noqa: PLC0415

        return set(holdout_ids(dict(data))), digest
    return {str(x) for x in data}, digest


def holdout_digest(ids: Iterable[str]) -> str:
    """The digest of an inline hold-out id list (order-free)."""
    return hashlib.sha256("\n".join(sorted({str(i) for i in ids})).encode()).hexdigest()


def check_holdout(rows: Sequence[Mapping[str, Any]], holdout: set[str], task_key: str | None) -> None:
    leaked = sorted({tid for tid in (row_task_id(r, task_key) for r in rows) if tid in holdout})
    if leaked:
        raise DispenseMismatch(f"dispense refused: {len(leaked)} hold-out task(s) in the training rows: "
                               f"{leaked[:10]}")


# -------------------------------------------------------------------- ledger


class StoreLedger:
    """One sampler process's view of the ledger (module doc)."""

    def __init__(self, objects: Any, namespace: str, *, rows: int, fingerprint: bytes, holder: str,
                 incarnation: str, seed: int = 0, passes: int = 1, lease_s: float = DEFAULT_LEASE_S,
                 max_redo: int = DEFAULT_MAX_REDO, holdout_sha256: str | None = None,
                 leases: Sequence[float | None] | None = None, slots: SlotMap | None = None,
                 lease_of_holder: Callable[[str], str | None] | None = None,
                 full_scan: bool | None = None) -> None:
        if int(passes) < 1:
            raise ValueError("passes must be >= 1")
        self.objects = objects
        self.ns = namespace.strip("/")
        self.rows, self.fingerprint = int(rows), bytes(fingerprint)
        self.holder, self.incarnation = str(holder), str(incarnation)
        self.seed, self.passes = int(seed), int(passes)
        self.lease_s, self.max_redo = float(lease_s), int(max_redo)
        self.holdout_sha256 = holdout_sha256
        self.slots = slots or SlotMap(self.rows, seed=self.seed, passes=self.passes)
        self.limit = self.slots.size
        self._leases = list(leases) if leases is not None else None
        # sampler id -> the incarnation its incarnation lease names (None: no lease)
        self.lease_of_holder = lease_of_holder
        self._bodies: dict[tuple[int, int], dict[str, Any]] = {}  # claim bodies (immutable)
        self.drain_seq: str | None = None  # an open drain request for this holder (last scan)
        self.retired: set[tuple[str, str]] = set()  # retire markers seen by the latest scan
        if full_scan is None:
            full_scan = os.environ.get(FULL_SCAN_ENV, "").strip().lower() in ("1", "true", "yes", "on")
        # range listings need an object API that takes them (a test double may not)
        self.incremental = not full_scan and lists_ranges(objects)
        self.settled = SettledSlots(self.limit)  # slots the scans of this process saw settled

    def key(self, *parts: str) -> str:
        return "/".join((self.ns, *parts))

    def lease_of(self, slot: int) -> float:
        if self._leases:
            value = self._leases[self.slots.row(slot)]
            if value:
                return float(value)
        return self.lease_s

    def plan(self) -> dict[str, Any]:
        return {"version": PLAN_VERSION, "rows": self.rows, "fingerprint": self.fingerprint.hex(),
                "seed": self.seed, "passes": self.passes, "slots": self.limit, "streams": self.slots.decl(),
                "max_redo": self.max_redo, "lease_s": self.lease_s, "holdout_sha256": self.holdout_sha256}

    def _create(self, key: str, body: bytes) -> bool:
        won = self.objects.create(key, body)
        if getattr(self.objects, "conditional_writes", True) is False:
            raise DispenseError("the object store answered NotImplemented to If-None-Match: the store "
                                "dispenser needs an atomic exclusive create")
        return bool(won)

    def open(self) -> dict[str, Any]:
        """Create or check the plan (every sampler of the run agrees on it)."""
        mine = self.plan()
        if self._create(self.key("plan.json"), json.dumps(mine, sort_keys=True).encode()):
            return mine
        theirs = json.loads(self.objects.read(self.key("plan.json")) or b"{}")
        for name in ("version", "rows", "fingerprint", "seed", "passes", "streams", "holdout_sha256"):
            if theirs.get(name) != mine[name]:
                raise DispenseMismatch(f"dispense mismatch: {name} is {mine[name]!r} here, "
                                       f"{theirs.get(name)!r} in {self.key('plan.json')}")
        self.max_redo = int(theirs.get("max_redo", self.max_redo))
        self.lease_s = float(theirs.get("lease_s", self.lease_s))
        return theirs

    def scan(self) -> tuple[dict[int, TaskState], float]:
        """``({slot: state}, server now)``. A slot this process has seen settled is
        not listed again (module doc, request budget): it is in ``self.settled``
        and, unless it shares a range with an unsettled slot, not in the states.
        :meth:`is_settled` answers for both. With ``incremental`` off (or an object
        API without range listings) the whole namespace is listed, as before."""
        mine = drain_key(self.ns, self.holder)
        if self.incremental:
            keys, now = scan_slots(self.objects, self.ns, self.settled)
            retired, stamp = self.objects.list_meta(self.key("retired") + "/")
            keys += retired
            now = max(now, float(stamp))
            raw = self.objects.read(mine)
        else:
            keys, now = self.objects.list_meta(self.ns + "/")
            keys = list(keys)
            raw = self.objects.read(mine) if any(k == mine for k, _ in keys) else None
        self.drain_seq = None
        if raw:
            try:
                body = json.loads(raw)
            except ValueError:
                body = {}
            if isinstance(body, dict) and body.get("drain"):
                self.drain_seq = str(body.get("seq"))
        self.retired = parse_retired(keys, self.ns)
        states = parse_ledger(keys, self.ns)
        for slot, state in states.items():
            if state.done is not None:
                self.settled.add(slot, state.done)
            elif slot in self.settled:  # in a listed range, settled by an earlier scan
                state.done = self.settled.above.get(slot, "trained")
        return states, now

    def is_settled(self, slot: int, states: Mapping[int, TaskState]) -> bool:
        """``slot`` is settled, by the states of a scan or by an earlier scan."""
        state = states.get(slot)
        return (state is not None and state.done is not None) or int(slot) in self.settled

    def is_retired(self) -> bool:
        """This process was retired (as of the latest scan)."""
        return (self.holder, self.incarnation) in self.retired

    def answer_drain(self, seq: str) -> None:
        self.objects.write(drained_key(self.ns, self.holder), self._body(seq=seq))

    def _body(self, **fields: Any) -> bytes:
        return json.dumps({"holder": self.holder, "incarnation": self.incarnation, **fields},
                          sort_keys=True, default=str).encode()

    def fail(self, slot: int, attempt: int, kind: str, **detail: Any) -> bool:
        assert kind in FAIL_KINDS
        return self._create(self.key("fail", _pkey(slot), f"{_akey(attempt)}.{kind}"),
                            self._body(attempt=attempt, **detail))

    def settle(self, slot: int, kind: str, **detail: Any) -> bool:
        assert kind in DONE_KINDS
        return self._create(self.key("done", f"{_pkey(slot)}.{kind}"), self._body(**detail))

    def claim(self, slot: int, attempt: int, **detail: Any) -> bool:
        won = self._create(self.key("claims", _pkey(slot), _akey(attempt)), self._body(attempt=attempt, **detail))
        if won:
            self._bodies[(int(slot), int(attempt))] = {"holder": self.holder, "incarnation": self.incarnation}
        return won

    def beat(self, slot: int, attempt: int) -> None:
        self.objects.write(self.key("beats", _pkey(slot), _akey(attempt)), self._body())

    def superseded(self, slot: int, attempt: int) -> bool:
        """A later attempt was claimed, or this attempt was ended by someone else."""
        if self.objects.exists(self.key("claims", _pkey(slot), _akey(attempt + 1))):
            return True
        return any(self.objects.exists(self.key("fail", _pkey(slot), f"{_akey(attempt)}.{kind}"))
                   for kind in ("expired", "release"))

    def claim_body(self, slot: int, attempt: int) -> dict[str, Any]:
        cached = self._bodies.get((int(slot), int(attempt)))
        if cached is not None:
            return cached
        # a body that is not a JSON object reads as {} (as status() and retire() read it):
        # no holder, so no fence verdict; the claim keeps its own slot until its lease runs out
        body = _json(self.objects, self.key("claims", _pkey(slot), _akey(attempt)))
        if body:
            self._bodies[(int(slot), int(attempt))] = body
        return body

    # -- the claim pass -----------------------------------------------------
    def _fenced_claim(self, slot: int, attempt: int, leases: dict[str, str | None]) -> str | None:
        """The incarnation now holding the claim holder's incarnation lease, when
        it is not the claim's (the holder was fenced or restarted); ``"retired"``
        when the controller retired the claim's process."""
        if self.lease_of_holder is None and not self.retired:
            return None
        try:
            body = self.claim_body(slot, attempt)
        except Exception as error:  # noqa: BLE001 - unknown this scan: its expiry still frees it
            logger.warning("store dispenser: claim %d/%d not read: %r", slot, attempt, error)
            return None
        holder, inc = body.get("holder"), body.get("incarnation")
        if not holder or inc is None:
            return None
        if (str(holder), str(inc)) in self.retired:
            return "retired"
        if self.lease_of_holder is None:
            return None
        if holder not in leases:
            try:
                leases[holder] = self.lease_of_holder(str(holder))
            except Exception as error:  # noqa: BLE001 - expiry still frees it
                logger.warning("store dispenser: lease read of %s failed: %r", holder, error)
                leases[holder] = None
        current = leases[holder]
        return current if current is not None and current != str(inc) else None

    def candidates(self, states: Mapping[int, TaskState], now: float,
                   accept: Callable[[int], bool] | None = None,
                   on_end: Callable[..., None] | None = None) -> tuple[list[tuple[int, int]], bool]:
        """``([(slot, next attempt)], ended)``: redo-able slots first (ascending),
        then never-claimed ones; on the way ends expired attempts (``expired``) and
        the claims of fenced holders (``release``), and settles slots past
        ``max_redo``. ``ended``: every slot is settled."""
        redo: list[tuple[int, int]] = []
        leases: dict[str, str | None] = {}
        settled = len(self.settled)  # seen settled by a scan; the states add the rest
        for p in sorted(states):
            state = states[p]
            if p >= self.limit:
                continue
            if p in self.settled:
                continue
            if state.done is not None:
                settled += 1
                continue
            a = state.latest
            if a is None:
                continue
            if a not in state.fails:
                alive = state.last_seen(a) + self.lease_of(p) > now
                by = self._fenced_claim(p, a, leases) if alive else None
                if alive and by is None:
                    continue
                kind = "expired" if not alive else "release"
                detail = ({"seen": state.last_seen(a), "now": now} if not alive
                          else {"reason": "retired"} if by == "retired"
                          else {"reason": "fenced", "superseded_by": by})
                if self.fail(p, a, kind, **detail):
                    logger.warning("store dispenser: slot %d attempt %d %s", p, a,
                                   f"expired (lease {self.lease_of(p):.0f} s)" if not alive else f"fenced by {by}")
                    if on_end is not None:
                        on_end(EXPIRED_EVENT if not alive else RELEASE_EVENT, slot=p, attempt=a, **detail)
                state.fails[a] = kind
            if state.counted() > self.max_redo:
                if self.settle(p, "exhausted", attempts=a + 1, fails=dict(sorted(state.fails.items()))):
                    logger.warning("store dispenser: slot %d exhausted after %d attempts", p, a + 1)
                    if on_end is not None:
                        on_end(SETTLE_EVENT, slot=p, state="exhausted", attempts=a + 1)
                state.done = "exhausted"
                settled += 1
                continue
            if accept is None or accept(p):
                redo.append((p, a + 1))
        # never claimed: gaps below the frontier first (a claim that raised), then the frontier on.
        # The frontier is past every claimed and every settled slot: a slot settled without a
        # claim (``dropped`` by a sandbox gate) is never given out, wherever it lies.
        frontier = max(max((p for p, s in states.items() if s.claims or s.done is not None), default=-1),
                       self.settled.top) + 1
        fresh = [(p, 0) for p in range(self.settled.floor, min(frontier, self.limit))
                 if (p not in states or not states[p].claims) and not self.is_settled(p, states)
                 and (accept is None or accept(p))]
        fresh_iter = ((p, 0) for p in range(frontier, self.limit) if accept is None or accept(p))
        ended = settled >= self.limit
        return redo + fresh + list(_take(fresh_iter, 4096)), ended


def _take(iterable: Iterable[Any], n: int) -> Iterable[Any]:
    for i, item in enumerate(iterable):
        if i >= n:
            return
        yield item


# ------------------------------------------------------------------ prompts


@dataclass
class _Held:
    attempt: int
    row: int
    started: float  # local monotonic
    lost: bool = False


class StoreDispensedPrompts:
    """The engine sampler's prompts from the store ledger (``DispensedPrompts``'s
    interface, module doc). ``fenced()``: this incarnation's lease is gone (no
    renewal, no group write); ``emit(event, **fields)``: the sampler's tape."""

    def __init__(self, path: str | os.PathLike[str], ledger: StoreLedger, *, input_key: str,
                 label_key: str | None, groups_per_round: int, render: Callable[[Any], str] = str,
                 task_key: str | None = None, rows: Sequence[Mapping[str, Any]] | None = None,
                 accept_streams: Sequence[int] | None = None,
                 scan_s: float = DEFAULT_SCAN_S, max_scan_s: float = DEFAULT_MAX_SCAN_S,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic,
                 renew: bool = True, fenced: Callable[[], bool] | None = None,
                 emit: Callable[..., None] | None = None, gate: Any = None) -> None:
        if rows is None:
            with open(os.path.expanduser(str(path)), encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
        self.rows = list(rows)
        if len(self.rows) != ledger.rows:
            raise DispenseMismatch(f"dispense mismatch: the prompt file has {len(self.rows)} rows, "
                                   f"the ledger {ledger.rows}")
        self.ledger = ledger
        self.slots = ledger.slots
        accept = None if accept_streams is None else frozenset(int(s) for s in accept_streams)
        self.accept = None if accept is None else (lambda slot: self.slots.stream(slot) in accept)
        self.input_key, self.label_key, self.task_key = input_key, label_key, task_key
        self.groups_per_round = int(groups_per_round)
        self.render = render
        self.scan_s, self.max_scan_s = float(scan_s), float(max_scan_s)
        self._sleep, self._clock = sleep, clock
        self.fenced = fenced
        self.emit = emit
        # sandbox gate (yeto.sandbox.gate.SandboxGate): readiness filter and free capacity.
        # A runtime filter only: the plan (and so every sampler's slot sequence) is unchanged.
        self.gate = gate
        self._gate_dropped: set[int] = set()
        self._task_slots: dict[str, list[int]] | None = None
        self._waiting_capacity = False
        self._drain_seen: str | None = None  # the drain request already reported / answered
        self._drain_answered: str | None = None
        self._held: dict[int, _Held] = {}
        holder = getattr(ledger, "holder", "")
        self._stagger = int.from_bytes(hashlib.sha256(str(holder).encode()).digest()[:4], "big")
        self._retired = False  # the controller retired this process (a marker in the ledger)
        self.superseded_check_every = SUPERSEDED_CHECK_EVERY
        self._renewals = 0
        self._lock = threading.RLock()
        self._next_scan = 0.0
        self._idle = self.scan_s
        self.drawn = 0
        self.stats = {"scans": 0, "claimed": 0, "claim_lost": 0, "trained": 0, "requeued": 0, "released": 0,
                      "dropped": 0, "lost": 0, "recovered": 0, "not_ready": 0, "capacity_waits": 0,
                      "scans_failed": 0, "hold_checks_failed": 0, "renew_checks_failed": 0}
        self._scan_failing = 0  # failed scans in a row
        self._stop = threading.Event()
        self._renewer: threading.Thread | None = None
        if renew:
            self._renewer = threading.Thread(target=self._renew_loop, name="yeto-store-dispense", daemon=True)
            self._renewer.start()

    def _emit(self, event: str, **fields: Any) -> None:
        if self.emit is not None:
            try:
                self.emit(event, backend=BACKEND, namespace=self.ledger.ns, **fields)
            except Exception as error:  # noqa: BLE001 - the tape is advisory here
                logger.warning("store dispenser: emit %s failed: %r", event, error)

    def _is_fenced(self) -> bool:
        if self._retired:
            return True
        try:
            return bool(self.fenced is not None and self.fenced())
        except Exception:  # noqa: BLE001 - an unreadable lease is not a fence
            return False

    # -- open ---------------------------------------------------------------
    @classmethod
    def open(cls, path: str | os.PathLike[str], config: Mapping[str, Any], *, store: Any, holder: str,
             incarnation: str, input_key: str, label_key: str | None, groups_per_round: int,
             render: Callable[[Any], str] = str, task_key: str | None = None,
             lease_of_holder: Callable[[str], str | None] | None = None,
             gate: Any = None, **kwargs: Any) -> "StoreDispensedPrompts":
        """``config`` (``prompts.dispense`` of the spec): ``backend: "store"``,
        ``namespace``, ``seed``, ``passes``, ``lease_s``, ``max_redo``, ``holdout``
        (a path on the sampler) or ``holdout_ids`` (inline), ``require_holdout``,
        ``mix`` / ``by_label`` / ``labels`` (streams), ``scan_s`` / ``max_scan_s``,
        ``full_scan`` (true: every scan lists the whole namespace; default: only the
        unsettled slots, :data:`FULL_SCAN_ENV` overrides the default)."""
        with open(os.path.expanduser(str(path)), encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        count, digest = file_fingerprint(path)
        holdout_sha = None
        if config.get("holdout"):
            ids, holdout_sha = load_holdout(config["holdout"])
            check_holdout(rows, ids, task_key)
        elif config.get("holdout_ids") is not None:
            ids = {str(i) for i in config["holdout_ids"]}
            holdout_sha = holdout_digest(ids)
            check_holdout(rows, ids, task_key)
        elif config.get("require_holdout"):
            raise DispenseMismatch("dispense refused: prompts.dispense.require_holdout is set without a holdout")
        leases = []
        for r in rows:
            meta = (r.get(task_key) if task_key else None) or r.get("metadata") or {}
            leases.append(meta.get(LEASE_KEY) if isinstance(meta, Mapping) else None)
        mix, labels = config.get("mix"), config.get("labels")
        by_label = bool(config.get("by_label")) or bool(labels)
        seed, passes = int(config.get("seed", 0) or 0), int(config.get("passes", 1) or 1)
        table, accept = None, None
        if mix or by_label:
            table = StreamTable.from_prompts(path, mix=list(mix) if mix else None, by_label=by_label)
            accept = table.accept(list(labels)) if labels else None
            if table.single and accept is None:
                table = None
        slots = SlotMap(count, seed=seed, passes=passes, table=table)
        ledger = StoreLedger(ledger_objects(store), str(config.get("namespace") or DEFAULT_NAMESPACE),
                             rows=count, fingerprint=digest, holder=holder, incarnation=incarnation,
                             seed=seed, passes=passes, lease_s=float(config.get("lease_s", DEFAULT_LEASE_S)),
                             max_redo=int(config.get("max_redo", DEFAULT_MAX_REDO)),
                             holdout_sha256=holdout_sha, leases=leases, slots=slots,
                             lease_of_holder=lease_of_holder,
                             full_scan=None if config.get("full_scan") is None else bool(config["full_scan"]))
        ledger.open()
        return cls(path, ledger, input_key=input_key, label_key=label_key, groups_per_round=groups_per_round,
                   render=render, task_key=task_key, rows=rows, accept_streams=accept,
                   scan_s=float(config.get("scan_s", DEFAULT_SCAN_S)),
                   max_scan_s=float(config.get("max_scan_s", DEFAULT_MAX_SCAN_S)), gate=gate, **kwargs)

    def recover(self, entries: Iterable[Any] = ()) -> dict[str, int]:
        """Restart: settle ``trained`` the slots this sampler's buffer ``entries``
        hold (written by an earlier incarnation, not yet settled), then release
        the open claims of its earlier incarnations."""
        written = set()
        for entry in entries:
            record = getattr(entry, "sampler_round", None) or {}
            if record.get("sampler") == self.ledger.holder and record.get("dispense_position") is not None:
                written.add(int(record["dispense_position"]))
        states, _now = self.ledger.scan()
        recovered = released = 0
        for p in sorted(written):
            if p < self.ledger.limit and not self.ledger.is_settled(p, states):
                if self.ledger.settle(p, "trained", reason="recovered_from_buffer"):
                    recovered += 1
                    states.setdefault(p, TaskState(p)).done = "trained"
        for p, state in states.items():
            a = state.open_attempt()
            if a is None:
                continue
            body = self.ledger.claim_body(p, a)
            if body.get("holder") == self.ledger.holder and body.get("incarnation") != self.ledger.incarnation:
                if self.ledger.fail(p, a, "release", reason="superseded_incarnation"):
                    released += 1
        self.stats["recovered"] += recovered
        sweep = getattr(self.gate, "sweep_stale", None)
        if callable(sweep):  # trial containers a killed earlier incarnation left on the sandbox host
            swept = sweep()
            if swept:
                self._emit("rl_sandbox_orphans_removed", removed=swept)
        if recovered or released:
            logger.info("store dispenser: recovered %d written slots, released %d earlier claims",
                        recovered, released)
        self._emit(RECOVER_EVENT, recovered=recovered, released=released)
        return {"recovered": recovered, "released": released}

    release_earlier = recover  # the WIP name

    # -- draw ---------------------------------------------------------------
    def holding(self) -> int:
        with self._lock:
            return len(self._held)

    def _slots_of_tasks(self) -> dict[str, list[int]]:
        if self._task_slots is None:
            ids = [row_task_id(r, self.task_key) for r in self.rows]
            out: dict[str, list[int]] = {}
            for slot in range(self.ledger.limit):
                tid = ids[self.slots.row(slot)]
                if tid is not None:
                    out.setdefault(tid, []).append(slot)
            self._task_slots = out
        return self._task_slots

    def _drop_failed(self, view: Any, states: Mapping[int, TaskState]) -> None:
        """Settle ``dropped`` the slots of tasks whose sandbox validation failed."""
        for tid, slots in self._slots_of_tasks().items():
            state, reason, _own = view.classify(tid)
            if state != "failed":
                continue
            for p in slots:
                st = states.get(p)
                if (p in self._gate_dropped or self.ledger.is_settled(p, states)
                        or (st is not None and st.open_attempt() is not None)):
                    continue
                if self.accept is not None and not self.accept(p):
                    continue
                self._gate_dropped.add(p)
                if self.ledger.settle(p, "dropped", reason=reason, task_id=tid):
                    self.stats["dropped"] += 1
                    self._emit(SETTLE_EVENT, slot=p, state="dropped", task_id=tid, reason=reason)
                states.setdefault(p, TaskState(p)).done = "dropped"  # type: ignore[union-attr]

    def _scan_failed(self, error: BaseException) -> None:
        """A scan the store did not answer: nothing claimed, nothing judged, the
        claims in flight stay; the next scan (``take``'s backoff) reads again.
        Refused credentials or no such bucket are no outage: raised as they are."""
        if is_fatal_store_error(error):
            raise error
        self.stats["scans_failed"] += 1
        self._scan_failing += 1
        logger.warning("store dispenser: scan failed (%d in a row), scanning again later: %r",
                       self._scan_failing, error)
        if self._scan_failing == 1:  # one event per outage, one at its end
            self._emit(SCAN_FAILED_EVENT, state="failed", holding=self.holding(),
                       error=f"{type(error).__name__}: {error}"[:300])

    def _claim(self, want: int) -> tuple[list[int], bool]:
        try:
            states, now = self.ledger.scan()
        except DispenseError:
            raise
        except Exception as error:  # noqa: BLE001 - a failed listing costs this scan, not the sampler
            self._scan_failed(error)
            return [], False
        if self._scan_failing:
            self._emit(SCAN_FAILED_EVENT, state="recovered", failed_scans=self._scan_failing)
            self._scan_failing = 0
        self.stats["scans"] += 1
        self._lost_by_scan(states)
        if self.ledger.is_retired():
            self._retired = True
            with self._lock:
                held = [(p, h) for p, h in self._held.items() if not h.lost]
            for p, h in held:
                self._mark_lost(p, h, "retired")
            return [], False
        seq = self.ledger.drain_seq
        if seq is not None:  # scale down: no new task; the claims in flight finish
            if self._drain_seen != seq:
                self._drain_seen = seq
                self._emit(DRAIN_EVENT, state="draining", seq=seq, holding=self.holding())
            self._answer_drain()
            return [], False
        if self._drain_seen is not None:
            self._emit(DRAIN_EVENT, state="resumed", seq=self._drain_seen)
            self._drain_seen = self._drain_answered = None
        accept = self.accept
        view = None
        if self.gate is not None:
            view = self.gate.view()
            self._drop_failed(view, states)
            base = self.accept

            def accept(slot: int) -> bool:
                if base is not None and not base(slot):
                    return False
                tid = row_task_id(self.rows[self.slots.row(slot)], self.task_key)
                if view.classify(tid)[0] == "ready":
                    return True
                self.stats["not_ready"] += 1
                return False

        try:
            candidates, ended = self.ledger.candidates(states, now, accept=accept, on_end=self._emit)
        except DispenseError:
            raise
        except Exception as error:  # noqa: BLE001 - an attempt it could not end: the next scan ends it
            self._scan_failed(error)
            return [], False
        if view is not None and view.host is not None:  # affinity: tasks ready on this sampler's host first
            candidates.sort(key=lambda c: not view.classify(
                row_task_id(self.rows[self.slots.row(c[0])], self.task_key))[2])
        got: list[int] = []
        queue, staggered, i = candidates, False, 0
        while i < len(queue):
            p, attempt = queue[i]
            i += 1
            if len(got) >= want:
                break
            with self._lock:
                mine = p in self._held
            if mine:  # a group of an earlier attempt is still in flight here (lost): it settles first
                continue
            row = self.slots.row(p)
            task_id = row_task_id(self.rows[row], self.task_key)
            try:
                won = self.ledger.claim(p, attempt, row=row, task_id=task_id)
            except DispenseError:
                raise
            except Exception as error:  # noqa: BLE001 - the slot stays a gap; the next scan retries it
                logger.warning("store dispenser: claim of slot %d failed: %r", p, error)
                break
            if won:
                with self._lock:
                    self._held[p] = _Held(attempt, row, self._clock())
                self.stats["claimed"] += 1
                got.append(p)
                self._emit(CLAIM_EVENT, slot=p, attempt=attempt, row=row, task_id=task_id,
                           stream=self.slots.stream(p))
                if view is not None and view.host is not None and not view.classify(task_id)[2]:
                    # affinity "prefer": prepared on another host only; its image may be pulled at trial start
                    self._emit("rl_dispense_task_off_host", slot=p, task_id=task_id, host=view.host)
            else:
                self.stats["claim_lost"] += 1
                if not staggered:
                    # Contention: every holder scanned the same order and races for the same
                    # next slots, losing ~one claim (one conditional PUT) per rival. Go on from a
                    # holder-specific offset within the next CLAIM_STAGGER_SLOTS candidates
                    # instead; an uncontended holder keeps the plan order exactly.
                    staggered = True
                    rest = queue[i:]
                    if rest:
                        k = self._stagger % min(len(rest), CLAIM_STAGGER_SLOTS)
                        queue, i = rest[k:] + rest[:k], 0
        return got, ended

    def _lost_by_scan(self, states: Mapping[int, TaskState]) -> None:
        """Mark the held slots the scan shows superseded (a later claim, or the
        attempt ended ``expired`` / ``release`` by someone else): what
        ``StoreLedger.superseded`` asks the store, from keys already listed. Those
        keys are never deleted, so a scan's view cannot report a loss that is not."""
        with self._lock:
            held = [(p, h) for p, h in self._held.items() if not h.lost]
        for p, h in held:
            state = states.get(p)
            if state is not None and ((state.latest or 0) > h.attempt
                                      or state.fails.get(h.attempt) in ("expired", "release")):
                self._mark_lost(p, h, "superseded")

    def _answer_drain(self) -> None:
        seq = self.ledger.drain_seq
        if seq is None or self.holding() or self._drain_answered == seq:
            return
        try:
            self.ledger.answer_drain(seq)
        except Exception as error:  # noqa: BLE001 - the pool's drain timeout bounds it
            logger.warning("store dispenser: drain answer failed: %r", error)
            return
        self._drain_answered = seq
        self._emit(DRAIN_EVENT, state="drained", seq=seq)

    def draining(self) -> bool:
        return self.ledger.drain_seq is not None

    def poll(self, want: int) -> tuple[str, list[int]]:
        """One claim attempt, never waiting, without the scan rate limit:
        ``("items", slots)``, ``("wait", [])``, ``("end", [])`` (every slot
        settled, none held here) or ``("fenced", [])`` (the WorkContract's
        ``claim``)."""
        if self._is_fenced():
            return "fenced", []
        got, ended = self._claim(want)
        if self._retired:
            return "fenced", []
        if got:
            return "items", got
        return ("end" if ended and not self.holding() else "wait"), []

    def take(self, want: int) -> list[int]:
        """Up to ``want`` slots. Holding groups in flight: one scan at most
        (rate-limited), possibly nothing. Holding nothing: wait for a task;
        :class:`DispenseEnded` when every slot is settled."""
        asked = want
        while True:
            if self._is_fenced():
                if self.holding():
                    return []
                raise DispenseFencedEnd("fenced: this incarnation takes no more tasks")
            want = asked
            if self.gate is not None:
                free = self.gate.free_groups()
                if free is not None and free < want:
                    want = max(0, int(free))
                if want == 0:  # no sandbox room: claim nothing (a claim would hold a lease while queued)
                    if not self._waiting_capacity:
                        self._waiting_capacity = True
                        self.stats["capacity_waits"] += 1
                        self._emit("rl_dispense_wait_capacity", free_groups=free, holding=self.holding(),
                                   **dict(getattr(self.gate, "last_probe", None) or {}))
                else:
                    self._waiting_capacity = False
            now = self._clock()
            if now >= self._next_scan:
                got, ended = self._claim(want)
                if got:
                    self._idle = self.scan_s
                    self._next_scan = 0.0 if len(got) == asked else now + self.scan_s
                    return got
                if ended and not self.holding():
                    raise DispenseEnded("every task of the store dispenser is settled")
                self._next_scan = now + self._idle
                # waiting for sandbox room is not an empty ledger: keep scanning at scan_s
                self._idle = self.scan_s if want == 0 else min(self._idle * 2, self.max_scan_s)
            if self.holding():
                return []
            self._sleep(max(0.05, self._next_scan - self._clock()))

    def __call__(self, rollout_id: int) -> list[Any]:
        from .engine_sampler import PromptItem  # noqa: PLC0415 - engine_sampler imports task_dispense

        del rollout_id
        items: list[Any] = []
        for slot in self.take(self.groups_per_round):
            index = self.slots.row(slot)
            row = self.rows[index]
            task = row.get(self.task_key) if self.task_key is not None else None
            if self.input_key not in row or (self.task_key is not None and not isinstance(task, Mapping)):
                self._end(slot, done="dropped", reason="malformed_row")
                continue
            metadata: dict[str, Any] = {"row": index, DISPENSE_POSITION_KEY: int(slot),
                                        ATTEMPT_KEY: self._held[slot].attempt}
            if task is not None:
                metadata["agent_task"] = dict(task)
            items.append(PromptItem(prompt_id=f"row{index}", prompt=self.render(row[self.input_key]),
                                    label=row.get(self.label_key) if self.label_key else None,
                                    metadata=metadata))
            self.drawn += 1
        return items

    def cursor(self, rollout_id: int) -> dict[str, int]:
        del rollout_id
        return {"dispensed": int(self.drawn)}

    # -- reports ------------------------------------------------------------
    def _mark_lost(self, slot: int, held: _Held, why: str) -> None:
        if not held.lost:
            held.lost = True
            self.stats["lost"] += 1
            logger.warning("store dispenser: slot %d attempt %d lost (%s)", slot, held.attempt, why)
            self._emit(LOST_EVENT, slot=slot, attempt=held.attempt, reason=why)

    def hold_verdict(self, slot: int) -> bool | None:
        """Whether this process still holds ``slot`` (not fenced, no later attempt,
        the attempt not ended by another sampler): True, the group may be written;
        False, it may not; None, the store did not answer (unknown: the group is
        neither written nor released now, the caller asks again)."""
        with self._lock:
            held = self._held.get(int(slot))
        if held is None or held.lost:
            return False
        if self._is_fenced():
            self._mark_lost(int(slot), held, "fenced")
            return False
        try:
            superseded = self.ledger.superseded(int(slot), held.attempt)
        except DispenseError:
            raise
        except Exception as error:  # noqa: BLE001 - unknown, not "lost": never a reason to end the sampler
            if is_fatal_store_error(error):  # ... but a refused credential or a missing bucket is
                raise
            self.stats["hold_checks_failed"] += 1
            logger.warning("store dispenser: slot %d attempt %d: hold check not answered: %r",
                           slot, held.attempt, error)
            return None
        if superseded:
            self._mark_lost(int(slot), held, "superseded")
            return False
        return True

    def still_holds(self, slot: int) -> bool:
        """The WorkContract's ``still_holds``: :meth:`hold_verdict` with "unknown"
        read as "do not write" (the claim itself stays as it is)."""
        return bool(self.hold_verdict(slot))

    def _end(self, slot: int, *, done: str | None = None, fail: str | None = None,
             **detail: Any) -> dict[str, Any] | None:
        with self._lock:
            held = self._held.pop(int(slot), None)
        if held is None:
            return None
        elapsed = round(self._clock() - held.started, 3)
        record = {"slot": int(slot), "position": int(self.slots.position(slot)), "row": held.row,
                  "attempt": held.attempt, "elapsed_s": elapsed,
                  "task_id": row_task_id(self.rows[held.row], self.task_key), **detail}
        if held.lost:  # superseded or fenced: the new holder settles it (the gate drops a fenced write)
            record["state"] = "lost"
            return record
        if fail is not None:
            self.ledger.fail(slot, held.attempt, fail, elapsed_s=elapsed, **detail)
            record["state"] = fail
        if done is not None:
            won = self.ledger.settle(slot, done, attempt=held.attempt, elapsed_s=elapsed, **detail)
            record["state"] = done
            if not won:
                record["duplicate"] = True  # another attempt settled it first
            self.stats["trained" if done == "trained" else "dropped"] += 1
        self._emit(SETTLE_EVENT if done else (REQUEUE_EVENT if fail == "infra" else RELEASE_EVENT),
                   **record)
        return record

    def written(self, rollout_id: int, positions: Iterable[int], *, unwritten: str = "release",
                keep: Iterable[int] = (), requeue: Iterable[int] = (),
                outcome: Mapping[int, Mapping[str, Any]] | None = None, **_: Any) -> dict[str, Any]:
        """The groups of ``positions`` (slots) are in the buffer (settled
        ``trained``, whatever their reward); ``requeue``: groups not written
        because their trials failed on infrastructure (attempt ends ``infra``,
        counted toward ``max_redo``); every other held slot not in ``keep`` is
        released (``unwritten``)."""
        outcome = outcome or {}
        done = {int(p) for p in positions}
        again = {int(p) for p in requeue} - done
        tasks = []
        for p in sorted(done):
            if (r := self._end(p, done="trained", **dict(outcome.get(p, {})))) is not None:
                tasks.append(r)
        for p in sorted(again):
            if (r := self._end(p, fail="infra", **dict(outcome.get(p, {})))) is not None:
                tasks.append(r)
                self.stats["requeued"] += 1
        released = 0
        if unwritten == "release":
            with self._lock:
                rest = set(self._held) - done - again - {int(p) for p in keep}
            for p in sorted(rest):
                detail = {"reason": "unwritten", **dict(outcome.get(p, {}))}
                if (r := self._end(p, fail="release", **detail)) is not None:
                    tasks.append(r)
                    released += 1
            self.stats["released"] += released
        return {"rollout_id": int(rollout_id), "trained": sum(1 for t in tasks if t.get("state") == "trained"),
                "requeued": sum(1 for t in tasks if t.get("state") == "infra"), "released": released,
                "acked": True, "backend": BACKEND, "tasks": tasks}

    # -- lease renewal ------------------------------------------------------
    def renew(self) -> int:
        """Rewrite the beat of every held slot (``lease / 3`` cadence); a fenced
        incarnation renews nothing (its claims run out or are released)."""
        with self._lock:
            held = [(p, h) for p, h in self._held.items() if not h.lost]
        if self._is_fenced():
            for p, h in held:
                self._mark_lost(p, h, "fenced")
            return 0
        # A beat on a superseded attempt changes nothing (module doc, request budget), so
        # the three existence checks run every SUPERSEDED_CHECK_EVERY-th renewal only.
        self._renewals += 1
        check = self._renewals % max(1, int(self.superseded_check_every)) == 0
        renewed = 0
        failed: BaseException | None = None
        for p, h in held:
            superseded = False
            if check:
                try:
                    superseded = self.ledger.superseded(p, h.attempt)
                except Exception as error:  # noqa: BLE001 - unknown: renew (a beat of an ended attempt counts for nothing)
                    if is_fatal_store_error(error):
                        raise
                    self.stats["renew_checks_failed"] += 1
            if superseded:
                self._mark_lost(p, h, "superseded")
                continue
            try:
                self.ledger.beat(p, h.attempt)
            except Exception as error:  # noqa: BLE001 - the other claims are still renewed
                failed = failed or error
                continue
            renewed += 1
        if failed is not None:  # the renew loop reports it and renews again a period later
            raise failed
        return renewed

    def renew_period(self) -> float:
        with self._lock:
            leases = [self.ledger.lease_of(p) for p in self._held] or [self.ledger.lease_s]
        return max(1.0, min(leases) / 3.0)

    def _renew_loop(self) -> None:
        while not self._stop.wait(self.renew_period()):
            try:
                self.renew()
            except Exception as error:  # noqa: BLE001 - retried at the next period
                logger.warning("store dispenser: lease renewal failed: %r", error)

    def close(self) -> dict[str, Any]:
        self._stop.set()
        with self._lock:
            held = sorted(self._held)
        for p in held:
            try:
                self._end(p, fail="release", reason="closing")
            except Exception as error:  # noqa: BLE001 - expiry gives it out anyway
                logger.warning("store dispenser: release at close failed: %r", error)
        return {"released": len(held), "backend": BACKEND, **self.stats}


# ------------------------------------------------------------------- retire


def retire(objects: Any, namespace: str, holder: str, incarnation: str | None = None, *,
           reason: str = "scale_down") -> dict[str, Any]:
    """Retire sampler ``holder`` now (the WorkContract's ``retire``; the
    controller's scale-down / evict): write its retire marker and release every
    open claim of it at once (``release``, reason ``retired``; not counted
    toward ``max_redo``), so the next claim of any sampler takes them.
    ``incarnation=None`` retires every incarnation of ``holder`` that holds an
    open claim. Idempotent: a second call releases nothing. It lists ``done/``
    and the claims and failures of the unsettled slots only."""
    ns = namespace.strip("/")
    if lists_ranges(objects):
        keys, _now = scan_slots(objects, ns, SettledSlots(), dirs=("claims", "fail"))
    else:
        keys, _now = objects.list_meta(ns + "/")
    states = parse_ledger(keys, ns)
    body = json.dumps({"holder": "controller", "reason": str(reason)}, sort_keys=True).encode()
    open_claims = []
    for p, state in sorted(states.items()):
        a = state.open_attempt()
        if a is None:
            continue
        claim = _json(objects, f"{ns}/claims/{_pkey(p)}/{_akey(a)}")
        if claim.get("holder") == holder and (incarnation is None or str(claim.get("incarnation")) == str(incarnation)):
            open_claims.append((p, a, str(claim.get("incarnation"))))
    incarnations = {str(incarnation)} if incarnation is not None else {inc for _p, _a, inc in open_claims}
    for inc in sorted(incarnations):
        objects.create(f"{ns}/retired/{_member_part(holder)}/{_member_part(inc)}", body)
    released = []
    for p, a, inc in open_claims:
        detail = json.dumps({"holder": "controller", "attempt": a, "reason": "retired", "retire_reason": str(reason),
                             "retired_holder": holder, "retired_incarnation": inc}, sort_keys=True).encode()
        if objects.create(f"{ns}/fail/{_pkey(p)}/{_akey(a)}.release", detail):
            released.append(p)
    if released:
        logger.info("store dispenser: retired %s (%s): released slots %s", holder, reason, released)
    return {"holder": holder, "incarnations": sorted(incarnations), "released": len(released), "slots": released}


# ------------------------------------------------------------------- status


def _json(objects: Any, key: str) -> dict[str, Any]:
    raw = objects.read(key)
    try:
        doc = json.loads(raw) if raw else {}
    except ValueError:
        return {}
    return doc if isinstance(doc, dict) else {}


def status(objects: Any, namespace: str = DEFAULT_NAMESPACE, *, details: bool = True,
           settled: bool = True) -> dict[str, Any]:
    """The ledger as a per-task table plus totals. Every task's latest claim body is
    read (task id, row, holder, incarnation: the table's columns); ``details`` also
    reads the failure and settlement bodies (elapsed time, failure reasons).

    ``settled=False``: the table holds the unsettled tasks only (running, stale,
    queued); the totals still count every task. The cost is then one listing of
    ``done/`` plus the keys and claim bodies of the unsettled tasks, instead of
    every key of the ledger and one read or more per task."""
    ns = namespace.strip("/")
    seen = SettledSlots()
    if not settled and lists_ranges(objects):
        keys, now = scan_slots(objects, ns, seen)
    else:
        keys, now = objects.list_meta(ns + "/")
    plan = _json(objects, f"{ns}/plan.json")
    states = parse_ledger(keys, ns)
    lease = float(plan.get("lease_s", DEFAULT_LEASE_S))
    totals = {k: 0 for k in ("trained", "exhausted", "dropped", "running", "stale", "queued")}
    tasks = []
    for p in sorted(states):
        s = states[p]
        a = s.latest
        if s.done is not None:
            state = s.done
        elif a is None or a in s.fails:
            state = "queued"
        else:
            state = "running" if s.last_seen(a) + lease > now else "stale"
        totals[state] = totals.get(state, 0) + 1
        if s.done is not None and not settled:
            continue
        row: dict[str, Any] = {"slot": p, "state": state, "attempts": 0 if a is None else a + 1,
                               "counted_failures": s.counted(),
                               "fails": {str(k): v for k, v in sorted(s.fails.items())},
                               "age_s": None if a is None else round(now - s.claims[a], 1),
                               "beat_age_s": None if a is None else round(now - s.last_seen(a), 1)}
        claim = _json(objects, f"{ns}/claims/{_pkey(p)}/{_akey(a)}") if a is not None else {}
        row.update(task_id=claim.get("task_id"), row=claim.get("row"), holder=claim.get("holder"),
                   incarnation=claim.get("incarnation"))
        if details:
            done = _json(objects, f"{ns}/done/{_pkey(p)}.{s.done}") if s.done else {}
            reasons = []
            for k, kind in sorted(s.fails.items()):
                body = _json(objects, f"{ns}/fail/{_pkey(p)}/{_akey(k)}.{kind}")
                reasons.append(f"{k}:{kind}:{body.get('reason') or ''}".rstrip(":"))
            row.update(elapsed_s=done.get("elapsed_s"), reason=done.get("reason"), failure_reasons=reasons)
        tasks.append(row)
    limit = int(plan.get("slots") or int(plan.get("rows", 0)) * int(plan.get("passes", 1)))
    touched = sum(1 for s in states.values() if s.claims or s.done is not None)
    totals["unclaimed"] = max(0, limit - touched)
    totals["slots"] = limit
    totals["ended"] = sum(totals[k] for k in DONE_KINDS) >= limit > 0
    return {"namespace": ns, "plan": plan, "server_time": now, "totals": totals, "tasks": tasks}


def format_table(report: Mapping[str, Any]) -> str:
    cols = ("slot", "state", "attempts", "task_id", "holder", "elapsed_s", "beat_age_s", "failure_reasons")
    lines = [json.dumps({"namespace": report["namespace"], "totals": report["totals"]}, sort_keys=True),
             "  ".join(f"{c:<12}" for c in cols)]
    for t in report["tasks"]:
        vals = [t.get(c) for c in cols]
        vals[-1] = ",".join(vals[-1] or []) or "-"
        lines.append("  ".join(f"{'-' if v is None else str(v):<12}" for v in vals))
    return "\n".join(lines)


def _objects_for(store: str) -> Any:
    if store.startswith(("s3://", "r2://")):
        from yeto.cloud.s3_store import head_objects  # noqa: PLC0415

        return head_objects(store.replace("r2://", "s3://", 1))
    return DirObjects(store)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m yeto.rl.engine.store_dispense")
    sub = parser.add_subparsers(dest="command", required=True)
    st = sub.add_parser("status", help="per-task table of a store dispenser")
    st.add_argument("--store", required=True, help="store directory or s3:// URI (the samplers' store)")
    st.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    st.add_argument("--log", help="write one JSON line per task here (the dashboard's dispense log)")
    st.add_argument("--json", action="store_true")
    st.add_argument("--open", action="store_true", dest="open_only",
                    help="list the unsettled tasks only (running, stale, queued); the totals still count "
                    "every task. Far fewer requests on a large ledger: no read per settled task")
    st.add_argument("--brief", action="store_true", help="one listing plus each task's latest claim (task id, holder); no settlement or "
                    "failure reads (elapsed_s and failure_reasons are not shown)")
    rt = sub.add_parser("retire", help="retire one sampler process now: release its claims, fence it")
    rt.add_argument("--store", required=True, help="store directory or s3:// URI (the samplers' store)")
    rt.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    rt.add_argument("--holder", required=True, help="the sampler id")
    rt.add_argument("--incarnation", help="only this incarnation (default: every one holding a claim)")
    rt.add_argument("--reason", default="operator")
    from yeto.cloud.s3_store import add_store_client_arguments, apply_store_client_arguments, describe_store_error

    add_store_client_arguments(st)
    add_store_client_arguments(rt)
    args = parser.parse_args(argv)
    apply_store_client_arguments(args)
    if args.command == "retire":
        print(json.dumps(retire(_objects_for(args.store), args.namespace, args.holder, args.incarnation,
                                reason=args.reason), sort_keys=True))
        return 0
    try:
        report = status(_objects_for(args.store), args.namespace, details=not args.brief,
                        settled=not args.open_only)
    except Exception as error:  # noqa: BLE001 - a store credential / endpoint error gets its hint
        message = describe_store_error(error, args.store) if args.store.startswith(("s3://", "r2://")) else None
        if message is None:
            raise
        print(f"[store_dispense] {message}", file=sys.stderr)
        return 2
    if args.log:
        with open(args.log, "w", encoding="utf-8") as handle:
            for task in report["tasks"]:
                handle.write(json.dumps({"event": "rl_dispense_task", **task}, sort_keys=True) + "\n")
    if args.json:
        json.dump(report, sys.stdout, indent=1, sort_keys=True)
        print()
    else:
        print(format_table(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
