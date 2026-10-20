"""Sampler island replacement: where a new incarnation continues (S25 C1).

A sampler island on spot capacity can be reclaimed at any time; the cloud (or
the launcher) then starts a new incarnation of the same island. The island's
sample-buffer writes must neither repeat nor lose groups, and the trainer keeps
training on what the buffer holds.

Each index entry a sampler writes carries a :class:`SamplerRound` record
(``SampleIndexEntry.sampler_round``): the sampler id (the island, stable across
incarnations), the incarnation, the rollout id, the data cursor after the round
drew its prompts, and ``last`` on the round's last group. A Miles round writes
its groups after the whole round is generated, so the buffer itself says how
far the island got:

* a round with any entry in the buffer is never written again
  (:class:`RoundGuard`); the new incarnation continues at the next rollout id
  from that round's data cursor, so no prompt is drawn twice;
* a round without its ``last`` entry (reclaimed while it wrote) keeps the
  groups it wrote (each whole and written once); its unwritten groups are lost
  (``partial_rounds`` counts such rounds);
* a round still generating when the island was reclaimed wrote nothing: the new
  incarnation generates that rollout id again from the cursor before it.

:func:`resume_point` reads the entries; the island loop starts at
``next_rollout_id`` and the rollout function seeks the data source to
``data_cursor``. The incarnation's arrival and departure are written as
``rl_sampler_join`` / ``rl_sampler_leave`` (reason: a ``LeaveReason`` name) and as
one member record per incarnation in the store (:func:`write_member`), which
outlives the island's own tape. Engine-neutral.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

JOIN_EVENT = "rl_sampler_join"
LEAVE_EVENT = "rl_sampler_leave"
DROPPED_EVENT = "rl_sampler_groups_dropped"  # groups not written after the leave
MEMBERS_DIR = "sample-buffer/members"  # one record per sampler incarnation, under the store root
RECLAIM_MARKER_ENV = "YETO_SPOT_RECLAIM_MARKER"  # written by the Modal runner on the reclaim signal
DEFAULT_RECLAIM_MARKER = "/tmp/yeto-reclaim-requested"
# Set where a host hook waits for the sampler's answer before the VM stops
# (Nebius: yeto.cloud.nebius_reclaim writes the marker and waits for this file).
RECLAIM_ACK_ENV = "YETO_RECLAIM_ACK_FILE"


class SamplerResumeError(RuntimeError):
    pass


def new_incarnation() -> str:
    """A fresh incarnation id (unique per island process)."""
    return uuid.uuid4().hex[:8]


@dataclass(frozen=True)
class SamplerRound:
    """The ``sampler_round`` record of one index entry."""

    sampler: str
    incarnation: str
    rollout_id: int
    data_cursor: Mapping[str, int] | None = None  # the data source after the round drew its prompts
    last: bool = False  # the round's last group

    def record(self) -> dict[str, Any]:
        out = {"sampler": self.sampler, "incarnation": self.incarnation,
               "rollout_id": int(self.rollout_id), "last": bool(self.last)}
        if self.data_cursor is not None:
            out["data_cursor"] = {k: int(v) for k, v in self.data_cursor.items()}
        return out


@dataclass(frozen=True)
class SamplerResume:
    """Where the next incarnation of a sampler continues."""

    next_rollout_id: int  # after the newest round with an entry in the buffer
    data_cursor: dict[str, int] | None  # that round's data cursor (None: start of the data)
    written_rounds: tuple[int, ...]  # rounds with at least one entry: never written again
    partial_rounds: tuple[int, ...]  # written rounds without their last group
    incarnations: tuple[str, ...]  # earlier incarnations, in the order they first wrote
    groups: int  # every group this sampler has in the buffer

    @property
    def resumed(self) -> bool:
        return bool(self.incarnations)

    def fields(self) -> dict[str, Any]:
        out = asdict(self)
        out["written_rounds"] = len(self.written_rounds)
        out["partial_rounds"] = list(self.partial_rounds)
        out["incarnations"] = list(self.incarnations)
        return out


def resume_point(entries: Iterable[Any], sampler: str) -> SamplerResume:
    """Read the ``sampler_round`` records of ``sampler``'s entries (other samplers'
    entries and entries without the record are skipped)."""
    cursors: dict[int, dict[str, int] | None] = {}
    ended: set[int] = set()
    incarnations: list[str] = []
    groups = 0
    for entry in entries:
        record = getattr(entry, "sampler_round", None)
        if not record or record.get("sampler") != sampler:
            continue
        groups += 1
        rid = int(record["rollout_id"])
        raw = record.get("data_cursor")
        cursor = None if raw is None else {k: int(v) for k, v in raw.items()}
        if rid in cursors and cursors[rid] != cursor:
            raise SamplerResumeError(f"sampler {sampler}: round {rid} has two data cursors "
                                     f"({cursors[rid]} and {cursor})")
        cursors[rid] = cursor
        if record.get("incarnation") not in incarnations:
            incarnations.append(str(record.get("incarnation")))
        if record.get("last"):
            if rid in ended:
                raise SamplerResumeError(f"sampler {sampler}: round {rid} has two last groups")
            ended.add(rid)
    newest = max(cursors) if cursors else None
    return SamplerResume(next_rollout_id=0 if newest is None else newest + 1,
                         data_cursor=None if newest is None else cursors[newest],
                         written_rounds=tuple(sorted(cursors)),
                         partial_rounds=tuple(sorted(set(cursors) - ended)),
                         incarnations=tuple(incarnations), groups=groups)


class RoundGuard:
    """Write-side check: a round with entries in the buffer is never written by
    another incarnation, and an incarnation never writes before its first round."""

    def __init__(self, resume: SamplerResume) -> None:
        self._written = set(resume.written_rounds)
        self._first = int(resume.next_rollout_id)

    def check(self, rollout_id: int) -> None:
        rid = int(rollout_id)
        if rid in self._written or rid < self._first:
            raise SamplerResumeError(
                f"round {rid} is already in the sample buffer (this incarnation starts at "
                f"round {self._first}); a replacement sampler must not write it again")

    def written(self, rollout_id: int) -> None:
        self._written.add(int(rollout_id))


def reclaim_marker(environ: Mapping[str, str]) -> Path:
    return Path(environ.get(RECLAIM_MARKER_ENV) or DEFAULT_RECLAIM_MARKER)


def write_reclaim_marker(environ: Mapping[str, str], signum: int) -> Path:
    """Write the reclaim marker (what the Modal runner writes on its signal). The
    runner may write it at the same moment: the temp file is per process, and a
    failed write is reported, never raised (it runs in a signal handler; the
    runner's marker serves as well)."""
    path = reclaim_marker(environ)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps({"signal": int(signum), "time": time.time()}) + "\n")
        tmp.replace(path)
    except OSError as error:
        print(f"[sampler-island] reclaim marker not written ({error})", flush=True)
    return path


def install_marker_signals(environ: Mapping[str, str], signal_mod: Any = None) -> list[str]:
    """On the island's main thread: SIGINT and SIGTERM write the reclaim marker
    instead of raising ``KeyboardInterrupt``. Modal signals the whole container,
    so the island process gets the reclaim signal next to the runner; without
    this it dies inside its round and leaves as INFRA_FAILED (T-C1 run a).
    Returns the signals handled ([] off the main thread)."""
    import signal as _signal

    sig = signal_mod or _signal
    if threading.current_thread() is not threading.main_thread():
        return []
    done = []
    for name in ("SIGINT", "SIGTERM"):
        num = getattr(sig, name, None)
        if num is not None:
            sig.signal(num, lambda signum, _frame: write_reclaim_marker(environ, signum))
            done.append(name)
    return done


def clear_reclaim_files(environ: Mapping[str, str]) -> None:
    """Remove a reclaim marker and ack left from an earlier boot (a restarted VM keeps /tmp)."""
    reclaim_marker(environ).unlink(missing_ok=True)
    ack = environ.get(RECLAIM_ACK_ENV)
    if ack:
        Path(ack).unlink(missing_ok=True)


def acknowledge_reclaim(environ: Mapping[str, str]) -> Path | None:
    """Write the ack file a host hook waits for (``RECLAIM_ACK_ENV``) once the
    LEAVE is durable, so the host lets the VM stop; None when no hook waits."""
    ack = environ.get(RECLAIM_ACK_ENV)
    if not ack:
        return None
    path = Path(ack)
    try:
        path.write_text(f"{time.time():.3f}\n")
    except OSError as error:
        print(f"[sampler-island] reclaim ack not written ({error})", flush=True)
        return None
    return path


def reclaim_requested(environ: Mapping[str, str]) -> bool:
    """The Modal runner wrote the reclaim marker (the container got SIGINT/SIGTERM)."""
    return reclaim_marker(environ).exists()


@dataclass
class Member:
    """One sampler incarnation's record in the store (``members/<sampler>/<incarnation>.json``)."""

    sampler: str
    incarnation: str
    joined_at: float
    resume: dict[str, Any]
    policy_version: int | None = None
    left_at: float | None = None
    leave_reason: str | None = None
    rounds: int | None = None  # rounds this incarnation completed
    detail: dict[str, Any] = field(default_factory=dict)


def member_key(member: Member) -> str:
    return f"{MEMBERS_DIR}/{member.sampler}/{member.incarnation}.json"


def write_member(store: Any, member: Member) -> Path:
    """Write (or rewrite) the incarnation's member record and commit the store.
    One file per incarnation: an old and a new incarnation never write the same
    file (a Modal Volume keeps the last commit of a file). A store with an object
    API (s3, OSS) gets the record by object: its staged mirror does not carry
    ``sample-buffer/``, so a file would stay on the island's disk."""
    data = (json.dumps(asdict(member), sort_keys=True) + "\n").encode()
    objects = getattr(store, "objects", None)
    if objects is not None:
        objects.write(member_key(member), data)
        return Path(member_key(member))
    path = Path(store.root) / member_key(member)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
    store.commit()
    return path


class SamplerMembership:
    """One incarnation's JOIN and LEAVE: the tape events and the member record.
    :meth:`leave` acts once (the reclaim watcher and the island's exit path may
    both call it); later calls return False."""

    def __init__(self, store: Any, sampler: str, incarnation: str, resume: SamplerResume,
                 emit: Callable[..., None], clock: Callable[[], float] = time.time) -> None:
        self.store = store
        self.emit = emit
        self.clock = clock
        self.member = Member(sampler=sampler, incarnation=incarnation, joined_at=clock(),
                             resume=resume.fields())
        self._lock = threading.Lock()
        self.left = False

    def join(self, **fields: Any) -> None:
        m = self.member
        self.emit(JOIN_EVENT, sampler=m.sampler, incarnation=m.incarnation, **m.resume, **fields)
        write_member(self.store, m)

    def leave(self, reason: str, *, rounds: int | None = None, policy_version: int | None = None,
              **detail: Any) -> bool:
        with self._lock:
            if self.left:
                return False
            self.left = True
            m = self.member
            m.left_at, m.leave_reason, m.rounds = self.clock(), str(reason), rounds
            m.policy_version, m.detail = policy_version, dict(detail)
            self.emit(LEAVE_EVENT, sampler=m.sampler, incarnation=m.incarnation, reason=str(reason),
                      rounds=rounds, policy_version=policy_version, **detail)
            write_member(self.store, m)
            return True


class ReclaimWatcher:
    """Polls ``reclaimed()`` on a daemon thread and calls ``on_reclaim()`` once, at
    once: the cloud stops the container a few seconds after its signal (Modal:
    30 s), while the island's loop may still be inside a round."""

    def __init__(self, reclaimed: Callable[[], bool], on_reclaim: Callable[[], Any], *,
                 poll_s: float = 0.5) -> None:
        self.reclaimed = reclaimed
        self.on_reclaim = on_reclaim
        self.poll_s = float(poll_s)
        self._stop = threading.Event()
        self.fired = threading.Event()
        self._thread = threading.Thread(target=self._run, name="sampler-reclaim-watcher", daemon=True)

    def start(self) -> "ReclaimWatcher":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            if self.reclaimed():
                self.fired.set()
                try:
                    self.on_reclaim()
                except Exception as error:  # noqa: BLE001 - the island loop still stops on the marker
                    print(f"[sampler-island] reclaim handler failed: {error!r}", flush=True)
                return
            self._stop.wait(self.poll_s)
