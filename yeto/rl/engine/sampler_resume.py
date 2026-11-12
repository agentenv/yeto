"""Sampler island replacement: where a new incarnation continues.

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
# one record per trainer whose run ended (sample_roles module doc: samplers that run until then)
TRAINERS_ENDED_DIR = "sample-buffer/trainers-ended"
TRAINERS_ENDED_EVENT = "rl_sampler_trainers_ended"
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
    # the task-dispenser position of the group's row (a sampler with a dispenser)
    dispense_position: int | None = None
    # the store dispenser's attempt of that task (``store_dispense.ATTEMPT_KEY``); only
    # a group of the store dispenser has it: there ``dispense_position`` is the slot,
    # one per task of the run, and the trainer's reader trains one group per slot
    dispense_attempt: int | None = None
    # the store dispenser that slot belongs to (``StoreDispensedPrompts.plan_id``)
    dispense_plan: str | None = None

    def record(self) -> dict[str, Any]:
        out = {"sampler": self.sampler, "incarnation": self.incarnation,
               "rollout_id": int(self.rollout_id), "last": bool(self.last)}
        if self.data_cursor is not None:
            out["data_cursor"] = {k: int(v) for k, v in self.data_cursor.items()}
        if self.dispense_position is not None:
            out["dispense_position"] = int(self.dispense_position)
        if self.dispense_attempt is not None:
            out["dispense_attempt"] = int(self.dispense_attempt)
        if self.dispense_plan is not None:
            out["dispense_plan"] = str(self.dispense_plan)
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


def _put_record(store: Any, key: str, record: dict[str, Any]) -> Path:
    """Write the JSON ``record`` at ``key`` under the store root and commit the
    store. A store with an object API (s3, OSS) gets it by object: its staged
    mirror does not carry ``sample-buffer/``, so a file would stay on the
    island's disk."""
    data = (json.dumps(record, sort_keys=True) + "\n").encode()
    objects = getattr(store, "objects", None)
    if objects is not None:
        objects.write(key, data)
        return Path(key)
    path = Path(store.root) / key
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
    store.commit()
    return path


def write_member(store: Any, member: Member) -> Path:
    """Write (or rewrite) the incarnation's member record and commit the store.
    One file per incarnation: an old and a new incarnation never write the same
    file (a Modal Volume keeps the last commit of a file)."""
    return _put_record(store, member_key(member), asdict(member))


def write_trainer_end(store: Any, trainer: str, **fields: Any) -> Path:
    """The trainer ``trainer`` (its learner id) ended its run: one record per
    trainer, read by the samplers that run until their trainers end."""
    return _put_record(store, f"{TRAINERS_ENDED_DIR}/{trainer}.json",
                       {"trainer": str(trainer), "ended_at": time.time(), **fields})


def trainers_ended(store: Any) -> int:
    """How many trainers wrote their end record. Reads the store as it is: the
    caller's sample buffer refresh reloads it (once per round and per backlog
    poll of the sampler loop)."""
    objects = getattr(store, "objects", None)
    if objects is not None:
        names = objects.list(f"{TRAINERS_ENDED_DIR}/")
    else:
        names = store.list(TRAINERS_ENDED_DIR)
    return sum(1 for name in names if name.endswith(".json"))


def stop_when_trainers_ended(store: Any, trainers: int) -> Callable[[], bool]:
    """``should_stop`` of a sampler that runs until ``trainers`` trainers that read
    ``store`` ended (``sample_roles`` module doc); never for 0."""
    if int(trainers) < 1:
        return lambda: False
    return lambda: trainers_ended(store) >= int(trainers)


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
    30 s), while the island's loop may still be inside a round. A poll that raises
    (the lease watcher reads the store) is no verdict: it is counted
    (``poll_failures``) and the next poll reads again, so one failed read never
    ends the thread."""

    def __init__(self, reclaimed: Callable[[], bool], on_reclaim: Callable[[], Any], *,
                 poll_s: float = 0.5, name: str = "sampler-reclaim-watcher") -> None:
        self.reclaimed = reclaimed
        self.on_reclaim = on_reclaim
        self.poll_s = float(poll_s)
        self._stop = threading.Event()
        self.fired = threading.Event()
        self.poll_failures = 0
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)

    def start(self) -> "ReclaimWatcher":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)

    def _run(self) -> None:
        failing = False
        while not self._stop.is_set():
            try:
                reclaimed = self.reclaimed()
                failing = False
            except Exception as error:  # noqa: BLE001 - unknown this time: poll again
                self.poll_failures += 1
                if not failing:  # one line per run of failures
                    print(f"[sampler-island] {self._thread.name}: poll failed, polling again: {error!r}",
                          flush=True)
                failing = True
                reclaimed = False
            if reclaimed:
                self.fired.set()
                try:
                    self.on_reclaim()
                except Exception as error:  # noqa: BLE001 - the island loop still stops on the marker
                    print(f"[sampler-island] reclaim handler failed: {error!r}", flush=True)
                return
            self._stop.wait(self.poll_s)


# ---------------------------------------------------------------------------- handoff
# A replacement sampler with its own shard (the NC lease pool: a new node, a new
# shard) continues the prompts of the sampler it replaces (``continues_shard``):
# it asks that sampler to stop (``handoffs/<predecessor>.json``), waits for its
# LEAVE record, and starts at the round after the predecessor's newest written
# round. Prompts are drawn by rollout id within the data slot, so the rounds
# continue the predecessor's rows with no repeat. The predecessor's shard keeps
# its groups; the trainer reads them as before.
HANDOFF_DIR = "sample-buffer/handoffs"
HANDOFF_EVENT = "rl_sampler_handoff"
DEFAULT_HANDOFF_WAIT_S = 900.0


def _objects(store: Any) -> Any:
    return getattr(store, "objects", None)


def _read_record(store: Any, key: str) -> dict[str, Any] | None:
    objects = _objects(store)
    if objects is not None:
        raw = objects.read(key)
    else:
        path = Path(store.root) / key
        raw = path.read_bytes() if path.is_file() else None
    return None if raw is None else json.loads(raw)


def _list_records(store: Any, prefix: str) -> list[str]:
    objects = _objects(store)
    if objects is not None:
        return [k for k in objects.list(prefix.rstrip("/") + "/") if k.endswith(".json")]
    root = Path(store.root) / prefix
    return sorted(f"{prefix}/{p.name}" for p in root.glob("*.json")) if root.is_dir() else []


def request_handoff(store: Any, predecessor: str, successor: str, clock: Callable[[], float] = time.time) -> None:
    _put_record(store, f"{HANDOFF_DIR}/{predecessor}.json",
                {"predecessor": predecessor, "successor": successor, "requested_at": clock()})


def handoff_requested(store: Any, sampler: str, *, on_error: Callable[[BaseException], None] | None = None) -> bool:
    """Whether a successor asked ``sampler`` to stop (its loop starts no new round).
    The sampler loop polls this before every round: a read that fails is no answer
    (False now, read again at the next poll; ``on_error(error)`` hears of it), never
    a reason to stop or to end the sampler. A record that is there but torn counts
    as a request."""
    try:
        return _read_record(store, f"{HANDOFF_DIR}/{sampler}.json") is not None
    except ValueError:
        return True
    except Exception as error:  # noqa: BLE001 - unknown this time
        from .object_store import is_fatal_store_error  # noqa: PLC0415

        if is_fatal_store_error(error):  # refused credentials, no such bucket: no outage
            raise
        if on_error is not None:
            on_error(error)
        return False


HANDOFF_ASIDE_DIR = "sample-buffer/handoffs-aside"


def _delete_record(store: Any, key: str) -> None:
    objects = _objects(store)
    if objects is not None:
        objects.delete(key)
    else:
        (Path(store.root) / key).unlink(missing_ok=True)


def set_aside_handoff(store: Any, sampler: str, *, keep: Iterable[str] = (), reason: str = "",
                      clock: Callable[[], float] = time.time) -> dict[str, Any] | None:
    """Move the stop request of ``sampler`` (``handoffs/<sampler>.json``) to
    ``handoffs-aside/`` unless its successor is in ``keep``; returns the record set
    aside (None: no record, or one that stays). A fenced shard of an earlier attempt
    of the run leaves its request behind; a new sampler started on the same shard
    name would obey it and stop. The record is copied before it is removed, so
    nothing is lost; only ``store`` (one run's store) is touched."""
    key = f"{HANDOFF_DIR}/{sampler}.json"
    record = _read_record(store, key)
    if record is None or str(record.get("successor")) in {str(k) for k in keep}:
        return None
    stamp = int(float(record.get("requested_at") or 0) * 1000)
    _put_record(store, f"{HANDOFF_ASIDE_DIR}/{sampler}-{stamp}.json",
                {**record, "set_aside_at": clock(), "set_aside_reason": reason})
    _delete_record(store, key)
    return record


def sampler_left(store: Any, sampler: str) -> bool:
    """Whether every incarnation record of ``sampler`` has its LEAVE (True: none is running)."""
    records = [_read_record(store, key) for key in _list_records(store, f"{MEMBERS_DIR}/{sampler}")]
    records = [r for r in records if r is not None]
    return bool(records) and all(r.get("left_at") is not None for r in records)


def handoff_start(store: Any, predecessor: str, successor: str,
                  predecessor_entries: Callable[[], Iterable[Any]], *,
                  wait_s: float = DEFAULT_HANDOFF_WAIT_S, poll_s: float = 5.0,
                  clock: Callable[[], float] = time.monotonic,
                  sleep: Callable[[float], None] = time.sleep) -> tuple[int, dict[str, Any]]:
    """The first round of ``successor`` (and the handoff detail). A predecessor that
    left: its next round. One still running after ``wait_s`` (or that never wrote a
    member record, e.g. killed hard): one round further, so its round in flight
    cannot repeat prompts (a round's rows are skipped instead)."""
    def has_left() -> bool:
        try:
            return sampler_left(store, predecessor)
        except Exception as error:  # noqa: BLE001 - unknown this time: not seen leaving, read again
            print(f"[sampler-island] member records of {predecessor} not read: {error!r}", flush=True)
            return False

    request_handoff(store, predecessor, successor)
    deadline = clock() + float(wait_s)
    left = has_left()
    while not left and clock() < deadline:
        sleep(poll_s)
        left = has_left()
    point = resume_point(predecessor_entries(), predecessor)
    start = point.next_rollout_id if left or not point.written_rounds else point.next_rollout_id + 1
    return start, {"predecessor": predecessor, "predecessor_left": left,
                   "predecessor_next_rollout_id": point.next_rollout_id,
                   "predecessor_groups": point.groups, "start_rollout_id": start}
