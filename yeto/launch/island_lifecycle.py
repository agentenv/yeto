"""The trainer island lifecycle every source and mode shares.

One state machine and one exit decision table for every trainer island, whatever
runs it (the fleet controller of ``yeto launch``, the trainer scheduler s1, a
resident ``ssh:`` island) and wherever it runs (sky, Modal, an SSH inventory):

* **states** (:data:`STATES`): ``ordering`` -> ``starting`` -> ``running`` <->
  ``recovering`` -> ``draining`` -> ``done`` / ``abandoned`` / ``failed``. The last
  three are terminal (:data:`TERMINAL`); :func:`allowed` says which moves exist;
* **decision table** (:func:`classify`, :data:`DECISIONS`): why an island ended ->
  what happens next. A configuration error (exit :data:`CONFIG_ERROR_EXIT`, its
  ``[yeto-config-error]`` line, LEAVE ``CONFIG_MISMATCH``, e.g.
  ``FullParameterLayoutMismatch``) and a strict failure are never relaunched (the
  island is ``failed``); an infrastructure failure, a lost node or cluster and a
  preemption are relaunched within the caller's relaunch budget (``recovering``); a
  normal end is ``done``, and so is a stop (STOP flag, LEAVE ``STOPPED``), which is
  never relaunched. A source only reports what it observed (exit code, LEAVE code,
  log line, job / cluster / call state); none decides on its own;
* **state record** (:class:`IslandStateStore`): each island's state is written to
  the run store at ``fleet/trainers/<k>/state.json`` (``k``: the island's learner
  id): state, source and its handle (cluster name and job id, Modal call id, SSH
  incarnation), incarnation, relaunch count, last exit, last resume point and the
  controller that wrote it. A launcher or a controller that dies leaves its islands
  readable from any machine (the record is plain JSON in the run store). Writes are
  best effort: a store that does not answer never stops supervision. The launcher's
  store does its I/O on one background thread (``background=True``): the fleet
  controller's poll loop never waits on the store (a write is queued, newest wins; an
  island's STOP is read from a cache refreshed every ``stop_poll_s``). A writer fence
  (:class:`Fence`: the run store's single-writer lock,
  ``yeto.launch.pool.store_lock.StoreLock``) stops a controller whose claim another
  one took over from writing any further record;
* **stop channel**: ``fleet/trainers/<k>/STOP`` (:meth:`IslandStateStore.request_stop`)
  stops island ``k`` only; the run's STOP flag stays for fleet-wide stops.

Standard library only (the launcher, ``yeto run up`` and the CLI import it).
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from ..config_errors import CONFIG_ERROR_EXIT, CONFIG_ERROR_MARKER

SCHEMA = 1

# -- states --------------------------------------------------------------------------------------
ORDERING = "ordering"  # the machines are being ordered
STARTING = "starting"  # machines up, the trainer is starting (image, model load, JOIN)
RUNNING = "running"
RECOVERING = "recovering"  # failed or lost; a relaunch is pending within the budget
DRAINING = "draining"  # asked to stop (STOP channel): finishing its segment, then released
DONE = "done"  # finished, or stopped on request
ABANDONED = "abandoned"  # out of relaunch budget (or a relaunch is refused for good)
FAILED = "failed"  # never relaunched: a configuration error or a strict failure

STATES: tuple[str, ...] = (ORDERING, STARTING, RUNNING, RECOVERING, DRAINING, DONE, ABANDONED, FAILED)
TERMINAL = frozenset({DONE, ABANDONED, FAILED})
ACTIVE = frozenset(STATES) - TERMINAL

_NEXT: dict[str, frozenset[str]] = {
    ORDERING: frozenset({STARTING, RUNNING, RECOVERING, DRAINING} | TERMINAL),
    STARTING: frozenset({RUNNING, RECOVERING, DRAINING} | TERMINAL),
    RUNNING: frozenset({RECOVERING, DRAINING} | TERMINAL),
    RECOVERING: frozenset({ORDERING, STARTING, RUNNING, DRAINING} | TERMINAL),
    DRAINING: TERMINAL,
    # a terminal island is only left by an explicit restart (``yeto island start``)
    DONE: frozenset({ORDERING}),
    ABANDONED: frozenset({ORDERING}),
    FAILED: frozenset({ORDERING}),
}


def allowed(old: str | None, new: str) -> bool:
    """Whether an island may move from ``old`` (None: no record yet) to ``new``."""
    if new not in STATES:
        return False
    if old is None or old == new:
        return True
    return new in _NEXT.get(old, frozenset())


# -- the decision table --------------------------------------------------------------------------
CONFIG_ERROR = "config_error"
STRICT = "strict"
INFRA = "infra"
LOST = "lost"
PREEMPTED = "preempted"
FINISHED = "finished"
STOPPED = "stopped"

FAIL = "fail"  # never relaunched
RELAUNCH = "relaunch"  # relaunched within the relaunch budget
FINISH = "finish"
STOP = "stop"  # ended on request: never relaunched

# cause -> (action, state)
DECISIONS: dict[str, tuple[str, str]] = {
    CONFIG_ERROR: (FAIL, FAILED),
    STRICT: (FAIL, FAILED),
    INFRA: (RELAUNCH, RECOVERING),
    LOST: (RELAUNCH, RECOVERING),
    PREEMPTED: (RELAUNCH, RECOVERING),
    FINISHED: (FINISH, DONE),
    STOPPED: (STOP, DONE),
}

# A strict RL failure, or the syncer refusing the island's identity / session.
STRICT_MARKERS: tuple[str, ...] = (
    "[yeto-rl-strict-failure]", "StrictRlInvariantError:",
    "backend identity mismatch, JOIN refused", "session mismatch (HELLO refused",
)
# LEAVE codes (``yeto.rl.leave_code.LeaveReason``) by name: no import of the RL package.
_LEAVE_CAUSE = {"CONFIG_MISMATCH": CONFIG_ERROR, "TRAINING_FAILED": STRICT, "STOPPED": STOPPED,
                "FINISHED": FINISHED, "PREEMPTED": PREEMPTED, "INFRA_FAILED": INFRA,
                "LEASE_EXPIRED": LOST, "FENCED": STOPPED}


@dataclass(frozen=True)
class Decision:
    """One row of the table: the cause found, its action and the island's next state."""

    cause: str
    action: str
    state: str
    detail: str = ""

    @property
    def relaunch(self) -> bool:
        return self.action == RELAUNCH

    def to_json(self) -> dict[str, str]:
        return {"cause": self.cause, "action": self.action, "state": self.state, "detail": self.detail[:500]}


def decision(cause: str, detail: str = "") -> Decision:
    action, state = DECISIONS[cause]
    return Decision(cause, action, state, detail)


def _leave_name(leave_reason: Any) -> str | None:
    if leave_reason is None:
        return None
    name = getattr(leave_reason, "name", None)
    return str(name if name is not None else leave_reason).upper()


def config_error_text(text: str | None) -> str | None:
    """The ``[yeto-config-error]`` line in ``text`` (None without one)."""
    if not text:
        return None
    for line in reversed(str(text).splitlines()):
        if CONFIG_ERROR_MARKER in line:
            return line.strip()
    return None


def strict_text(text: str | None) -> str | None:
    """The strict-failure line in ``text`` (None without one)."""
    if not text:
        return None
    for line in reversed(str(text).splitlines()):
        if any(m in line for m in STRICT_MARKERS):
            return line.strip()
    return None


def classify(*, exit_code: int | None = None, leave_reason: Any = None, log_text: str | None = None,
             succeeded: bool = False, stopped: bool = False, preempted: bool = False,
             lost: bool = False, detail: str = "") -> Decision:
    """The decision for an island that ended (module doc). The first matching row wins:
    configuration error, strict failure, stop, normal end, preemption, lost, and any
    other failure is an infrastructure failure. ``leave_reason``: a ``LeaveReason`` or
    its name; ``log_text``: the island's last log lines (its markers)."""
    leave = _leave_name(leave_reason)
    line = config_error_text(log_text)
    if line or leave == "CONFIG_MISMATCH" or (exit_code == CONFIG_ERROR_EXIT and not isinstance(exit_code, bool)):
        return decision(CONFIG_ERROR, line or detail or (f"exit {exit_code}" if exit_code is not None else leave or ""))
    line = strict_text(log_text)
    if line or leave == "TRAINING_FAILED":
        return decision(STRICT, line or detail or str(leave))
    if stopped or _LEAVE_CAUSE.get(leave or "") == STOPPED:
        return decision(STOPPED, detail or (leave or "stop requested"))
    if succeeded or leave == "FINISHED" or (exit_code == 0 and not isinstance(exit_code, bool)):
        return decision(FINISHED, detail)
    if preempted or leave == "PREEMPTED":
        return decision(PREEMPTED, detail or "preempted")
    if lost or leave == "LEASE_EXPIRED":
        return decision(LOST, detail or "lost")
    return decision(INFRA, detail or (f"exit {exit_code}" if exit_code is not None else leave or "failed"))


def classify_error(error: BaseException | None, *, detail: str = "") -> Decision:
    """The decision for an island process that stopped with ``error`` (None: finished):
    a :class:`~yeto.config_errors.ConfigError` in its chain is a configuration error,
    else its LEAVE code (``yeto.rl.engine.leave_reason.leave_reason_for``) decides."""
    from ..config_errors import config_error_in  # noqa: PLC0415

    if error is not None and config_error_in(error) is not None:
        return decision(CONFIG_ERROR, detail or f"{type(error).__name__}: {error}")
    from yeto.rl.engine.leave_reason import leave_reason_for  # noqa: PLC0415 - no torch

    reason = leave_reason_for(error)
    return classify(leave_reason=reason,
                    detail=detail or ("" if error is None else f"{type(error).__name__}: {error}"))


# -- the state record in the store ---------------------------------------------------------------
STATE_DIR = "fleet/trainers"
HISTORY = 20  # transitions kept in a record


def state_key(island: int) -> str:
    return f"{STATE_DIR}/{int(island)}/state.json"


def stop_key(island: int) -> str:
    return f"{STATE_DIR}/{int(island)}/STOP"


def controller_identity() -> dict[str, Any]:
    """Who supervises: host, PID and a fencing token unique to this process."""
    return {"host": socket.gethostname(), "pid": os.getpid(), "token": uuid.uuid4().hex[:16],
            "since": round(time.time(), 3)}


STOP_POLL_S = 15.0  # a background store re-reads the watched islands' STOP this often
FLUSH_S = 10.0  # a background store's queued writes get this long at the launcher's end


class Fence(Protocol):
    """A single-writer claim on the island records: the interface of the run store's lock
    (``yeto.launch.pool.store_lock.StoreLock``). ``token``: its fencing token, written into
    each record. ``guard()``: before a write; raises an error with a ``by`` attribute (the
    higher token, ``LockLost``) once another controller took over, and another error
    (``LockUnconfirmed``) when the store did not answer; it renews the lease when due."""

    token: int

    def guard(self) -> None: ...


class IslandStateStore:
    """Each island's state record in the run store (module doc).

    ``objects``: the store's object API (``read`` / ``write``; ``delete`` and ``list``
    when present). Writes keep the record's earlier fields and append a history entry
    on each state change; every failure is logged (once per kind) and swallowed: the
    record is the view others read, never the reason supervision stops.

    ``background``: writes are queued to one worker thread (the newest body per key wins)
    and :meth:`stop_requested` answers from a cache that thread refreshes every
    ``stop_poll_s``: the caller never waits on the store. :meth:`flush` / :meth:`close`
    wait (bounded) for the queue. ``fence``: a :class:`Fence`; once it is lost no record
    is written any more (sticky); while the store cannot confirm it, a write waits for
    the next one. The background thread also guards it every ``stop_poll_s`` (the
    lock's lease renewal). The STOP channel is not fenced: anyone may stop an island."""

    def __init__(self, objects: Any, *, controller: Mapping[str, Any] | None = None,
                 clock: Callable[[], float] = time.time, log: Callable[[str], None] | None = None,
                 background: bool = False, stop_poll_s: float = STOP_POLL_S,
                 fence: Fence | None = None) -> None:
        self.objects = objects
        self.controller = dict(controller or controller_identity())
        self.clock = clock
        self.log = log or (lambda m: print(m, file=sys.stderr, flush=True))
        self._records: dict[int, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._warned: set[str] = set()
        self.background = bool(background)
        self.stop_poll_s = float(stop_poll_s)
        self.fence = fence
        self.fenced: BaseException | None = None
        self._pending: dict[str, bytes] = {}
        self._watched: set[int] = set()
        self._stops: dict[int, dict[str, Any] | None] = {}
        self._stops_due = True
        self._wake = threading.Event()
        self._idle = threading.Event()
        self._idle.set()
        self._thread: threading.Thread | None = None
        self._closed = False

    def _warn(self, what: str, error: BaseException) -> None:
        kind = f"{what}:{type(error).__name__}"
        if kind not in self._warned:
            self._warned.add(kind)
            self.log(f"[launcher] island state store: {what} failed ({type(error).__name__}: {error})"[:400])

    # -- reading
    def read(self, island: int) -> dict[str, Any] | None:
        """Island ``island``'s record as stored (None: none, or the read failed)."""
        record = self._read_unlocked(island)
        if record is not None:
            with self._lock:
                self._records.setdefault(int(island), dict(record))  # what a first write merges into
        return record

    def _read_unlocked(self, island: int) -> dict[str, Any] | None:
        try:
            raw = self.objects.read(state_key(island))
        except Exception as error:  # noqa: BLE001
            self._warn("read", error)
            return None
        if raw is None:
            return None
        try:
            record = json.loads(raw)
        except ValueError as error:
            self._warn("parse", error)
            return None
        return record if isinstance(record, dict) else None

    def read_all(self, islands: Iterable[int] | None = None) -> dict[int, dict[str, Any]]:
        """Every island's record: those of ``islands``, else every one the store lists."""
        if islands is None:
            islands = self.listed()
        out = {}
        for k in islands:
            record = self.read(int(k))
            if record is not None:
                out[int(k)] = record
        return out

    def listed(self) -> list[int]:
        """The island ids with a record (by listing ``fleet/trainers/``)."""
        keys: list[str] = []
        try:
            lister = getattr(self.objects, "list", None)
            if callable(lister):
                keys = list(lister(f"{STATE_DIR}/"))
            elif callable(getattr(self.objects, "list_meta", None)):
                keys = [k for k, _ in self.objects.list_meta(f"{STATE_DIR}/")[0]]
        except Exception as error:  # noqa: BLE001
            self._warn("list", error)
        ids = set()
        for key in keys:
            parts = str(key).split("/")
            if len(parts) >= 4 and parts[-1] == "state.json" and parts[-2].isdigit():
                ids.add(int(parts[-2]))
        return sorted(ids)

    # -- writing
    def write(self, island: int, state: str | None = None, **fields: Any) -> dict[str, Any]:
        """Merge ``fields`` (and ``state``) into island ``island``'s record and store it.
        A state move :func:`allowed` refuses is still written (the record shows what
        happened) and logged."""
        if state is not None and state not in STATES:
            raise ValueError(f"unknown island state {state!r} (known: {', '.join(STATES)})")
        with self._lock:
            record = self._records.get(int(island))
            if record is None:  # a background store never reads here: read() / read_all() first
                record = {} if self.background else (self._read_unlocked(island) or {})
            record = dict(record)
            now = round(self.clock(), 3)
            old = record.get("state")
            record.update({k: v for k, v in fields.items()})
            record.update(schema=SCHEMA, island=int(island), updated=now, controller=self.controller)
            if state is not None and state != old:
                if old is not None and not allowed(old, state):
                    self.log(f"[launcher] island {island}: state {old} -> {state} is not a lifecycle move; "
                             "recorded as observed")
                record["state"] = state
                record["since"] = now
                entry = {"at": now, "from": old, "to": state}
                if fields.get("reason"):
                    entry["reason"] = str(fields["reason"])[:300]
                record["history"] = [*list(record.get("history") or ())[-(HISTORY - 1):], entry]
            if self.fence is not None:
                record["fence_token"] = int(self.fence.token)
            self._records[int(island)] = record
            data = json.dumps(record, sort_keys=True, default=str).encode()
        self._put(state_key(island), data)
        return record

    # -- the I/O: inline, or on the background thread
    def _put(self, key: str, data: bytes) -> None:
        if not self.background:
            self._store(key, data)
            return
        with self._lock:
            self._pending[key] = data
            self._idle.clear()
            self._start()
        self._wake.set()

    def _fence_holds(self) -> bool:
        if self.fence is None:
            return True
        if self.fenced is not None:
            return False
        try:
            self.fence.guard()
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as error:  # noqa: BLE001 - the lock's errors are BaseExceptions
            if hasattr(error, "by"):
                self.fenced = error
                self.log(f"[launcher] island state store: fenced ({error}); this controller writes no "
                         "island record any more"[:400])
            else:
                self._warn("fence check", error)  # not confirmed: this write waits for the next one
            return False
        return True

    def _store(self, key: str, data: bytes) -> None:
        if not self._fence_holds():
            return
        try:
            self.objects.write(key, data)
        except Exception as error:  # noqa: BLE001
            self._warn("write", error)

    def _start(self) -> None:  # under self._lock
        if self._thread is None and not self._closed:
            self._thread = threading.Thread(target=self._work, name="island-state-store", daemon=True)
            self._thread.start()

    def _work(self) -> None:
        stops_at = float("-inf")
        while True:
            self._wake.wait(self.stop_poll_s)
            self._wake.clear()
            with self._lock:
                pending, self._pending = self._pending, {}
                watched = sorted(self._watched)
                due, self._stops_due = self._stops_due, False
            for key, data in pending.items():
                self._store(key, data)
            if not pending:
                self._fence_holds()  # the lease renewal of a quiet fleet
            if watched and (due or time.monotonic() - stops_at >= self.stop_poll_s):
                for k in watched:
                    self._stops[k] = self._read_stop(k)
                stops_at = time.monotonic()
            with self._lock:
                if not self._pending:
                    self._idle.set()
                if self._closed and not self._pending:
                    return

    def flush(self, timeout: float = FLUSH_S) -> bool:
        """Wait up to ``timeout`` s for the queued writes (True once none is left)."""
        if not self.background:
            return True
        self._wake.set()
        return self._idle.wait(timeout)

    def close(self, timeout: float = FLUSH_S) -> bool:
        """Flush (bounded) and end the background thread; True when every write landed."""
        if not self.background:
            return True
        with self._lock:
            self._closed = True
            thread = self._thread
        self._wake.set()
        if thread is not None:
            thread.join(timeout)
        done = self._idle.is_set()
        if not done:
            self.log(f"[launcher] island state store: queued record writes not done after {timeout:.0f} s; "
                     "left behind")
        return done

    # -- the stop channel
    def stop_requested(self, island: int) -> dict[str, Any] | None:
        """Island ``island``'s STOP request (``{"reason": ..., "by": ..., "at": ...}``), else None.
        A background store answers from its cache: None until its first read of the island."""
        if not self.background:
            return self._read_stop(island)
        k = int(island)
        with self._lock:
            if k not in self._watched:
                self._watched.add(k)
                self._stops_due = True
                self._start()
                self._wake.set()
        return self._stops.get(k)

    def _read_stop(self, island: int) -> dict[str, Any] | None:
        try:
            raw = self.objects.read(stop_key(island))
        except Exception as error:  # noqa: BLE001
            self._warn("stop read", error)
            return None
        if raw is None:
            return None
        try:
            request = json.loads(raw)
        except ValueError:
            request = {"reason": raw.decode(errors="replace").strip() or "STOP"}
        return request if isinstance(request, dict) else {"reason": str(request)}

    def request_stop(self, island: int, reason: str, *, by: str = "operator") -> bool:
        """Write island ``island``'s STOP (True once written)."""
        body = {"reason": str(reason), "by": str(by), "at": round(self.clock(), 3)}
        try:
            self.objects.write(stop_key(island), json.dumps(body, sort_keys=True).encode())
            return True
        except Exception as error:  # noqa: BLE001
            self._warn("stop write", error)
            return False

    def clear_stop(self, island: int) -> bool:
        """Remove island ``island``'s STOP (True when none is left)."""
        try:
            delete = getattr(self.objects, "delete", None)
            if callable(delete):
                delete(stop_key(island))
            elif self.objects.read(stop_key(island)) is not None:
                raise RuntimeError("this store cannot delete objects")
            return True
        except FileNotFoundError:
            return True
        except Exception as error:  # noqa: BLE001
            self._warn("stop clear", error)
            return False


def store_objects(uri: str | None, environ: Mapping[str, str] | None = None) -> Any:
    """The launcher's object API of a run store: ``s3://`` / ``r2://`` / ``oss://`` by
    object key, a local path as a directory; None for any other store (a Modal Volume)
    or none."""
    if not uri:
        return None
    text = str(uri)
    if text.startswith(("s3://", "r2://", "oss://")):
        from .trainer_island import head_store_objects  # noqa: PLC0415

        return head_store_objects(text.replace("r2://", "s3://", 1), environ)
    if "://" in text:
        return None
    from yeto.rl.engine.object_store import DirectoryObjectStore  # noqa: PLC0415

    return DirectoryObjectStore(os.path.expanduser(text))


def open_state_store(uri: str | None, environ: Mapping[str, str] | None = None, *,
                     log: Callable[[str], None] | None = None, background: bool = False,
                     fence: Fence | None = None) -> IslandStateStore | None:
    """:class:`IslandStateStore` of the run store ``uri`` (None when it has no object API)."""
    try:
        objects = store_objects(uri, environ)
    except Exception as error:  # noqa: BLE001 - no record, supervision goes on
        (log or (lambda m: print(m, file=sys.stderr, flush=True)))(
            f"[launcher] island state store: {uri}: {type(error).__name__}: {error}"[:400])
        return None
    return None if objects is None else IslandStateStore(objects, log=log, background=background, fence=fence)


@dataclass(frozen=True)
class IslandRef:
    """What the fleet controller knows of one launch island besides its record:
    its learner id (the record's ``k``), its source and its role."""

    island: int
    source: str  # "sky" | "modal" | "ssh" | "verda" | ...
    role: str = "trainer"  # "trainer" | "sampler"
    tape: str | None = None  # the island's echoed event tape on the launcher's machine


class FleetRecorder:
    """The fleet controller's writer of island records: launch island name ->
    :class:`IslandRef`, and the store. A name it does not know (the syncer, a
    renamed island it was not told about) is not recorded."""

    def __init__(self, store: IslandStateStore, islands: Mapping[str, IslandRef]) -> None:
        self.store = store
        self.islands = dict(islands)

    def ref(self, name: str) -> IslandRef | None:
        return self.islands.get(name)

    def rename(self, old: str, new: str) -> None:
        if old in self.islands:
            self.islands[new] = self.islands.pop(old)

    def record(self, name: str, state: str | None = None, **fields: Any) -> dict[str, Any] | None:
        ref = self.islands.get(name)
        if ref is None:
            return None
        if ref.tape is not None:
            fields.setdefault("tape", ref.tape)
        return self.store.write(ref.island, state, name=name, source=ref.source, role=ref.role, **fields)

    def stop_requested(self, name: str) -> dict[str, Any] | None:
        ref = self.islands.get(name)
        if ref is None or ref.role == "sampler":
            return None
        return self.store.stop_requested(ref.island)


def handle_of(source: str, name: str, job_id: Any) -> dict[str, Any]:
    """The machine handle of an island in its record: the cluster (or island) name
    and the source's job id (Modal: the function call id; SSH: the incarnation)."""
    out: dict[str, Any] = {"cluster": name}
    if job_id is None:
        return out
    if source == "modal":
        out["call_id"] = str(job_id)
    elif source == "ssh":
        out["incarnation"] = int(job_id) if str(job_id).isdigit() else str(job_id)
    else:
        out["job_id"] = int(job_id) if str(job_id).isdigit() else str(job_id)
    return out


def active_islands(records: Mapping[int, Mapping[str, Any]]) -> list[int]:
    """The islands whose record is not terminal (sampler islands excluded)."""
    return sorted(k for k, r in records.items()
                  if r.get("state") in ACTIVE and r.get("role") != "sampler")
