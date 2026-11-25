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
  ``FullParameterLayoutMismatch``), a strict failure and a trainer breaker's stop
  (:data:`BREAKER`: e.g. no new group within ``trainer.sample_wait``; exit
  :data:`ISLAND_STOP_EXIT`, its ``[yeto-island-stop]`` line, or the island's own
  ``rl_breaker`` stop record) are never relaunched (the island is ``failed``: a
  relaunch stops the same way and bills a machine meanwhile); an infrastructure failure, a lost node or cluster and a
  preemption are relaunched within the caller's relaunch budget (``recovering``); a
  normal end is ``done``, and so is a stop (STOP flag, LEAVE ``STOPPED``), which is
  never relaunched. A source only reports what it observed (exit code, LEAVE code,
  log line, job / cluster / call state); none decides on its own;
* **state record** (:class:`IslandStateStore`): each island's state is written to
  the run store at ``fleet/trainers/<k>/state.json`` (``k``: the island's learner
  id): state, source and its handle (cluster name and job id, Modal call id, SSH
  incarnation), incarnation, relaunch count, last exit, last resume point and the
  controller that wrote it. A launcher or a controller that dies leaves its islands
  readable from any machine (the record is plain JSON in the run store) and
  adoptable by the next launcher of the run (``yeto.launch.island_adopt``). Writes are
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

from ..config_errors import CONFIG_ERROR_EXIT, CONFIG_ERROR_MARKER, ISLAND_STOP_EXIT, ISLAND_STOP_MARKER

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
BREAKER = "breaker"  # a trainer breaker stopped the island (rl_breaker action stop)

FAIL = "fail"  # never relaunched
RELAUNCH = "relaunch"  # relaunched within the relaunch budget
FINISH = "finish"
STOP = "stop"  # ended on request: never relaunched
# who wrote an island STOP: the per-island fleet's trainer worker writes one to move, remove
# or give up the island, or to forward the fleet's stop; ``yeto run up`` and ``yeto island
# stop`` write theirs under their own names (``IslandStateStore.request_stop(by=...)``)
STOP_BY_TRAINER_WORKER = "trainer worker"

# cause -> (action, state)
DECISIONS: dict[str, tuple[str, str]] = {
    CONFIG_ERROR: (FAIL, FAILED),
    STRICT: (FAIL, FAILED),
    BREAKER: (FAIL, FAILED),
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
# A trainer breaker's stop marks the error it raises with the breaker's name (this attribute,
# ``yeto.rl.engine.trainer_breakers.stop_error``): the reader's stops (trainer.sample_wait,
# trainer.buffer_damaged, trainer.sample_alignment, trainer.compaction_round). The mark survives
# pickling, so an error a Ray actor raised still carries it (``RayTaskError.cause``).
BREAKER_ATTR = "yeto_breaker"
# The trainer breakers whose stop record (``rl_breaker`` action stop) alone makes the island's
# stop deterministic (:func:`classify_records`): the ones that mark their error (above). Any
# other trainer breaker's stop goes by its strict record or its LEAVE; a new transient breaker
# is not read as final unless listed here (tests/test_island_deterministic_stop.py checks the list).
BREAKER_STOPS = frozenset({"trainer.sample_wait", "trainer.buffer_damaged", "trainer.sample_alignment",
                           "trainer.compaction_round", "trainer.release_drain"})
# The strict failures that end an island the same way on every attempt (module, class): a
# strict trainer check (``trainer.strict_invariant`` -- including a learning rate 0 before the
# final round --, ``trainer.nan``, ``trainer.gradient_flow``, ``trainer.policy_hash``; the
# syncer's session refusals: HELLO profile / session / identity mismatch, island auth, epoch
# fenced, init digest), groups older than ``max_policy_age`` that the rollout cannot drop, a
# malformed policy-age limit, an optimizer without a readable or finite learning rate. A whitelist: the other LEAVE ``TRAINING_FAILED`` errors (a
# failed optimizer step, a publication whose engine update raised, a cross-channel commit, a
# plain ``DriverError``) may be an NCCL, CUDA, OOM, storage or network fault underneath, and
# keep their retries and relaunches.
DETERMINISTIC_ERRORS: tuple[tuple[str, str], ...] = (
    ("yeto.rl.core", "StrictRlInvariantError"),
    ("yeto.rl.engine.driver", "PolicyAgeExceeded"),
    ("yeto.rl.engine.policy_age", "PolicyAgeError"),
    ("yeto.rl.applied_lr", "AppliedLrError"),
)
DETERMINISTIC_ERROR_NAMES = frozenset(name for _module, name in DETERMINISTIC_ERRORS)
# The deterministic stops a faulty card may cause (a silent data error on one of the island's
# GPUs): a non-finite gradient norm or the NaN-skip limit (``trainer.nan``), a policy that does
# not hash the same after an in-memory apply or rebuild (``trainer.policy_hash``). The island
# gets one relaunch on a newly ordered machine for each (:data:`NEW_MACHINE_RETRIES`, counted in
# its record and its tape); the next stop of the same breaker ends it as any deterministic stop.
# A data or algorithm cause stops again soon after the relaunch's resume: the waste is bounded.
# Not here: ``trainer.nonfinite_reward`` (the reward), ``trainer.xhw_guard`` (the samplers'
# numerics), ``trainer.gradient_flow`` (a zero gradient: frozen or detached parameters),
# ``trainer.strict_invariant`` and every breaker of the reader (sample_wait, ...).
NEW_MACHINE_BREAKERS = frozenset({"trainer.nan", "trainer.policy_hash"})
NEW_MACHINE_RETRIES = 1
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


def island_stop_text(text: str | None) -> str | None:
    """The ``[yeto-island-stop]`` line in ``text`` (None without one)."""
    if not text:
        return None
    for line in reversed(str(text).splitlines()):
        if ISLAND_STOP_MARKER in line:
            return line.strip()
    return None


def _stop_line_cause(line: str) -> str:
    """The cause a ``[yeto-island-stop] <cause> ...`` line names (:data:`BREAKER` when it names
    none: the exit code alone says only that a relaunch cannot help)."""
    word = line.split(ISLAND_STOP_MARKER, 1)[1].split(None, 1)[:1]
    return word[0] if word and word[0] in (BREAKER, STRICT, CONFIG_ERROR) else BREAKER


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
    line = island_stop_text(log_text)
    if line or (exit_code == ISLAND_STOP_EXIT and not isinstance(exit_code, bool)):
        return decision(_stop_line_cause(line) if line else BREAKER, line or detail or f"exit {exit_code}")
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


def breaker_error_in(error: BaseException | None) -> tuple[str, BaseException] | None:
    """The trainer breaker stop in ``error``'s chain (cause, context and a Ray task error's
    ``cause``): its breaker's name and the error that carries the mark (:data:`BREAKER_ATTR`);
    None without one."""
    seen, todo = set(), [error]
    while todo:
        error = todo.pop()
        if error is None or id(error) in seen:
            continue
        seen.add(id(error))
        name = getattr(error, BREAKER_ATTR, None)
        if isinstance(name, str) and name:
            return name, error
        inner = getattr(error, "cause", None)
        todo += [e for e in (error.__cause__, error.__context__, inner) if isinstance(e, BaseException)]
    return None


def classify_error(error: BaseException | None, *, detail: str = "", deterministic: bool = True) -> Decision:
    """The decision for an island process that stopped with ``error`` (None: finished):
    a :class:`~yeto.config_errors.ConfigError` in its chain is a configuration error, a
    deterministic stop (:func:`island_stop_decision`: a trainer breaker, a LEAVE
    ``CONFIG_MISMATCH``, a whitelisted strict failure) is never relaunched, any other LEAVE
    ``TRAINING_FAILED`` is an infrastructure failure, else its LEAVE code
    (``yeto.rl.engine.leave_reason.leave_reason_for``) decides."""
    from ..config_errors import config_error_in  # noqa: PLC0415

    if error is not None and config_error_in(error) is not None:
        return decision(CONFIG_ERROR, detail or f"{type(error).__name__}: {error}")
    if not deterministic:  # launch.deterministic_stop off: the LEAVE code decides, as before
        from yeto.rl.engine.leave_reason import leave_reason_for  # noqa: PLC0415 - no torch

        return classify(leave_reason=leave_reason_for(error),
                        detail=detail or ("" if error is None else f"{type(error).__name__}: {error}"))
    stop = island_stop_decision(error)
    if stop is not None:
        return decision(stop.cause, detail or stop.detail)
    from yeto.rl.engine.leave_reason import leave_reason_for  # noqa: PLC0415 - no torch

    reason = leave_reason_for(error)
    text = detail or ("" if error is None else f"{type(error).__name__}: {error}")
    if _leave_name(reason) == "TRAINING_FAILED":
        # not a whitelisted strict failure (DETERMINISTIC_ERRORS): it may be an NCCL, CUDA,
        # storage or network fault underneath, so it keeps its relaunches
        return decision(INFRA, text)
    return classify(leave_reason=reason, detail=text)


def _chain(error: BaseException | None) -> list[BaseException]:
    """``error`` and its cause / context chain, and a Ray task error's ``cause``."""
    seen, todo, out = set(), [error], []
    while todo:
        error = todo.pop()
        if error is None or id(error) in seen:
            continue
        seen.add(id(error))
        out.append(error)
        inner = getattr(error, "cause", None)
        todo += [e for e in (error.__cause__, error.__context__, inner) if isinstance(e, BaseException)]
    return out


def deterministic_error_in(error: BaseException | None) -> BaseException | None:
    """The whitelisted deterministic strict failure (:data:`DETERMINISTIC_ERRORS`) in
    ``error``'s chain; None without one. Classes are looked up in ``sys.modules`` only."""
    classes = tuple(c for c in (getattr(sys.modules.get(m), n, None) for m, n in DETERMINISTIC_ERRORS)
                    if isinstance(c, type))
    return next((e for e in _chain(error) if classes and isinstance(e, classes)), None)


def island_stop_decision(error: BaseException | None) -> Decision | None:
    """The deterministic stop ``error`` is, else None (the error keeps its retries and
    relaunches): a trainer breaker's stop (:func:`breaker_error_in`), a LEAVE
    ``CONFIG_MISMATCH`` (an identity, profile or layout the run does not accept), or a
    whitelisted strict failure (:data:`DETERMINISTIC_ERRORS`). A
    :class:`~yeto.config_errors.ConfigError` is None here: it has its own exit."""
    from ..config_errors import config_error_in  # noqa: PLC0415

    if error is None or config_error_in(error) is not None:
        return None
    stop = breaker_error_in(error)
    if stop is not None:
        name, marked = stop
        return decision(BREAKER, f"{name}: {type(marked).__name__}: {marked}")
    from yeto.rl.engine.leave_reason import leave_reason_for  # noqa: PLC0415 - no torch

    if _leave_name(leave_reason_for(error)) == "CONFIG_MISMATCH":
        return decision(CONFIG_ERROR, f"{type(error).__name__}: {error}")
    strict = deterministic_error_in(error)
    if strict is not None:
        name = _strict_breaker(getattr(strict, "metric", None))
        return decision(STRICT, f"{name + ': ' if name else ''}{type(strict).__name__}: {strict}")
    return None


def _strict_breaker(metric: Any) -> str | None:
    """The trainer breaker a ``StrictRlInvariantError`` of ``metric`` trips
    (``yeto.rl.engine.trainer_breakers.strict_breaker``); None without a metric."""
    if not metric:
        return None
    try:
        from yeto.rl.engine.trainer_breakers import strict_breaker  # noqa: PLC0415 - import-light
    except Exception:  # noqa: BLE001 - the stop is read without its breaker's name
        return None
    return strict_breaker(str(metric))


def new_machine_breaker(stop: Decision | None) -> str | None:
    """The breaker of deterministic stop ``stop`` when a new machine may cure it
    (:data:`NEW_MACHINE_BREAKERS`), else None. ``stop.detail`` names the breaker first
    (``trainer.nan: ...``), as :func:`island_stop_decision`, :func:`classify_records` and an
    ``[yeto-island-stop]`` line (``<marker> strict trainer.nan: ...``) write it."""
    if stop is None or stop.cause not in (BREAKER, STRICT):
        return None
    text = str(stop.detail or "")
    if ISLAND_STOP_MARKER in text:
        words = text.split(ISLAND_STOP_MARKER, 1)[1].split(None, 1)
        text = words[1] if len(words) > 1 and words[0] in (BREAKER, STRICT, CONFIG_ERROR) else " ".join(words)
    head = text.split(":", 1)[0].strip()
    return head if head in NEW_MACHINE_BREAKERS else None


def island_stop_line(error: BaseException) -> str | None:
    """The ``[yeto-island-stop]`` line of an island process that ends with ``error`` when the
    error is a deterministic stop (:func:`island_stop_decision`); None for any other error."""
    found = island_stop_decision(error)
    if found is None:
        return None
    return f"{ISLAND_STOP_MARKER} {found.cause} {found.detail} (exit {ISLAND_STOP_EXIT}, not retried)"


def _error_type_name(text: Any) -> str:
    """The class name an ``island_leave`` record's ``error_type`` names (a Ray task error is
    written ``RayTaskError(Name)``)."""
    name = str(text or "")
    if name.endswith(")") and "(" in name:
        name = name[name.rindex("(") + 1:-1]
    return name.rsplit(".", 1)[-1]


def classify_records(records: Mapping[str, Any] | None) -> Decision | None:
    """The decision the island's own tape records give for its ended incarnation (None: they
    give none, the caller's reading stands). ``records``: the last trainer ``rl_breaker`` stop
    (``breaker``), ``rl_strict_failure`` (``strict``) and ``island_leave`` (``leave``) records
    of the incarnation (``yeto.rl.event_echo.TapeCollector.island_stop``). The stop of a breaker
    of :data:`BREAKER_STOPS` is :data:`BREAKER`; a strict failure (``rl_strict_failure``) or a LEAVE
    ``TRAINING_FAILED`` of a whitelisted error (:data:`DETERMINISTIC_ERRORS`) is :data:`STRICT`;
    a LEAVE ``CONFIG_MISMATCH`` is a configuration error. Any other LEAVE (an infrastructure
    failure, a ``TRAINING_FAILED`` that may be an NCCL / CUDA / storage fault) gives none."""
    if not records:
        return None
    breaker = records.get("breaker")
    if isinstance(breaker, Mapping) and str(breaker.get("name") or "") in BREAKER_STOPS:
        name = str(breaker.get("name"))
        text = f"{name}: {breaker.get('detail') or breaker.get('stop_reason') or 'stop'}"
        return decision(BREAKER, text)
    leave = records.get("leave") if isinstance(records.get("leave"), Mapping) else {}
    error = f"{leave.get('error_type') or 'error'}: {leave.get('error') or ''}".strip()
    strict = records.get("strict")
    if isinstance(strict, Mapping):
        named = _strict_breaker(strict.get("metric")) or leave.get("breaker")
        text = str(strict.get("error") or strict.get("detail") or error)
        return decision(STRICT, f"{named}: {text}" if named else text)
    name = str(leave.get("reason_name") or "").upper()
    if name == "TRAINING_FAILED" and _error_type_name(leave.get("error_type")) in DETERMINISTIC_ERROR_NAMES:
        return decision(STRICT, f"{leave['breaker']}: {error}" if leave.get("breaker") else error)
    if name == "CONFIG_MISMATCH":
        return decision(CONFIG_ERROR, error)
    if leave.get("breaker") in BREAKER_STOPS:  # the driver's LEAVE record names the breaker whose stop ended it
        return decision(BREAKER, f"{leave['breaker']}: {error}")
    return None


# -- the state record in the store ---------------------------------------------------------------
STATE_DIR = "fleet/trainers"
HISTORY = 20  # transitions kept in a record
SYNCER_KEY = "fleet/syncer.json"


def state_key(island: int) -> str:
    return f"{STATE_DIR}/{int(island)}/state.json"


def stop_key(island: int) -> str:
    return f"{STATE_DIR}/{int(island)}/STOP"


def start_key(island: int) -> str:
    return f"{STATE_DIR}/{int(island)}/START"


def controller_identity() -> dict[str, Any]:
    """Who supervises: host, PID and a fencing token unique to this process."""
    return {"host": socket.gethostname(), "pid": os.getpid(), "token": uuid.uuid4().hex[:16],
            "since": round(time.time(), 3)}


STOP_POLL_S = 15.0  # a background store re-reads the watched islands' STOP this often
FLUSH_S = 10.0  # a background store's queued writes get this long at the launcher's end
RETRY_MAX_S = 60.0  # the longest back-off between two tries of a write the store did not take


class Fence(Protocol):
    """A single-writer claim on the island records: the interface of the run store's lock
    (``yeto.launch.pool.store_lock.StoreLock``). ``token``: its fencing token, written into
    each record. ``guard()``: before a write; raises an error with a ``by`` attribute (the
    higher token, ``LockLost``) once another controller took over, and another error
    (``LockUnconfirmed``) when the store did not answer; it renews the lease when due."""

    token: int

    def guard(self) -> None: ...


class IslandRecordsFenced(RuntimeError):
    """This controller lost the island records' single-writer lock (another launcher took
    them over): it orders, relaunches and tears down nothing of the fleet any more."""


class IslandRecordsUnconfirmed(RuntimeError):
    """The store did not confirm the island records' lock before an order (and the lease is
    nearly over): nothing is ordered now. ``lease_over``: this holder's own lease has run out
    too, so another launcher may hold the records meanwhile: nothing is torn down either.
    Never sticky: the next check confirms the lock again (only a higher token is lost)."""

    def __init__(self, message: str, *, lease_over: bool = False) -> None:
        super().__init__(message)
        self.lease_over = bool(lease_over)


class IslandRecordsUnreadable(RuntimeError):
    """An island record (or the syncer's) could not be read: whether its machine still runs
    is unknown, so nothing is ordered for it (an order could start it twice)."""


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
        self._starts: dict[int, dict[str, Any] | None] = {}
        self._stops_due = True
        self._wake = threading.Event()
        self._idle = threading.Event()
        self._idle.set()
        self._thread: threading.Thread | None = None
        self._closed = False
        self._ended = False  # closed: nothing more is written or guarded (the lock may be released)

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

    def _read_unlocked(self, island: int, *, strict: bool = False) -> dict[str, Any] | None:
        try:
            raw = self.objects.read(state_key(island))
        except Exception as error:  # noqa: BLE001
            if strict:
                raise IslandRecordsUnreadable(f"island {island}: its record did not read "
                                              f"({type(error).__name__}: {error})"[:400]) from error
            self._warn("read", error)
            return None
        if raw is None:
            return None
        try:
            record = json.loads(raw)
        except ValueError as error:
            if strict:
                raise IslandRecordsUnreadable(f"island {island}: its record is not JSON ({error})"[:400]) from error
            self._warn("parse", error)
            return None
        if strict and not isinstance(record, dict):
            raise IslandRecordsUnreadable(f"island {island}: its record is not an object")
        return record if isinstance(record, dict) else None

    def cached(self, island: int) -> dict[str, Any] | None:
        """Island ``island``'s record as this store last read or wrote it (no store call)."""
        with self._lock:
            record = self._records.get(int(island))
        return None if record is None else dict(record)

    def read_all(self, islands: Iterable[int] | None = None, *, strict: bool = False) -> dict[int, dict[str, Any]]:
        """Every island's record: those of ``islands``, else every one the store lists.
        ``strict``: a record that does not read raises :class:`IslandRecordsUnreadable`
        (a missing one is no record)."""
        if islands is None:
            islands = self.listed()
        out = {}
        for k in islands:
            record = self._read_unlocked(int(k), strict=strict)
            if record is not None:
                with self._lock:
                    self._records.setdefault(int(k), dict(record))  # what a first write merges into
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
        if self._ended:
            return False
        if self.fence is None:
            return True
        if self.fenced is not None:
            return False
        try:
            self.fence.guard()
        except (KeyboardInterrupt, SystemExit):
            if threading.current_thread() is threading.main_thread():
                raise
            self._warn("fence check", RuntimeError("interrupted"))  # never ends the background thread
            return False
        except BaseException as error:  # noqa: BLE001 - the lock's errors are BaseExceptions
            if hasattr(error, "by"):
                self.fenced = error
                self.log(f"[launcher] island state store: fenced ({error}); this controller writes no "
                         "island record any more"[:400])
            else:
                self._warn("fence check", error)  # not confirmed: this write waits for the next one
            return False
        return True

    def _store(self, key: str, data: bytes) -> bool:
        """Write one body; False when it should be tried again (the fence not confirmed, the
        store did not take it); True once written, or for good not (the fence is lost)."""
        if not self._fence_holds():
            return self.fenced is not None or self._ended
        try:
            self.objects.write(key, data)
        except Exception as error:  # noqa: BLE001
            self._warn("write", error)
            return False
        return True

    def keep_lease(self) -> None:
        """Start the background thread now (not at the first write): it guards the fence every
        ``stop_poll_s``, which renews the lock's lease while nothing is written yet (a syncer
        launch, a Modal deploy)."""
        if self.background:
            with self._lock:
                self._start()

    def guard_order(self, what: str) -> None:
        """Right before an order of the fleet (an island, its relaunch, the syncer): the fence
        still holds, checked now. :class:`IslandRecordsFenced` once another launcher holds the
        records (sticky), :class:`IslandRecordsUnconfirmed` when the store does not confirm it
        and the lease is nearly over. Nothing to check without a fence."""
        if self.fence is None:
            return
        if not self._ended:
            self.keep_lease()  # a background thread that died is started again (the lease renewal)
        if self.fenced is None and not self._ended:
            try:
                self.fence.guard()
                return
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as error:  # noqa: BLE001 - the lock's errors are BaseExceptions
                if not hasattr(error, "by"):
                    left = getattr(self.fence, "lease_left", None)
                    over = callable(left) and left() <= 0  # another launcher may hold it meanwhile
                    raise IslandRecordsUnconfirmed(
                        f"{what}: the island records' lock is not confirmed ({error})"
                        + ("; this launcher's lease ran out: nothing ordered or torn down now" if over
                           else "; nothing ordered now"), lease_over=over) from None
                self.fenced = error
                self.log(f"[launcher] island state store: fenced ({error}); this controller writes no "
                         "island record any more"[:400])
        raise IslandRecordsFenced(f"{what}: this launcher no longer holds the island records "
                                  f"({self.fenced or 'store closed'}); the launcher that holds them "
                                  "orders and tears down"[:400])

    def _start(self) -> None:  # under self._lock
        if (self._thread is None or not self._thread.is_alive()) and not self._closed:
            if self._thread is not None:  # it died: the lease renewal and the queued writes need it
                self.log("[launcher] island state store: the background thread had ended; started again")
            self._thread = threading.Thread(target=self._work, name="island-state-store", daemon=True)
            self._thread.start()

    def _work(self) -> None:
        while True:
            try:
                self._work_loop()
                return
            except Exception as error:  # noqa: BLE001 - one bad pass never ends the lease renewal
                self._warn("background pass", error)
                time.sleep(min(1.0, self.stop_poll_s))

    def _work_loop(self) -> None:
        stops_at = float("-inf")
        retry_at, backoff = float("-inf"), self.stop_poll_s
        while True:
            self._wake.wait(self.stop_poll_s)
            self._wake.clear()
            with self._lock:
                pending, self._pending = self._pending, {}
                watched = sorted(self._watched)
                due, self._stops_due = self._stops_due, False
                closing = self._closed
            if pending and time.monotonic() < retry_at and not closing:
                retry = pending  # backing off: nothing is tried before retry_at
                self._fence_holds()  # ... but the lease is renewed meanwhile
            else:
                retry = {key: data for key, data in pending.items() if not self._store(key, data)}
                if retry:  # the fence not confirmed, or the store did not take it: back off, try again
                    retry_at, backoff = time.monotonic() + backoff, min(backoff * 2, RETRY_MAX_S)
                else:
                    backoff = self.stop_poll_s
            if retry:
                with self._lock:  # unless a newer body of the key is queued meanwhile
                    for key, data in retry.items():
                        self._pending.setdefault(key, data)
            if not pending:
                self._fence_holds()  # the lease renewal of a quiet fleet
            if watched and (due or time.monotonic() - stops_at >= self.stop_poll_s):
                for k in watched:
                    self._stops[k] = self._read_stop(k)
                    self._starts[k] = self._read_request(start_key(k))
                stops_at = time.monotonic()
            with self._lock:
                if not self._pending:
                    self._idle.set()
                if closing and (not self._pending or retry):
                    return  # closing: a body the store does not take now is left behind (close() says which)

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
        with self._lock:
            left = {key: data for key, data in self._pending.items()}
            self._ended = True  # a write still in flight lands or not; nothing after it
        if not left and self._idle.is_set():
            return True
        what = []
        for key, data in sorted(left.items()):
            try:
                state = json.loads(data).get("state")
            except (ValueError, AttributeError):
                state = None
            what.append(f"{key}" + (f" ({state})" if state else ""))
        self.log(f"[launcher] ERROR: island state store: {len(left) or 'a'} record write(s) did not land "
                 f"within {timeout:.0f} s at the end ({', '.join(what) or 'in flight'}"
                 f"{'; fenced: ' + str(self.fenced) if self.fenced is not None else ''}): the run store shows "
                 "an older state of these islands")
        return False

    # -- the syncer's record (fleet/syncer.json: its cluster, job and address)
    def write_syncer(self, **fields: Any) -> None:
        body = {**fields, "updated": round(self.clock(), 3), "controller": self.controller}
        if self.fence is not None:
            body["fence_token"] = int(self.fence.token)
        self._put(SYNCER_KEY, json.dumps(body, sort_keys=True, default=str).encode())

    def read_syncer(self, *, strict: bool = False) -> dict[str, Any] | None:
        """The syncer's record; ``strict``: one that does not read raises
        :class:`IslandRecordsUnreadable` (a missing one is None)."""
        try:
            raw = self.objects.read(SYNCER_KEY)
            record = None if raw is None else json.loads(raw)
        except Exception as error:  # noqa: BLE001
            if strict:
                raise IslandRecordsUnreadable(f"the syncer's record did not read "
                                              f"({type(error).__name__}: {error})"[:400]) from error
            self._warn("syncer read", error)
            return None
        return record if isinstance(record, dict) else None

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

    def start_requested(self, island: int) -> dict[str, Any] | None:
        """Island ``island``'s START request (``yeto island start``), else None; a background
        store answers from the cache it keeps with the STOP's."""
        if not self.background:
            return self._read_request(start_key(island))
        self.stop_requested(island)  # watches the island
        return self._starts.get(int(island))

    def request_start(self, island: int, reason: str, *, by: str = "operator") -> bool:
        """Write island ``island``'s START request (True once written)."""
        body = {"reason": str(reason), "by": str(by), "at": round(self.clock(), 3)}
        try:
            self.objects.write(start_key(island), json.dumps(body, sort_keys=True).encode())
            return True
        except Exception as error:  # noqa: BLE001
            self._warn("start write", error)
            return False

    def _read_stop(self, island: int) -> dict[str, Any] | None:
        return self._read_request(stop_key(island))

    def _read_request(self, key: str) -> dict[str, Any] | None:
        try:
            raw = self.objects.read(key)
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
        return self._delete(stop_key(island), "stop clear")

    def clear_start(self, island: int) -> bool:
        """Remove island ``island``'s START request (True when none is left)."""
        return self._delete(start_key(island), "start clear")

    def _delete(self, key: str, what: str) -> bool:
        try:
            delete = getattr(self.objects, "delete", None)
            if callable(delete):
                delete(key)
            elif self.objects.read(key) is not None:
                raise RuntimeError("this store cannot delete objects")
            return True
        except FileNotFoundError:
            return True
        except Exception as error:  # noqa: BLE001
            self._warn(what, error)
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

    def start_requested(self, name: str) -> dict[str, Any] | None:
        ref = self.islands.get(name)
        if ref is None or ref.role == "sampler":
            return None
        return self.store.start_requested(ref.island)


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


# -- the trainer scheduler s1 (``yeto.rl.engine.trainer_loop``) ------------------------------------
# s1 runs one trainer island in incarnations it starts and stops itself; its journal
# (``<run dir>/trainer-loop/journal.jsonl``, kind ``trainer_loop``) stays the source of its
# own restart. With the island lifecycle on, each of its records also moves the island's
# state record (:class:`LoopRecorder`), and its failures go through the decision table
# (:func:`loop_classifier`). Names of ``yeto.rl.engine.trainer_loop``: no import (stdlib only).
LOOP_KIND = "trainer_loop"


def loop_classifier(reason: Any, error: BaseException | None, *,
                    deterministic: bool = False) -> tuple[str, bool]:
    """``TrainerLoop(classify_stop=...)``: a failed s1 incarnation's (cause, relaunch).
    ``deterministic``: ``launch.deterministic_stop`` is on (:func:`classify_error`)."""
    d = classify_error(error, deterministic=deterministic) if error is not None else classify(leave_reason=reason)
    if d.cause in (FINISHED, STOPPED):  # a failure (error or INFRA_FAILED) the table read as an end
        d = decision(INFRA, d.detail)
    return d.cause, d.relaunch


def loop_event(record: Mapping[str, Any]) -> tuple[str, Decision | None] | None:
    """The island state (and decision) one s1 journal record means; None for a record
    that moves no state (scheduler state, ``first_policy``, other kinds).

    ``started`` -> ordering; ``ready`` -> running; ``stopped``: by its ``cause`` (the
    decision table, when the loop had it), else an error or ``INFRA_FAILED`` -> an
    infrastructure failure (recovering), ``FINISHED`` -> done, any other stop (an idle
    stop of the scheduler, a STOP) -> done, never a relaunch by the table: s1 starts the
    next incarnation itself (done -> ordering); ``orphaned`` (the head died with it up)
    -> recovering (lost); ``gave_up`` -> abandoned; ``finished`` -> done."""
    if record.get("kind", LOOP_KIND) != LOOP_KIND:
        return None
    event = record.get("event")
    incarnation = record.get("incarnation")
    if event == "started":
        return ORDERING, None
    if event == "ready":
        return RUNNING, None
    if event == "orphaned":
        return RECOVERING, decision(LOST, f"incarnation {incarnation}: the head that ran it died")
    if event == "gave_up":
        return ABANDONED, decision(INFRA, f"{record.get('failures')} incarnations failed in a row "
                                          f"(limit {record.get('limit')})")
    if event == "finished":
        return DONE, decision(FINISHED, "the data ended")
    if event != "stopped":
        return None
    reason, error, cause = record.get("reason"), record.get("error"), record.get("cause")
    detail = str(error or reason or "")
    if cause in DECISIONS:
        d = decision(str(cause), detail)
    elif error is not None and "ConfigError(" in str(error):  # a journal of a loop without the table
        d = decision(CONFIG_ERROR, detail)
    elif error is not None or reason == "INFRA_FAILED":
        d = classify(log_text=str(error or ""), detail=detail)
        if not d.relaunch and d.action != FAIL:
            d = decision(INFRA, detail)
    elif reason == "FINISHED":
        d = decision(FINISHED, detail)
    elif reason == "PREEMPTED":
        d = decision(PREEMPTED, detail)
    else:
        d = decision(STOPPED, detail or "scheduled stop")
    return d.state, d


class LoopRecorder:
    """An s1 journal writer that also keeps the island's state record: wraps
    ``journal.append`` (``TrainerLoop(record=...)``); the journal is written first and
    stays the loop's only input. ``island``: the record's ``k``; ``source`` / ``name``:
    the held trainer island's. ``store``: the launcher's own (fenced, under the island
    records' lock), so a launcher that lost the lock writes nothing.

    The island's record is read (strictly) before the first write, so a write merges into
    it: until a read answers (the record, or none), nothing is written (the journal goes on;
    the record catches up at the next journal record). An s1 idle stop is ``done`` with
    ``idle: true`` (the scheduler starts the island again when samples wait)."""

    def __init__(self, append: Callable[..., Any], store: IslandStateStore, island: int, *,
                 source: str, name: str | None = None) -> None:
        self._append = append
        self.store = store
        self.island = int(island)
        self.source = str(source)
        self.name = name
        self._incarnations: set[int] = set()
        self._last: Decision | None = None
        self._known = False  # the island's record read once (it, or none): writes merge into it
        self._pending: list[Mapping[str, Any]] | None = None  # an older journal to import once known
        self.imported: dict[str, Any] | None = None

    def _ready(self) -> bool:
        """The island's record read strictly once (True: writes may go on). The first read
        that answers imports a pending older journal when the store holds no record; one that
        fails leaves both for the next journal record (:class:`IslandRecordsUnreadable` is
        not "no record": an import then would write over the record)."""
        if self._known:
            return True
        try:
            existing = self.store.read_all([self.island], strict=True).get(self.island)
        except IslandRecordsUnreadable as error:
            self.store._warn("s1 record read", error)  # noqa: SLF001
            return False
        self._known = True
        pending, self._pending = self._pending, None
        if existing is None and pending:
            self.imported = self._import(pending)
        return True

    def __call__(self, kind: str, **fields: Any) -> Any:
        out = self._append(kind, **fields)
        try:
            self.observe({"kind": kind, **fields})
        except Exception as error:  # noqa: BLE001 - the record is a view, never the reason the loop stops
            self.store._warn("s1 record", error)
        return out

    def observe(self, record: Mapping[str, Any], **extra: Any) -> dict[str, Any] | None:
        moved = loop_event(record)
        if moved is None or not self._ready():
            return None
        state, d = moved
        incarnation = record.get("incarnation")
        if incarnation is not None:
            self._incarnations.add(int(incarnation))
        if record.get("event") == "gave_up" and self._last is not None and self._last.action == FAIL:
            state, d = FAILED, self._last  # the loop ended on a failure the table never relaunches
        if d is not None:
            self._last = d
        fields: dict[str, Any] = {"source": self.source, "role": "trainer", "mode": "s1",
                                  "relaunches": max(0, len(self._incarnations) - 1), **extra}
        if self.name:
            fields["name"] = self.name
        if incarnation is not None:
            fields["incarnation"] = int(incarnation)
            fields["handle"] = {"incarnation": int(incarnation), **({"cluster": self.name} if self.name else {})}
        if record.get("resume_cut") is not None or record.get("resume_version") is not None:
            fields["resume"] = {"cut": record.get("resume_cut"), "version": record.get("resume_version")}
        if record.get("event") == "stopped":
            fields["exit"] = str(record.get("error") or record.get("reason") or "stopped")[:500]
        # an idle stop of the scheduler: done for now, started again when samples wait
        fields["idle"] = bool(state == DONE and d is not None and d.cause == STOPPED)
        fields["decision"] = None if d is None else d.to_json()
        fields["reason"] = None if d is None else d.detail
        return self.store.write(self.island, state, **fields)

    def import_journal(self, records: Iterable[Mapping[str, Any]]) -> dict[str, Any] | None:
        """Write the island's record from an s1 journal written before the lifecycle
        was on: once (only when the store holds no record of the island), read only
        (the journal is not touched). Returns the record written, else None. A store read
        that fails is not "no record": the import waits for the first read that answers
        (:meth:`_ready`, tried again at each journal record)."""
        if self._known:
            return None  # the record was read already: imported then, or the store has one
        self._pending = list(records)
        self._ready()
        return self.imported

    def _import(self, records: Iterable[Mapping[str, Any]]) -> dict[str, Any] | None:
        last, resume = None, None
        for record in records:
            moved = loop_event(record)
            if moved is None:
                continue
            last = record
            if record.get("event") == "started":
                resume = {"cut": record.get("resume_cut"), "version": record.get("resume_version")}
            if record.get("incarnation") is not None:
                self._incarnations.add(int(record["incarnation"]))
            if moved[1] is not None:
                self._last = moved[1]
        if last is None:
            return None
        extra: dict[str, Any] = {"imported_from_journal": True}
        if resume is not None and (resume["cut"] is not None or resume["version"] is not None):
            extra["resume"] = resume
        return self.observe(last, **extra)


def loop_enabled(args: Any) -> bool:
    """Whether s1 keeps the island's state record: ``--island-lifecycle on`` only (s1
    runs a single trainer island, which takes no new path by default)."""
    return getattr(args, "island_lifecycle", "auto") == "on" and not getattr(args, "dry_run", False)
