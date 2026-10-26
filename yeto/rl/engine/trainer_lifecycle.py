"""Trainer start/stop: the lifecycle, the durable sample buffer, the scheduler.

rl-trainer-start-stop design D1/D2. Mechanism and policy are separate:

* :class:`TrainerLifecycle` does what it is told (start, run, drain and cut, stop);
* :class:`TrainerScheduler` decides ``Start`` / ``Stop`` / ``Hold`` from backlog
  statistics, the lifecycle state and prices (a pure function of its inputs);
* :class:`DurableSampleBuffer` holds the samples across trainer stops.

A trainer that starts once is the special case: a scheduler that holds until the
data ends, then stops with ``FINISHED`` (:class:`AlwaysOnScheduler`). Scheduler S1
(start on backlog, ski-rental stop) is in :mod:`.trainer_scheduler`.

:class:`DriverLifecycle` is the lifecycle over the existing :class:`IslandDriver`
(D2). Each start builds a fresh driver (new trainer ports, new sync session) for a
new incarnation, so ``IslandDriver.rebuild_trainer`` (a same-driver rebuild) is
never needed across starts. The driver runs on its own thread; the lifecycle
wraps its sync session to observe the states and to hold the driver after a
publication until it is told to train on or to drain:

* JOIN as a new incarnation before restoring (``Membership.join``);
* restore the last cut through the sync session's own resume path (round cut ->
  ``CuttableTrainer.restore_cut``);
* the first publication after the restore is the warm-up; the trainer is then
  ``TRAINING`` and waits for :meth:`DriverLifecycle.run`;
* a drain ends the driver where it waits: the final cut takes the
  trigger of the stop reason (``STOPPED`` -> ``SCHEDULED``) and LEAVE carries the
  P5 reason code (``Membership.leave``).

This module is neutral core: it knows ports and the LEAVE code, no backend.
"""

from __future__ import annotations

import enum
import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from yeto.rl.elastic_client import LeaveReason

from . import recovery
from .leave_reason import leave_reason_for

if TYPE_CHECKING:
    from .driver import IslandDriver


class LifecycleError(RuntimeError):
    """A lifecycle verb was called in a state that does not allow it, or a start
    did not continue from the cut it was given."""


class LifecycleState(str, enum.Enum):
    COLD = "cold"
    ACQUIRING = "acquiring"
    JOINING = "joining"
    RESTORING = "restoring"
    WARMING = "warming"
    TRAINING = "training"
    DRAINING = "draining"
    STOPPED = "stopped"


_ORDER = tuple(LifecycleState)


def allowed_transition(old: LifecycleState, new: LifecycleState) -> bool:
    """The states move only forward one step, and from any state to STOPPED."""
    if new is LifecycleState.STOPPED:
        return old is not LifecycleState.STOPPED
    return _ORDER.index(new) == _ORDER.index(old) + 1


# Trigger of the final cut for each stop reason (D3): a scheduler stop is SCHEDULED,
# a spot reclaim PREEMPT, the end of the data (or a user stop) MANUAL.
CUT_TRIGGER_FOR_REASON: Mapping[LeaveReason, str] = {
    LeaveReason.STOPPED: recovery.SCHEDULED,
    LeaveReason.PREEMPTED: recovery.PREEMPT,
    LeaveReason.FINISHED: recovery.MANUAL,
}


# ---------------------------------------------------------------------------
# Neutral records
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CutRef:
    """A committed cut a later start can restore. ``policy_version`` is the
    published version at the cut (the next start publishes it again first);
    ``watermark`` is the consumed buffer watermark in the same record (D4)."""

    cut_id: str
    trigger: str
    policy_version: int
    watermark: int | None = None

    def __post_init__(self) -> None:
        if self.trigger not in recovery.TRIGGERS:
            raise ValueError(f"cut trigger {self.trigger!r} is not one of {recovery.TRIGGERS}")


@dataclass(frozen=True)
class StartPlan:
    incarnation: int
    spec: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StopCondition:
    """``rounds``: train this many more rounds, then wait at the round boundary for
    the next verb. None: train until the data ends."""

    rounds: int | None = None

    def __post_init__(self) -> None:
        if self.rounds is not None and self.rounds < 1:
            raise ValueError("StopCondition.rounds must be positive")


@dataclass(frozen=True)
class RunReceipt:
    rounds: int  # rounds trained by this run call
    first_version: int  # trained version before the first of these rounds
    trained_version: int  # trained version after the last of these rounds
    data_ended: bool  # the data ended: the driver drained on its own (FINISHED)


@dataclass(frozen=True)
class SampleIndexEntry:
    """One sample in the durable buffer: its id, the policy version that produced it
    (behavior version, P0-S schema 4) and where its payload is. One sample per
    entry; the cross-island group index is ``sample_pool.SampleIndexEntry``.

    ``provenance``: the source of the sample (``sampler_engine.Provenance``
    record: device family, card, engine and version, precision, quantization
    scheme, topology); None for entries written before it existed or by a
    sampler that does not set it (read as ``unknown``). Data only: it never
    enters a hash or the version gate (rl-hetero-sampler-engines D3).

    ``sampler_round``: which sampler round wrote the entry
    (``sampler_resume.SamplerRound`` record: sampler, incarnation, rollout id,
    last group of the round, data cursor after the round); None for entries
    written before it existed. A replacement sampler island reads it to resume
    after the last complete round (S25 C1)."""

    sample_id: str
    behavior_version: int
    payload_ref: str
    tokens: int = 0
    appended_at: float = 0.0
    provenance: Mapping[str, str] | None = field(default=None, compare=False, hash=False)
    sampler_round: Mapping[str, Any] | None = field(default=None, compare=False, hash=False)


@dataclass(frozen=True)
class Selected:
    """A selected sample and its buffer position (the cursor to commit)."""

    position: int
    entry: SampleIndexEntry


@dataclass(frozen=True)
class BacklogStats:
    """``batches``: unconsumed samples in units of one training batch;
    ``version_histogram``: behavior version -> unconsumed samples;
    ``arrival_rate``: samples per second over the recent window;
    ``data_ended``: the samplers will append nothing more;
    ``max_version_gap``: current trainer version minus ``oldest_version``."""

    batches: float
    samples: int
    oldest_version: int | None = None
    version_histogram: Mapping[int, int] = field(default_factory=dict)
    arrival_rate: float = 0.0
    data_ended: bool = False
    max_version_gap: int | None = None


# ---------------------------------------------------------------------------
# Scheduler decisions
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Start:
    pass


@dataclass(frozen=True)
class Hold:
    pass


@dataclass(frozen=True)
class Stop:
    reason: LeaveReason = LeaveReason.STOPPED


Decision = Start | Hold | Stop


# ---------------------------------------------------------------------------
# Ports
# ---------------------------------------------------------------------------
@runtime_checkable
class DurableSampleBuffer(Protocol):
    def append(self, entry: SampleIndexEntry) -> None: ...
    def backlog(self, current_version: int) -> BacklogStats: ...
    def select(self, consumer: str, limit: int) -> list[Selected]: ...
    def commit_consumed(self, cursor: int) -> None: ...  # written into the cut
    def on_arrival(self, callback: Callable[[], None]) -> None: ...


@runtime_checkable
class TrainerScheduler(Protocol):
    def decide(self, backlog: BacklogStats, lifecycle_state: LifecycleState,
               prices: Mapping[str, float] | None = None) -> Decision: ...


@runtime_checkable
class Membership(Protocol):
    """JOIN / LEAVE toward the coordinator (D2: the elastic client's ``Join`` with a
    new incarnation per start, ``Leave`` with the P5 reason code). ``join`` returns
    the coordinator's base version, or None when there is no coordinator."""

    def join(self, incarnation: int) -> int | None: ...
    def leave(self, incarnation: int, reason: LeaveReason) -> None: ...


@runtime_checkable
class CutLog(Protocol):
    """Where the lifecycle reads the newest committed cut back."""

    def latest(self) -> CutRef | None: ...


@runtime_checkable
class TrainerLifecycle(Protocol):
    def plan_start(self, spec: Mapping[str, Any]) -> StartPlan: ...
    def start(self, plan: StartPlan, *, resume_from: CutRef | None) -> TrainerHandle: ...
    def ready(self, handle: TrainerHandle, deadline: float) -> bool: ...
    def run(self, handle: TrainerHandle, source: DurableSampleBuffer | None,
            until: StopCondition) -> RunReceipt: ...
    def drain_and_cut(self, handle: TrainerHandle, reason: LeaveReason) -> CutRef | None: ...
    def stop(self, handle: TrainerHandle, reason: LeaveReason) -> None: ...


class AlwaysOnScheduler:
    """"Start once" (D1): start while the trainer is cold, hold while it trains,
    stop with ``FINISHED`` when the data ended and the backlog is empty."""

    def decide(self, backlog: BacklogStats, lifecycle_state: LifecycleState,
               prices: Mapping[str, float] | None = None) -> Decision:
        del prices  # S1 and this scheduler ignore prices
        if lifecycle_state is LifecycleState.COLD:
            return Start()
        if backlog.data_ended and backlog.samples == 0:
            return Stop(LeaveReason.FINISHED)
        return Hold()


# ---------------------------------------------------------------------------
# Handle
# ---------------------------------------------------------------------------
class TrainerHandle:
    """One incarnation of the trainer. Thread-safe: the driver thread moves the
    state; the caller's thread gives commands and waits on ``_cond``."""

    def __init__(self, plan: StartPlan, *, resume_from: CutRef | None,
                 clock: Callable[[], float]) -> None:
        self.plan = plan
        self.incarnation = plan.incarnation
        self.resume_from = resume_from
        self.state = LifecycleState.COLD
        self.history: list[tuple[str, float]] = [(self.state.value, clock())]
        self.driver: IslandDriver | None = None
        self.source: DurableSampleBuffer | None = None
        self.start_version: int | None = None  # version restored (or 0) and published first
        self.published_version: int | None = None
        self.rounds = 0  # rounds trained by this incarnation
        self.data_ended = False
        self.joined = False
        self.requested_reason: LeaveReason | None = None
        self.leave_reason: LeaveReason | None = None
        self.cut: CutRef | None = None
        self.error: BaseException | None = None
        self._clock = clock
        self._cond = threading.Condition()
        self._target_rounds = 0
        self._drain_trigger: str | None = None
        self._parks = 0  # times the driver waited for a command
        self._parked = False
        self._done = False  # the driver thread ended
        self._thread: threading.Thread | None = None

    # -- state ---------------------------------------------------------------
    def advance(self, new: LifecycleState) -> None:
        with self._cond:
            if not allowed_transition(self.state, new):
                raise LifecycleError(f"incarnation {self.incarnation}: {self.state.value} -> {new.value}")
            self.state = new
            self.history.append((new.value, self._clock()))
            self._cond.notify_all()

    @property
    def states(self) -> list[str]:
        return [s for s, _ in self.history]

    @property
    def trained_version(self) -> int | None:
        return None if self.start_version is None else self.start_version + self.rounds

    # -- driver side ---------------------------------------------------------
    def park(self) -> str | None:
        """Wait at a round boundary for a command. Returns the cut trigger of a
        drain, or None to train on."""
        with self._cond:
            self._parked = True
            self._parks += 1
            self._cond.notify_all()
            self._cond.wait_for(lambda: self._drain_trigger is not None
                                or self.rounds < self._target_rounds)
            self._parked = False
            return self._drain_trigger

    def wants_park(self) -> bool:
        with self._cond:
            return self.rounds >= self._target_rounds or self._drain_trigger is not None


# ---------------------------------------------------------------------------
# Sync wrapper: the lifecycle's view into the driver
# ---------------------------------------------------------------------------
class _Drained(Exception):
    """Ends the driver after a drain: the cut is written, nothing more to train."""


class _LifecycleSync:
    """Wraps the driver's sync session; everything else is delegated.

    The driver waits for the next verb right after each publication (the restored
    policy, then every trained version), so the samplers already hold the newest
    version while the trainer waits (D5). A drain there finishes the inner session
    (final cut, with the trigger of the stop reason) and ends the driver."""

    def __init__(self, inner: Any, handle: TrainerHandle, membership: Membership | None) -> None:
        self.inner = inner
        self.handle = handle
        self.membership = membership
        self.closed_reason: LeaveReason | None = None
        self.drained = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def start(self, driver: IslandDriver) -> Any:
        h = self.handle
        h.advance(LifecycleState.JOINING)
        if self.membership is not None:
            self.membership.join(h.incarnation)
        h.joined = True
        h.advance(LifecycleState.RESTORING)
        start = self.inner.start(driver)
        if h.resume_from is not None and int(start.rollout_id) != h.resume_from.policy_version:
            raise LifecycleError(
                f"incarnation {h.incarnation} restored version {start.rollout_id}, "
                f"the cut {h.resume_from.cut_id} is at version {h.resume_from.policy_version}")
        h.start_version = int(start.rollout_id)
        h.advance(LifecycleState.WARMING)
        return start

    def published(self, driver: IslandDriver, *, rollout_id: int, policy_hash: str) -> None:
        self.inner.published(driver, rollout_id=rollout_id, policy_hash=policy_hash)
        h = self.handle
        h.published_version = int(rollout_id)
        if h.state is LifecycleState.WARMING:
            # the restored policy is on the samplers: warm-up done
            h.advance(LifecycleState.TRAINING)
        if h.state is not LifecycleState.TRAINING or not h.wants_park():
            return
        trigger = h.park()
        if trigger is None:
            return
        h.advance(LifecycleState.DRAINING)
        if hasattr(self.inner, "final_trigger"):
            self.inner.final_trigger = trigger
        self.inner.finish(driver)  # the final cut at the published version
        self.drained = True
        raise _Drained()

    def boundary(self, driver: IslandDriver, *, rollout_id: int, stats: Any) -> Any:
        out = self.inner.boundary(driver, rollout_id=rollout_id, stats=stats)
        h = self.handle
        commit = getattr(driver.rollout, "commit_round", None)
        if callable(commit):  # the round is trained: its samples are consumed (D4)
            commit(rollout_id)
        with h._cond:
            h.rounds += 1
        if out.stop:  # the data ended: drain on our own
            h.data_ended = True
            h.advance(LifecycleState.DRAINING)
        return out

    def close(self, reason: int = LeaveReason.FINISHED) -> None:
        """The driver's departure code (P5) unless the driver ended cleanly or by
        our own stop: then the reason the lifecycle was given."""
        code = LeaveReason(reason)
        if (code is LeaveReason.FINISHED or self.drained) and self.handle.requested_reason is not None:
            code = self.handle.requested_reason
        self.closed_reason = code
        self.inner.close(code)


# ---------------------------------------------------------------------------
# Lifecycle over IslandDriver
# ---------------------------------------------------------------------------
DriverFactory = Callable[[TrainerHandle], "IslandDriver"]


class DriverLifecycle:
    """:class:`TrainerLifecycle` over :class:`IslandDriver` (see the module doc).

    ``build_driver(handle)`` acquires the trainer for one incarnation and returns
    a fresh driver whose sync session restores ``handle.resume_from`` in
    ``start`` (e.g. ``LocalOnlySync`` with round cuts). ``release(handle)`` gives
    the cards back after LEAVE. ``cuts`` reads the cut a drain committed."""

    def __init__(self, build_driver: DriverFactory, *, membership: Membership | None = None,
                 cuts: CutLog | None = None, release: Callable[[TrainerHandle], None] | None = None,
                 first_incarnation: int = 0, clock: Callable[[], float] = time.monotonic) -> None:
        self.build_driver = build_driver
        self.membership = membership
        self.cuts = cuts
        self.release = release
        self.clock = clock
        self._next_incarnation = int(first_incarnation)
        self._lock = threading.Lock()

    def plan_start(self, spec: Mapping[str, Any]) -> StartPlan:
        with self._lock:
            incarnation = self._next_incarnation
            self._next_incarnation += 1
        return StartPlan(incarnation=incarnation, spec=dict(spec))

    def start(self, plan: StartPlan, *, resume_from: CutRef | None) -> TrainerHandle:
        handle = TrainerHandle(plan, resume_from=resume_from, clock=self.clock)
        handle.advance(LifecycleState.ACQUIRING)
        try:
            driver = self.build_driver(handle)
        except BaseException as error:
            handle.error = error
            self._end(handle, leave_reason_for(error))
            raise
        sync = _LifecycleSync(driver.sync, handle, self.membership)
        driver.sync = sync
        handle.driver = driver
        thread = threading.Thread(target=self._drive, args=(handle, sync),
                                  name=f"trainer-incarnation-{plan.incarnation}", daemon=True)
        handle._thread = thread
        thread.start()
        return handle

    def ready(self, handle: TrainerHandle, deadline: float) -> bool:
        """Wait until the trainer can train (True) or failed (False), at most until
        ``deadline`` on the lifecycle clock."""
        with handle._cond:
            while not (handle._parked and handle.state is LifecycleState.TRAINING) \
                    and handle.state is not LifecycleState.STOPPED:
                left = deadline - self.clock()
                if left <= 0:
                    return False
                handle._cond.wait(timeout=min(left, 1.0))
            return handle.state is LifecycleState.TRAINING

    def run(self, handle: TrainerHandle, source: DurableSampleBuffer | None,
            until: StopCondition) -> RunReceipt:
        """Train ``until.rounds`` more rounds (or to the end of the data); return at
        the round boundary with the trainer still in ``TRAINING``, or after it
        drained on its own because the data ended."""
        handle.source = source
        with handle._cond:
            self._wait_parked_or_done(handle)
            self._raise_if_done(handle)
            first = handle.rounds
            parks = handle._parks
            handle._target_rounds = math.inf if until.rounds is None else first + until.rounds
            handle._cond.notify_all()
            handle._cond.wait_for(lambda: handle._done or (handle._parked and handle._parks > parks))
            if handle.error is not None:
                raise LifecycleError(f"incarnation {handle.incarnation} failed") from handle.error
            assert handle.start_version is not None
            return RunReceipt(rounds=handle.rounds - first, first_version=handle.start_version + first,
                              trained_version=handle.start_version + handle.rounds,
                              data_ended=handle.data_ended)

    def drain_and_cut(self, handle: TrainerHandle, reason: LeaveReason) -> CutRef | None:
        """End the driver where it waits (after a publication): final cut with the
        trigger of ``reason``, close the sync session. Returns the newest cut (the
        restored one when nothing was trained since; None without a cut log)."""
        reason = LeaveReason(reason)
        trigger = CUT_TRIGGER_FOR_REASON.get(reason)
        if trigger is None:
            raise LifecycleError(f"no cut trigger for LEAVE reason {reason.name}")
        with handle._cond:
            if not handle._done:
                self._wait_parked_or_done(handle)
            if not handle._done:
                handle.requested_reason = reason
                handle._drain_trigger = trigger
                handle._cond.notify_all()
        self._join(handle)
        if handle.error is not None:
            raise LifecycleError(f"incarnation {handle.incarnation} failed") from handle.error
        return handle.cut

    def stop(self, handle: TrainerHandle, reason: LeaveReason) -> None:
        """LEAVE with ``reason`` and release the trainer. Drains first when the
        trainer still waits at a round boundary."""
        reason = LeaveReason(reason)
        if handle.state is LifecycleState.STOPPED:
            return
        if not handle._done:
            self.drain_and_cut(handle, reason)
        if handle.state is LifecycleState.STOPPED:
            return
        self._end(handle, handle.requested_reason or reason)

    # -- internals -------------------------------------------------------------
    def _drive(self, handle: TrainerHandle, sync: _LifecycleSync) -> None:
        assert handle.driver is not None
        try:
            handle.driver.run()
        except _Drained:
            pass
        except BaseException as error:  # noqa: BLE001 - reported on the handle
            handle.error = error
        if handle.error is None:
            if self.cuts is not None:
                handle.cut = self.cuts.latest()
        with handle._cond:
            handle._done = True
            handle._cond.notify_all()
        if handle.error is not None:
            # from any state to STOPPED on failure; LEAVE with the driver's code
            self._end(handle, sync.closed_reason or leave_reason_for(handle.error))

    def _end(self, handle: TrainerHandle, reason: LeaveReason) -> None:
        handle.leave_reason = LeaveReason(reason)
        try:
            if handle.joined and self.membership is not None:
                self.membership.leave(handle.incarnation, handle.leave_reason)
        finally:
            try:
                if self.release is not None:
                    self.release(handle)
            finally:
                handle.advance(LifecycleState.STOPPED)

    @staticmethod
    def _wait_parked_or_done(handle: TrainerHandle) -> None:
        handle._cond.wait_for(lambda: handle._parked or handle._done)

    @staticmethod
    def _raise_if_done(handle: TrainerHandle) -> None:
        if handle._done:
            if handle.error is not None:
                raise LifecycleError(f"incarnation {handle.incarnation} failed") from handle.error
            raise LifecycleError(f"incarnation {handle.incarnation} already drained")

    @staticmethod
    def _join(handle: TrainerHandle) -> None:
        if handle._thread is not None:
            handle._thread.join()
