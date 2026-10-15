"""The controller loop of the trainer scheduler (rl-trainer-start-stop L5.3b, D1/D8).

:class:`TrainerLoop` connects the policy to the mechanism. Each :meth:`TrainerLoop.tick`:

1. reads the inputs: the backlog of the :class:`~.trainer_lifecycle.DurableSampleBuffer`
   at the current trained version, the data-end signal, and the lifecycle state.
   ``T_s`` (the start-phase events) is read by the scheduler itself;
2. asks the :class:`~.trainer_lifecycle.TrainerScheduler` for a decision;
3. executes it on the :class:`~.trainer_lifecycle.TrainerLifecycle`:

   * ``Start``: plan a new incarnation, start it from the newest cut
     (``CutLog.latest``; the lifecycle JOINs and restores it), wait until ready;
   * ``Stop``: ``lifecycle.stop`` (drain, cut with the trigger of the reason --
     ``STOPPED`` -> ``SCHEDULED`` --, LEAVE with the P5 code, release the cards);
   * ``Hold`` while ``TRAINING`` with at least one batch: train ``rounds_per_tick``
     rounds and return at the round boundary.

A ``Stop(FINISHED)`` with no trainer running (the data ended with less than one
batch left) ends the loop too; it is journaled as ``finished`` (no incarnation).

After ``max_consecutive_failures`` failed incarnations in a row (no trained round
between them) the loop releases the trainer, journals ``gave_up`` and raises
:class:`TrainerLoopGaveUp`; a trained round resets the count.

The loop journals each start and stop (kind ``trainer_loop``) in the controller
journal next to the scheduler state (kind ``trainer_scheduler_state``). After a head
restart, :meth:`TrainerLoop.restore` reads both back: the scheduler continues with
its idle time, the next incarnation number follows the last one, and a trainer the
journal shows as started and not stopped is an orphan of the dead head. The
in-process handle of an orphan is gone, so the loop releases it through
``release_orphan`` (when given) and journals it as orphaned. The next start restores
the newest committed cut. The caller resets the head's buffer to the watermark of
that cut after :meth:`TrainerLoop.restore` (the buffer is not this module's).

:class:`TrainerLoopThread` runs ticks at a fixed period on its own thread, next to
the fleet controller's poll loop (``yeto.launch``).

Neutral core: no backend, no cloud.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from typing import Any

from yeto.rl.elastic_client import LeaveReason

from .trainer_lifecycle import (
    BacklogStats,
    CutLog,
    Decision,
    DurableSampleBuffer,
    Hold,
    LifecycleError,
    LifecycleState,
    Start,
    Stop,
    StopCondition,
    TrainerHandle,
    TrainerLifecycle,
    TrainerScheduler,
)

LOOP_KIND = "trainer_loop"

# trainer_loop records: ``event`` is one of these.
STARTED = "started"
READY = "ready"
STOPPED = "stopped"
ORPHANED = "orphaned"
FINISHED = "finished"  # the data ended with no trainer running (incarnation None)
GAVE_UP = "gave_up"  # too many incarnations failed in a row: the loop stopped
FIRST_POLICY = "first_policy"  # a start (or a kept trainer) for the samplers' first policy

DEFAULT_MAX_CONSECUTIVE_FAILURES = 3


class TrainerLoopGaveUp(RuntimeError):
    """``max_consecutive_failures`` incarnations failed in a row with no trained
    round between them; the loop stopped starting new ones."""

Record = Callable[..., Any]  # journal.append(kind, **fields)


def next_incarnation(records: Iterable[Mapping[str, Any]]) -> int:
    """The incarnation after the last one the journal names (0 for a new run).
    Build the lifecycle with it (``DriverLifecycle(first_incarnation=...)``)."""
    last = -1
    for r in records:
        if r.get("kind") == LOOP_KIND and r.get("incarnation") is not None:
            last = max(last, int(r["incarnation"]))
    return last + 1


class TrainerLoop:
    """One trainer, one scheduler, one buffer. Not thread-safe: one caller ticks.

    ``record`` / ``records``: the controller journal (``Journal.append`` and
    ``Journal.records``). ``data_ended`` tells the samplers will append nothing more.
    ``release_orphan(incarnation)`` gives back the cards of a trainer the dead head
    started (cloud layer). ``needs_first_policy()`` is True while the samplers wait
    for a first policy release (a LoRA sampler serves no adapter before it): the
    buffer then stays empty and S1 would never start the trainer, so the loop
    starts one, and keeps it until it trained a round (no idle stop of a trainer
    that only published the first policy)."""

    def __init__(self, *, scheduler: TrainerScheduler, lifecycle: TrainerLifecycle,
                 buffer: DurableSampleBuffer, cuts: CutLog,
                 record: Record | None = None,
                 data_ended: Callable[[], bool] = lambda: False,
                 release_orphan: Callable[[int], None] | None = None,
                 needs_first_policy: Callable[[], bool] | None = None,
                 spec: Mapping[str, Any] | None = None,
                 rounds_per_tick: int = 1, ready_timeout_s: float = 1800.0,
                 max_consecutive_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if rounds_per_tick < 1:
            raise ValueError("rounds_per_tick must be positive")
        if max_consecutive_failures < 1:
            raise ValueError("max_consecutive_failures must be positive")
        self.scheduler = scheduler
        self.lifecycle = lifecycle
        self.buffer = buffer
        self.cuts = cuts
        self._record = record
        self._data_ended = data_ended
        self._release_orphan = release_orphan
        self._needs_first_policy = needs_first_policy
        self._first_policy_incarnation: int | None = None
        self._start_for_first_policy = False
        self.spec = dict(spec or {})
        self.rounds_per_tick = int(rounds_per_tick)
        self.ready_timeout_s = float(ready_timeout_s)
        self._clock = clock
        self.handle: TrainerHandle | None = None
        self.started = 0  # incarnations started by this loop or a head before it
        self.finished = False  # stopped with FINISHED: nothing more to do
        self._journaled_stops: set[int] = set()  # incarnations with a stopped record
        # failed incarnations since the last trained round; at the limit the loop gives up
        self.max_consecutive_failures = int(max_consecutive_failures)
        self.consecutive_failures = 0

    # ------------------------------------------------------------ state
    @property
    def state(self) -> LifecycleState:
        if self.handle is not None:
            return self.handle.state
        return LifecycleState.STOPPED if self.started else LifecycleState.COLD

    def current_version(self) -> int:
        """The trained version: the running trainer's, else the newest cut's, else 0."""
        if self.handle is not None and self.handle.trained_version is not None:
            return self.handle.trained_version
        cut = self.cuts.latest()
        return 0 if cut is None else cut.policy_version

    def backlog(self) -> BacklogStats:
        stats = self.buffer.backlog(self.current_version())
        return replace(stats, data_ended=stats.data_ended or bool(self._data_ended()))

    # ------------------------------------------------------------ tick
    def tick(self) -> Decision:
        """Read the inputs, decide, execute. Returns the decision."""
        if self.finished:
            return Hold()
        self._note_failure()
        self._give_up_at_limit()
        backlog = self.backlog()
        decision = self._for_first_policy(self.scheduler.decide(backlog, self.state), backlog)
        if isinstance(decision, Start):
            self._start()
        elif isinstance(decision, Stop):
            self._stop(decision.reason)
        elif self.state is LifecycleState.TRAINING and backlog.batches >= 1:
            self._train()
        self._give_up_at_limit()
        return decision

    def _for_first_policy(self, decision: Decision, backlog: BacklogStats) -> Decision:
        """Start a trainer for the samplers' first policy; keep that incarnation
        until it trained a round (see the class doc)."""
        if self._needs_first_policy is None or backlog.data_ended:
            return decision
        handle = self.handle
        if (isinstance(decision, Stop) and decision.reason is LeaveReason.STOPPED
                and handle is not None and handle.rounds == 0
                and handle.incarnation == self._first_policy_incarnation):
            return Hold()
        if (isinstance(decision, Hold) and self.state in (LifecycleState.COLD, LifecycleState.STOPPED)
                and self._needs_first_policy()):
            self._start_for_first_policy = True
            return Start()
        return decision

    def _give_up_at_limit(self) -> None:
        """Stop the loop after ``max_consecutive_failures`` failed incarnations in a
        row: release a trainer still up, journal it and raise (no new Start)."""
        if self.consecutive_failures < self.max_consecutive_failures:
            return
        self.close(LeaveReason.INFRA_FAILED)
        self.finished = True
        self._journal(GAVE_UP, None if self.handle is None else self.handle.incarnation,
                      failures=self.consecutive_failures, limit=self.max_consecutive_failures)
        raise TrainerLoopGaveUp(
            f"{self.consecutive_failures} trainer incarnations failed in a row "
            f"(limit {self.max_consecutive_failures}); the trainer loop stopped")

    def close(self, reason: LeaveReason = LeaveReason.STOPPED) -> None:
        """Stop a running trainer (run end or head shutdown)."""
        if self.handle is not None and self.handle.state is not LifecycleState.STOPPED:
            self._stop(reason)

    # ------------------------------------------------------------ verbs
    def _start(self) -> None:
        resume = self.cuts.latest()
        plan = self.lifecycle.plan_start(self.spec)
        if self._start_for_first_policy:
            self._start_for_first_policy = False
            self._first_policy_incarnation = plan.incarnation
            self._journal(FIRST_POLICY, plan.incarnation)
        # journal first: a head that dies during the start sees the orphan
        self._journal(STARTED, plan.incarnation,
                      resume_cut=None if resume is None else resume.cut_id,
                      resume_version=None if resume is None else resume.policy_version)
        self.started += 1
        try:
            self.handle = self.lifecycle.start(plan, resume_from=resume)
        except Exception as error:  # noqa: BLE001 - the lifecycle stopped it with its P5 code
            self.handle = None
            self.consecutive_failures += 1
            self._journal(STOPPED, plan.incarnation, reason=None, error=repr(error))
            return
        if self.lifecycle.ready(self.handle, self._clock() + self.ready_timeout_s):
            self._journal(READY, plan.incarnation, version=self.handle.start_version)
            return
        if self.handle.state is not LifecycleState.STOPPED:
            try:
                self.lifecycle.stop(self.handle, LeaveReason.INFRA_FAILED)  # not ready in time
            except LifecycleError:
                pass  # no cut for a failure: the lifecycle released it; journaled below
        self._note_failure()

    def _stop(self, reason: LeaveReason) -> None:
        handle = self.handle
        if handle is None or handle.state is LifecycleState.STOPPED:
            if reason is LeaveReason.FINISHED:  # the data ended, nothing left to train
                self.finished = True
                self._journal(FINISHED, None, remainder=self.buffer.backlog(self.current_version()).samples)
            return
        try:
            self.lifecycle.stop(handle, reason)
        except LifecycleError:
            pass  # the handle carries the error; journaled below
        self._journal_stop(handle)
        if handle.error is None and handle.leave_reason is LeaveReason.FINISHED:
            self.finished = True

    def _train(self) -> None:
        assert self.handle is not None
        try:
            receipt = self.lifecycle.run(self.handle, self.buffer, StopCondition(self.rounds_per_tick))
        except LifecycleError:
            self._note_failure()
            return
        if receipt.rounds > 0:
            self.consecutive_failures = 0  # a trained round: the trainer works
        if receipt.data_ended:
            self._stop(LeaveReason.FINISHED)

    def _note_failure(self) -> None:
        """Journal a trainer that stopped on its own (failure) once."""
        handle = self.handle
        if handle is not None and handle.state is LifecycleState.STOPPED \
                and handle.incarnation not in self._journaled_stops:
            self._journal_stop(handle)

    def _journal_stop(self, handle: TrainerHandle) -> None:
        self._journaled_stops.add(handle.incarnation)
        if handle.error is not None or handle.leave_reason is LeaveReason.INFRA_FAILED:
            self.consecutive_failures += 1
        cut = handle.cut
        reason = handle.leave_reason
        self._journal(STOPPED, handle.incarnation,
                      reason=None if reason is None else reason.name,
                      cut=None if cut is None else cut.cut_id,
                      trigger=None if cut is None else cut.trigger,
                      version=handle.trained_version,
                      released=getattr(handle, "released", None),
                      error=None if handle.error is None else repr(handle.error))

    # ------------------------------------------------------------ journal (D8)
    def _journal(self, event: str, incarnation: int | None, **fields: Any) -> None:
        if self._record is not None:
            self._record(LOOP_KIND, event=event,
                         incarnation=None if incarnation is None else int(incarnation), **fields)

    def restore(self, records: Iterable[Mapping[str, Any]]) -> bool:
        """Continue after a head restart. Returns False for a new run (no records)."""
        records = list(records)
        restored = self.scheduler.restore(records) if hasattr(self.scheduler, "restore") else False
        open_incarnation = None
        for r in records:
            if r.get("kind") != LOOP_KIND:
                continue
            if r["event"] == FINISHED:
                self.finished = True
            elif r["event"] == STARTED:
                self.started += 1
                open_incarnation = int(r["incarnation"])
            elif r["event"] in (STOPPED, ORPHANED):
                if r["event"] == STOPPED and r.get("reason") == LeaveReason.FINISHED.name:
                    self.finished = True
                open_incarnation = None
        if open_incarnation is not None:
            if self._release_orphan is not None:
                self._release_orphan(open_incarnation)
            self._journal(ORPHANED, open_incarnation)
        return bool(restored or self.started)


class TrainerLoopThread:
    """Ticks a :class:`TrainerLoop` every ``interval_s`` on a daemon thread until
    :meth:`stop` or the loop finished. A tick error ends the thread; ``error``
    holds it."""

    def __init__(self, loop: TrainerLoop, interval_s: float,
                 on_error: Callable[[BaseException], None] | None = None,
                 on_finished: Callable[[], None] | None = None) -> None:
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        self.loop = loop
        self.interval_s = float(interval_s)
        self._on_error = on_error  # told once, on this thread, after a tick error
        self._on_finished = on_finished  # told once, on this thread, when the loop finished
        self.error: BaseException | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="trainer-scheduler-loop", daemon=True)

    def start(self) -> TrainerLoopThread:
        self._thread.start()
        return self

    def stop(self, *, close_reason: LeaveReason | None = LeaveReason.STOPPED,
             timeout: float | None = None) -> None:
        """End the ticks; then stop a running trainer with ``close_reason``
        (None keeps it, e.g. for a head that hands over)."""
        self._stop.set()
        self._thread.join(timeout)
        if close_reason is not None and not self._thread.is_alive():
            self.loop.close(close_reason)

    def join(self, timeout: float | None = None) -> bool:
        """Wait up to ``timeout`` seconds for the loop to end on its own (finished
        or a tick error); True when it ended."""
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def _run(self) -> None:
        try:
            while not self.loop.finished and not self._stop.is_set():
                self.loop.tick()
                self._stop.wait(self.interval_s)
            if self.loop.finished and self._on_finished is not None:
                self._on_finished()
        except BaseException as error:  # noqa: BLE001 - reported to the launcher
            self.error = error
            if self._on_error is not None:
                self._on_error(error)
