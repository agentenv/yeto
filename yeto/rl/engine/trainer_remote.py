"""The trainer lifecycle across a process boundary (rl-trainer-start-stop L5.3c).

The head runs the scheduler loop; the trainer runs on its own island (a Modal
function, a sky cluster). Each incarnation is a fresh island. Two channels connect
them, both existing paths:

* commands, head -> island: a JSON-lines file per incarnation in the run's
  checkpoint store (``trainer-control/inc-<n>.jsonl``). The island already mounts
  that store for its cuts and the sample buffer; the head writes it through the
  store's own client (Modal Volume upload, bucket, local directory);
* events, island -> head: ``rl_trainer_lifecycle`` records in the island's event
  tape, which the island already echoes on its log stream (``yeto.rl.event_echo``).

Why not an elastic control frame: the elastic protocol has no head -> island
command frame (the syncer sends JoinAck, Base, Finished, Verdict), the trainer of a
start/stop run restores round cuts without a syncer, and a new frame changes the
Rust syncer and the protocol version (the v3 boundary of L9.7).

Island side: :class:`IslandTrainerAgent` runs the island's driver inside a
:class:`~.trainer_lifecycle.DriverLifecycle` (JOIN through the sync session,
restore of the newest cut, warm-up publication) and executes the commands at round
boundaries: train N rounds, or drain with the cut trigger of the stop reason and
LEAVE with the P5 code. It reports each step as an event.

Head side: :class:`RemoteTrainerLifecycle` is the
:class:`~.trainer_lifecycle.TrainerLifecycle` of the scheduler loop. It acquires an
island per incarnation through a :class:`TrainerIslands` port (cloud layer), sends
the commands, follows the events, and releases the island after its LEAVE.

:class:`StoreCutLog` reads the newest round-cut pointer from the store (snapshot
or flat layout); its watermark is the consumed position the trainer committed.

Neutral core: no backend, no cloud SDK.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from yeto.rl.elastic_client import LeaveReason

from . import recovery
from .resume import FLAT_MANIFEST, LATEST, STATE_DIR, CheckpointStore
from .trainer_lifecycle import (
    CUT_TRIGGER_FOR_REASON,
    CutLog,
    CutRef,
    DriverLifecycle,
    LifecycleError,
    LifecycleState,
    RunReceipt,
    StartPlan,
    StopCondition,
    TrainerHandle,
)

CONTROL_DIR = "trainer-control"
EVENT = "rl_trainer_lifecycle"
POINTER = "round-cut.json"  # adapters.miles.round_cut.POINTER (state dir, synced to the store)

# Island environment, set per incarnation by the cloud layer.
CONTROL_ENV = "YETO_TRAINER_CONTROL"  # "1": the island waits for head commands
INCARNATION_ENV = "YETO_TRAINER_INCARNATION"
RESUME_ENV = "YETO_TRAINER_RESUME"  # JSON of the CutRef to restore; empty for a first start
POLL_ENV = "YETO_TRAINER_COMMAND_POLL_S"

DEFAULT_POLL_S = 5.0
# A log stream may deliver the island's last lines after the cloud reports the call
# ended: wait this long for the stopped event before calling the exit a failure.
DEFAULT_EXIT_GRACE_S = 60.0

RUN, STOP = "run", "stop"

# Event phases (field ``phase``).
STARTED, READY, TRAINED, STOPPED = "started", "ready", "trained", "stopped"


class RemoteTrainerError(RuntimeError):
    """The trainer island failed or ended without a stopped event."""


# ---------------------------------------------------------------------------
# Store files
# ---------------------------------------------------------------------------
@runtime_checkable
class StoreFiles(Protocol):
    """Small whole-file reads and writes in the run's checkpoint store, by path
    relative to the store root. ``read`` returns None for a missing file and sees
    what other hosts committed; ``write`` is visible to them when it returns."""

    def read(self, rel: str) -> bytes | None: ...
    def write(self, rel: str, data: bytes) -> None: ...


class LocalStoreFiles:
    """:class:`StoreFiles` over a mounted :class:`CheckpointStore` (the island's
    mount, a bucket mount, a local directory)."""

    def __init__(self, store: CheckpointStore) -> None:
        self.store = store

    def read(self, rel: str) -> bytes | None:
        self.store.reload()
        path = Path(self.store.root) / rel
        return path.read_bytes() if path.is_file() else None

    def write(self, rel: str, data: bytes) -> None:
        path = Path(self.store.root) / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        self.store.commit()


# ---------------------------------------------------------------------------
# Cuts in the store
# ---------------------------------------------------------------------------
def cut_ref_to_json(cut: CutRef | None) -> str:
    return "" if cut is None else json.dumps(asdict(cut), sort_keys=True)


def cut_ref_from_json(text: str | None) -> CutRef | None:
    return CutRef(**json.loads(text)) if text else None


def cut_ref_of_pointer(pointer: Mapping[str, Any]) -> CutRef:
    """The cut a round-cut pointer names. ``policy_version`` is the version the next
    start publishes first: the pointer's next rollout id."""
    cut = recovery.ConsistentCut.from_record(pointer[recovery.CUT_KEY])
    version = pointer.get("next_rollout_id", pointer.get("policy_version"))
    return CutRef(cut_id=cut.checkpoint_id, trigger=cut.trigger, policy_version=int(version),
                  watermark=cut.buffer_watermark)


class StoreCutLog:
    """:class:`~.trainer_lifecycle.CutLog` over the store: the round-cut pointer of
    the newest snapshot (``LATEST`` -> ``state/<seq>/round-cut.json``) or, in the
    flat layout, the pointer in the store root."""

    def __init__(self, files: StoreFiles) -> None:
        self.files = files

    def pointer(self) -> dict[str, Any] | None:
        latest = self.files.read(LATEST)
        if latest is not None:
            seq = int(json.loads(latest)["seq"])
            raw = self.files.read(f"{STATE_DIR}/{seq:08d}/{POINTER}")
        elif self.files.read(FLAT_MANIFEST) is not None:
            raw = self.files.read(POINTER)
        else:
            raw = None
        return None if raw is None else dict(json.loads(raw))

    def latest(self) -> CutRef | None:
        pointer = self.pointer()
        if pointer is None or recovery.CUT_KEY not in pointer:
            return None
        return cut_ref_of_pointer(pointer)

    def watermark(self) -> int:
        """The consumed buffer position of the newest cut (0 without a cut)."""
        cut = self.latest()
        return 0 if cut is None or cut.watermark is None else int(cut.watermark)


# ---------------------------------------------------------------------------
# Command channel
# ---------------------------------------------------------------------------
def command_path(incarnation: int) -> str:
    return f"{CONTROL_DIR}/inc-{int(incarnation):06d}.jsonl"


@dataclass(frozen=True)
class Command:
    """``run``: train ``rounds`` more rounds (None: to the end of the data);
    ``stop``: drain, cut with the trigger of ``reason`` and LEAVE with it."""

    seq: int
    verb: str
    rounds: int | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.verb not in (RUN, STOP):
            raise ValueError(f"unknown trainer command {self.verb!r}")
        if self.verb == STOP and self.reason not in LeaveReason.__members__:
            raise ValueError(f"stop command needs a LEAVE reason name, got {self.reason!r}")


class CommandWriter:
    """Head side: appends commands to one incarnation's file (rewritten whole: the
    store has no append)."""

    def __init__(self, files: StoreFiles, incarnation: int) -> None:
        self.files = files
        self.path = command_path(incarnation)
        self.sent: list[Command] = []

    def send(self, verb: str, *, rounds: int | None = None, reason: LeaveReason | None = None) -> int:
        command = Command(seq=len(self.sent) + 1, verb=verb, rounds=rounds,
                          reason=None if reason is None else LeaveReason(reason).name)
        lines = [json.dumps(asdict(c), sort_keys=True) for c in (*self.sent, command)]
        self.files.write(self.path, ("\n".join(lines) + "\n").encode())
        self.sent.append(command)
        return command.seq


class CommandReader:
    """Island side: the commands of one incarnation not seen yet, in order."""

    def __init__(self, files: StoreFiles, incarnation: int) -> None:
        self.files = files
        self.path = command_path(incarnation)
        self.seen = 0

    def poll(self) -> list[Command]:
        raw = self.files.read(self.path)
        if raw is None:
            return []
        out = []
        for line in raw.decode().splitlines():
            if not line.strip():
                continue
            command = Command(**json.loads(line))
            if command.seq > self.seen:
                out.append(command)
                self.seen = command.seq
        return out


# ---------------------------------------------------------------------------
# Island side
# ---------------------------------------------------------------------------
def commanded(environ: Mapping[str, str] | None = None) -> bool:
    return (os.environ if environ is None else environ).get(CONTROL_ENV) == "1"


def lifecycle_event(handle: TrainerHandle, phase: str, *, seq: int,
                    watermark: int | None = None) -> dict[str, Any]:
    """The fields of one ``rl_trainer_lifecycle`` event. ``watermark``: the
    consumed buffer position after the trained rounds (committed on the island)."""
    cut = handle.cut
    return {
        "phase": phase, "incarnation": handle.incarnation, "seq": int(seq), "watermark": watermark,
        "state": handle.state.value, "start_version": handle.start_version,
        "rounds": handle.rounds, "trained_version": handle.trained_version,
        "data_ended": handle.data_ended,
        "leave_reason": None if handle.leave_reason is None else handle.leave_reason.name,
        "cut": None if cut is None else asdict(cut),
        "error": None if handle.error is None else repr(handle.error)[:2000],
    }


class IslandTrainerAgent:
    """Runs a built island driver as one commanded incarnation (see the module doc).

    ``emit(event, **fields)`` writes the island's event tape (``driver.emit``);
    ``cuts`` reads back the cut a drain committed (``StoreCutLog`` over the island's
    store mount); ``watermark(version)`` commits and returns the consumed buffer
    position below ``version`` (``RoundCutCheckpoint.buffer_watermark``), so the head
    counts the backlog between two cuts."""

    def __init__(self, driver: Any, *, commands: CommandReader, incarnation: int,
                 resume_from: CutRef | None, emit: Callable[..., Any], cuts: CutLog | None = None,
                 watermark: Callable[[int], int | None] | None = None,
                 poll_s: float = DEFAULT_POLL_S, sleep: Callable[[float], None] = time.sleep) -> None:
        self.lifecycle = DriverLifecycle(lambda _handle: driver, cuts=cuts,
                                         first_incarnation=int(incarnation))
        self.commands = commands
        self.resume_from = resume_from
        self.emit = emit
        self.watermark = watermark
        self.poll_s = float(poll_s)
        self.sleep = sleep
        self.seq = 0  # last command applied
        self.handle: TrainerHandle | None = None

    def run(self) -> TrainerHandle:
        lc = self.lifecycle
        handle = self.handle = lc.start(lc.plan_start({}), resume_from=self.resume_from)
        self._event(STARTED)
        if lc.ready(handle, math.inf):
            self._event(READY)
        while handle.state is not LifecycleState.STOPPED:
            commands = self.commands.poll()
            if not commands:
                self.sleep(self.poll_s)
            for command in commands:
                self._apply(command)
                if handle.state is LifecycleState.STOPPED:
                    break
        self._event(STOPPED)
        if handle.error is not None:
            raise handle.error
        return handle

    def _apply(self, command: Command) -> None:
        lc, handle = self.lifecycle, self.handle
        assert handle is not None
        self.seq = command.seq
        try:
            if command.verb == RUN:
                receipt = lc.run(handle, None, StopCondition(command.rounds))
                self._event(TRAINED)
                if receipt.data_ended:
                    lc.stop(handle, LeaveReason.FINISHED)
            else:
                lc.stop(handle, LeaveReason[command.reason])
        except LifecycleError:
            pass  # the handle is STOPPED and carries the error: reported by run()

    def _event(self, phase: str) -> None:
        handle = self.handle
        assert handle is not None
        watermark = None
        if phase in (READY, TRAINED) and self.watermark is not None and handle.trained_version is not None:
            watermark = self.watermark(handle.trained_version)
        self.emit(EVENT, **lifecycle_event(handle, phase, seq=self.seq,
                                           watermark=None if watermark is None else int(watermark)))


def run_commanded_island(driver: Any, *, store: CheckpointStore,
                         environ: Mapping[str, str] | None = None) -> TrainerHandle:
    """Island entry: the incarnation and the cut to restore come from the
    environment the cloud layer set for this incarnation."""
    environ = os.environ if environ is None else environ
    files = LocalStoreFiles(store)
    incarnation = int(environ[INCARNATION_ENV])
    watermark = getattr(getattr(driver.sync, "round_cuts", None), "buffer_watermark", None)
    return IslandTrainerAgent(
        driver, commands=CommandReader(files, incarnation), incarnation=incarnation,
        resume_from=cut_ref_from_json(environ.get(RESUME_ENV)), emit=driver.emit,
        cuts=StoreCutLog(files), watermark=watermark if callable(watermark) else None,
        poll_s=float(environ.get(POLL_ENV) or DEFAULT_POLL_S)).run()


# ---------------------------------------------------------------------------
# Head side
# ---------------------------------------------------------------------------
@runtime_checkable
class TrainerIslands(Protocol):
    """Cloud layer port: one trainer island per incarnation.

    ``launch`` starts the island with the incarnation environment (``island_env``)
    and returns its job; ``events`` returns the island events that arrived since
    the last call (non-blocking); ``alive`` is False once the island ended;
    ``release`` gives the cards back and returns True when the cloud confirmed it;
    ``release_incarnation`` does the same for an incarnation a dead head started."""

    def launch(self, incarnation: int, resume_from: CutRef | None) -> Any: ...
    def events(self, job: Any) -> list[Mapping[str, Any]]: ...
    def alive(self, job: Any) -> bool: ...
    def release(self, job: Any) -> bool: ...
    def release_incarnation(self, incarnation: int) -> bool: ...


def island_env(incarnation: int, resume_from: CutRef | None, *,
               poll_s: float | None = None) -> dict[str, str]:
    env = {CONTROL_ENV: "1", INCARNATION_ENV: str(int(incarnation)),
           RESUME_ENV: cut_ref_to_json(resume_from)}
    if poll_s is not None:
        env[POLL_ENV] = str(float(poll_s))
    return env


class RemoteHandle(TrainerHandle):
    """A :class:`TrainerHandle` the head moves from the island's events."""

    def __init__(self, plan: StartPlan, *, resume_from: CutRef | None, clock: Callable[[], float],
                 commands: CommandWriter) -> None:
        super().__init__(plan, resume_from=resume_from, clock=clock)
        self.commands = commands
        self.job: Any = None
        self.acked = 0  # last command the island reported done
        self.ready_seen = False
        self.watermark: int | None = None  # consumed position the island reported
        self.island_ended = False  # stopped event seen, or the island exited
        self.released: bool | None = None  # cloud confirmed the cards are back
        self.exited_at: float | None = None  # first poll that saw the island not alive


class RemoteTrainerLifecycle:
    """:class:`~.trainer_lifecycle.TrainerLifecycle` over a trainer island per
    incarnation (see the module doc). Waits poll the events every ``poll_s``; an
    island that ended without a stopped event is a failure after ``exit_grace_s``."""

    def __init__(self, islands: TrainerIslands, files: StoreFiles, *, first_incarnation: int = 0,
                 poll_s: float = DEFAULT_POLL_S, exit_grace_s: float = DEFAULT_EXIT_GRACE_S,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.islands = islands
        self.files = files
        self.poll_s = float(poll_s)
        self.exit_grace_s = float(exit_grace_s)
        self.clock = clock
        self.sleep = sleep
        self._next_incarnation = int(first_incarnation)
        self.current: RemoteHandle | None = None

    def running_watermark(self) -> int | None:
        """The consumed position the running incarnation reported (None when no
        trainer runs: then the newest cut holds it)."""
        handle = self.current
        if handle is None or handle.state is LifecycleState.STOPPED:
            return None
        return handle.watermark

    def plan_start(self, spec: Mapping[str, Any]) -> StartPlan:
        plan = StartPlan(incarnation=self._next_incarnation, spec=dict(spec))
        self._next_incarnation += 1
        return plan

    def start(self, plan: StartPlan, *, resume_from: CutRef | None) -> RemoteHandle:
        handle = RemoteHandle(plan, resume_from=resume_from, clock=self.clock,
                              commands=CommandWriter(self.files, plan.incarnation))
        self.current = handle
        handle.advance(LifecycleState.ACQUIRING)
        try:
            handle.job = self.islands.launch(plan.incarnation, resume_from)
        except BaseException as error:
            handle.error = error
            handle.leave_reason = LeaveReason.INFRA_FAILED
            handle.advance(LifecycleState.STOPPED)  # nothing acquired, nothing to release
            raise
        return handle

    def ready(self, handle: RemoteHandle, deadline: float) -> bool:
        self._wait(handle, lambda: handle.ready_seen, deadline)
        return handle.ready_seen and handle.state is LifecycleState.TRAINING

    def run(self, handle: RemoteHandle, source: Any, until: StopCondition) -> RunReceipt:
        """Train ``until.rounds`` more rounds on the island (``source`` is the island's
        own buffer reader; the head's view is not sent)."""
        del source
        self._raise_if_ended(handle)
        first = handle.rounds
        seq = handle.commands.send(RUN, rounds=until.rounds)
        self._wait(handle, lambda: handle.acked >= seq, math.inf)
        if handle.error is not None:
            raise LifecycleError(f"incarnation {handle.incarnation} failed") from handle.error
        assert handle.start_version is not None
        return RunReceipt(rounds=handle.rounds - first, first_version=handle.start_version + first,
                          trained_version=handle.start_version + handle.rounds,
                          data_ended=handle.data_ended)

    def drain_and_cut(self, handle: RemoteHandle, reason: LeaveReason) -> CutRef | None:
        reason = LeaveReason(reason)
        if reason not in CUT_TRIGGER_FOR_REASON:
            raise LifecycleError(f"no cut trigger for LEAVE reason {reason.name}")
        if not handle.island_ended:
            handle.requested_reason = reason
            handle.commands.send(STOP, reason=reason)
            self._wait(handle, lambda: False, math.inf)
        if handle.error is not None:
            raise LifecycleError(f"incarnation {handle.incarnation} failed") from handle.error
        return handle.cut

    def stop(self, handle: RemoteHandle, reason: LeaveReason) -> None:
        """Drain (when the island still runs), then release the island."""
        reason = LeaveReason(reason)
        if handle.state is LifecycleState.STOPPED:
            return
        try:
            if not handle.island_ended:
                self.drain_and_cut(handle, reason)
        finally:
            if handle.state is not LifecycleState.STOPPED:
                self._end(handle, reason)

    @staticmethod
    def _raise_if_ended(handle: RemoteHandle) -> None:
        if handle.island_ended or handle.state is LifecycleState.STOPPED:
            if handle.error is not None:
                raise LifecycleError(f"incarnation {handle.incarnation} failed") from handle.error
            raise LifecycleError(f"incarnation {handle.incarnation} already drained")

    # -- events ----------------------------------------------------------------
    def _wait(self, handle: RemoteHandle, done: Callable[[], bool], deadline: float) -> None:
        """Follow the events until ``done()``, the island ended, or ``deadline``."""
        while True:
            self._pump(handle)
            if done() or handle.island_ended or self.clock() >= deadline:
                return
            self.sleep(self.poll_s)

    def _pump(self, handle: RemoteHandle) -> None:
        events = [e for e in self.islands.events(handle.job)
                  if e.get("event") == EVENT and int(e.get("incarnation", -1)) == handle.incarnation]
        for event in events:
            self._apply(handle, event)
        if not handle.island_ended and not events and not self.islands.alive(handle.job):
            if handle.exited_at is None:
                handle.exited_at = self.clock()
            if self.clock() - handle.exited_at >= self.exit_grace_s:
                handle.island_ended = True
                handle.error = handle.error or RemoteTrainerError(
                    f"trainer island of incarnation {handle.incarnation} ended without a "
                    "stopped event")
        if handle.island_ended and handle.state is not LifecycleState.STOPPED:
            self._end(handle, _first_reason(handle.leave_reason, handle.requested_reason,
                                            LeaveReason.INFRA_FAILED))

    @staticmethod
    def _apply(handle: RemoteHandle, event: Mapping[str, Any]) -> None:
        for key in ("start_version", "rounds"):
            if event.get(key) is not None:
                setattr(handle, key, int(event[key]))
        handle.data_ended = bool(event.get("data_ended"))
        if event.get("watermark") is not None:
            handle.watermark = int(event["watermark"])
        handle.acked = max(handle.acked, int(event.get("seq") or 0))
        _catch_up(handle, LifecycleState(event["state"]))
        phase = event.get("phase")
        if phase == READY:
            handle.ready_seen = True
        elif phase == STOPPED:
            handle.island_ended = True
            if event.get("cut"):
                handle.cut = CutRef(**event["cut"])
            if event.get("leave_reason"):
                handle.leave_reason = LeaveReason[event["leave_reason"]]
            if event.get("error"):
                handle.error = RemoteTrainerError(event["error"])

    def _end(self, handle: RemoteHandle, reason: LeaveReason) -> None:
        """The island left (its own LEAVE): give the cards back."""
        handle.leave_reason = _first_reason(handle.leave_reason, LeaveReason(reason))
        try:
            handle.released = bool(self.islands.release(handle.job))
        finally:
            handle.advance(LifecycleState.STOPPED)
        if not handle.released:
            print(f"[trainer] incarnation {handle.incarnation}: card release not confirmed by the "
                  "cloud layer", flush=True)


def _first_reason(*reasons: LeaveReason | None) -> LeaveReason:
    """The first given reason (FINISHED is 0: never test a reason for truth)."""
    return next(LeaveReason(r) for r in reasons if r is not None)


def consumed_watermark(cuts: StoreCutLog, lifecycle: RemoteTrainerLifecycle | None) -> int:
    """Where the head counts the backlog from: the position the running trainer
    reported, else the newest cut's (D4: a failed trainer's rounds after its last
    cut are trained again)."""
    running = None if lifecycle is None else lifecycle.running_watermark()
    return max(cuts.watermark(), running or 0)


def _catch_up(handle: TrainerHandle, target: LifecycleState) -> None:
    """Advance the head's copy of the state step by step up to the island's
    (events can skip states). STOPPED is the head's own step, after the release."""
    if target is LifecycleState.STOPPED:
        return
    order = list(LifecycleState)
    while order.index(handle.state) < order.index(target):
        handle.advance(order[order.index(handle.state) + 1])
