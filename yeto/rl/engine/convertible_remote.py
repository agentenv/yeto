"""Convertible nodes across a process boundary (rl-trainer-start-stop L7.5).

A *convertible pool* is a launch whose islands are all declared convertible
(``--rl-node-class islandM=convertible``). Every island boots as a rollout node
of the run: it generates rounds into the durable sample buffer. The head's
trainer loop (S1) decides when a wave of samples is worth training; then the
best node by the selection rule (:func:`.convertible.select_group`) converts
to trainer in place, trains from the buffer, and converts back to rollout with
the trained weights loaded in its engines. No island is started or stopped for
this, so the start time ``T_s`` is the conversion time, not a cold start.

Channels, all in the run's checkpoint store (the store every island mounts;
``trainer_remote.StoreFiles``), under ``convertible/<node>/``:

* ``commands.jsonl`` (head -> node): ``to_trainer`` with the trainer incarnation
  and the cut to restore, or ``end`` (the run is over). The trainer commands of
  one incarnation (``run`` / ``stop``) use the trainer channel of
  :mod:`.trainer_remote` (``trainer-control/inc-<n>.jsonl``) unchanged;
* ``events.jsonl`` (node -> head): ``rl_trainer_lifecycle`` events of each
  trainer incarnation, the ``node_conversion`` receipts, and a
  ``trainer_start_summary`` per incarnation (``total_s`` = command seen ->
  trainer ready; the S1 scheduler reads it as ``T_s``);
* ``status.json`` (node -> head, rewritten every few seconds): role, in-flight
  requests, measured rollout capacity, served version, generation done.

One file for the whole pool, ``convertible/backlog.json`` (head -> nodes,
rewritten at every backlog count): the groups the trainer can train that it has
not read yet (the count S1 starts a trainer on). A node does not start a rollout
round while this backlog is at its cap (:class:`PoolBacklogGate`), so the nodes
do not run through their rounds ahead of the trainer: while one node trains, the
others still have rounds left to generate.

Weights: every trainer incarnation releases each published version into the
store (``policy_release``); a node in rollout loads the newest release before
each round (the ``follow`` hook), so a node that has not converted generates
with the trained policy too, not with the version it last trained.

Samples: each node appends to its own shard of the buffer
(``sample_buffer.shard_index``); the head merges the shards into the index
(``sample_buffer.merge_shards``) before it counts the backlog, so the index has
one writer.

Island side: :class:`ConvertibleNodeAgent`. Head side: :class:`ConvertibleIslands`
(the ``trainer_remote.TrainerIslands`` port of :class:`.trainer_remote.RemoteTrainerLifecycle`:
"launch" an incarnation = convert the selected node, "release" = wait until the
node is back to rollout) and :class:`ConvertibleGate` (holds a start when no node
may convert). Neutral core: no backend, no cloud SDK.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from .convertible import (
    CONVERSION_DECISION_KIND,
    ConversionReceipt,
    ConvertibleNode,
    NodeProfile,
    Selection,
    TrainerNeeds,
    select_group,
)
from .trainer_lifecycle import CutRef, Decision, Hold, LifecycleState, Start, TrainerHandle
from .trainer_remote import (
    EVENT,
    READY,
    STOP,
    Command,
    RemoteTrainerError,
    StoreFiles,
    command_path,
    cut_ref_from_json,
    cut_ref_to_json,
)
from .trainer_scheduler import START_SUMMARY_EVENT

CONVERTIBLE_DIR = "convertible"
TO_TRAINER, END = "to_trainer", "end"
NODE_VERBS = (TO_TRAINER, END)

# node roles in status.json
ROLLOUT, TRAINER, CONVERTING, FAILED, EXITED = "rollout", "trainer", "converting", "failed", "exited"

DEFAULT_POLL_S = 5.0
DEFAULT_STATUS_S = 10.0
# a node whose status is older than this is not alive (head side)
DEFAULT_STALE_S = 300.0
# max seconds the head waits for a node to report it is back to rollout
DEFAULT_RELEASE_TIMEOUT_S = 600.0
# a node whose generation is done exits on its own after this long without a command
DEFAULT_IDLE_EXIT_S = 1800.0
# a pool backlog record older than this does not hold a node back (the head may be gone)
DEFAULT_BACKLOG_STALE_S = 120.0
POOL_BACKLOG_PATH = f"{CONVERTIBLE_DIR}/backlog.json"


def node_path(node: str, name: str) -> str:
    return f"{CONVERTIBLE_DIR}/{node}/{name}"


# ---------------------------------------------------------------------------
# Channel records
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NodeCommand:
    """``to_trainer``: convert and run trainer ``incarnation``, restoring ``resume``
    (``cut_ref_to_json``; empty for none); ``end``: exit after the current step."""

    seq: int
    verb: str
    incarnation: int | None = None
    resume: str = ""

    def __post_init__(self) -> None:
        if self.verb not in NODE_VERBS:
            raise ValueError(f"unknown node command {self.verb!r}")
        if self.verb == TO_TRAINER and self.incarnation is None:
            raise ValueError("to_trainer needs the trainer incarnation")


class NodeCommandWriter:
    """Head side: the command file of one node, rewritten whole (no append in the
    store). Restarted heads continue after the commands already in the file."""

    def __init__(self, files: StoreFiles, node: str) -> None:
        self.files = files
        self.path = node_path(node, "commands.jsonl")
        self.sent = [NodeCommand(**json.loads(line)) for line in _lines(files.read(self.path))]

    def send(self, verb: str, *, incarnation: int | None = None, resume: CutRef | None = None) -> int:
        command = NodeCommand(seq=len(self.sent) + 1, verb=verb, incarnation=incarnation,
                              resume=cut_ref_to_json(resume))
        self.files.write(self.path, _jsonl([*map(asdict, self.sent), asdict(command)]))
        self.sent.append(command)
        return command.seq


class NodeCommandReader:
    """Island side: the node's commands not seen yet. ``seen`` starts at the commands
    already in the file when the island starts (a command of an earlier island of
    the same name is not replayed)."""

    def __init__(self, files: StoreFiles, node: str, *, skip_existing: bool = True) -> None:
        self.files = files
        self.path = node_path(node, "commands.jsonl")
        self.seen = len(_lines(files.read(self.path))) if skip_existing else 0

    def poll(self) -> list[NodeCommand]:
        out = []
        for line in _lines(self.files.read(self.path)):
            command = NodeCommand(**json.loads(line))
            if command.seq > self.seen:
                out.append(command)
                self.seen = command.seq
        return out


class NodeEventLog:
    """Island side: the node's event file (rewritten whole; one writer). Thread-safe."""

    def __init__(self, files: StoreFiles, node: str) -> None:
        self.files = files
        self.path = node_path(node, "events.jsonl")
        self._lock = threading.Lock()
        self.records = [json.loads(line) for line in _lines(files.read(self.path))]

    def append(self, record: Mapping[str, Any]) -> None:
        with self._lock:
            self.records.append({"seq": len(self.records) + 1, **record})
            write_retrying(self.files, self.path, _jsonl(self.records))


WRITE_ATTEMPTS = 4
WRITE_RETRY_DELAY_S = 1.0


def write_retrying(files: StoreFiles, rel: str, data: bytes, *,
                   wait: Callable[[float], Any] = threading.Event().wait) -> None:
    """``files.write`` retried on an OS error: a Modal Volume mount refused the
    rename of the temporary file now and then (``PermissionError``, GPU run
    s24-conv-20261015b) while the container reloaded the volume."""
    for attempt in range(WRITE_ATTEMPTS):
        try:
            files.write(rel, data)
            return
        except OSError:
            if attempt + 1 == WRITE_ATTEMPTS:
                raise
            wait(WRITE_RETRY_DELAY_S * (attempt + 1))


def write_pool_backlog(files: StoreFiles, *, groups: int, version: int, at: float) -> None:
    """Head side: the trainable groups not read yet (``groups``) at policy ``version``."""
    files.write(POOL_BACKLOG_PATH, json.dumps(
        {"groups": int(groups), "version": int(version), "at": round(float(at), 3)},
        sort_keys=True).encode())


class PoolBacklogGate:
    """Island side: may this node start a rollout round? Yes while the pool backlog
    the head wrote last is below ``max_groups``, or when there is no fresh record
    (no count yet, or older than ``stale_s``: a head that stopped writing must not
    stall the nodes). ``last`` is the record read by the last call."""

    def __init__(self, files: StoreFiles, max_groups: int, *,
                 stale_s: float = DEFAULT_BACKLOG_STALE_S,
                 clock: Callable[[], float] = time.time) -> None:
        if int(max_groups) < 1:
            raise ValueError("max_groups must be >= 1")
        self.files = files
        self.max_groups = int(max_groups)
        self.stale_s = float(stale_s)
        self.clock = clock
        self.last: dict[str, Any] | None = None

    def __call__(self) -> bool:
        raw = self.files.read(POOL_BACKLOG_PATH)
        self.last = None if not raw else json.loads(raw)
        if self.last is None or self.clock() - float(self.last.get("at", 0.0)) > self.stale_s:
            return True
        return int(self.last["groups"]) < self.max_groups


def append_trainer_command(files: StoreFiles, incarnation: int, verb: str, *,
                           rounds: int | None = None, reason: str | None = None) -> int:
    """Append one trainer command to an incarnation's file keeping the commands
    already there (a restarted head stops the orphan of the dead one)."""
    path = command_path(incarnation)
    sent = [json.loads(line) for line in _lines(files.read(path))]
    command = Command(seq=len(sent) + 1, verb=verb, rounds=rounds, reason=reason)
    files.write(path, _jsonl([*sent, asdict(command)]))
    return command.seq


def _lines(raw: bytes | None) -> list[str]:
    return [line for line in (raw or b"").decode().splitlines() if line.strip()]


def _jsonl(records: Iterable[Mapping[str, Any]]) -> bytes:
    return "".join(json.dumps(r, sort_keys=True) + "\n" for r in records).encode()


# ---------------------------------------------------------------------------
# Island side
# ---------------------------------------------------------------------------
@dataclass
class RolloutRound:
    """One rollout round of the node: groups written to its shard and seconds taken."""

    groups: int
    seconds: float


StintRunner = Callable[[int, CutRef | None, Callable[..., None]], TrainerHandle]


class ConvertibleNodeAgent:
    """Runs one convertible island (see the module doc).

    ``generate()`` runs one rollout round into the node's buffer shard;
    ``node``: the :class:`~.convertible.ConvertibleNode` over the backend's port;
    ``run_stint(incarnation, resume_from, emit)`` runs one trainer incarnation on
    this island until the head stops it (``trainer_remote.IslandTrainerAgent`` over
    a fresh driver) and returns its handle; ``published_checksum()`` is the engine
    checksum of the last publication of that stint; ``inflight()`` the requests on
    the engines (None: unknown); ``emit(event, **fields)`` writes the island tape.

    Before each rollout round the node calls ``follow()`` (load the newest policy
    release; returns the version it now serves, or None when nothing changed) and
    ``admit()`` (False: the pool backlog is at its cap; the node waits ``poll_s``
    and reads its commands again). ``after_stint()`` runs when the node is back
    to rollout after a trainer incarnation, before its next round (its engines
    serve the version it trained and released itself).

    The node takes a command at a rollout round boundary. It generates at most
    ``max_rounds`` rounds; after that it reports ``generation_done`` and still
    converts on command until ``end`` (or ``idle_exit_s`` without a command)."""

    def __init__(self, *, name: str, files: StoreFiles, node: ConvertibleNode,
                 generate: Callable[[], RolloutRound], run_stint: StintRunner,
                 published_checksum: Callable[[], str | None],
                 inflight: Callable[[], int | None] = lambda: None,
                 emit: Callable[..., None] = lambda event, **fields: None,
                 follow: Callable[[], int | None] = lambda: None,
                 admit: Callable[[], bool] = lambda: True,
                 after_stint: Callable[[], None] = lambda: None,
                 max_rounds: int | None = None, poll_s: float = DEFAULT_POLL_S,
                 status_s: float = DEFAULT_STATUS_S, idle_exit_s: float = DEFAULT_IDLE_EXIT_S,
                 served_version: int = 0, clock: Callable[[], float] = time.time,
                 wait: Callable[[float], None] | None = None) -> None:
        self.name = name
        self.files = files
        self.node = node
        self.generate = generate
        self.run_stint = run_stint
        self.published_checksum = published_checksum
        self.inflight = inflight
        self.emit_tape = emit
        self.follow = follow
        self.admit = admit
        self.after_stint = after_stint
        self.max_rounds = max_rounds
        self.poll_s = float(poll_s)
        self.status_s = float(status_s)
        self.idle_exit_s = float(idle_exit_s)
        self.clock = clock
        self.commands = NodeCommandReader(files, name)
        self.events = NodeEventLog(files, name)
        self.served_version = int(served_version)
        self.rounds = 0
        self.groups = 0
        self.capacity: float | None = None  # groups per second of the last round
        self.role = ROLLOUT
        self.stints = 0
        self.waiting_since: float | None = None  # held back by the pool backlog cap
        self._stop = threading.Event()
        self._wait = wait or self._stop.wait
        self._status_lock = threading.Lock()

    # -- loop --------------------------------------------------------------------
    def run(self) -> int:
        """Run until ``end``; returns the trainer incarnations this node ran."""
        heartbeat = threading.Thread(target=self._heartbeat, name=f"node-status-{self.name}",
                                     daemon=True)
        heartbeat.start()
        idle_since: float | None = None
        try:
            self._write_status()
            while True:
                commands = self.commands.poll()
                if any(c.verb == END for c in commands):
                    break
                for command in commands:
                    self._convert_and_train(command)
                    idle_since = None
                if commands:
                    continue
                if self.generation_done:
                    idle_since = self.clock() if idle_since is None else idle_since
                    if self.clock() - idle_since >= self.idle_exit_s:
                        self._event("node_idle_exit", idle_s=round(self.clock() - idle_since, 3))
                        break
                    self._wait(self.poll_s)
                    continue
                self._follow()
                if not self._admitted():
                    self._wait(self.poll_s)
                    continue
                self._rollout_round()
        except BaseException as error:
            self.role = FAILED
            self._write_status(error=repr(error)[:2000])
            raise
        finally:
            self._stop.set()
            heartbeat.join(timeout=self.status_s + 5)
        self.role = EXITED
        self._write_status()
        return self.stints

    @property
    def generation_done(self) -> bool:
        return self.max_rounds is not None and self.rounds >= self.max_rounds

    def _follow(self) -> None:
        version = self.follow()
        if version is not None and int(version) != self.served_version:
            self.served_version = int(version)
            self._write_status()

    def _admitted(self) -> bool:
        if self.admit():
            if self.waiting_since is not None:
                self._event("node_backlog_admitted",
                            waited_s=round(self.clock() - self.waiting_since, 3))
                self.waiting_since = None
            return True
        if self.waiting_since is None:
            self.waiting_since = self.clock()
            self._event("node_backlog_full")
            self._write_status()
        return False

    def _rollout_round(self) -> None:
        result = self.generate()
        self.rounds += 1
        self.groups += int(result.groups)
        if result.seconds > 0:
            self.capacity = round(result.groups / result.seconds, 6)
        self._write_status()

    def _convert_and_train(self, command: NodeCommand) -> None:
        incarnation = int(command.incarnation)  # type: ignore[arg-type]
        seen = self.clock()
        self.role = CONVERTING
        self._write_status()
        receipt = self.node.to_trainer(_NoBuffer())
        self._receipt(receipt, incarnation)
        self.role = TRAINER
        self._write_status()
        ready_at: list[float] = []

        def emit(event: str, **fields: Any) -> None:
            self.emit_tape(event, **fields)
            self._event(event, **fields)
            if event == EVENT and fields.get("phase") == READY and not ready_at:
                ready_at.append(self.clock())
                self._event(START_SUMMARY_EVENT, incarnation=incarnation, kind="conversion",
                            total_s=round(ready_at[0] - seen, 3),
                            to_trainer_s=round(receipt.seconds, 3))

        handle = self.run_stint(incarnation, cut_ref_from_json(command.resume), emit)
        self.stints += 1
        if handle.error is not None:
            raise RemoteTrainerError(f"node {self.name}: trainer incarnation {incarnation} "
                                     f"failed: {handle.error!r}") from handle.error
        version = handle.published_version
        if version is None:
            raise RemoteTrainerError(f"node {self.name}: trainer incarnation {incarnation} "
                                     "published no version")
        self.role = CONVERTING
        self._write_status()
        back = self.node.to_rollout(int(version), expected_checksum=self.published_checksum())
        self.served_version = int(version)
        self.after_stint()
        self._receipt(back, incarnation)
        self.role = ROLLOUT
        self._write_status()

    # -- reports -----------------------------------------------------------------
    def _receipt(self, receipt: ConversionReceipt, incarnation: int) -> None:
        record = {**receipt.event(), "incarnation": incarnation}
        record.pop("event")
        self.emit_tape(receipt.event()["event"], **record)
        self._event(receipt.event()["event"], **record)

    def _event(self, event: str, **fields: Any) -> None:
        self.events.append({"event": event, "node": self.name, "at": round(self.clock(), 3),
                            **fields})

    def status(self) -> dict[str, Any]:
        try:
            inflight = self.inflight()
        except Exception:  # noqa: BLE001 - an observation; unknown is reported as None
            inflight = None
        # a node that finished generating gives no rollout capacity away by converting
        capacity = 0.0 if self.generation_done else self.capacity
        return {"node": self.name, "role": self.role, "at": round(self.clock(), 3),
                "inflight": inflight, "capacity": capacity, "rounds": self.rounds,
                "groups": self.groups, "served_version": self.served_version,
                "generation_done": self.generation_done, "stints": self.stints,
                "backlog_wait": self.waiting_since is not None}

    def _write_status(self, **extra: Any) -> None:
        """Best effort: the status is an observation the head reads again; a write
        that fails after its retries is reported and the next one replaces it."""
        with self._status_lock:
            try:
                write_retrying(self.files, node_path(self.name, "status.json"),
                               json.dumps({**self.status(), **extra}, sort_keys=True).encode())
            except OSError as error:
                print(f"[convertible] node {self.name}: status write failed: {error!r}", flush=True)

    def _heartbeat(self) -> None:
        while not self._stop.wait(self.status_s):
            self._write_status()


class _NoBuffer:
    """The cut rule's sink on a node whose in-flight trajectories have no finished
    segment of their own (a Miles group is returned whole or regenerated)."""

    def append(self, entry: Any) -> None:
        raise RemoteTrainerError(f"unexpected finished segment {entry!r} on a convertible node")


# ---------------------------------------------------------------------------
# Head side
# ---------------------------------------------------------------------------
@dataclass
class _NodeJob:
    node: str
    incarnation: int
    events_seen: int = 0


@dataclass
class ConvertibleIslands:
    """:class:`~.trainer_remote.TrainerIslands` over a convertible pool (module doc).

    ``nodes``: the declared profile of every node (spot, precision, GPU type and
    count, network domain); the measured fields come from each node's status.
    ``needs``: what the trainer needs (precision, GPUs, network domains of the
    parameter source). ``record(kind, tx_id=None, **fields)``: the controller
    journal; every selection is journaled with its ranking."""

    files: StoreFiles
    nodes: Sequence[NodeProfile]
    needs: TrainerNeeds
    group_size: int = 1
    record: Callable[..., None] | None = None
    stale_s: float = DEFAULT_STALE_S
    release_timeout_s: float = DEFAULT_RELEASE_TIMEOUT_S
    poll_s: float = DEFAULT_POLL_S
    clock: Callable[[], float] = time.time
    sleep: Callable[[float], None] = time.sleep
    pending: Selection | None = None
    jobs: list[_NodeJob] = field(default_factory=list)
    _events: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.group_size != 1:
            raise ValueError("a trainer on a convertible pool runs on one node (group size 1): "
                             "one trainer process does not span islands")
        names = [n.name for n in self.nodes]
        if len(set(names)) != len(names):
            raise ValueError(f"convertible node names repeat: {names}")

    # -- reads ---------------------------------------------------------------------
    def statuses(self) -> dict[str, dict[str, Any]]:
        out = {}
        for node in self.nodes:
            raw = self.files.read(node_path(node.name, "status.json"))
            if raw:
                out[node.name] = json.loads(raw)
        return out

    def node_events(self, node: str) -> list[dict[str, Any]]:
        records = [json.loads(line) for line in _lines(self.files.read(node_path(node, "events.jsonl")))]
        self._events[node] = records
        return records

    def _alive(self, status: Mapping[str, Any] | None) -> bool:
        return (status is not None and status.get("role") not in (FAILED, EXITED)
                and self.clock() - float(status.get("at", 0.0)) < self.stale_s)

    # -- selection -----------------------------------------------------------------
    def plan(self) -> Selection:
        """The node to convert now by the selection rule, from the declared profiles
        and the newest status of each node (a node not alive or not in rollout is
        not a candidate and does not count as doing rollout)."""
        statuses = self.statuses()
        rolling = [n for n in self.nodes if self._alive(statuses.get(n.name))
                   and statuses[n.name].get("role") == ROLLOUT]
        candidates = [replace(n, inflight=statuses[n.name].get("inflight"),
                              capacity=statuses[n.name].get("capacity")) for n in rolling]
        return select_group(candidates, self.needs, group_size=self.group_size,
                            rollout_nodes=[n.name for n in rolling])

    def data_ended(self) -> bool:
        """Every node finished generating (or exited): no new samples will come."""
        statuses = self.statuses()
        return len(statuses) == len(self.nodes) and all(
            s.get("generation_done") or s.get("role") in (EXITED, FAILED) for s in statuses.values())

    # -- TrainerIslands ------------------------------------------------------------
    def launch(self, incarnation: int, resume_from: CutRef | None) -> _NodeJob:
        selection, self.pending = self.pending or self.plan(), None
        if not selection.group:
            raise RemoteTrainerError(f"no convertible node for incarnation {incarnation}: "
                                     f"{selection.note}")
        node = selection.group[0]
        known = len(self.node_events(node))
        NodeCommandWriter(self.files, node).send(TO_TRAINER, incarnation=incarnation,
                                                 resume=resume_from)
        job = _NodeJob(node, int(incarnation), events_seen=known)
        self.jobs.append(job)
        self._journal("convert", incarnation=int(incarnation), node=node, selection=selection)
        print(f"[launcher] trainer incarnation {incarnation}: convert node {node} "
              f"(resume {None if resume_from is None else resume_from.cut_id})", flush=True)
        return job

    def events(self, job: _NodeJob) -> list[Mapping[str, Any]]:
        records = self.node_events(job.node)
        new, job.events_seen = records[job.events_seen:], len(records)
        return [r for r in new if r.get("event") == EVENT
                and int(r.get("incarnation", -1)) == job.incarnation]

    def alive(self, job: _NodeJob) -> bool:
        return self._alive(self.statuses().get(job.node))

    def release(self, job: _NodeJob) -> bool:
        """The trainer left: wait until the node reports it is back to rollout."""
        deadline = self.clock() + self.release_timeout_s
        while True:
            for record in self.node_events(job.node):
                if record.get("event") == "node_conversion" and record.get("direction") == "to_rollout" \
                        and int(record.get("incarnation", -1)) == job.incarnation:
                    return True
            status = self.statuses().get(job.node)
            if not self._alive(status) or self.clock() >= deadline:
                return False
            self.sleep(self.poll_s)

    def release_incarnation(self, incarnation: int) -> bool:
        """A trainer the dead head started: tell it to stop (SCHEDULED cut); the
        node converts back on its own."""
        from yeto.rl.elastic_client import LeaveReason

        append_trainer_command(self.files, incarnation, STOP, reason=LeaveReason.STOPPED.name)
        return True

    def start_events(self) -> list[Mapping[str, Any]]:
        """Every event of every node read so far (``trainer_start_summary`` gives T_s)."""
        return [e for records in self._events.values() for e in records]

    def finish(self) -> None:
        """The run is over: every node exits."""
        for node in self.nodes:
            NodeCommandWriter(self.files, node.name).send(END)

    def _journal(self, action: str, *, selection: Selection, **fields: Any) -> None:
        if self.record is not None:
            self.record(CONVERSION_DECISION_KIND, tx_id=None, action=action,
                        selection=selection.record(), **fields)


class ConvertibleGate:
    """A :class:`~.trainer_lifecycle.TrainerScheduler` decorator: a ``Start`` of
    the inner scheduler (S1) holds when no convertible node may convert (none
    eligible, or it would be the last rollout node). Otherwise the selection is
    kept for the launch that follows, so the node the journal names is the node
    that converts."""

    def __init__(self, inner: Any, islands: ConvertibleIslands) -> None:
        self.inner = inner
        self.islands = islands

    def decide(self, backlog: Any, lifecycle_state: LifecycleState,
               prices: Mapping[str, float] | None = None) -> Decision:
        decision = self.inner.decide(backlog, lifecycle_state, prices)
        if not isinstance(decision, Start):
            return decision
        selection = self.islands.plan()
        if not selection.group:
            self.islands._journal("hold", selection=selection, batches=backlog.batches)
            return Hold()
        self.islands.pending = selection
        return decision

    def restore(self, records: Iterable[Mapping[str, Any]]) -> bool:
        restore = getattr(self.inner, "restore", None)
        return bool(restore(records)) if callable(restore) else False

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)
