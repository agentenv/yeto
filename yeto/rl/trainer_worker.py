"""The trainer worker: the trainer islands of the fleet's desired state, reconciled.

A fleet launched island by island (``fleet.launch: per_island``,
:mod:`yeto.rl.island_launches`) has one trainer member per island in the desired state
(``fleet/desired.json``, kind ``trainer``). :class:`TrainerWorker` runs under the rollout
controller's supervisor (``RolloutController.add_worker``), next to the pool workers:
the supervisor restarts it when its thread dies. Each :meth:`TrainerWorker.step` reads
the newest generation and, island by island (keyed by learner id):

* **new member**: once the fleet's syncer is recorded (``fleet/island_syncer.json``, written
  by the syncer launch), start its ``yeto launch`` (:func:`island_launches.launch_argv`:
  its argv, ``--gpu``, cluster prefix, ``--rl-fleet-syncer``, ``--rl-syncer-epoch``);
* **running member, same spec**: nothing. A worker that restarted (its thread died,
  or ``yeto run up`` itself restarted) finds the launch alive through its run record
  and adopts it: no second launch, no new incarnation;
* **launch ended**: the island lifecycle's decision table
  (:func:`yeto.launch.island_lifecycle.classify` of its exit code and log tail) decides.
  An infrastructure failure is relaunched within ``relaunch_budget`` (only this island:
  the other launches never notice); a configuration error or strict failure is
  ``failed`` and a normal end ``done``, neither relaunched;
* **spec changed** (``gpu``, ``argv``, ``fleet_shape``: a move to other machines or
  another source): the island's STOP (``fleet/trainers/<k>/STOP``) ends its launch
  (no RETIRE: the syncer sees the island leave and come back under its learner id,
  a short absence); once it ended, the next incarnation starts on the new machines
  under prefix ``<prefix>-m<n>``, same learner id, from the store's ``trainer<k>/``;
* **member removed**: its STOP ends its launch; once ended the island is ``removed``.
  Its sampler pools leave with it (the pool reconcile drains them and retires their
  shards, ``yeto.launch.pool.fleet_reconcile``).

The fleet's own STOP (the run's STOP flag of the fleet launch, a local file) is checked
first at every step, before anything that reads the run store, and forwarded:

* to every island launch of this host, through the launch ops (``yeto launch``'s own
  run STOP): the ones this worker started or adopted, and the ones the host's run
  registry knows under the fleet's island prefixes (``local_launches``), so a worker
  that cannot read the store, or restarted, still stops them;
* to every island's STOP in the store (``fleet/trainers/<k>/STOP``), which also ends an
  island launch of another host; a write that fails is tried again at every step (an
  error every ``unknown_alarm_steps`` steps), and a STOP that a starting launch cleared
  is written again.

When in doubt the islands stop rather than keep spending. While an island is moving
or being removed and its launch still runs, its island STOP is written again whenever
it is missing (a launch that was just starting clears its STOP once when it comes up).

Only one worker drives the islands: before each step it checks that its controller
still holds the fleet (``fleet/applied.json`` still names its writer; the
reconciler itself only looks when it applies or writes), and one whose controller was
fenced (another controller took the fleet over) does nothing more. Whether it still
holds it may not read (``unknown``): that step does nothing, a warning the first time
and an error every ``unknown_alarm_steps`` steps in a row.

The launches are processes of the host that started them; a launch recorded by another
host (``host`` in its record) is never judged gone or started again from here. Taking
the islands of a lost host over is not supported yet. ``yeto run down`` (on any host)
tears every island launch down and records each one it confirmed down as ``down``
(:func:`mark_down`); the next ``yeto run up``, on any host, starts the next incarnation
of a ``down`` island (``<prefix>-m<n>``, same learner id, from the store's
``trainer<k>/``).

What the worker did is kept per island in ``fleet/trainers/<k>/launch.json``
(:data:`LAUNCH_RECORD`): name, prefix and incarnation, the spec it launched, state,
relaunches, the generation applied, and its billing: ``billed_s`` and ``billed_usd``
of the incarnations that ended (each at its own price), ``unpriced_s`` of the ones of
them that ran without a price (not in ``billed_usd``), ``usd_per_h`` and
``started_at`` of the one that runs. :func:`billed` is what the fleet's spend counts for its trainer islands, added
and removed ones too. The record is what a restarted worker adopts from;
the island launch itself keeps the lifecycle record ``state.json`` next to it.

The launches are processes of this host (:class:`LocalLaunchOps`, ``yeto launch
--controller local``), found again through yeto's run registry. Tests pass fake ops.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from yeto.launch import island_lifecycle as lifecycle
from yeto.launch.pool import desired as dz
from yeto.rl import island_launches as il

LAUNCH_RECORD = "launch.json"
SCHEMA = "yeto.fleet.trainer-launch/v1"
STARTING, RUNNING, MOVING, STOPPING = "starting", "running", "moving", "stopping"
REMOVED = "removed"
DOWN = "down"  # torn down by `yeto run down`: the next `yeto run up` starts its next incarnation
DONE, FAILED, ABANDONED = lifecycle.DONE, lifecycle.FAILED, lifecycle.ABANDONED
TERMINAL = frozenset({DONE, FAILED, ABANDONED, REMOVED})
DEFAULT_RELAUNCH_BUDGET = 3
START_GRACE_S = 600.0  # a launch just started may take this long to register its run
# yeto.launcher HANDS_OFF_EXITS: another live launcher holds the island's records lock (12),
# or whether its machine runs did not read (13); the launch ordered nothing and tore nothing down
HANDS_OFF_EXITS = frozenset({12, 13})


def launch_key(learner_id: int) -> str:
    return f"{lifecycle.STATE_DIR}/{int(learner_id)}/{LAUNCH_RECORD}"


def save_record(objects: Any, learner_id: int, record: dict[str, Any], now: float) -> dict[str, Any]:
    record = dict(record, schema=SCHEMA, learner_id=int(learner_id), updated=round(now, 3))
    objects.write(launch_key(learner_id), json.dumps(record, sort_keys=True).encode())
    return record


def closed(record: dict[str, Any] | None, now: float) -> dict[str, Any]:
    """``record`` with its running incarnation's time added to ``billed_s`` (once): to
    ``billed_usd`` at its price, or to ``unpriced_s`` without one."""
    record = dict(record or {})
    if record.get("started_at") is not None and record.get("closed_at") is None:
        seconds = max(0.0, now - float(record["started_at"]))
        record["billed_s"] = round(float(record.get("billed_s") or 0) + seconds, 3)
        price = record.get("usd_per_h")
        if price is None:  # its dollars are not known: counted apart (the spend prices it)
            record["unpriced_s"] = round(float(record.get("unpriced_s") or 0) + seconds, 3)
        else:
            record["billed_usd"] = round(float(record.get("billed_usd") or 0) + seconds / 3600.0 * float(price), 6)
        record["closed_at"] = round(now, 3)
    return record


def spec_of(member: dict[str, Any]) -> str:
    """What a change of moves the island: its machines, argv and fleet shape."""
    body = {"gpu": member.get("gpu"), "argv": member.get("argv"), "fleet_shape": member.get("fleet_shape")}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]


class LaunchOps(Protocol):
    def alive(self, prefix: str) -> bool: ...
    def start(self, prefix: str, argv: list[str]) -> int | None: ...
    def ended(self, prefix: str) -> dict[str, Any] | None: ...
    def request_stop(self, prefix: str, why: str) -> None: ...


@dataclass
class TrainerWorker:
    """The trainer members of the desired state, reconciled (module doc)."""

    desired: dz.DesiredStore
    objects: Any  # the run store's object API (launch records, the syncer record)
    ops: LaunchOps
    states: lifecycle.IslandStateStore
    fleet_stop: Callable[[], str | None] = lambda: None
    # a syncer record older than this is an earlier run's (a number, or a callable read each step)
    syncer_since: float | Callable[[], float | None] | None = None
    relaunch_budget: int = DEFAULT_RELAUNCH_BUDGET
    log: Callable[[str], None] = print
    clock: Callable[[], float] = time.time
    # why this worker's controller drives the fleet no more (None: it does)
    fenced: Callable[[], str | None] = lambda: None
    host: str = field(default_factory=socket.gethostname)  # recorded with each launch it starts
    # the island launches this host's run registry knows under the fleet's island prefixes (no store read)
    local_launches: Callable[[], Any] = lambda: ()
    unknown_alarm_steps: int = 20  # steps in a row a store failure lasts before an ERROR (and every so many after)
    steps: int = 0
    _stop_forwarded: set[str] = field(default_factory=set)
    _stop_written: set[int] = field(default_factory=set)
    _seen: set[str] = field(default_factory=set)  # island launch prefixes this worker started or adopted
    _waiting_logged: bool = False
    _fenced_logged: bool = False
    _unknown_steps: int = 0
    _stop_failures: int = 0
    _foreign_logged: set[int] = field(default_factory=set)

    # -- records
    def load(self, learner_id: int) -> dict[str, Any] | None:
        raw = self.objects.read(launch_key(learner_id))
        if raw is None:
            return None
        try:
            doc = json.loads(raw)
        except ValueError:
            return None
        return doc if isinstance(doc, dict) else None

    def save(self, learner_id: int, record: dict[str, Any]) -> dict[str, Any]:
        return save_record(self.objects, learner_id, record, self.clock())

    def known(self) -> list[int]:
        """Learner ids with a launch record."""
        ids = set()
        for key in self.objects.list(f"{lifecycle.STATE_DIR}/"):
            parts = str(key).split("/")
            if len(parts) >= 4 and parts[-1] == LAUNCH_RECORD and parts[-2].isdigit():
                ids.add(int(parts[-2]))
        return sorted(ids)

    def syncer(self) -> dict[str, Any] | None:
        record = il.read_syncer_record(self.objects)
        if record is None:
            return None
        since = self.syncer_since() if callable(self.syncer_since) else self.syncer_since
        if since is not None and float(record.get("at") or 0) < float(since):
            return None  # an earlier run's syncer
        return record

    def _event(self, learner_id: int, what: str, **fields: Any) -> None:
        self.log(f"fleet: trainer island {learner_id}: {what}"
                 + "".join(f", {k} {v}" for k, v in fields.items() if v is not None))

    # -- one pass
    def _alarm(self, count: int) -> str | None:
        """The level of a failure seen ``count`` steps in a row: a warning the first time,
        an error every ``unknown_alarm_steps`` steps; None (not logged) in between."""
        if count == 1:
            return "WARNING"
        return "ERROR" if count % max(1, self.unknown_alarm_steps) == 0 else None

    def step(self) -> None:
        self.steps += 1
        stop = self.fleet_stop()  # a local flag: read before anything that needs the store
        if stop:
            self._forward_stop(stop)
            return
        fenced = self.fenced()
        if fenced:
            if fenced.startswith("unknown"):  # not known this step: nothing is done until it is
                self._unknown_steps += 1
                level = self._alarm(self._unknown_steps)
                if level:
                    self.log(f"fleet: {level}: trainer worker: whether this controller still drives the fleet is "
                             f"{fenced}; nothing is started, stopped or relaunched until it reads "
                             f"({self._unknown_steps} step(s) in a row)")
                return
            if not self._fenced_logged:
                self._fenced_logged = True
                self.log(f"fleet: CRITICAL: trainer worker: {fenced}: it starts, stops and relaunches no island "
                         "launch any more")
            return
        self._unknown_steps = 0
        doc = self.desired.read()
        members = {int(m["learner_id"]): m for m in dz.trainers(doc).values()}
        syncer = self.syncer()
        if syncer is None and members and not self._waiting_logged:
            self._waiting_logged = True
            self.log("fleet: trainer islands wait for the fleet's syncer (fleet/island_syncer.json)")
        generation = None if doc is None else int(doc.get("generation") or 0)
        for k in sorted(members):
            self._reconcile(k, members[k], syncer, generation)
        for k in self.known():
            if k not in members:
                self._retire(k)

    def _forward_stop(self, why: str) -> None:
        """The fleet STOP to every island launch (module doc): this host's through the
        launch ops first (no store read), then each island's STOP in the store."""
        local = set(self._seen)
        try:
            local |= {str(p) for p in self.local_launches() or ()}
        except Exception as error:  # noqa: BLE001 - the launches this worker knows still stop
            self.log(f"fleet: WARNING: trainer worker: the run registry was not read ({type(error).__name__}: "
                     f"{error}); the fleet stop goes to the {len(local)} island launch(es) this worker knows")
        for prefix in sorted(local - self._stop_forwarded):
            if self.ops.alive(prefix):
                self._stop_forwarded.add(prefix)
                self.ops.request_stop(prefix, f"fleet stop: {why}")
                self.log(f"fleet: fleet stop forwarded to island launch {prefix} ({why})")
        try:
            for k in self.known():
                record = self.load(k)
                if record is None or record.get("state") in TERMINAL or record.get("state") == DOWN:
                    continue
                prefix = str(record.get("prefix_now"))
                if (record.get("host") in (None, self.host) and prefix not in self._stop_forwarded
                        and self.ops.alive(prefix)):  # one this worker did not know yet (it restarted)
                    self._stop_forwarded.add(prefix)
                    self.ops.request_stop(prefix, f"fleet stop: {why}")
                    self._event(k, "fleet stop forwarded to its launch", prefix=prefix)
                if self.states.stop_requested(k) is None:  # missing, or cleared by a launch that was starting
                    self.states.request_stop(k, f"fleet stop: {why}", by="trainer worker")
                    if k not in self._stop_written:
                        self._event(k, "fleet stop written to its island STOP", prefix=record.get("prefix_now"))
                    self._stop_written.add(k)
        except Exception as error:  # noqa: BLE001 - tried again at the next step
            self._stop_failures += 1
            level = self._alarm(self._stop_failures)
            if level:
                self.log(f"fleet: {level}: trainer worker: the fleet stop was not written to the islands' STOPs in "
                         f"the store ({type(error).__name__}: {error}; {self._stop_failures} step(s) in a row); "
                         f"this host's {len(self._stop_forwarded)} island launch(es) were stopped directly, "
                         "the store is tried again at the next step")
            return
        self._stop_failures = 0

    def _closed(self, record: dict[str, Any] | None) -> dict[str, Any]:
        return closed(record, self.clock())

    def _start(self, k: int, member: dict[str, Any], syncer: dict[str, Any], record: dict[str, Any] | None, *,
               incarnation: int, relaunches: int, generation: int | None, why: str) -> dict[str, Any]:
        prefix = il.island_prefix(str(member["prefix"]), incarnation)
        self.states.clear_stop(k)  # a STOP of the incarnation before (a move) is void for this one
        argv = il.launch_argv(member, incarnation=incarnation, syncer=syncer)
        pid = self.ops.start(prefix, argv)
        self._seen.add(prefix)
        self._event(k, f"launch started ({why})", name=member.get("name"), prefix=prefix, pid=pid)
        record = self._closed(record) if record else record
        return self.save(k, {**(record or {}), "name": member.get("name"), "prefix": member["prefix"],
                             "prefix_now": prefix, "incarnation": incarnation, "spec": spec_of(member),
                             "gpu": member.get("gpu"), "state": STARTING, "relaunches": relaunches,
                             "host": self.host, "usd_per_h": member.get("usd_per_h"), "closed_at": None, "argv": argv,
                             "generation": generation, "pid": pid, "started_at": round(self.clock(), 3),
                             "why": why, "ended": None,
                             "last_end": (record or {}).get("ended") or (record or {}).get("last_end")})

    def _reconcile(self, k: int, member: dict[str, Any], syncer: dict[str, Any] | None,
                   generation: int | None) -> None:
        record = self.load(k)
        spec = spec_of(member)
        if record is None or record.get("state") in (REMOVED, DOWN):
            if syncer is not None:
                incarnation = 0 if record is None else int(record.get("incarnation") or 0) + 1
                why = ("added" if record is None else
                       "started again after `yeto run down`" if record.get("state") == DOWN else "added again")
                self._start(k, member, syncer, record, incarnation=incarnation, relaunches=0,
                            generation=generation, why=why)
            return
        prefix = str(record["prefix_now"])
        state = record.get("state")
        if self._foreign(k, record):
            return
        alive = self.ops.alive(prefix)
        if alive:
            self._seen.add(prefix)
        if state == MOVING:
            if alive:  # its STOP is ending the launch on the old machines
                self._assert_stop(k, f"moving to {member.get('gpu')}")
                return
            if syncer is None:
                return
            self._start(k, member, syncer, record, incarnation=int(record.get("incarnation") or 0) + 1,
                        relaunches=0, generation=generation, why=f"moved to {member.get('gpu')}")
            return
        if alive:
            if record.get("spec") != spec:
                self.states.request_stop(k, f"moving to {member.get('gpu')}", by="trainer worker")
                self.save(k, {**record, "state": MOVING, "moving_to": member.get("gpu"), "generation": generation})
                self._event(k, "moving: its launch stops on the old machines first", to=member.get("gpu"))
                return
            if state != RUNNING:  # up
                self.save(k, {**record, "state": RUNNING})
                if state != STARTING:
                    self._event(k, "launch adopted (already running)", prefix=prefix)
            return
        if state in TERMINAL:
            if record.get("spec") != spec and syncer is not None:  # the operator changed a finished island
                self._start(k, member, syncer, record, incarnation=int(record.get("incarnation") or 0) + 1,
                            relaunches=0, generation=generation, why="changed after it ended")
            return
        ended = self.ops.ended(prefix)
        if ended is None and state == STARTING and self.clock() - float(record.get("started_at") or 0) < START_GRACE_S:
            return  # just started: its run is not registered yet
        ended = ended or {"exit_code": None, "log": "", "detail": "launch gone without a run record"}
        if ended.get("exit_code") in HANDS_OFF_EXITS:
            self._hand_back(k, record, ended)
            return
        stop = self.states.stop_requested(k)
        decision = lifecycle.classify(exit_code=ended.get("exit_code"), log_text=ended.get("log"),
                                      stopped=stop is not None, detail=str(ended.get("detail") or ""))
        relaunches = int(record.get("relaunches") or 0)
        if decision.relaunch and relaunches < self.relaunch_budget and syncer is not None:
            self._start(k, member, syncer, {**record, "ended": ended}, incarnation=int(record.get("incarnation") or 0),
                        relaunches=relaunches + 1, generation=generation,
                        why=f"relaunch {relaunches + 1}/{self.relaunch_budget}: {decision.cause} {decision.detail}"[:200])
            return
        final = decision.state if not decision.relaunch else ABANDONED
        self.save(k, {**self._closed(record), "state": final, "ended": ended, "decision": decision.to_json()})
        self._event(k, f"launch ended: {final}", cause=decision.cause, detail=decision.detail[:200] or None)

    def _hand_back(self, k: int, record: dict[str, Any], ended: dict[str, Any]) -> None:
        """The launch handed off (:data:`HANDS_OFF_EXITS`): another launcher holds island
        ``k``'s records lock, or its machine's state did not read. It ordered nothing; it
        is started again (the lock asked again)."""
        self.save(k, {**record, "state": STARTING, "started_at": round(self.clock(), 3), "ended": ended})
        self._event(k, f"its launch handed off ({ended.get('detail')}); started again at the next step")
        self.ops.start(str(record["prefix_now"]), list(record.get("argv") or ()))

    def _foreign(self, k: int, record: dict[str, Any]) -> bool:
        """Island ``k``'s launch runs on another host (not terminal there): this worker
        cannot see it, so it neither adopts, judges nor starts it."""
        host = record.get("host")
        if not host or host == self.host or record.get("state") in TERMINAL or record.get("state") == DOWN:
            return False
        if k not in self._foreign_logged:
            self._foreign_logged.add(k)
            self.log(f"fleet: ERROR: trainer island {k}: its launch {record.get('prefix_now')} was started on host "
                     f"{host}, not this one ({self.host}); it is left alone (no second launch under learner id {k}). "
                     f"Run `yeto run up` on {host} again. Taking a lost host's islands over is not supported yet: "
                     f"`yeto run down --config <run.json>` (any host) tears them down and records them down; once "
                     "none of the run's launches runs, `yeto run up` here starts them again")
        return True

    def _assert_stop(self, k: int, why: str) -> None:
        """Island ``k``'s STOP, written again when it is missing (module doc)."""
        if self.states.stop_requested(k) is None:
            self.states.request_stop(k, why, by="trainer worker")
            self._event(k, "island STOP written again (its launch cleared it as it started)")

    def _retire(self, k: int) -> None:
        record = self.load(k)
        if record is None or record.get("state") == REMOVED:
            return
        if record.get("state") == DOWN:  # torn down: nothing runs
            self.save(k, {**record, "state": REMOVED, "removed_at": round(self.clock(), 3)})
            self._event(k, "removed (it was down)")
            return
        if self._foreign(k, record):
            return
        prefix = str(record["prefix_now"])
        if self.ops.alive(prefix):
            if record.get("state") != STOPPING:
                self.states.request_stop(k, "removed from the fleet's desired state", by="trainer worker")
                self.save(k, {**record, "state": STOPPING})
                self._event(k, "removed: its launch stops", prefix=prefix)
            else:
                self._assert_stop(k, "removed from the fleet's desired state")
            return
        self.states.clear_stop(k)
        self.save(k, {**self._closed(record), "state": REMOVED, "removed_at": round(self.clock(), 3)})
        self._event(k, "removed: its launch ended", prefix=prefix)


def billed(objects: Any, now: float | None = None) -> list[tuple[int, float, float, float]]:
    """``(learner id, billed hours, known usd, unpriced hours)`` of every trainer island
    launch record: the incarnations that ended, each at the price it ran at, plus the one
    that runs up to ``now`` at its own; the hours of incarnations without a price are in
    ``unpriced hours`` (and in the billed hours), not in the dollars."""
    now = time.time() if now is None else now
    out = []
    for key in sorted(objects.list(f"{lifecycle.STATE_DIR}/")):
        parts = str(key).split("/")
        if not (len(parts) >= 4 and parts[-1] == LAUNCH_RECORD and parts[-2].isdigit()):
            continue
        try:
            record = json.loads(objects.read(key) or b"null")
        except ValueError:
            continue
        if not isinstance(record, dict):
            continue
        seconds = float(record.get("billed_s") or 0)
        usd = float(record.get("billed_usd") or 0)
        unpriced = float(record.get("unpriced_s") or 0)
        if record.get("started_at") is not None and record.get("closed_at") is None:
            running = max(0.0, now - float(record["started_at"]))
            seconds += running
            price = record.get("usd_per_h")
            if price is None:
                unpriced += running
            else:
                usd += running / 3600.0 * float(price)
        out.append((int(parts[-2]), seconds / 3600.0, usd, unpriced / 3600.0))
    return out


def mark_down(objects: Any, prefixes: Any, *, log: Callable[[str], None] = print,
              clock: Callable[[], float] = time.time) -> list[int]:
    """``yeto run down`` confirmed the island launches ``prefixes`` torn down: each
    island whose launch runs under one of them (not ended) is recorded ``down`` and its
    incarnation closed for the spend; the next ``yeto run up``, on any host, starts its
    next incarnation. Returns the learner ids recorded down."""
    gone, out = {str(p) for p in prefixes}, []
    for key in sorted(objects.list(f"{lifecycle.STATE_DIR}/")):
        parts = str(key).split("/")
        if not (len(parts) >= 4 and parts[-1] == LAUNCH_RECORD and parts[-2].isdigit()):
            continue
        try:
            record = json.loads(objects.read(key) or b"null")
        except ValueError:
            continue
        if (not isinstance(record, dict) or record.get("state") in TERMINAL or record.get("state") == DOWN
                or str(record.get("prefix_now")) not in gone):
            continue
        k, now = int(parts[-2]), clock()
        save_record(objects, k, {**closed(record, now), "state": DOWN, "down_at": round(now, 3)}, now)
        log(f"fleet: trainer island {k}: its launch {record.get('prefix_now')} is down; the next `yeto run up` "
            "starts it again")
        out.append(k)
    return out


# -- the launches as processes of this host ------------------------------------------------------

@dataclass
class LocalLaunchOps:
    """``yeto launch --controller local`` processes, found again through yeto's run
    registry (``yeto.runs``): a launch is alive while its worker (or the submitting CLI
    that has not registered the run yet) runs."""

    python: str = sys.executable
    env: dict[str, str] | None = None
    log_dir: Path | None = None
    procs: dict[str, subprocess.Popen] = field(default_factory=dict)

    def _log_path(self, prefix: str) -> Path:
        base = Path(self.log_dir) if self.log_dir is not None else Path("~/.yeto/islands").expanduser()
        base.mkdir(parents=True, exist_ok=True)
        return base / f"launch-{prefix}.log"

    def alive(self, prefix: str) -> bool:
        from yeto import runs  # noqa: PLC0415

        proc = self.procs.get(prefix)
        if proc is not None and proc.poll() is None:
            return True
        if runs.worker_pids(prefix):
            return True
        meta = runs.load_run(prefix)
        return bool(meta) and runs.is_alive(meta.get("pid"))

    def start(self, prefix: str, argv: list[str]) -> int | None:
        path = self._log_path(prefix)
        with open(path, "a") as fh:
            fh.write(f"\n### {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} yeto launch {' '.join(argv)}\n")
            fh.flush()
            proc = subprocess.Popen([self.python, "-m", "yeto.cli", "launch", *argv], env=self.env, stdout=fh,
                                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        self.procs[prefix] = proc
        return proc.pid

    def ended(self, prefix: str) -> dict[str, Any] | None:
        from yeto import runs  # noqa: PLC0415

        meta = runs.load_run(prefix)
        tail = _tail(self._log_path(prefix))
        if meta is not None and meta.get("state") not in (runs.PENDING, runs.RUNNING):
            return {"exit_code": meta.get("exit_code", 0 if meta.get("state") == runs.SUCCEEDED else 1),
                    "state": meta.get("state"), "log": _tail(runs.log_path(prefix)) + "\n" + tail,
                    "detail": str(meta.get("stop_reason") or meta.get("state"))}
        proc = self.procs.get(prefix)
        if proc is not None and proc.poll() is not None:
            # the CLI refused before it registered the run: before anything was ordered (its
            # checks: a fleet shape that differs, a bad flag): a configuration error, never
            # relaunched (moving or changing the island again starts it again)
            from yeto.config_errors import CONFIG_ERROR_EXIT  # noqa: PLC0415

            code = proc.returncode
            if code in HANDS_OFF_EXITS:  # another launcher holds the island, or its state did not read
                return {"exit_code": code, "log": tail, "detail": f"yeto launch handed off (exit {code})"}
            return {"exit_code": CONFIG_ERROR_EXIT if code else 0, "log": tail,
                    "detail": f"yeto launch refused before ordering anything (exit {code})"}
        if meta is not None and meta.get("state") in (runs.PENDING, runs.RUNNING):
            return {"exit_code": None, "log": _tail(runs.log_path(prefix)), "detail": "worker exited without a result"}
        return None

    def request_stop(self, prefix: str, why: str) -> None:
        from yeto import runs  # noqa: PLC0415

        runs.request_stop(prefix)


def _tail(path: os.PathLike[str] | str, size: int = 20_000) -> str:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - size))
            return fh.read().decode(errors="replace")
    except OSError:
        return ""


def seed_trainers(store: dz.DesiredStore, members: list[dict[str, Any]], *, seed: dict[str, Any] | None = None
                  ) -> dict[str, Any]:
    """Add the run config's trainer members to the desired state once (none there yet):
    a new generation; a desired state that already has trainer members is kept."""
    current = store.read()
    if current is not None and dz.trainers(current):
        return current

    def change(doc: dict[str, Any]) -> None:
        if dz.trainers(doc):
            return
        doc.setdefault("members", []).extend(members)

    return store.update(change, seed=seed, note="trainer islands seeded from the run config", attempts=3)
