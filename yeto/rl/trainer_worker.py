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
  under prefix ``<prefix>-m<n>``, same learner id, from the store's ``trainer<k>/``.
  A change applied while the launch was already ending (an infrastructure failure
  ordering the old machines) is a move too: the next incarnation, its relaunch count
  reset; only a launch whose spec is unchanged is relaunched within the budget;
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
A launch that reads no STOP yet (still ordering its machines, e.g. a sky launch waiting
for SSH) is torn down :data:`STOP_UNREAD_S` after its STOP, any other one that still runs
:data:`STOP_IGNORED_S` after it (``ops.tear_down``: ``yeto down``), so a move or a removal
never waits for a provisioning timeout. A launch being relaunched on its old machines
(after an infrastructure failure) orders them again: it is torn down the same way, which
is what a move or a removal wants.

A launch recorded by another control host is left alone (an error naming that host), except
under an on-site control plane (``takeover``, ``fleet.syncer_service``): there a launch of a
lost host is taken over (:meth:`TrainerWorker._take_over`) once the fleet's syncer is known,
its records lock's lease ran out ``takeover_margin_s`` ago, and this host claims that lock
(``StoreLock.acquire``: the new token fences the old launch, whose next renewal fails); only
an SSH or Modal island (:data:`TAKEOVER_SOURCES`; a sky cluster lives in the other host's sky
state). An island whose launch ended there (its own island record ``done`` / ``failed`` /
``abandoned``) is closed as that host would have closed it, nothing started; a moving or
stopping one goes on with its move or removal. Otherwise the same launch is started again
(same prefix and incarnation) as a takeover launch (:data:`TAKEOVER_FLAG`: it never starts the
island afresh, never clears its STOP), counted against the relaunch budget, and adopts the
running island. A moving or stopping island keeps its STOP: its launch here reattaches, reads
the STOP and releases the island before the move or the removal goes on; a takeover launch
that cannot take its island over (:data:`TAKEOVER_BLOCKED_MARKER`) blocks the island
(``island_takeover_blocked``): its records stay, no new incarnation starts.

Only one worker drives the islands: before each step it checks that its controller
still holds the fleet (``fleet/applied.json`` still names its writer; the
reconciler itself only looks when it applies or writes), and one whose controller was
fenced (another controller took the fleet over) does nothing more. Whether it still
holds it may not read (``unknown``): that step does nothing, a warning the first time
and an error every ``unknown_alarm_steps`` steps in a row.

The launches are processes of the host that started them; a launch recorded by another
host (``host`` in its record) is never judged gone or started again from here, except
by an on-site control plane's takeover (above). ``yeto run down`` (on any host)
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
HANDOFF_LIMIT = 3  # launches in a row that hand off (exit 12/13) before the island is given up
START_GRACE_S = 600.0  # a launch just started may take this long to register its run
TAKEOVER_MARGIN_S = 60.0  # yeto.rl.fleet.DEFAULT_TAKEOVER_MARGIN_S
TAKEOVER_LOCK_TTL_S = 120.0  # the takeover's own claim of the records lock (released at once)
# the line an island launch that may only reattach prints when it cannot (yeto.launch.ssh_resident)
TAKEOVER_BLOCKED_MARKER = "[yeto-takeover-blocked]"
# the launch option of a launch that takes a lost control host's island over (yeto.launcher):
# it never starts its island afresh, never clears its STOP, and refuses an island record that ended
TAKEOVER_FLAG = "--rl-fleet-takeover"
# A launch that still orders its machines this long after its island STOP was written (it
# reads no STOP before they are up: a sky launch waiting for SSH) is torn down
STOP_UNREAD_S = 90.0
# Any launch still running this long after its island STOP was written is torn down
STOP_IGNORED_S = 1800.0
# A move whose old launch ended leaving resources open waits this long for their teardown
# (`yeto down` of the old prefix) before the next incarnation orders anything: a leftover
# machine would hold the target's quota and fail the new order as an infrastructure failure.
CLEAR_TIMEOUT_S = 600.0
# The resource ledger kinds that never hold a move: they hold no quota and bill nothing
# (yeto.cost.ledger: ``sky_storage`` is "Not compute: billed 0"). A bucket left open on purpose
# (``release_skipped``: it holds data, a person decides) or one cleanup could not confirm
# (``release_failed``) is reported, not waited for. Every other kind, known or not (a provider's
# lease kind, one added later), holds a move while it is open: an unknown kind is taken as
# compute rather than ordering new machines next to a live one.
NON_BLOCKING_LEDGER_KINDS = frozenset({"sky_storage"})


def split_open_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(rows a move waits for, rows it only reports) of an old launch's open ledger rows. A
    row is only reported when it was left open on purpose (``release_skipped``: `yeto down`
    never releases it, waiting would only time out) or its kind is non-billing
    (:data:`NON_BLOCKING_LEDGER_KINDS`); every other row waits, unknown kinds included."""
    waits = [r for r in rows if r.get("kind") not in NON_BLOCKING_LEDGER_KINDS
             and not str(r.get("release_error") or "").startswith("release_skipped")]
    return waits, [r for r in rows if r not in waits]


def _row_names(rows: list[dict[str, Any]]) -> str:
    return ", ".join(sorted({f"{r.get('kind')} {r.get('id') or r.get('name')}"
                             + (f" ({r['release_error']})" if r.get("release_error") else "") for r in rows})[:5])
# yeto.launcher HANDS_OFF_EXITS: another live launcher holds the island's records lock (12),
# or whether its machine runs did not read (13); the launch ordered nothing and tore nothing down
HANDS_OFF_EXITS = frozenset({12, 13})


def launch_key(learner_id: int) -> str:
    return f"{lifecycle.STATE_DIR}/{int(learner_id)}/{LAUNCH_RECORD}"


def save_record(objects: Any, learner_id: int, record: dict[str, Any], now: float) -> dict[str, Any]:
    record = dict(record, schema=SCHEMA, learner_id=int(learner_id), updated=round(now, 3))
    objects.write(launch_key(learner_id), json.dumps(record, sort_keys=True).encode())
    return record


def _last_flag(argv: Any, flag: str) -> float | None:
    """The number after the last ``flag`` (``flag N`` or ``flag=N``) of ``argv`` (None: absent or not a number)."""
    value = None
    items = [str(a) for a in (argv or ())]
    for i, a in enumerate(items):
        if a == flag and i + 1 < len(items):
            value = items[i + 1]
        elif a.startswith(flag + "="):
            value = a.split("=", 1)[1]
    try:
        return None if value is None else float(value)
    except ValueError:
        return None


def table_usd_per_h(gpu: Any, argv: Any = None) -> float | None:
    """The unified price table's USD/h of an island launch's ``--gpu`` (every node, from a
    billing row of :mod:`yeto.cost.prices`); None when the table has no billing row for it.
    A Modal node also bills the CPU and memory it requests: ``--modal-cpu`` /
    ``--modal-memory-gib`` / ``--modal-ephemeral-disk-gib`` of the launch's ``argv``, else
    the per-GPU default request (:func:`yeto.cost.billing.default_modal_host`)."""
    from yeto.cost import prices  # noqa: PLC0415
    from yeto.gpu_spec import parse_gpu_spec  # noqa: PLC0415

    try:
        specs = parse_gpu_spec(str(gpu))
    except Exception:  # noqa: BLE001 - not a cluster spec: not priced here
        return None
    total = 0.0
    for spec in specs:
        node = prices.node_usd_h(str(spec.cloud), spec.gpu, spec.gpus_per_node, region=spec.region)
        if node is None:
            return None
        if str(spec.cloud).lower() == "modal":
            from yeto.cost.billing import default_modal_host  # noqa: PLC0415

            dcpu, dmem = default_modal_host(spec.gpus_per_node)
            cpu, mem = _last_flag(argv, "--modal-cpu"), _last_flag(argv, "--modal-memory-gib")
            hostpart = prices.modal_host_usd_h(dcpu if cpu is None else cpu, dmem if mem is None else mem,
                                               _last_flag(argv, "--modal-ephemeral-disk-gib"))
            if hostpart is None:
                return None  # no host.modal row: the GPU line alone would undercount
            node += hostpart
        total += node * int(spec.num_nodes)
    return total


def incarnation_price(record: dict[str, Any]) -> float | None:
    """The price of the incarnation that runs in ``record``: its member's ``usd_per_h``,
    else the unified price table's for its ``--gpu`` (:func:`table_usd_per_h`); None when
    neither prices it."""
    price = record.get("usd_per_h")
    return float(price) if price is not None else table_usd_per_h(record.get("gpu"), record.get("argv"))


def closed(record: dict[str, Any] | None, now: float) -> dict[str, Any]:
    """``record`` with its running incarnation's time added to ``billed_s`` (once): to
    ``billed_usd`` at its price (:func:`incarnation_price`), or to ``unpriced_s`` without one."""
    record = dict(record or {})
    if record.get("started_at") is not None and record.get("closed_at") is None:
        seconds = max(0.0, now - float(record["started_at"]))
        record["billed_s"] = round(float(record.get("billed_s") or 0) + seconds, 3)
        price = incarnation_price(record)
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
    def tear_down(self, prefix: str, why: str) -> None: ...


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
    # an island given up for good (FAILED / ABANDONED): its own sampler pools are released
    # (``Run.stop_island_pools``); called with its learner id, its member and why
    on_given_up: Callable[[int, dict[str, Any], str], None] | None = None
    steps: int = 0
    _stop_forwarded: set[str] = field(default_factory=set)
    _stop_written: set[int] = field(default_factory=set)
    _seen: set[str] = field(default_factory=set)  # island launch prefixes this worker started or adopted
    _waiting_logged: bool = False
    _fenced_logged: bool = False
    _unknown_steps: int = 0
    _stop_failures: int = 0
    _stop_write_failed: dict[int, int] = field(default_factory=dict)  # island -> STOP writes failed in a row
    _foreign_logged: set[int] = field(default_factory=set)
    # an on-site control plane (``fleet.syncer_service``): the launches of a lost control host are
    # taken over (:meth:`_take_over`); off: a launch of another host is left alone, as before
    takeover: bool = False
    takeover_margin_s: float = TAKEOVER_MARGIN_S  # past the lost launch's lease end (clock skew)
    on_event: Callable[[dict[str, Any]], None] | None = None  # the run's tape (``island_takeover_blocked``)

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
        # an on-site control plane: a control host another one replaced forwards no fleet STOP either
        fenced = self.fenced() if self.takeover else None
        stop = None if fenced else self.fleet_stop()  # a local flag: read before anything that needs the store
        if stop:
            self._forward_stop(stop)
            return
        if not self.takeover:
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
                self._retire(k, syncer)

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
        failed: list[str] = []
        try:
            islands = self.known()
        except Exception as error:  # noqa: BLE001 - tried again at the next step
            islands, failed = [], [f"the island records were not listed ({type(error).__name__}: {error})"]
        for k in islands:  # each island on its own: one that fails never keeps the next from its STOP
            try:
                self._forward_stop_to(k, why)
            except Exception as error:  # noqa: BLE001 - this island again at the next step
                failed.append(f"island {k}: {type(error).__name__}: {error}")
        if failed:
            self._stop_failures += 1
            level = self._alarm(self._stop_failures)
            if level:
                self.log(f"fleet: {level}: trainer worker: the fleet stop was not written to every island's STOP in "
                         f"the store ({'; '.join(failed)[:500]}; {self._stop_failures} step(s) in a row); this "
                         f"host's {len(self._stop_forwarded)} island launch(es) were stopped directly, the others "
                         "got theirs, the failed ones are tried again at the next step")
            return
        self._stop_failures = 0

    def _forward_stop_to(self, k: int, why: str) -> None:
        """The fleet STOP to island ``k`` (:meth:`_forward_stop`): its launch through the
        launch ops when this worker did not know it yet, and its STOP in the store. An
        island given up (``abandoned``) gets it too: a launcher may still hold it."""
        record = self.load(k)
        if record is None or record.get("state") in (DONE, FAILED, REMOVED, DOWN):
            return
        prefix = str(record.get("prefix_now"))
        if (record.get("host") in (None, self.host) and prefix not in self._stop_forwarded
                and self.ops.alive(prefix)):  # one this worker did not know yet (it restarted)
            self._stop_forwarded.add(prefix)
            self.ops.request_stop(prefix, f"fleet stop: {why}")
            self._event(k, "fleet stop forwarded to its launch", prefix=prefix)
        if self.states.stop_requested(k) is None:  # missing, or cleared by a launch that was starting
            if not self._write_stop(k, f"fleet stop: {why}", quiet=True):
                raise OSError(f"its STOP was not written")
            if k not in self._stop_written:
                self._event(k, "fleet stop written to its island STOP", prefix=record.get("prefix_now"))
            self._stop_written.add(k)

    def _write_stop(self, k: int, why: str, *, quiet: bool = False) -> bool:
        """Island ``k``'s STOP written to the store; False when the write failed (the store
        swallows the error): a warning the first time, an error every
        ``unknown_alarm_steps`` failures in a row, unless ``quiet`` (the caller reports)."""
        if self.states.request_stop(k, why, by=lifecycle.STOP_BY_TRAINER_WORKER):
            self._stop_write_failed.pop(k, None)
            return True
        failed = self._stop_write_failed[k] = self._stop_write_failed.get(k, 0) + 1
        level = None if quiet else self._alarm(failed)
        if level:
            self.log(f"fleet: {level}: trainer island {k}: its STOP ({why}) was not written to the store "
                     f"({failed} time(s) in a row); written again at the next step")
        return False

    def _closed(self, record: dict[str, Any] | None) -> dict[str, Any]:
        return closed(record, self.clock())

    def _start(self, k: int, member: dict[str, Any], syncer: dict[str, Any], record: dict[str, Any] | None, *,
               incarnation: int, relaunches: int, generation: int | None, why: str,
               clear_stop: bool = True, takeover: bool = False) -> dict[str, Any]:
        """Start island ``k``'s launch. ``takeover``: it takes a lost control host's launch over
        (:data:`TAKEOVER_FLAG` on its command line, not in the recorded argv; its STOP is never
        cleared)."""
        prefix = il.island_prefix(str(member["prefix"]), incarnation)
        if clear_stop and not takeover:
            self.states.clear_stop(k)  # a STOP of the incarnation before (a move) is void for this one
        argv = il.launch_argv(member, incarnation=incarnation, syncer=syncer)
        pid = self.ops.start(prefix, [*argv, TAKEOVER_FLAG] if takeover else argv)
        self._seen.add(prefix)
        self._event(k, f"launch started ({why})", name=member.get("name"), prefix=prefix, pid=pid)
        record = self._closed(record) if record else record
        return self.save(k, {**(record or {}), "name": member.get("name"), "prefix": member["prefix"],
                             "prefix_now": prefix, "incarnation": incarnation, "spec": spec_of(member),
                             "gpu": member.get("gpu"), "state": STARTING, "relaunches": relaunches,
                             "host": self.host, "usd_per_h": member.get("usd_per_h"), "closed_at": None, "argv": argv,
                             "generation": generation, "pid": pid, "started_at": round(self.clock(), 3),
                             "why": why, "ended": None, "handoffs": 0, "billing_open": None,
                             "reattach": None, "takeover_blocked": None, "takeover_launch": bool(takeover) or None,
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
        if self.takeover and self._lost_host(k, record):
            self._take_over_step(k, record, syncer, member=member, generation=generation)
            return  # nothing more for this island this step (the launch started here is judged next step)
        if self._foreign(k, record):
            return
        if record.get("takeover_blocked"):
            return  # its records stay as they are until an operator resolves it (module doc)
        alive = self.ops.alive(prefix)
        if alive:
            self._seen.add(prefix)
        if state == MOVING:
            if alive:  # its STOP is ending the launch on the old machines
                self._assert_stop(k, f"moving to {member.get('gpu')}")
                self._stuck(k, record, prefix, f"moving to {member.get('gpu')}")
                return
            if syncer is None:
                return
            if self._reattach_pending(k, record, prefix):
                return  # the launch taken over has not confirmed its island released yet
            if not record.get("clearing"):
                # its launch ended through its STOP: what it ordered must be released before
                # the new machines are ordered (a leftover would hold the target's quota)
                left = self._left_open(prefix, self.ops.ended(prefix) or {"exit_code": None}, k)
                if left:
                    self._event(k, "moving: its launch ended leaving resources; its old prefix is torn down "
                                "before the next incarnation orders anything", to=member.get("gpu"), left=left)
                    self.save(k, {**record, "relaunches": 0, "clearing": {
                        "prefix": prefix, "since": round(self.clock(), 3), "left": left, "attempts": 1}})
                    self._tear_down_old(k, prefix, member, left)
                    return
            elif not self._cleared(k, record, member):
                return  # the old prefix is still being torn down: nothing is ordered yet
            # a move starts its relaunch count again; failures to tear its old prefix down count
            relaunches = int(record.get("relaunches") or 0) if record.get("clearing") else 0
            self._start(k, member, syncer, {**record, "clearing": None},
                        incarnation=int(record.get("incarnation") or 0) + 1,
                        relaunches=relaunches, generation=generation, why=f"moved to {member.get('gpu')}")
            return
        if alive:
            if record.get("spec") != spec:
                if not self._write_stop(k, f"moving to {member.get('gpu')}"):
                    return  # not moving until its STOP is written: asked again at the next step
                self.save(k, {**record, "state": MOVING, "moving_to": member.get("gpu"), "generation": generation,
                              "stop_at": round(self.clock(), 3), "torn_down_at": None})
                self._event(k, "moving: its launch stops on the old machines first", to=member.get("gpu"))
                return
            if state != RUNNING:  # up
                self.save(k, {**record, "state": RUNNING, "handoffs": 0, "billing_open": None})
                if state != STARTING:
                    self._event(k, "launch adopted (already running)", prefix=prefix)
            return
        if state in TERMINAL:
            if record.get("spec") != spec and syncer is not None:  # the operator changed a finished island
                if record.get("pools_stopped"):
                    self.log(f"fleet: WARNING: trainer island {k}: started again, but its pools "
                             f"{record['pools_stopped']} were stopped when it was given up: it gets no samples "
                             "until they are added back (`yeto fleet island remove`, then `add`)")
                self._start(k, member, syncer, record, incarnation=int(record.get("incarnation") or 0) + 1,
                            relaunches=0, generation=generation, why="changed after it ended")
            return
        ended = self.ops.ended(prefix)
        if ended is None and state == STARTING and self.clock() - float(record.get("started_at") or 0) < START_GRACE_S:
            return  # just started: its run is not registered yet
        ended = ended or {"exit_code": None, "log": "", "detail": "launch gone without a run record"}
        if record.get("takeover_launch") and TAKEOVER_BLOCKED_MARKER in str(ended.get("log") or ""):
            self._block_takeover(k, record, prefix, ended)  # it could not take its island over: blocked
            return
        if ended.get("exit_code") in HANDS_OFF_EXITS:
            self._hand_back(k, record, ended)
            return
        if record.get("spec") != spec and syncer is not None:
            # The island was changed (another target or argv) while this launch was ending
            # (e.g. failing to order its old machines): a move, never a relaunch of the new
            # spec under the old incarnation's prefix and cluster names, and never counted
            # against the infrastructure relaunch budget. (A hand-off above is started again
            # as it was and moves once it runs.)
            left = self._left_open(prefix, ended, k)
            if left:  # what the old incarnation ordered is not all released: torn down first, then the move
                self._event(k, "moving: its launch had ended leaving resources; its old prefix is torn down "
                            "before the next incarnation orders anything", to=member.get("gpu"), left=left)
                self.save(k, {**record, "state": MOVING, "moving_to": member.get("gpu"), "generation": generation,
                              "ended": ended, "relaunches": 0, "clearing": {"prefix": prefix, "since": round(self.clock(), 3),
                                                           "left": left, "attempts": 1}})
                self._tear_down_old(k, prefix, member, left)
                return
            self._event(k, "moving: its launch had ended, the next incarnation starts on the new machines",
                        to=member.get("gpu"), ended=ended.get("detail"))
            self._start(k, member, syncer, {**record, "ended": ended},
                        incarnation=int(record.get("incarnation") or 0) + 1, relaunches=0, generation=generation,
                        why=f"moved to {member.get('gpu')} (changed while its launch ended)")
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
        give_up = final in (FAILED, ABANDONED) and self.on_given_up is not None and stop is None
        self.save(k, {**self._closed(record), "state": final, "ended": ended, "decision": decision.to_json(),
                      **({"pools_stopped": list(member.get("pools") or ())} if give_up else {})})
        self._event(k, f"launch ended: {final}", cause=decision.cause, detail=decision.detail[:200] or None)
        if give_up:  # nobody trains on its samples any more: its pools stop billing now, not at the fleet's end
            try:
                self.on_given_up(k, member, f"{final}: {decision.cause} {decision.detail}"[:200])
            except Exception as error:  # noqa: BLE001 - reported; the other islands go on
                self.log(f"fleet: CRITICAL: trainer island {k}: given up ({final}) but its pools "
                         f"{list(member.get('pools') or ())} were not asked to stop ({error!r}); they bill until "
                         "`yeto run down` (or `yeto fleet island remove` it)"[:500])

    def _tear_down_old(self, k: int, prefix: str, member: dict[str, Any], left: str) -> bool:
        """``yeto down`` of a moved island's old prefix (:meth:`_cleared` waits for it)."""
        try:
            self.ops.tear_down(prefix, f"moving to {member.get('gpu')}: {left}")
            return True
        except Exception as error:  # noqa: BLE001 - reported; asked again at the next step
            self.log(f"fleet: ERROR: trainer island {k}: tearing its old launch {prefix} down failed ({error!r}); "
                     "asked again at the next step")
            return False

    def _cleared(self, k: int, record: dict[str, Any], member: dict[str, Any]) -> bool:
        """Whether a moved island's old prefix is torn down (its resource ledger has no open
        row), so the next incarnation may order. Not yet within :data:`CLEAR_TIMEOUT_S`:
        False, nothing ordered. Past it: an infrastructure failure of the move (counted
        against the relaunch budget), the leftovers reported and torn down again; the
        budget spent, the island is given up with its bill open (the leftovers may run)."""
        clearing = dict(record["clearing"])
        prefix = str(clearing["prefix"])
        reader = getattr(self.ops, "open_resources", None)
        try:
            waits, _reported = split_open_rows([] if reader is None else reader(prefix))
            left = None if not waits else f"{len(waits)} ledger row(s) still open that may hold quota or bill ({_row_names(waits)})"
        except Exception as error:  # noqa: BLE001 - not known: not cleared
            left = f"its resource ledger unread ({type(error).__name__}: {error})"
        if left is None:
            self._event(k, "old prefix torn down; the move goes on", prefix=prefix)
            return True
        waited = self.clock() - float(clearing.get("since") or 0)
        if waited < CLEAR_TIMEOUT_S:
            return False
        relaunches = int(record.get("relaunches") or 0) + 1
        self.log(f"fleet: ERROR: trainer island {k}: its old launch {prefix} is still not torn down "
                 f"{waited:.0f}s after `yeto down` ({left}); the move to {member.get('gpu')} orders nothing "
                 f"while it may hold the quota (infrastructure failure {relaunches}/{self.relaunch_budget}). "
                 f"Check `yeto down {prefix}` on this host")
        if relaunches >= self.relaunch_budget:
            self.save(k, {**self._closed(record), "state": ABANDONED, "relaunches": relaunches, "clearing": None,
                          "billing_open": f"its old launch {prefix} was not torn down: billed until removed or run down"})
            self._event(k, "launch ended: abandoned", cause="old prefix not torn down", detail=left)
            return False
        self.save(k, {**record, "relaunches": relaunches,
                      "clearing": {**clearing, "since": round(self.clock(), 3), "left": left,
                                   "attempts": int(clearing.get("attempts") or 1) + 1}})
        self._tear_down_old(k, prefix, member, left)
        return False

    def _left_open(self, prefix: str, ended: dict[str, Any], k: int | None = None) -> str | None:
        """What an ended launch ``prefix`` may have left ordered (None: nothing): no run
        record (its teardown is not known), resource ledger rows still open (an INIT sky
        cluster, a sky storage bucket, a Modal app...), or a ledger that does not read."""
        if ended.get("exit_code") is None:
            return "no run record"
        reader = getattr(self.ops, "open_resources", None)
        if reader is None:
            return None  # launch ops without a ledger (tests, remote ops): its own teardown is trusted
        try:
            rows = reader(prefix)
        except Exception as error:  # noqa: BLE001 - not known: torn down
            return f"its resource ledger unread ({type(error).__name__}: {error})"
        waits, reported = split_open_rows(list(rows or ()))
        if reported:
            skipped = [r for r in reported if str(r.get("release_error") or "").startswith("release_skipped")]
            note = ("the old prefix has bucket(s) left for a person to decide (release_skipped)" if skipped
                    else "the old prefix has bucket(s) cleanup could not confirm released")
            self.log(f"fleet: WARNING: trainer island {k}: {note}: {_row_names(reported)}; they do not block "
                     "the move (see `yeto cleanup` / the run's resource ledger)")
        if not waits:
            return None
        return f"{len(waits)} ledger row(s) still open that may hold quota or bill ({_row_names(waits)})"

    def _hand_back(self, k: int, record: dict[str, Any], ended: dict[str, Any]) -> None:
        """The launch handed off (:data:`HANDS_OFF_EXITS`): another launcher holds island
        ``k``'s records lock, or its machine's state did not read. It ordered nothing. It is
        started again now (the lock asked again), at most :data:`HANDOFF_LIMIT` times in a
        row; then the island is given up (``abandoned``) with an error: another launcher
        drives it. Its island STOP is written first (the launcher that holds it stops; given
        up only once the STOP is written, else asked again at the next step), and its spend
        stays open (``billing_open``): what that launcher runs is billed until the island is
        removed or ``yeto run down`` confirms it down."""
        handoffs = int(record.get("handoffs") or 0) + 1
        if handoffs > HANDOFF_LIMIT:
            if not self._write_stop(k, "given up: another launcher holds the island's records lock"):
                return  # not given up until its STOP is written: asked again at the next step
            self.save(k, {**record, "state": ABANDONED, "ended": ended, "handoffs": handoffs,
                          "billing_open": "another launcher holds the island: billed until removed or run down"})
            self.log(f"fleet: ERROR: trainer island {k}: its launch {record.get('prefix_now')} handed off "
                     f"{handoffs - 1} times in a row ({ended.get('detail')}): another launcher holds the island's "
                     "records lock; it is not started again, its island STOP is written and it is still billed (it counts "
                     "toward run.budget until it is removed or `yeto run down` confirms it down). "
                     f"Stop that launcher (`yeto down {record.get('prefix_now')}` on its host), then move or add the "
                     "island again")
            return
        # its machine may still run (a launcher that lost the lock exits 12/13 and tears nothing
        # down): the time billed so far is closed first, the next launch's time opens after it
        self.save(k, {**self._closed(record), "state": STARTING, "started_at": round(self.clock(), 3),
                      "closed_at": None, "ended": ended, "handoffs": handoffs})
        self.log(f"fleet: WARNING: trainer island {k}: its launch handed off ({ended.get('detail')}); started "
                 f"again now ({handoffs}/{HANDOFF_LIMIT})")
        argv = list(record.get("argv") or ())
        self.ops.start(str(record["prefix_now"]), [*argv, TAKEOVER_FLAG] if record.get("takeover_launch") else argv)

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
                     f"Run `yeto run up` on {host} again. Only an on-site control plane (fleet.syncer_service) "
                     f"takes a lost host's islands over: otherwise `yeto run down --config <run.json>` (any host) "
                     "tears them down and records them down; once none of the run's launches runs, `yeto run up` "
                     "here starts them again")
        return True

    def _take_over_step(self, k: int, record: dict[str, Any], syncer: dict[str, Any] | None, *,
                        member: dict[str, Any] | None, generation: int | None) -> None:
        """A launch of a lost control host (:meth:`_lost_host`), this step: taken over
        (:meth:`_take_over`) once the fleet's syncer is known. Then, by what its island launch
        recorded itself (``IslandStateStore``, :meth:`_ended_there`):

        * it ended there (``done`` / ``failed`` / ``abandoned``): closed the way that host
          would have closed it, nothing started (:meth:`_close_taken_over`); a moving or
          stopping one goes on with its move or removal;
        * otherwise its launch is started again here as a takeover launch
          (:data:`TAKEOVER_FLAG`: it never starts the island afresh and never clears its STOP),
          counted against the relaunch budget (spent: given up, :meth:`_give_up_taken_over`):
          a starting or running island as it ran (same prefix and incarnation); a moving or
          stopping one, or one removed from the desired state (``member`` None), with its own
          argv and its STOP kept (:meth:`_reattach`)."""
        if syncer is None:  # not before the fleet's syncer is known: the launch would wait for it
            return
        state = record.get("state")
        taken = self._take_over(k, record)
        if taken is None:
            return
        why = f"taken over from host {taken['taken_over']['from']}"
        ended = taken["taken_over"].get("ended")
        if ended:
            self._close_taken_over(k, taken, ended, member, why)
            return
        relaunches = int(taken.get("relaunches") or 0)
        if relaunches >= self.relaunch_budget:
            self._give_up_taken_over(k, taken, why)
            return
        if member is not None and state in (STARTING, RUNNING):
            self._start(k, member, syncer, taken, incarnation=int(taken.get("incarnation") or 0),
                        relaunches=relaunches + 1, generation=generation,
                        why=f"{why} (relaunch {relaunches + 1}/{self.relaunch_budget})", clear_stop=False,
                        takeover=True)
        elif state in (STARTING, RUNNING, MOVING, STOPPING):
            self._reattach(k, {**taken, "relaunches": relaunches + 1}, why)

    def _ended_there(self, k: int, record: dict[str, Any]) -> dict[str, Any] | None:
        """Island ``k``'s own record (``IslandStateStore``, written by its island launch) when
        the launch ``record`` names ended there: terminal (``done`` / ``failed`` /
        ``abandoned``) and of that launch's island (its cluster name: an earlier
        incarnation's record, before this launch wrote one, is not its end). None: it did not
        end, or that is not known."""
        try:
            life = self.states.read(k)
        except Exception:  # noqa: BLE001 - not known: taken over as a launch that may run
            return None
        if not isinstance(life, dict) or life.get("state") not in lifecycle.TERMINAL:
            return None
        name = island_name(record)
        if name is None or life.get("name") != name:
            return None
        return {"state": life.get("state"), "reason": str(life.get("reason") or "")[:300],
                "since": life.get("since"), "name": name}

    def _close_taken_over(self, k: int, record: dict[str, Any], ended: dict[str, Any],
                          member: dict[str, Any] | None, why: str) -> None:
        """A launch taken over whose island launch had ended on the lost host
        (:meth:`_ended_there`): nothing is started. A starting or running island is closed as
        that host would have closed it: its island record's terminal state, its sampler pools
        released when it was given up without a STOP (``on_given_up``). A moving or stopping
        one (or one removed from the desired state) goes on with its move or removal at the
        next step, as on that host."""
        state = record.get("state")
        detail = (f"its launch ended on host {(record.get('taken_over') or {}).get('from')}: its island record is "
                  f"{ended.get('state')}" + (f" ({ended['reason']})" if ended.get("reason") else ""))
        outcome = {"exit_code": None, "log": "", "detail": detail, "island_state": ended.get("state")}
        if member is None or state in (MOVING, STOPPING):
            self.save(k, {**record, "ended": outcome, "reattach": None})
            self._event(k, f"{why}: {detail}; its {state if member is not None else 'removal'} goes on",
                        prefix=record.get("prefix_now"))
            return
        final = str(ended.get("state"))
        stop = self.states.stop_requested(k)
        give_up = final in (FAILED, ABANDONED) and self.on_given_up is not None and stop is None
        self.save(k, {**record, "state": final, "ended": outcome,
                      "decision": {"state": final, "cause": "ended on the lost control host", "detail": detail},
                      **({"pools_stopped": list(member.get("pools") or ())} if give_up else {})})
        self._event(k, f"launch ended: {final}", cause=f"{why}; nothing started", detail=detail[:200])
        if give_up:
            try:
                self.on_given_up(k, member, f"{final}: {detail}"[:200])
            except Exception as error:  # noqa: BLE001 - reported; the other islands go on
                self.log(f"fleet: CRITICAL: trainer island {k}: given up ({final}) but its pools "
                         f"{list(member.get('pools') or ())} were not asked to stop ({error!r})"[:500])

    def _give_up_taken_over(self, k: int, record: dict[str, Any], why: str) -> None:
        """A launch taken over with its relaunch budget spent: no launch is started again; the
        island is given up (``abandoned``), its STOP written, and its spend stays open
        (``billing_open``): what the lost launch left may still run."""
        self._write_stop(k, "given up: taken over from a lost control host with no relaunch left")
        self.save(k, {**record, "state": ABANDONED, "billing_open": "taken over with its relaunch budget spent: "
                      "what its lost launch left may run; billed until removed or run down"})
        self.log(f"fleet: ERROR: trainer island {k}: {why}, but its relaunch budget is spent "
                 f"({int(record.get('relaunches') or 0)}/{self.relaunch_budget}): it is not started again, its "
                 f"island STOP is written and it is still billed. Check `yeto cleanup --run {record.get('prefix_now')}` "
                 "on a host that reaches its machines, then `yeto fleet island remove` / `add` it again")

    def _reattach(self, k: int, record: dict[str, Any], why: str) -> None:
        """Island ``k``'s launch, taken over while it was moving or stopping (or removed from
        the desired state), started again on this host as it was: its prefix, incarnation and
        argv (plus :data:`TAKEOVER_FLAG`), its STOP kept. It reattaches to the island, reads
        the STOP and releases it; the move or the removal goes on once it ended
        (:meth:`_reattach_pending`). Its wait for the STOP starts again here (``stop_at``):
        the launch that read it is this one."""
        prefix = str(record["prefix_now"])
        argv = [a for a in (record.get("argv") or ()) if a != TAKEOVER_FLAG]
        pid = self.ops.start(prefix, [*argv, TAKEOVER_FLAG])
        self._seen.add(prefix)
        now = round(self.clock(), 3)
        self.save(k, {**record, "argv": argv, "pid": pid, "takeover_launch": True, "stop_at": now,
                      "torn_down_at": None, "reattach": {"at": now, "state": record.get("state"), "why": why}})
        self._event(k, f"launch started again to finish its {record.get('state')} ({why}; its STOP is kept)",
                    prefix=prefix, pid=pid)

    def _reattach_pending(self, k: int, record: dict[str, Any], prefix: str) -> bool:
        """A launch started again by :meth:`_reattach` that ended: True while the move or the
        removal must wait (it has not registered yet), or for good when it could not
        reattach (:data:`TAKEOVER_BLOCKED_MARKER` in its log, or no run record): the island is
        blocked (``takeover_blocked``, an ``island_takeover_blocked`` event), its records
        kept and no new incarnation started. False: it released the island (or there was
        nothing to reattach), the move or the removal goes on."""
        if not record.get("reattach"):
            return False
        ended = self.ops.ended(prefix)
        if ended is None and self.clock() - float(record["reattach"].get("at") or 0) < START_GRACE_S:
            return True
        ended = ended or {"exit_code": None, "log": "", "detail": "launch gone without a run record"}
        if TAKEOVER_BLOCKED_MARKER in str(ended.get("log") or "") or ended.get("exit_code") is None:
            self._block_takeover(k, record, prefix, ended)
            return True
        self.save(k, {**record, "reattach": None})
        return False

    def _block_takeover(self, k: int, record: dict[str, Any], prefix: str, ended: dict[str, Any]) -> None:
        """A takeover launch that could not take its island over (module doc): the island is
        blocked, its records kept, no new incarnation started; one event."""
        detail = str(ended.get("detail") or "")[:300]
        marked = [line for line in str(ended.get("log") or "").splitlines() if TAKEOVER_BLOCKED_MARKER in line]
        if marked:
            detail = marked[-1].strip()[:300]
        self.save(k, {**record, "takeover_blocked": {"at": round(self.clock(), 3), "ended": ended,
                                                     "state": record.get("state")}})
        self.log(f"fleet: ERROR: trainer island {k}: island_takeover_blocked: its {record.get('state')} launch "
                 f"{prefix}, taken over from host {(record.get('taken_over') or {}).get('from')}, could not "
                 f"take its island over ({detail}); its records are kept and no new incarnation starts. Check "
                 f"the island's nodes, release it (`yeto cleanup --run {prefix}` on a host that reaches them), "
                 "then `yeto fleet island remove` / `add` it again")
        if self.on_event is not None:
            try:
                self.on_event({"event": "island_takeover_blocked", "island": k, "prefix": prefix,
                               "state": record.get("state"), "detail": detail})
            except Exception:  # noqa: BLE001 - the block holds whether the event is written or not
                pass

    def _lost_host(self, k: int, record: dict[str, Any]) -> bool:
        """Island ``k``'s launch was started by another control host and is not over."""
        host = record.get("host")
        return bool(host) and host != self.host and record.get("state") not in TERMINAL \
            and record.get("state") != DOWN

    def _take_over(self, k: int, record: dict[str, Any]) -> dict[str, Any] | None:
        """Take island ``k``'s launch over from the control host that started it, once that
        host is gone (an on-site control plane, ``takeover``). Only:

        * an island on a source another host can adopt: ``ssh`` (the island launch reattaches
          by the incarnation its store record names, :data:`TAKEOVER_SOURCES`) or Modal (the
          function call id). A sky cluster is in the starting host's own sky state: never;
        * once the lease of its island launch's records lock (``fleet/trainers/<k>/lock``)
          ended ``takeover_margin_s`` ago (the two hosts' clocks may differ);
        * once its launch started (``started_at``, the last takeover's time included) at
          least :data:`START_GRACE_S` ago: a launch that has not claimed its lock yet is
          waited for, so a host that takes over and dies is taken over in turn after that;
        * when this host claims that lock (``StoreLock.acquire``): its token is higher than
          the lost launch's, whose next renewal fails (``LockLost``); a lease that came back
          meanwhile, or another host's claim, makes it fail (None). Under the claim the launch
          record is read again: one another host took over meanwhile (and released its claim)
          is left alone (None). The claim is released at once: the launch started here claims
          the next token.
          ``fleet/trainers/<k>/takeover/<token>.json`` records it (information only).

        The record then names this host (``taken_over``); None: not taken over (yet)."""
        from yeto.launch.pool.store_lock import StoreLock, StoreLockError  # noqa: PLC0415

        host = str(record.get("host"))
        source = _source(record.get("gpu"))
        if source not in TAKEOVER_SOURCES:
            self._foreign_once(k, f"its launch {record.get('prefix_now')} was started on host {host}; a {source} "
                                  "island is never taken over from another host (its cluster is in that host's own "
                                  f"sky state). Run `yeto run up` on {host} again, or `yeto run down`")
            return None
        lock = lifecycle.lock_prefix(k)
        now = self.clock()
        margin = max(0.0, float(self.takeover_margin_s))
        holder = StoreLock.holder_of(self.objects, lock, now=now - margin)
        if holder is not None:
            self._foreign_once(k, f"its launch {record.get('prefix_now')} on host {host} still holds its records "
                                  f"lock (lease until {float(holder.get('until') or 0):.0f}, taken over {margin:.0f}s "
                                  "after it ends): not taken over yet")
            return None
        if now - float(record.get("started_at") or 0) < START_GRACE_S:
            return None  # started (or taken over) just now: its launch may not have claimed its lock yet
        try:
            claimed = StoreLock.acquire(self.objects, lock, holder=f"takeover of island {k} by {self.host}",
                                        ttl_s=TAKEOVER_LOCK_TTL_S, clock=self.clock)
        except StoreLockError as error:
            self._foreign_once(k, f"its launch was started on host {host}; this store has no exclusive create "
                                  f"({error}), so no launch is taken over from another host")
            return None
        if claimed is None:
            self._foreign_once(k, f"its launch {record.get('prefix_now')} on host {host}: its records lock is held "
                                  "again (its lease came back, or another host claimed it): not taken over")
            return None
        try:
            current = self.load(k)
            if not _same_launch(current, record):
                # another host took it over (and released its claim) between this worker's read
                # and its claim: its record names that host now, whose launch is not lost
                self._foreign_once(k, f"its launch {record.get('prefix_now')} was taken over by host "
                                      f"{(current or {}).get('host')} meanwhile: not taken over here")
                return None
            # read under the claim: the lost launch writes its island record no more
            ended = self._ended_there(k, record)
            claim = {"host": self.host, "from": host, "token": claimed.token, "at": round(now, 3),
                     "prefix": record.get("prefix_now"), "incarnation": record.get("incarnation"),
                     "ended": ended}
            try:  # information only: the lock's token is the fence
                self.objects.write(lifecycle.takeover_key(k, claimed.token), json.dumps(claim, sort_keys=True).encode())
            except Exception:  # noqa: BLE001
                pass
            self._foreign_logged.discard(k)
            if ended is None:  # the lost launch's time is billed up to now; the launch started here bills on
                base = {**self._closed(record), "started_at": round(now, 3), "closed_at": None}
            else:  # it ended there: billed up to its end (its own host's clock, as its start), nothing reopens
                base = closed(record, _ended_at(ended.get("since"), record.get("started_at"), now))
            taken = self.save(k, {**base, "host": self.host, "taken_over": claim})
        finally:
            claimed.release()  # the island launch started next claims the token after it
        self._event(k, f"taken over from host {host}: its launch's lease ran out (records lock token "
                       f"{claimed.token} claimed and released for the launch started here)",
                    prefix=record.get("prefix_now"), incarnation=record.get("incarnation"))
        return taken

    def _foreign_once(self, k: int, why: str) -> None:
        if k not in self._foreign_logged:
            self._foreign_logged.add(k)
            self.log(f"fleet: ERROR: trainer island {k}: {why}")

    def _stuck(self, k: int, record: dict[str, Any], prefix: str, why: str) -> None:
        """Island ``k``'s launch still runs after its STOP (moving or removed): torn down
        (``ops.tear_down``: ``yeto down``, the launcher's own teardown) once it still orders
        its machines :data:`STOP_UNREAD_S` after the STOP was written (it reads no STOP before
        they are up: a sky launch waiting for SSH would hold the move for its whole timeout,
        then fail and be relaunched on the same target), or :data:`STOP_IGNORED_S` after it
        whatever it does. Once per STOP; the move or removal goes on when it ended."""
        since = record.get("stop_at")
        if since is None:  # a record from before this field: its wait starts now
            self.save(k, {**record, "stop_at": round(self.clock(), 3)})
            return
        if record.get("torn_down_at") is not None:
            return
        waited = self.clock() - float(since)
        life = self.states.read(k) or {}
        ordering = life.get("state") == lifecycle.ORDERING
        if not ((ordering and waited >= STOP_UNREAD_S) or waited >= STOP_IGNORED_S):
            return
        what = (f"still ordering its machines {waited:.0f}s after its STOP (it reads none before they are up)"
                if ordering and waited < STOP_IGNORED_S else f"still running {waited:.0f}s after its STOP")
        self.save(k, {**record, "torn_down_at": round(self.clock(), 3)})
        self.log(f"fleet: WARNING: trainer island {k}: its launch {prefix} is {what}: torn down now "
                 f"(`yeto down {prefix}`; the island is {why})")
        try:
            self.ops.tear_down(prefix, f"{why}: {what}")
        except Exception as error:  # noqa: BLE001 - its STOP still stands; reported
            self.log(f"fleet: ERROR: trainer island {k}: tearing its launch {prefix} down failed ({error!r}); "
                     f"run `yeto down {prefix}` on this host")

    def _assert_stop(self, k: int, why: str) -> None:
        """Island ``k``'s STOP, written again when it is missing (module doc)."""
        if self.states.stop_requested(k) is None and self._write_stop(k, why):
            self._event(k, "island STOP written again (its launch cleared it as it started)")

    def _retire(self, k: int, syncer: dict[str, Any] | None = None) -> None:
        record = self.load(k)
        if record is None or record.get("state") == REMOVED:
            return
        if record.get("state") == DOWN:  # torn down: nothing runs
            self.save(k, {**record, "state": REMOVED, "removed_at": round(self.clock(), 3)})
            self._event(k, "removed (it was down)")
            return
        if self.takeover and self._lost_host(k, record):
            self._take_over_step(k, record, syncer, member=None, generation=None)
            return
        if self._foreign(k, record):
            return
        if record.get("takeover_blocked"):
            return
        prefix = str(record["prefix_now"])
        if not self.ops.alive(prefix) and self._reattach_pending(k, record, prefix):
            return
        if self.ops.alive(prefix):
            if record.get("state") != STOPPING:
                if not self._write_stop(k, "removed from the fleet's desired state"):
                    return  # not stopping until its STOP is written: asked again at the next step
                self.save(k, {**record, "state": STOPPING, "stop_at": round(self.clock(), 3), "torn_down_at": None})
                self._event(k, "removed: its launch stops", prefix=prefix)
            else:
                self._assert_stop(k, "removed from the fleet's desired state")
                self._stuck(k, record, prefix, "removed from the fleet's desired state")
            return
        if not record.get("billing_open"):  # a launcher that still holds a given-up island keeps its STOP
            self.states.clear_stop(k)
        self.save(k, {**self._closed(record), "state": REMOVED, "removed_at": round(self.clock(), 3),
                      "billing_open": None})
        self._event(k, "removed: its launch ended", prefix=prefix)


# the sources whose running island another control host can adopt (yeto.launch.island_adopt):
# ssh (the incarnation in the island's store record; the launch reattaches when every node runs
# it) and Modal (the function call id). A sky cluster lives in the starting host's sky state.
TAKEOVER_SOURCES = frozenset({"ssh", "modal"})


def _same_launch(current: dict[str, Any] | None, record: dict[str, Any]) -> bool:
    """``current`` (the launch record read under a takeover's claim) is still the launch
    ``record`` names: the same host, start, incarnation, prefix and takeover."""
    if not isinstance(current, dict):
        return False
    return all(current.get(f) == record.get(f) for f in ("host", "started_at", "incarnation", "prefix_now",
                                                          "taken_over"))


def island_name(record: dict[str, Any]) -> str | None:
    """The island (cluster) name the launch ``record`` names writes its island record under
    (``yeto.launcher``: its ``--gpu`` cluster under its prefix); None when it does not parse."""
    try:
        from yeto.gpu_spec import parse_gpu_spec  # noqa: PLC0415
        from yeto.launch.sky_ops import learner_cluster_names  # noqa: PLC0415

        return str(learner_cluster_names(str(record["prefix_now"]), parse_gpu_spec(str(record.get("gpu"))))[0])
    except Exception:  # noqa: BLE001 - not known: no island record is its end
        return None


def _ended_at(since: Any, started_at: Any, now: float) -> float:
    """When a launch that ended on a lost host stopped billing: its island record's end
    (``since``), within its start and now."""
    try:
        end = float(since)
    except (TypeError, ValueError):
        return now
    try:
        end = max(end, float(started_at))
    except (TypeError, ValueError):
        pass
    return min(end, now)


def _source(gpu: Any) -> str:
    """The source (cloud) of an island's ``--gpu`` cluster; ``?`` when it does not parse."""
    try:
        from yeto.gpu_spec import parse_gpu_spec  # noqa: PLC0415

        return str(parse_gpu_spec(str(gpu))[0].cloud)
    except Exception:  # noqa: BLE001 - an unknown source is never taken over
        return "?"


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
            price = incarnation_price(record)
            if price is None:
                unpriced += running
            else:
                usd += running / 3600.0 * float(price)
        out.append((int(parts[-2]), seconds / 3600.0, usd, unpriced / 3600.0))
    return out


def island_spend(rows: Any, declared: Any) -> dict[int, tuple[float, float, bool]]:
    """``{learner id: (billed hours, usd, fallback)}`` from :func:`billed` rows: the dollars
    each incarnation was billed at its price, plus the hours no price covered at the
    island's declared price (``declared``: learner id -> USD/h; 0 when absent); ``fallback``
    when some hours were. What ``run.budget`` (``FleetRuntime.spend``) and the dashboard's
    cost panel both count."""
    out = {}
    for k, hours, usd, unpriced in rows:
        fallback = unpriced > 0
        if fallback:
            usd += unpriced * float(declared.get(k, 0.0) or 0.0)
        out[int(k)] = (float(hours), float(usd), fallback)
    return out


def mark_down(objects: Any, prefixes: Any, *, log: Callable[[str], None] = print,
              clock: Callable[[], float] = time.time) -> list[int]:
    """``yeto run down`` confirmed the island launches ``prefixes`` torn down: each
    island whose launch runs under one of them (not ended) is recorded ``down`` and its
    incarnation closed for the spend; the next ``yeto run up``, on any host, starts its
    next incarnation. An island given up and still billed (``billing_open``) gets its bill
    closed and stays given up: it is never restarted. Returns the learner ids recorded down."""
    gone, out = {str(p) for p in prefixes}, []
    for key in sorted(objects.list(f"{lifecycle.STATE_DIR}/")):
        parts = str(key).split("/")
        if not (len(parts) >= 4 and parts[-1] == LAUNCH_RECORD and parts[-2].isdigit()):
            continue
        try:
            record = json.loads(objects.read(key) or b"null")
        except ValueError:
            continue
        if (not isinstance(record, dict) or record.get("state") == DOWN
                or (record.get("state") in TERMINAL and not record.get("billing_open"))
                or str(record.get("prefix_now")) not in gone):
            continue
        k, now = int(parts[-2]), clock()
        if record.get("state") in TERMINAL:  # given up and still billed: its bill closes, it is never restarted
            save_record(objects, k, {**closed(record, now), "billing_open": None, "down_at": round(now, 3)}, now)
            log(f"fleet: trainer island {k}: its launch {record.get('prefix_now')} is down; it stays "
                f"{record.get('state')}")
            continue
        save_record(objects, k, {**closed(record, now), "state": DOWN, "down_at": round(now, 3), "billing_open": None}, now)
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
        base = (Path(self.log_dir) if self.log_dir is not None  # $YETO_ISLANDS_LOG_DIR moves the default
                else Path(os.environ.get("YETO_ISLANDS_LOG_DIR") or "~/.yeto/islands").expanduser())
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
            # the CLI ended. Only its configuration error exit (CONFIG_ERROR_EXIT) is a
            # configuration error, never relaunched (moving or changing the island again
            # starts it again); any other exit before it registered the run ordered nothing
            # and is an infrastructure failure (relaunched within the budget)
            from yeto.config_errors import CONFIG_ERROR_EXIT  # noqa: PLC0415

            code = proc.returncode
            if code in HANDS_OFF_EXITS:  # another launcher holds the island, or its state did not read
                return {"exit_code": code, "log": tail, "detail": f"yeto launch handed off (exit {code})"}
            if meta is None and code == CONFIG_ERROR_EXIT:
                return {"exit_code": code, "log": tail,
                        "detail": f"yeto launch refused its configuration before ordering anything (exit {code})"}
            if meta is None:  # never registered: it ordered nothing
                return {"exit_code": code, "log": tail,
                        "detail": f"yeto launch ended before it registered its run (exit {code})"}
            # registered (it may have ordered and trained): its worker died without a result
            # (killed); an infrastructure failure, relaunched -- never a configuration error
        if meta is not None and meta.get("state") in (runs.PENDING, runs.RUNNING):
            return {"exit_code": None, "log": _tail(runs.log_path(prefix)), "detail": "worker exited without a result"}
        return None

    def request_stop(self, prefix: str, why: str) -> None:
        from yeto import runs  # noqa: PLC0415

        runs.request_stop(prefix)

    def open_resources(self, prefix: str) -> list[dict[str, Any]]:
        """The rows of launch ``prefix``'s resource ledger that are not released (none
        when it has no ledger: it ordered nothing it recorded)."""
        from yeto.cost.ledger import ResourceLedger  # noqa: PLC0415

        ledger = ResourceLedger.for_run(prefix)
        if not ledger.path.exists():
            return []
        return [{"kind": r.kind, "id": r.id, "name": r.name, "release_error": r.release_error}
                for r in ledger.open_resources()]

    def tear_down(self, prefix: str, why: str) -> None:
        """The launch torn down now, without waiting for it to read a STOP: ``yeto down
        <prefix>`` in the background (it signals the launch's worker, whose teardown releases
        what it ordered, then confirms its clusters down). A launch that has not registered its
        run yet ordered nothing: its process is terminated."""
        from yeto import runs  # noqa: PLC0415

        proc = self.procs.get(prefix)
        if runs.load_run(prefix) is None:
            if proc is not None and proc.poll() is None:
                proc.terminate()
            return
        path = self._log_path(prefix)
        with open(path, "a") as fh:
            fh.write(f"\n### {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} yeto down {prefix} ({why})\n")
            fh.flush()
            subprocess.Popen([self.python, "-m", "yeto.cli", "down", "--missing-ok", prefix], env=self.env,
                             stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)


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
