"""The seeded scheduler and the invariant checks of the store simulation.

One :class:`Simulation` is one run: N sampler shards, one trainer and the pool,
all on one :class:`~.fault_s3.S3Server`. Phase 1 runs ``steps`` scheduler steps
with faults on; phase 2 turns every fault off, heals partitions and lets the
supervisor restart what died until the trainer has consumed the data (or a step
bound is hit: a liveness violation). :meth:`Simulation.check` then compares the
store and what the trainer consumed with what the participants intended.

A scheduler step: the clock moves, background work that is due runs (lease
renewals, lease polls, supervisor restarts, the pool's drain follow-ups), maybe
a fault event starts (a partition, a pool action, a trainer restart), then one
participant takes one foreground action. Inside an action any store call may
fail, be answered late, kill the process, or let another participant take an
action first (operation-level interleaving without threads).

Everything random comes from ``random.Random(seed)``: a seed replays exactly.
"""

from __future__ import annotations

import collections
import contextlib
import json
import random
import shutil
import tempfile
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from yeto.cloud import s3_store
from yeto.launch.nc import elastic
from yeto.launch.pool.data_path import R2DataPath
from yeto.rl.engine import sample_buffer, store_dispense
from yeto.rl.engine.buffer_payload import decode_group

from .actors import CUT_KEY, NAMESPACE, TRANSPORT_ERRORS, URI, Blocked, SamplerProc, TrainerProc
from .fault_s3 import ActorClock, Crash, FaultConfig, FaultyS3Client, Injector, S3Server, SimClock

MERGE_DROPS_FENCED_WRITES = hasattr(sample_buffer, "ShardFenced")
INDEX_PREFIX = f"{sample_buffer.BUFFER_DIR}/{sample_buffer.INDEX_OBJECTS}/"
# violations that only show with a fault the product does not claim to survive yet
KNOWN_DESIGN_GAPS = ("two_trainers",)


@dataclass
class SimConfig:
    seed: int = 0
    samplers: int = 3
    slots: int = 24  # rows of the prompt file (one pass)
    capacity: int = 2  # groups a sampler generates at once (groups_per_round)
    steps: int = 400  # scheduler steps with faults on
    converge_steps: int = 3000  # bound of the fault-free phase
    dt_max_s: float = 2.0  # the clock moves up to this much per step
    lease_s: float = 30.0  # dispenser lease
    max_redo: int = 2
    lease_poll_s: float = 5.0
    write_check_s: float = 1.0
    reload_max_defer_s: float = 60.0
    groups_per_step: int = 3
    max_policy_age: int = 12
    release_every: int = 1
    cut_every: int = 2
    keep_releases: int = 2
    trainer_wait_s: float = 600.0
    part_bytes: int = 96  # multipart part / download range size
    page_size: int = 7  # keys per listing page
    trainer_lease: bool = True  # the trainer claims a generation (where the tree has trainer_lease)
    legacy_samplers: int = 0  # the first shards run the sampler rules of before shard ownership records
    faults: FaultConfig = field(default_factory=FaultConfig)
    # more invariants, as ``package.module:name`` (:func:`load_check`): each is called with the
    # :class:`Simulation` when it is built (``on_start``), after every scheduler step
    # (``on_step``) and after the run converged (``on_check``, or the callable itself), and
    # reports through ``sim.violation`` / ``sim.metric``
    checks: tuple[str, ...] = ()

    def describe(self) -> str:
        on = {k: v for k, v in asdict(self.faults).items() if v and k not in ("retry_fault", "slow_max_s",
                                                                              "partition_max_s")}
        return (f"seed={self.seed} samplers={self.samplers} slots={self.slots} steps={self.steps} "
                f"faults={json.dumps(on, sort_keys=True)}")


@dataclass
class Take:
    trainer: int
    rollout_id: int
    zombie: bool
    trained: list[dict[str, Any]]
    skipped: list[dict[str, Any]]
    report: dict[str, Any]
    rolled_back: bool = False


@dataclass
class Result:
    config: SimConfig
    violations: dict[str, list[dict[str, Any]]]
    metrics: dict[str, int]
    errors: dict[str, int]
    steps: int
    trace: list[str]
    trimmed: bool = False  # ``trace`` is already the minimal one (a result sent between processes)

    @property
    def ok(self) -> bool:
        return not self.violations

    def kinds(self) -> list[str]:
        return sorted(self.violations)

    def report(self, *, trace: int = 60) -> str:
        lines = [f"store-sim {'OK' if self.ok else 'VIOLATED'}: {self.config.describe()} "
                 f"(replay: python -m tests.store_sim.run --seeds {self.config.seed}:{self.config.seed + 1} ...)"]
        for kind, items in sorted(self.violations.items()):
            lines.append(f"  {kind} x{len(items)}: {json.dumps(items[0], sort_keys=True, default=str)[:400]}")
        lines.append(f"  metrics: {json.dumps(self.metrics, sort_keys=True)}")
        if self.errors:
            lines.append(f"  process errors: {json.dumps(self.errors, sort_keys=True)}")
        if not self.ok and trace:
            lines.append(f"  trace (events naming the violating records, then the last {trace}):")
            lines += [f"    {line}" for line in (self.trace if self.trimmed else self.minimal_trace(trace))]
        return "\n".join(lines)

    def minimal_trace(self, tail: int) -> list[str]:
        tokens: set[str] = set()
        for items in self.violations.values():
            for item in items[:3]:
                for name in ("sample_id", "key", "actor"):
                    if item.get(name) is not None:
                        tokens.add(str(item[name]))
                if item.get("slot") is not None:
                    tokens.add(f"slot={item['slot']} ")
                    tokens.add(f"/{int(item['slot']):012d}")
        named = [line for line in self.trace if any(t in line for t in tokens)]
        return named[-120:] + ["    ..."] + self.trace[-tail:]


def load_check(spec: str) -> Any:
    """``package.module:name`` -> the check: a callable ``(sim)`` run after the run
    converged, or an object with any of ``on_start(sim)`` / ``on_step(sim)`` /
    ``on_check(sim)``."""
    import importlib

    module, sep, name = spec.partition(":")
    if not sep:
        raise ValueError(f"check {spec!r}: expected package.module:name")
    return getattr(importlib.import_module(module), name)


@contextlib.contextmanager
def _patched(clock: SimClock, rng: random.Random) -> Iterator[None]:
    """The few places the code under test reaches for real time, real randomness or
    threads on its own: a range read's retry sleep, the concurrent record reads."""

    class _Time:
        time = staticmethod(lambda: clock.now)
        monotonic = perf_counter = staticmethod(lambda: clock.now)
        sleep = staticmethod(clock.advance)

    saved = (s3_store.time, s3_store.random, sample_buffer._READ_WORKERS)
    s3_store.time, s3_store.random, sample_buffer._READ_WORKERS = _Time, rng, 1  # type: ignore[assignment]
    try:
        yield
    finally:
        s3_store.time, s3_store.random, sample_buffer._READ_WORKERS = saved


class Simulation:
    def __init__(self, config: SimConfig, tmp: Path) -> None:
        self.config, self.tmp = config, Path(tmp)
        self.rng = random.Random(config.seed)
        self.clock = SimClock()
        self.server = S3Server(self.clock, page_size=config.page_size)
        self._trace: collections.deque[str] = collections.deque(maxlen=50_000)
        self.injector = Injector(self.rng, self.clock, config.faults, self.trace)
        self.injector.preempt = self._preempt
        self.steps = 0
        self.metrics: collections.Counter[str] = collections.Counter()
        self.errors: collections.Counter[str] = collections.Counter()
        self.violations: dict[str, list[dict[str, Any]]] = {}
        self.shards = [f"s{i}" for i in range(config.samplers)]
        self.prompt_path = self.tmp / "prompts.jsonl"
        self.prompt_path.write_text("".join(
            json.dumps({"prompt": f"task {i}", "metadata": {"task_id": f"t{i}"}}) + "\n"
            for i in range(config.slots)), encoding="utf-8")
        self.dispense_config = {"backend": "store", "namespace": NAMESPACE, "seed": 0, "passes": 1,
                                "lease_s": config.lease_s, "max_redo": config.max_redo, "scan_s": 2.0,
                                "max_scan_s": 8.0}
        # participants
        self.samplers: dict[str, SamplerProc | None] = {sid: None for sid in self.shards}
        self.zombies: list[SamplerProc] = []  # superseded sampler processes that still run
        self.incarnations = collections.Counter()
        self.nodes = {sid: f"{sid}-node0" for sid in self.shards}
        self.start_at: dict[str, float] = {sid: 0.0 for sid in self.shards}  # supervisor / pool (re)starts
        self.trainer: TrainerProc | None = None
        self.trainer_zombies: list[tuple[TrainerProc, int]] = []
        self.trainers = 0
        self.trainer_start_at: float | None = 0.0
        self._stack: list[Any] = []  # participants inside an action (never re-entered)
        # the pool
        pool = s3_store.S3ObjectStore(URI, client=FaultyS3Client(self.server, "pool", self.injector))
        self.pool_clock = ActorClock(self.clock)
        self.pool_objects = pool
        self.data_path = R2DataPath(elastic.ObjectRecords(pool), (pool, NAMESPACE), clock=self.pool_clock.time)
        self.drains: dict[str, tuple[str, float]] = {}  # sid -> (seq, give up at)
        self.tickets = 0
        self.fenced_shards = 0
        # the oracle
        self.oracle = s3_store.S3ObjectStore(URI, client=FaultyS3Client(self.server, "oracle"))
        self.intents: dict[str, dict[str, Any]] = {}  # sample id -> what its writer meant
        self.fenced_truth: dict[tuple[str, str], float] = {}  # (sid, incarnation) -> fenced at
        self.takes: list[Take] = []
        self._refusals_seen: set[str] = set()
        self.checks = [load_check(spec) for spec in config.checks]
        self._hook("on_start")
        self.errors_after_faults: list[str] = []  # processes that exited on an error with every fault off

    def _hook(self, name: str) -> None:
        for check in self.checks:
            call = getattr(check, name, None)
            if call is None and name == "on_check" and callable(check):
                call = check
            if call is not None:
                call(self)

    # ------------------------------------------------------------------ records
    def trace(self, event: str, **fields: Any) -> None:
        detail = " ".join(f"{k}={v}" for k, v in fields.items())
        self._trace.append(f"#{self.steps} t={self.clock.now - 1_800_000_000.0:8.2f} {event} {detail} ")

    def metric(self, name: str, count: int = 1) -> None:
        self.metrics[name] += count

    def violation(self, kind: str, **detail: Any) -> None:
        self.violations.setdefault(kind, []).append(detail)
        self.trace("VIOLATION", kind=kind, **detail)

    # -------------------------------------------------------------------- truth
    def truth_holds(self, slot: int, attempt: int, sid: str, incarnation: str) -> bool:
        """The ledger, read without faults: this process holds the slot's open attempt."""
        base = f"{NAMESPACE}/claims/{slot:012d}/"
        claims = self.server.keys(base)
        if not claims or int(claims[-1][len(base):]) != attempt:
            return False
        body = json.loads(self.server.get(claims[-1]) or b"{}")
        if body.get("holder") != sid or str(body.get("incarnation")) != incarnation:
            return False
        return not (self.server.keys(f"{NAMESPACE}/fail/{slot:012d}/{attempt:04d}.")
                    or self.server.keys(f"{NAMESPACE}/done/{slot:012d}."))

    def truth_fenced(self, sid: str, incarnation: str) -> bool:
        """The incarnation lease of the shard names another incarnation."""
        raw = self.server.get(f"sample-buffer/leases/{sid}.json")
        return bool(raw) and str(json.loads(raw).get("incarnation")) != incarnation

    def intent(self, sample_id: str, proc: SamplerProc, gen: Any) -> None:
        self.intents[sample_id] = {
            "slot": gen.slot, "attempt": gen.attempt, "sampler": proc.sid, "incarnation": proc.incarnation,
            "version": gen.version, "policy_hash": gen.policy_hash,
            "owned": self.truth_holds(gen.slot, gen.attempt, proc.sid, proc.incarnation),
            "fenced": self.truth_fenced(proc.sid, proc.incarnation),
            "retired": (proc.sid, proc.incarnation) in self.fenced_truth}
        if not self.intents[sample_id]["owned"]:
            self.metric("writes_without_ownership")
        if self.intents[sample_id]["fenced"]:
            self.metric("writes_by_fenced_incarnation")
        if self.intents[sample_id]["retired"]:
            self.metric("writes_by_retired_incarnation")

    def took(self, trainer: TrainerProc, rollout_id: int, groups: list, report: Any, *,
             start: int = 0, end: int = 0) -> None:
        if report is not None:
            start, end = int(report.start), int(report.end)

        def describe(position: int, group: list | None) -> dict[str, Any]:
            record = json.loads(self.server.get(f"{INDEX_PREFIX}{position:012d}.json") or b"{}")
            out = {"position": position, "sample_id": record.get("sample_id"), "shard": record.get("shard"),
                   "shard_pos": record.get("shard_pos"), "behavior_version": record.get("behavior_version"),
                   "slot": (record.get("sampler_round") or {}).get("dispense_position"),
                   "trainer_hash": trainer.hashes.get(int(record.get("behavior_version") or 0))}
            if group is not None:
                out["payload"] = group[0]["metadata"]["sim"]
            return out

        taken = {int(position) for position, _ in groups}
        trained = [describe(int(position), group) for position, group in groups]
        skipped = [describe(p, None) for p in range(start, end) if p not in taken]
        self.takes.append(Take(trainer.number, rollout_id, trainer.zombie, trained, skipped,
                               {"stale": getattr(report, "stale_groups", 0),
                                "stale_incarnation": getattr(report, "stale_incarnation_groups", 0),
                                "passed_over": len(skipped), "start": start, "end": end}))
        self.trace("take", actor=trainer.actor, rollout_id=rollout_id, start=start, end=end,
                   trained=[t["sample_id"] for t in trained], skipped=[s["sample_id"] for s in skipped])

    def restored(self, trainer: TrainerProc, cut: dict[str, Any]) -> None:
        """A trainer starts from ``cut``: the rounds after it are trained again."""
        for take in self.takes:
            if not take.zombie and take.rollout_id >= int(cut["rollout_id"]):
                take.rolled_back = True
        self.trace("trainer_start", actor=trainer.actor, watermark=cut["watermark"], rollout_id=cut["rollout_id"])

    # ---------------------------------------------------------------- scheduling
    def _procs(self) -> list[Any]:
        out: list[Any] = [p for p in self.samplers.values() if p is not None and p.state == "running"]
        out += [p for p in self.zombies if p.state == "running"]
        if self.trainer is not None and self.trainer.state == "running":
            out.append(self.trainer)
        out += [t for t, _ in self.trainer_zombies if t.state == "running"]
        return out

    def _run(self, proc: Any, action: Callable[[], None], *, crash: bool = False) -> None:
        """One action of one participant; what leaves it ends the process."""
        if proc in self._stack:
            return
        self._stack.append(proc)
        if crash and self.rng.random() < self.config.faults.crash:
            self.injector.crash_in[proc.actor] = (self.rng.randint(0, 10), self.rng.random() < 0.5)
        try:
            action()
        except Blocked:
            pass
        except Crash:
            proc.killed()
            self.metric(f"{proc.kind}_crashes")
        except Exception as error:  # noqa: BLE001 - the process exits on it
            transport = isinstance(error, TRANSPORT_ERRORS)
            if type(error).__name__ == "TrainerFenced":
                self.metric("trainers_fenced_by_a_higher_generation")
            elif not transport:
                frame = next((f for f in reversed(traceback.extract_tb(error.__traceback__))
                              if "/yeto/" in f.filename), None)
                where = "" if frame is None else f" at {Path(frame.filename).name}:{frame.name}"
                self.errors[f"{proc.kind}: {type(error).__name__}: {str(error)[:110]}{where}"] += 1
            if not self.injector.enabled and type(error).__name__ != "TrainerFenced":
                self.errors_after_faults.append(f"{proc.actor}: {type(error).__name__}: {str(error)[:160]}")
            self.metric(f"{proc.kind}_exits_on_{'store_error' if transport else 'error'}")
            self.trace("process_error", actor=proc.actor, error=f"{type(error).__name__}: {str(error)[:200]}")
            try:
                proc.failed(error)
            except Crash:
                proc.killed()
        finally:
            self.injector.crash_in.pop(proc.actor, None)
            self._stack.pop()
        if proc.state != "running":
            self._left(proc)

    def _left(self, proc: Any) -> None:
        """A process ended: who starts the next one, and when."""
        delay = self.rng.uniform(1.0, 20.0)
        if proc.kind == "trainer":
            if proc is self.trainer and proc.state == "dead":
                self.trainer_start_at = self.clock.now + delay
            return
        if self.samplers.get(proc.sid) is proc and proc.state in ("dead", "fenced"):
            # a crash or an error: the supervisor restarts it; a fenced / retired process stays
            # down until the pool brings the shard back; a finished one is done
            self.samplers[proc.sid] = None
            self.start_at[proc.sid] = self.clock.now + (delay if proc.state == "dead" else 3 * delay)

    def _preempt(self, actor: str) -> None:
        """Another participant acts in the middle of ``actor``'s store call."""
        others = [p for p in self._procs() if p not in self._stack]
        if others:
            proc = self.rng.choice(others)
            self._run(proc, self.rng.choice(proc.actions()))

    def yield_from(self, proc: Any) -> None:
        """``proc`` sleeps in a real wait loop: the others go on."""
        if self.injector._depth:
            return
        self.injector._depth += 1
        try:
            self._preempt(proc.actor)
            self._due()
        finally:
            self.injector._depth -= 1

    def _start_sampler(self, sid: str) -> None:
        self.incarnations[sid] += 1
        proc = SamplerProc(self, sid, f"i{self.incarnations[sid]}", self.nodes[sid],
                           legacy=self.shards.index(sid) < self.config.legacy_samplers)
        self.samplers[sid] = proc
        del self.start_at[sid]
        self.trace("sampler_start", actor=proc.actor, node=proc.node)
        self._run(proc, proc.start, crash=self.injector.enabled)

    def _start_trainer(self) -> None:
        self.trainers += 1
        self.trainer = TrainerProc(self, self.trainers)
        self.trainer_start_at = None
        self._run(self.trainer, self.trainer.start, crash=self.injector.enabled)

    def _shard_ended(self, sid: str) -> bool:
        keys = self.server.keys(sample_buffer.shard_prefix(sid))
        return bool(keys) and bool(json.loads(self.server.get(keys[-1]) or b"{}").get("end"))

    def _due(self) -> None:
        now = self.clock.now
        for sid in [s for s, at in self.start_at.items() if at <= now]:
            if self.samplers[sid] is not None:
                continue
            if self._shard_ended(sid):  # nothing more can be written to it
                del self.start_at[sid]
                continue
            self._start_sampler(sid)
        if self.trainer_start_at is not None and self.trainer_start_at <= now:
            self._start_trainer()
        for proc in [p for p in self._procs() if p.kind == "sampler"]:
            if proc.state == "running" and proc.next_renew <= now:
                self._run(proc, proc.renew)
            if proc.state == "running" and proc.next_lease_poll <= now:
                self._run(proc, proc.lease_poll)
        for sid, (seq, give_up) in list(self.drains.items()):
            self._drain_follow_up(sid, seq, give_up)

    # ------------------------------------------------------------------ the pool
    def _pool(self, what: str, call: Callable[[], Any]) -> tuple[bool, Any]:
        """A pool operation, retried like the pool's next ticks do."""
        for _ in range(8):
            try:
                return True, call()
            except Crash:
                raise
            except Exception as error:  # noqa: BLE001
                self.metric("pool_call_retries")
                self.trace("pool_retry", what=what, error=type(error).__name__)
                self.clock.advance(5.0)
        self.metric("pool_call_gave_up")
        return False, None

    def _ticket(self) -> str:
        self.tickets += 1
        return f"ticket{self.tickets}"

    def _control(self) -> None:
        running = [p for p in self.samplers.values() if p is not None and p.state == "running"
                   and p not in self._stack]
        if not running:
            return
        proc = self.rng.choice(running)
        sid = proc.sid
        gone = self.injector.is_partitioned(proc.node)
        kinds = ["retire", "drain", "restart_in_place"] + (["fence_gone"] * 3 if gone else [])
        kind = self.rng.choice(kinds)
        if kind == "retire":
            ok, _ = self._pool("retire", lambda: store_dispense.retire(
                self.pool_objects, NAMESPACE, sid, proc.incarnation, reason="scale_down"))
            if ok:
                self.fenced_truth.setdefault((sid, proc.incarnation), self.clock.now)
                self.trace("pool_retire", actor=proc.actor)
                self.metric("pool_retire")
        elif kind == "drain" and sid not in self.drains:
            seq = f"{self._ticket()}:{self.clock.now:.3f}"
            ok, _ = self._pool("drain", lambda: self.data_path.request_drain(sid, seq))
            if ok:
                self.drains[sid] = (seq, self.clock.now + self.rng.uniform(20.0, 120.0))
                self.trace("pool_drain", actor=proc.actor, seq=seq)
                self.metric("pool_drain")
        elif kind == "restart_in_place":
            # a stalled engine: fence the incarnation, restart the container on the same node
            ok, _ = self._pool("fence_incarnation", lambda: self.data_path.fence_incarnation(sid, self._ticket()))
            if ok:
                self.fenced_truth.setdefault((sid, proc.incarnation), self.clock.now)
                proc.killed()
                self.samplers[sid] = None
                self.start_at[sid] = self.clock.now + self.rng.uniform(1.0, 10.0)
                self.trace("pool_restart_in_place", actor=proc.actor)
                self.metric("pool_restart_in_place")
        elif kind == "fence_gone" and self.fenced_shards < max(0, len(self.shards) - 1):
            # the node is taken for gone: fence its shard; a new node takes the shard name
            # later, while the old process may still run behind its partition
            ok, _ = self._pool("fence_shard", lambda: self.data_path.fence_shard(sid, self._ticket()))
            if ok:
                self.fenced_shards += 1
                self._replace_node(proc)
                self.trace("pool_fence_gone", actor=proc.actor)
                self.metric("pool_fence_gone")

    def _replace_node(self, proc: SamplerProc, *, kill: bool = False) -> None:
        sid = proc.sid
        self.fenced_truth.setdefault((sid, proc.incarnation), self.clock.now)
        self.samplers[sid] = None
        if kill:
            proc.killed()
        else:
            self.zombies.append(proc)
        self.nodes[sid] = f"{sid}-node{self.incarnations[sid]}"
        self.start_at[sid] = self.clock.now + self.rng.uniform(5.0, 60.0)

    def _drain_follow_up(self, sid: str, seq: str, give_up: float) -> None:
        proc = self.samplers.get(sid)
        if proc is None or proc.state != "running" or proc in self._stack:
            if proc is None or proc.state != "running":
                del self.drains[sid]
                self._pool("cancel_drain", lambda: self.data_path.cancel_drain(sid))
            return
        ok, answered = self._pool("drain_answered", lambda: self.data_path.drain_answered(sid, seq))
        if ok and answered:
            # drained: the pool fences the shard and deletes the node
            del self.drains[sid]
            ok, _ = self._pool("fence_shard", lambda: self.data_path.fence_shard(sid, self._ticket()))
            if ok:
                self._replace_node(proc, kill=True)
                self.metric("pool_scale_down_after_drain")
                self.trace("pool_scale_down", actor=proc.actor)
            self._pool("cancel_drain", lambda: self.data_path.cancel_drain(sid))
        elif self.clock.now >= give_up:
            del self.drains[sid]
            self._pool("cancel_drain", lambda: self.data_path.cancel_drain(sid))
            self.metric("pool_drain_cancelled")

    def _restart_trainer(self) -> None:
        old = self.trainer
        if old is None or old.state != "running" or old in self._stack:
            return
        self.metric("trainer_restarts")
        if self.config.faults.two_trainers:
            old.zombie = True
            self.trainer_zombies.append((old, self.steps + self.rng.randint(5, 60)))
            self.trace("trainer_superseded_still_running", actor=old.actor)
        else:
            old.killed()
            self.trace("trainer_stopped", actor=old.actor)
        self.trainer = None
        self.trainer_start_at = self.clock.now + self.rng.uniform(0.0, 10.0)

    def _clear_handoff_before_start(self, sid: str) -> None:
        """The pool's ``clear_stale_handoff`` for a new member on a shard name."""
        self._pool("clear_stale_handoff", lambda: self.data_path.clear_stale_handoff(sid, keep=()))

    # ---------------------------------------------------------------------- run
    def step(self) -> None:
        self.steps += 1
        faults = self.config.faults if self.injector.enabled else FaultConfig()
        self.clock.advance(self.rng.uniform(0.05, self.config.dt_max_s))
        for sid in [s for s, at in self.start_at.items() if at <= self.clock.now and self.samplers[s] is None]:
            if not self._shard_ended(sid) and self.server.get(f"sample-buffer/handoffs/{sid}.json"):
                self._clear_handoff_before_start(sid)
        self._due()
        self.trainer_zombies = [(t, until) for t, until in self.trainer_zombies
                                if t.state == "running" and until > self.steps]
        if faults.partition and self.rng.random() < faults.partition:
            nodes = sorted({p.node for p in self._procs()})
            if nodes:
                node = self.rng.choice(nodes)
                self.injector.partitioned[node] = self.clock.now + self.rng.uniform(5.0, faults.partition_max_s)
                self.metric("partitions")
                self.trace("partition", node=node, until=round(self.injector.partitioned[node] - 1_800_000_000.0, 1))
        if faults.read_outage and self.rng.random() < faults.read_outage:
            nodes = sorted({p.node for p in self._procs()})
            node = "*" if not nodes or self.rng.random() < 0.5 else self.rng.choice(nodes)
            self.injector.read_outage[node] = self.clock.now + self.rng.uniform(5.0, faults.read_outage_max_s)
            self.metric("read_outages")
            self.trace("read_outage", node=node,
                       until=round(self.injector.read_outage[node] - 1_800_000_000.0, 1))
        if faults.control and self.rng.random() < faults.control:
            self._control()
        if faults.trainer_restart and self.rng.random() < faults.trainer_restart:
            self._restart_trainer()
        procs = self._procs()
        if procs:
            proc = self.rng.choice(procs)
            self._run(proc, self.rng.choice(proc.actions()), crash=True)
            self._check_lost_claims()
        self._hook("on_step")

    def _check_lost_claims(self) -> None:
        """A claim whose create this sampler's own retry was refused for (PR #354):
        the ledger names it the holder; it must know."""
        for key, actor in list(self.server.own_retry_refusals.items()):
            if key in self._refusals_seen:
                continue
            self._refusals_seen.add(key)
            self.metric("creates_refused_after_own_retry")
            if not key.startswith(f"{NAMESPACE}/claims/"):
                continue
            slot, attempt = int(key.split("/")[-2]), int(key.split("/")[-1])
            proc = next((p for p in [*self.samplers.values(), *self.zombies]
                         if p is not None and p.actor == actor), None)
            if (actor, "get", key) in self.server.failed_calls:
                # the read-back that tells a retried create from a lost race failed too: the
                # claim stays unknown to its holder until its lease runs out
                self.metric("design_loss_claims_unknown_until_expiry")
                continue
            if (proc is not None and proc.state == "running" and slot not in proc.prompts._held
                    and self.truth_holds(slot, attempt, proc.sid, proc.incarnation)):
                self.violation("claim_held_without_the_holder_knowing", key=key, actor=actor, slot=slot)

    def done(self) -> bool:
        return self.trainer is not None and self.trainer.state == "done"

    def run(self) -> Result:
        with _patched(self.clock, self.rng):
            try:
                for _ in range(self.config.steps):
                    if self.done():
                        break
                    self.step()
                self._converge()
                self.check()
                self._hook("on_check")
            except Crash as error:  # pragma: no cover - a crash outside any participant
                self.violation("harness_crash_escaped", error=str(error))
        return Result(self.config, self.violations, dict(sorted(self.metrics.items())),
                      dict(sorted(self.errors.items())), self.steps, list(self._trace))

    def _converge(self) -> None:
        """Faults stop; what is down comes back; the run must end."""
        self.injector.enabled = False
        self.injector.partitioned.clear()
        self.injector.read_outage.clear()
        self.injector.crash_in.clear()
        self.trace("faults_off")
        for trainer, _ in self.trainer_zombies:
            trainer.killed()
        self.trainer_zombies = []
        for sid in list(self.drains):
            del self.drains[sid]
            self.data_path.cancel_drain(sid)
        for sid in self.shards:  # the pool brings every shard that can still write back
            if self.samplers[sid] is None and sid not in self.start_at:
                self.start_at[sid] = self.clock.now
        begun = self.steps
        while not self.done():
            if self.steps - begun >= self.config.converge_steps or len(self.errors_after_faults) >= 20:
                self.violation("run_did_not_end", **self._stuck())
                return
            self.step()
        self.metric("converge_steps", self.steps - begun)

    def _stuck(self) -> dict[str, Any]:
        totals = store_dispense.status(self.oracle, NAMESPACE, details=False)["totals"]
        return {"ledger": totals, "exits_with_faults_off": len(self.errors_after_faults),
                "last_exit": self.errors_after_faults[-1] if self.errors_after_faults else None,
                "shards_ended": {sid: self._shard_ended(sid) for sid in self.shards},
                "samplers": {sid: None if p is None else p.state for sid, p in self.samplers.items()},
                "trainer": None if self.trainer is None else
                {"state": self.trainer.state, "read": self.trainer.buffer.read_cursor,
                 "index": len(self.trainer.buffer), "ended": self.trainer.buffer.data_ended}}

    # -------------------------------------------------------------------- checks
    def check(self) -> None:
        if "run_did_not_end" in self.violations:
            return
        server = self.server
        # -- every shard: each record once
        shard_records: dict[str, list[dict[str, Any]]] = {}
        groups_of_slot: dict[int, list[str]] = collections.defaultdict(list)
        for sid in self.shards:
            records = [json.loads(server.get(k)) for k in server.keys(sample_buffer.shard_prefix(sid))]
            records = [r for r in records if "sample_id" in r]  # not the end, not an ownership record
            shard_records[sid] = records
            seen: dict[str, int] = {}
            for record in records:
                if record["sample_id"] in seen:
                    self.violation("shard_record_written_twice", shard=sid, sample_id=record["sample_id"],
                                   positions=[seen[record["sample_id"]], record["pos"]])
                seen[record["sample_id"]] = record["pos"]
                slot = (record.get("sampler_round") or {}).get("dispense_position")
                if slot is not None and record["sample_id"] not in groups_of_slot[int(slot)]:
                    groups_of_slot[int(slot)].append(record["sample_id"])
        # -- the merged index: each shard record exactly once, each shard in order
        merged = [json.loads(server.get(k)) for k in server.keys(INDEX_PREFIX)]
        ended = bool(merged) and bool(merged[-1].get("end"))
        if not ended:
            self.violation("merged_index_has_no_end")
        by_shard: dict[str, list[int]] = collections.defaultdict(list)
        for i, record in enumerate(merged):
            if int(record["pos"]) != i:
                self.violation("merged_index_position_gap", position=i, record_pos=record["pos"])
            if record.get("end"):
                continue
            by_shard[record["shard"]].append(int(record["shard_pos"]))
        for sid in self.shards:
            got, at = by_shard.get(sid, []), {int(r["pos"]): r["sample_id"] for r in shard_records[sid]}
            twice = sorted({p for p in got if got.count(p) > 1})
            if twice:
                self.violation("shard_record_merged_twice", shard=sid, shard_positions=twice[:10],
                               sample_id=at.get(twice[0]))
            missing = sorted(set(at) - set(got))
            # a merge that reads the shard's ownership records leaves a fenced writer's group out
            dropped = [p for p in missing if (self.intents.get(at[p]) or {}).get("fenced")
                       or (self.intents.get(at[p]) or {}).get("retired")] if MERGE_DROPS_FENCED_WRITES else []
            self.metric("design_loss_fenced_writes_left_out_of_the_merge", len(dropped))
            missing = [p for p in missing if p not in dropped]
            if missing and ended:
                self.violation("shard_record_never_merged", shard=sid, shard_positions=missing[:10],
                               sample_id=at[missing[0]])
            if got != sorted(got):
                self.violation("shard_merged_out_of_order", shard=sid)
        # -- what the trainer consumed (rounds after a restored cut were trained again: by design)
        takes = [t for t in self.takes if not t.zombie and not t.rolled_back]
        trained = [g for t in takes for g in t.trained]
        skipped = [g for t in takes for g in t.skipped]
        for name, key in (("sample_id", lambda g: g["sample_id"]),
                          ("record", lambda g: (g["shard"], g["shard_pos"])),
                          ("slot", lambda g: g["slot"])):
            counts = collections.Counter(key(g) for g in trained if key(g) is not None)
            for value, n in counts.items():
                if n > 1:
                    first = next(g for g in trained if key(g) == value)
                    self.violation(f"trained_twice_same_{name}", value=value, times=n,
                                   sample_id=first["sample_id"], slot=first["slot"])
        consumed = {g["position"] for g in trained} | {g["position"] for g in skipped}
        unread = [i for i, r in enumerate(merged) if not r.get("end") and i not in consumed]
        if unread:
            self.violation("merged_record_never_consumed", positions=unread[:10],
                           sample_id=merged[unread[0]].get("sample_id"))
        # -- versions and ownership of what was trained
        for group in trained:
            intent = self.intents.get(group["sample_id"])
            payload = group["payload"]
            if intent is None:
                self.violation("trained_group_nobody_wrote", sample_id=group["sample_id"])
                continue
            if not (int(group["behavior_version"]) == int(payload["version"]) == int(intent["version"])):
                self.violation("trained_with_another_behavior_version", sample_id=group["sample_id"],
                               index=group["behavior_version"], payload=payload["version"], meant=intent["version"])
            if intent["fenced"]:
                self.violation("fenced_writer_group_trained", sample_id=group["sample_id"], slot=group["slot"],
                               actor=f"{intent['sampler']}/{intent['incarnation']}")
            elif intent["retired"]:  # the pool retired (or fenced) the process before it wrote
                self.violation("retired_writer_group_trained", sample_id=group["sample_id"], slot=group["slot"],
                               actor=f"{intent['sampler']}/{intent['incarnation']}")
            if intent["policy_hash"] is not None and group["trainer_hash"] not in (None, intent["policy_hash"]):
                self.metric("design_loss_trained_groups_of_a_rolled_back_lineage")
        # -- the ledger: every slot settled; a trained slot's group reached the trainer
        keys, _now = self.oracle.list_meta(NAMESPACE + "/")
        states = store_dispense.parse_ledger(keys, NAMESPACE)
        trained_slots = {g["slot"] for g in trained}
        skipped_slots = {g["slot"] for g in skipped}
        for slot in range(self.config.slots):
            done = states[slot].done if slot in states else None
            if done is None:
                self.violation("slot_never_settled", slot=slot)
            elif done != "trained":
                self.metric(f"design_loss_slots_{done}")
                if slot in trained_slots:
                    self.metric("slots_trained_but_settled_otherwise")
            elif slot in trained_slots:
                self.metric("slots_trained")
            elif slot in skipped_slots:
                self.metric("design_loss_slots_trained_in_ledger_but_passed_over")
            elif not groups_of_slot.get(slot):
                self.violation("slot_settled_trained_without_a_group", slot=slot)
            else:
                self.violation("slot_settled_trained_but_never_consumed", slot=slot,
                               sample_id=groups_of_slot[slot][0])
        self.metric("design_loss_groups_passed_over", sum(t.report["passed_over"] for t in takes))
        self.metric("design_loss_groups_stale_policy_age", sum(t.report["stale"] for t in takes))
        self.metric("design_loss_groups_stale_incarnation", sum(t.report["stale_incarnation"] for t in takes))
        self.metric("groups_trained", len(trained))
        self.metric("groups_written", sum(len(r) for r in shard_records.values()))
        self.metric("trainer_rounds_rolled_back", sum(1 for t in self.takes if t.rolled_back and not t.zombie))
        # -- storage: releases the trainer should have pruned
        names = {k.split("/")[1] for k in server.keys("policy-releases/") if k.count("/") >= 2}
        self.metric("release_dirs_left", len(names))
        if len(names) > self.config.keep_releases:
            self.metric("release_dirs_never_pruned", len(names) - self.config.keep_releases)
        for name, count in self.injector.counts.items():
            self.metric(f"fault_{name}", count)
        self.metric("store_requests", server.requests)


def run_seed(config: SimConfig, *, keep_dir: bool = False) -> Result:
    tmp = Path(tempfile.mkdtemp(prefix=f"store-sim-{config.seed}-"))
    try:
        return Simulation(config, tmp).run()
    finally:
        if not keep_dir:
            shutil.rmtree(tmp, ignore_errors=True)


__all__ = ["CUT_KEY", "KNOWN_DESIGN_GAPS", "Result", "SimConfig", "Simulation", "run_seed"]
