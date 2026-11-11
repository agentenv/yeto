"""Eviction-tolerant ("elastic") policy pieces of the National Compute sampler pool.

:class:`~.lease_pool.LeasePool` keeps ``target`` samplers writing one buffer
across 4 h leases. This module holds what it needs to survive a lease that goes
away *early* or *late*, and nothing that talks to a cloud:

What National Compute does (protocol reference section 9.3 plus measurements):

* a ticket is ``held``/``queued`` (free) -> ``provisioning`` -> ``active``
  (billed per GPU hour) -> ``releasing`` -> ``ended`` with ``end_reason`` one of
  ``released`` (we deleted it), ``preempted``, ``lost`` (supply failure, also
  before a grant), ``balance`` (organisation balance ran out), ``admin``;
* ``preempt_at`` is set on an active ticket the platform will take back;
* **a lease is not ended at ``lease_until``**: a ticket read ``active`` with
  ``ended_at=null`` 19.5 minutes past its ``lease_until`` and ended only when it
  was deleted (``end_reason=released``). So the pool never relies on the platform
  to end a lease: it drains before ``lease_until`` and a patrol deletes any ticket
  of the run found ``active`` past it;
* a node can drop off the network for good while its ticket stays ``active``
  (billed): reachability is the sampler pool's job, not the platform's;
* there is no renewal (no extend call; ``max_lease_hours`` 4): "renewing" is a
  replacement order on another account placed ``replace_lead_s`` before the end.

Pieces:

* :func:`classify` - one member's ticket status + node probe -> a
  :class:`Verdict` (healthy, network blip, unreachable/lost, sampler crashed,
  evicted, expired, overdue, balance, ...);
* :class:`Breaker` - consecutive provisioning failures open it (exponential
  cooldown); too many trips stop ordering for good;
* :class:`Budget` - run spend cap, concurrent node cap, order cap;
* :class:`EventLog` - JSONL events (``nc_elastic_*``) plus an optional hook;
* :func:`fence_shard` - isolate the shard of a node taken for gone: a handoff
  request (a live sampler stops at its next round) and a claim of the shard's
  incarnation lease (``yeto.rl.engine.incarnation_lease``) by the pool, so a
  node that comes back cannot add groups the trainer would take (the trainer's
  ``IncarnationGate`` drops groups of a superseded incarnation appended after
  the claim).
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

# -- ticket status ------------------------------------------------------------

EVICTION_REASONS = frozenset({"preempted", "lost", "admin"})


@dataclass(frozen=True)
class LeaseStatus:
    """One ticket as the order tool reports it. ``state``: ``active``, ``waiting``
    (held/queued/provisioning), ``ended``, or ``unknown`` (the status call failed:
    a control-plane blip, never acted on alone)."""

    state: str
    lease_until: float | None = None
    end_reason: str | None = None
    preempt_at: float | None = None
    gpus: int | None = None
    rate_ucr_per_gpu_hour: int | None = None

    @classmethod
    def of(cls, value: Any) -> LeaseStatus:
        """A :class:`LeaseStatus`, or the legacy ``(state, lease_until)`` pair."""
        if isinstance(value, LeaseStatus):
            return value
        state, until = value
        return cls(state=state, lease_until=until)


# -- node probe ---------------------------------------------------------------

PROBE_OK, PROBE_UNREACHABLE, PROBE_CRASHED, PROBE_STARTING = "ok", "unreachable", "crashed", "starting"


@dataclass(frozen=True)
class Probe:
    """One look at the node (one ssh round trip). ``node``: ``ok`` / ``unreachable``;
    ``sampler``: ``running`` / ``exited`` / ``absent``; ``exit_code`` of an exited
    container; ``joined``: the sampler wrote its JOIN (engine up, first round on its
    way); ``progress_age_s``: seconds since its event tape last grew (None: unknown).
    From the engine's ``/metrics`` (None: not answered): ``gen_tokens`` (the
    generation-token counter), ``running`` / ``queued`` (requests in the engine);
    ``trainer_wait``: the newest tape line is an ``rl_sampler_waiting`` heartbeat."""

    node: str
    sampler: str = "absent"
    exit_code: int | None = None
    joined: bool = False
    progress_age_s: float | None = None
    gen_tokens: float | None = None
    running: int | None = None
    queued: int | None = None
    trainer_wait: bool = False

    @classmethod
    def parse(cls, text: str, returncode: int) -> Probe:
        """The output of :func:`probe_script` (``returncode`` 255 / a timeout: no node)."""
        if returncode == 255 or "YETO_NC_PROBE" not in text:
            return cls(node="unreachable")
        doc: dict[str, str] = {}
        for line in text.splitlines():
            if line.startswith("YETO_NC_PROBE "):
                for part in line.split()[1:]:
                    key, _, value = part.partition("=")
                    doc[key] = value
        status = doc.get("status", "absent")
        sampler = "running" if status == "running" else ("absent" if status == "absent" else "exited")
        code = doc.get("exit")
        age = doc.get("age")

        def num(key: str) -> float | None:
            try:
                return float(doc[key])
            except (KeyError, ValueError):
                return None

        running, queued = num("run"), num("wq")
        return cls(node="ok", sampler=sampler, exit_code=int(code) if code not in (None, "", "-") else None,
                   joined=doc.get("joined") == "1",
                   progress_age_s=float(age) if age not in (None, "", "-") else None,
                   gen_tokens=num("tok"), running=None if running is None else int(running),
                   queued=None if queued is None else int(queued), trainer_wait=doc.get("tw") == "1")


def resolve_metrics_port(value: int | str | None, spec: Mapping[str, Any] | None) -> int | None:
    """The port whose ``/metrics`` the probe reads: an explicit port, else (``"spec"``) the
    sampler spec's ``engine.server.port``; None when neither gives one."""
    if value is None:
        return None
    if value != "spec":
        return int(value)
    port = (((spec or {}).get("engine") or {}).get("server") or {}).get("port")
    return int(port) if port is not None else None


def probe_script(container: str, tape_glob: str, join_pattern: str = "rl_sampler_join",
                 metrics_port: int | None = None) -> str:
    """Node shell: container state, exit code, JOIN seen, age of the newest tape,
    whether its last line is a trainer-wait heartbeat, and (``metrics_port``) the
    engine's generation-token counter and running / waiting requests from vLLM's
    ``/metrics`` on the node (the sampler runs ``--network host``). One line
    ``YETO_NC_PROBE status=... exit=... joined=0|1 age=... tw=0|1 tok=... run=... wq=...``;
    read-only."""
    metrics = "t=- r=- w=-; "
    if metrics_port is not None:
        metrics = (f"set -- $(curl -s -m 5 http://127.0.0.1:{int(metrics_port)}/metrics 2>/dev/null | awk "
                   "'/^vllm:generation_tokens_total/{t+=$NF;f=1} /^vllm:num_requests_running/{r+=$NF} "
                   "/^vllm:num_requests_waiting/{w+=$NF} END{if(f) printf \"%.0f %d %d\", t, r, w; "
                   "else print \"- - -\"}'); t=${1:--} r=${2:--} w=${3:--}; ")
    return (f"s=$(sudo docker inspect -f '{{{{.State.Status}}}} {{{{.State.ExitCode}}}}' {container} "
            "2>/dev/null || echo 'absent -'); set -- $s; "
            f"j=0; ls {tape_glob} >/dev/null 2>&1 && grep -qs {join_pattern} {tape_glob} && j=1; "
            f"[ $j = 1 ] || {{ sudo docker logs --tail 2000 {container} 2>&1 | grep -qs {join_pattern} && j=1; }}; "
            f"n=$(ls -t {tape_glob} 2>/dev/null | head -1); "
            "a=-; [ -n \"$n\" ] && a=$(( $(date +%s) - $(stat -c %Y \"$n\") )); "
            "tw=0; [ -n \"$n\" ] && tail -n 1 \"$n\" | grep -qs rl_sampler_waiting && tw=1; "
            "st=$1 ex=$2; " + metrics +
            "echo \"YETO_NC_PROBE status=$st exit=$ex joined=$j age=$a tw=$tw tok=$t run=$r wq=$w\"")


# -- config -------------------------------------------------------------------

@dataclass
class ElasticConfig:
    """The eviction-handling knobs (customer guide: ``nc-elastic`` section).

    Detection: ``unreachable_grace_s`` of failed node probes in a row before a
    node counts as lost (shorter failures are network blips and change nothing);
    ``control_plane_alert_s`` of failed status calls before an alert (the pool
    never acts on a failed status call); ``stall_s`` without event-tape progress
    of a joined sampler before it counts as crashed (None: off; set it above the
    longest round); ``engine_stall_s``: a ready sampler whose engine has requests
    running or queued (vLLM ``/metrics``) but whose generation-token counter has not
    moved for that long is stalled (a hung collective, e.g. an NCCL/RCCL kernel out
    of memory, leaves the container up and the engine silent); an idle engine
    (no requests: the sampler waits for the trainer, ``rl_sampler_waiting``) never
    is (None: off). A stalled engine is fenced and restarted in place; after
    ``max_stall_restart_failures`` restarts in a row with no token progress
    between them the node is retired. Recovery: ``max_sampler_restarts`` in-place restarts of a
    crashed sampler on a reachable node before the node is replaced (the count restarts at the
    node's first join, at a ``pool-retry``, and after ``sampler_restart_reset_s`` healthy);
    ``start_timeout_s`` from the order to a joined sampler (model pull + engine
    start); ``overdue_grace_s`` past ``lease_until`` before the patrol deletes a
    ticket still ``active``; ``lost_handoff_wait_s``: how long a replacement of a
    lost node waits for its predecessor's LEAVE (it never comes);
    ``stop_on_finish``: a sampler exiting 0 stops the pool. Limits: see
    :class:`Budget` and :class:`Breaker`."""

    unreachable_grace_s: float = 300.0
    control_plane_alert_s: float = 600.0
    stall_s: float | None = None
    engine_stall_s: float | None = 600.0
    # the engine server's port on the node (its /metrics): "spec" reads the sampler spec's
    # engine.server.port; an int overrides it; None: not read
    metrics_port: int | str | None = "spec"
    max_stall_restart_failures: int = 3
    max_sampler_restarts: int = 2
    # a sampler that ran this long since it joined starts a fresh restart count when it
    # crashes (None: the count never resets); restarts before the node's first join and
    # before a pool-retry never count
    sampler_restart_reset_s: float | None = 1800.0
    start_timeout_s: float = 3600.0
    overdue_grace_s: float = 0.0
    lost_handoff_wait_s: float = 60.0
    breaker_failures: int = 3
    breaker_cooldown_s: float = 600.0
    breaker_max_cooldown_s: float = 3600.0
    breaker_max_trips: int = 3
    max_run_spend_ucr: int | None = None
    max_nodes: int | None = None  # billed nodes at once (default: 2 x target)
    max_orders: int | None = None  # orders over the run
    stop_on_finish: bool = True  # a sampler that exits 0 ends the pool (the run is over)

    @classmethod
    def from_dict(cls, doc: Mapping[str, Any] | None) -> ElasticConfig:
        doc = dict(doc or {})
        known = {f.name for f in fields(cls)}
        unknown = set(doc) - known
        if unknown:
            raise ValueError(f"elastic config: unknown keys {sorted(unknown)} (known: {sorted(known)})")
        cfg = cls(**doc)
        for name in ("unreachable_grace_s", "control_plane_alert_s", "start_timeout_s", "overdue_grace_s",
                     "lost_handoff_wait_s", "breaker_cooldown_s", "breaker_max_cooldown_s"):
            if float(getattr(cfg, name)) < 0:
                raise ValueError(f"elastic config: {name} must be >= 0")
        if cfg.sampler_restart_reset_s is not None and float(cfg.sampler_restart_reset_s) < 0:
            raise ValueError("elastic config: sampler_restart_reset_s must be >= 0 (or null: never reset)")
        if cfg.engine_stall_s is not None and float(cfg.engine_stall_s) <= 0:
            raise ValueError("elastic config: engine_stall_s must be > 0 (or null: off)")
        if isinstance(cfg.metrics_port, str) and cfg.metrics_port != "spec":
            raise ValueError('elastic config: metrics_port must be a port, "spec" or null')
        if cfg.max_stall_restart_failures < 1:
            raise ValueError("elastic config: max_stall_restart_failures must be >= 1")
        if cfg.breaker_failures < 1 or cfg.breaker_max_trips < 1:
            raise ValueError("elastic config: breaker_failures and breaker_max_trips must be >= 1")
        return cfg


# -- verdicts -----------------------------------------------------------------

HEALTHY = "healthy"
WAITING = "waiting"  # not granted yet / sampler still starting
CONTROL_BLIP = "control_plane_blip"  # the status call failed
NETWORK_BLIP = "network_blip"  # node probe failed, inside the grace
LOST = "lost"  # active ticket, node unreachable past the grace
CRASHED = "sampler_crashed"  # node reachable, sampler exited / gone / stalled
FINISHED = "sampler_finished"  # the sampler exited 0 (its run is over)
CONFIG_ERROR = "sampler_config_error"  # the sampler refused its spec (exit 2): the same on any node
# exit codes of a deterministic refusal (engine_sampler: spec rejected / bad arguments)
CONFIG_EXIT_CODES = frozenset({2})
EVICTED = "evicted"  # ended: preempted / lost / admin
PREEMPT_NOTICE = "preempt_notice"  # active with preempt_at set
EXPIRED = "expired"  # ended at / after its lease end
OVERDUE = "overdue"  # active past lease_until (still billed)
BALANCE = "balance"  # ended for the balance
RELEASED = "released"  # ended released (by us, or a person in the console)


@dataclass(frozen=True)
class Verdict:
    kind: str
    detail: str = ""


def effective_end(status: LeaseStatus) -> float | None:
    ends = [t for t in (status.lease_until, status.preempt_at) if t is not None]
    return min(ends) if ends else None


def classify(status: LeaseStatus, probe: Probe | None, *, now: float, unreachable_since: float | None,
             cfg: ElasticConfig, joined: bool) -> Verdict:
    """The member's verdict this tick. ``probe`` None: not probed (not started yet);
    ``unreachable_since``: when node probes started failing (None: the last one was ok);
    ``joined``: the sampler joined before (a stall only counts then)."""
    if status.state == "unknown":
        return Verdict(CONTROL_BLIP)
    if status.state == "ended":
        reason = status.end_reason
        if reason in EVICTION_REASONS:
            return Verdict(EVICTED, reason)
        if reason == "balance":
            return Verdict(BALANCE, reason)
        if reason == "released":
            return Verdict(RELEASED, reason)
        if status.lease_until is not None and now >= status.lease_until:
            return Verdict(EXPIRED, reason or "")
        return Verdict(EVICTED, reason or "ended")
    if status.state != "active":
        return Verdict(WAITING, status.state)
    if status.lease_until is not None and now > status.lease_until + cfg.overdue_grace_s:
        return Verdict(OVERDUE, f"{now - status.lease_until:.0f}s past lease_until")
    if probe is not None and probe.node == PROBE_UNREACHABLE:
        since = now if unreachable_since is None else unreachable_since
        if now - since >= cfg.unreachable_grace_s:
            return Verdict(LOST, f"unreachable for {now - since:.0f}s")
        return Verdict(NETWORK_BLIP, f"unreachable for {now - since:.0f}s")
    if status.preempt_at is not None:
        return Verdict(PREEMPT_NOTICE, f"preempt_at in {status.preempt_at - now:.0f}s")
    if probe is not None and probe.sampler == "exited" and probe.exit_code == 0:
        return Verdict(FINISHED)
    if probe is not None and probe.sampler == "exited" and probe.exit_code in CONFIG_EXIT_CODES:
        return Verdict(CONFIG_ERROR, f"sampler exit={probe.exit_code} (spec or arguments refused)")
    if probe is not None and joined:
        if probe.sampler != "running":
            return Verdict(CRASHED, f"sampler {probe.sampler} exit={probe.exit_code}")
        if cfg.stall_s is not None and probe.progress_age_s is not None and probe.progress_age_s > cfg.stall_s:
            return Verdict(CRASHED, f"no progress for {probe.progress_age_s:.0f}s")
    if probe is not None and not joined and probe.sampler not in ("running",) and probe.node == PROBE_OK:
        return Verdict(CRASHED, f"sampler {probe.sampler} exit={probe.exit_code} before join")
    return Verdict(HEALTHY if joined or probe is None else WAITING)


# -- breaker and budget -------------------------------------------------------

@dataclass
class Breaker:
    """Consecutive provisioning failures (an order that fails, a ticket lost before
    it is granted, a node that never gets a sampler up) open the breaker for a
    cooldown that doubles per trip; ``max_trips`` trips stop ordering for good."""

    failures: int = 0
    trips: int = 0
    open_until: float = 0.0
    tripped: bool = False

    def allows(self, now: float) -> bool:
        return not self.tripped and now >= self.open_until

    def success(self) -> None:
        self.failures = 0

    def failure(self, now: float, cfg: ElasticConfig) -> str | None:
        """Count one failure; returns ``"open"`` / ``"tripped"`` when that happened."""
        self.failures += 1
        if self.failures < cfg.breaker_failures:
            return None
        self.failures = 0
        self.trips += 1
        if self.trips >= cfg.breaker_max_trips:
            self.tripped = True
            return "tripped"
        cooldown = min(cfg.breaker_cooldown_s * 2 ** (self.trips - 1), cfg.breaker_max_cooldown_s)
        self.open_until = now + cooldown
        return "open"


@dataclass
class Budget:
    """Spend of the run in ucr: the active seconds of every member times its rate
    (an estimate from the ticket's rate; the organisation balance is shared, so the
    balance cannot tell the run's spend)."""

    spent_ucr: float = 0.0
    orders: int = 0

    def charge(self, seconds: float, hourly_ucr: int) -> None:
        if seconds > 0:
            self.spent_ucr += seconds / 3600.0 * hourly_ucr

    def refusal(self, cfg: ElasticConfig, *, billed_nodes: int, target: int, hourly_ucr: int) -> str | None:
        """Why a new order is refused now (None: allowed)."""
        cap_nodes = cfg.max_nodes if cfg.max_nodes is not None else 2 * target
        if billed_nodes >= cap_nodes:
            return f"{billed_nodes} nodes billed, max_nodes {cap_nodes}"
        if cfg.max_orders is not None and self.orders >= cfg.max_orders:
            return f"{self.orders} orders placed, max_orders {cfg.max_orders}"
        if cfg.max_run_spend_ucr is not None and self.spent_ucr + hourly_ucr > cfg.max_run_spend_ucr:
            return (f"spend {self.spent_ucr:.0f} + one node hour {hourly_ucr} > "
                    f"max_run_spend_ucr {cfg.max_run_spend_ucr}")
        return None


# -- events -------------------------------------------------------------------

@dataclass
class EventLog:
    """``nc_elastic_*`` events: one JSON line each in ``path`` (None: not written),
    the pool log, and ``hook`` (e.g. a pager command; its failure is logged, never raised)."""

    path: Path | None = None
    run: str | None = None
    log: Callable[[str], None] = print
    hook: Callable[[dict[str, Any]], None] | None = None
    clock: Callable[[], float] = time.time
    events: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, event: str, **detail: Any) -> dict[str, Any]:
        doc = {"t": self.clock(), "event": f"nc_elastic_{event}", **({"run": self.run} if self.run else {}),
               **detail}
        self.events.append(doc)
        self.log(f"pool event {doc['event']} " + " ".join(f"{k}={v}" for k, v in detail.items()))
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(doc, sort_keys=True) + "\n")
        if self.hook is not None:
            try:
                self.hook(doc)
            except Exception as error:  # noqa: BLE001 - an alert path never stops the pool
                self.log(f"pool event hook failed: {error!r}")
        return doc


def command_hook(command: str, *, run: Callable[..., Any] | None = None) -> Callable[[dict[str, Any]], None]:
    """An event hook running ``command`` (shell) with the event JSON on stdin."""
    import subprocess  # noqa: PLC0415

    runner = run or subprocess.run

    def hook(doc: dict[str, Any]) -> None:
        runner(command, shell=True, input=json.dumps(doc), text=True, timeout=60, check=False,
               capture_output=True)

    return hook


# -- fencing ------------------------------------------------------------------

FENCE_PREFIX = "pool-fence"


def fence_incarnation(store: Any, shard: str, *, by: str, clock: Callable[[], float] = time.time) -> dict[str, Any]:
    """Supersede the running incarnation of ``shard`` without asking the shard to stop
    (no handoff record): the sampler restarted in place on the same shard claims the
    lease anew at its join. Used for a stalled engine: its claims are released at once."""
    from yeto.rl.engine.incarnation_lease import IncarnationLease  # noqa: PLC0415

    return IncarnationLease(store, shard, f"{FENCE_PREFIX}:{by}", clock=clock).claim()


def fence_shard(store: Any, shard: str, *, by: str, clock: Callable[[], float] = time.time) -> dict[str, Any]:
    """Isolate ``shard`` (its node is taken for gone): ask its sampler to stop
    (handoff record) and claim its incarnation lease as ``pool-fence:<by>``. A
    sampler of the shard that is still running (a node back from a network
    partition) is fenced at its next lease read and writes no further group; a
    group it wrote anyway after the claim (minus the gate's clock margin) is a
    stale incarnation to the trainer. Groups written before stay (they are whole:
    a group is one object). Returns the lease record."""
    from yeto.rl.engine.incarnation_lease import IncarnationLease  # noqa: PLC0415
    from yeto.rl.engine.sampler_resume import request_handoff  # noqa: PLC0415

    request_handoff(store, shard, f"{FENCE_PREFIX}:{by}", clock=clock)
    return IncarnationLease(store, shard, f"{FENCE_PREFIX}:{by}", clock=clock).claim()


class ObjectRecords:
    """The record view ``fence_shard`` needs over an object API (``read``/``write``/``list``)."""

    def __init__(self, objects: Any):
        self.objects = objects


def s3_fence_store(uri: str, env_file: str | os.PathLike[str] | None,
                   environ: Mapping[str, str] | None = None) -> ObjectRecords:
    """The run's store as records for :func:`fence_shard`, from the launcher: the
    buffer URI and an env file in the sampler's format (``AWS_ACCESS_KEY_ID``,
    ``AWS_SECRET_ACCESS_KEY``, ``AWS_ENDPOINT_URL``)."""
    from yeto.cloud import s3_store  # noqa: PLC0415

    env = dict(os.environ if environ is None else environ)
    if env_file:
        env.update(read_env_file(env_file))
    env.setdefault(s3_store.CREDENTIALS_ENV, s3_store.CREDENTIALS_AWS_ENV)
    if env.get(s3_store.AWS_ENDPOINT_ENV):
        env.setdefault(s3_store.ENDPOINT_ENV, env[s3_store.AWS_ENDPOINT_ENV])
    env.setdefault(s3_store.REGION_ENV, "auto")
    return ObjectRecords(s3_store.head_objects(uri, environ=env))


def read_env_file(path: str | os.PathLike[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.removeprefix("export ").strip()] = value.strip().strip("'\"")
    return out


def snapshot(obj: Any) -> dict[str, Any]:
    return asdict(obj)
