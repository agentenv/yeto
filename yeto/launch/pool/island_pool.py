"""Provider-neutral, eviction-tolerant rollout island pool.

:class:`IslandPool` keeps ``target`` samplers ("islands") writing one buffer, on
machines a provider (:mod:`yeto.launch.pool.provider`) supplies. It was written as
the National Compute lease pool and keeps every behaviour of it
(``yeto.launch.nc.lease_pool.LeasePool`` is this class):

* lease-bounded providers (National Compute): a member whose lease ends within
  ``replace_lead_s`` gets a replacement ordered on a free slot (make before
  break); the replacement writes a new shard (``<base>-<n>``) of the same buffer
  and continues the prompts of the shard it replaces (``continues_shard``); the
  old member is drained once the replacement is ready, and anyway
  ``end_margin_s`` before its lease end. A provider without leases reports no
  ``lease_until`` and none of this applies;
* every tick reads each member's status and, once its sampler is started, probes
  its node; :func:`~.status.classify` tells an eviction, a preemption notice, an
  expiry, an overdue lease, a lost node, a network blip, a crashed or stalled
  sampler and a failed status call apart;
* a gone member is retired: its shard is fenced (the work contract releases its
  claims at once), its machine is released and confirmed ended, and its data
  slot and shard go to a pending handoff that the next order continues;
* a crashed sampler on a reachable node is restarted in place; a stalled engine
  is fenced (incarnation only) and restarted in place;
* orders are gated by a circuit breaker, a budget (run spend, billed nodes,
  orders) and an optional external gate (the controller's run budget);
* the patrol adopts any open machine of this run no live member owns;
* ``scale_path`` (a control file: target, retire list, seq, run id) changes the
  target of a running pool; a lowered target fences and drains the chosen
  members.

The pool is pure policy over :class:`PoolOps` (provision, start, probe, fence,
drain, balance) and a clock; :class:`ProviderPoolOps` builds the ops from an
:class:`~.provider.IslandProvider`, a :class:`~.provider.NodeRoute` and a
:class:`~.provider.Fencer`. A member's ``account`` is its provider slot (an int
for one provider; ``"<provider>:<slot>"`` for a :class:`~.provider.MultiProvider`).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from . import status as el

LEASE_HOURS = 4  # the default lease of a lease-bounded provider (National Compute)

METRICS_MISS_ALERT = 3  # probes in a row without engine /metrics before an event
PROVISIONING, STARTING, READY, DRAINING, GONE = "provisioning", "starting", "ready", "draining", "gone"


@dataclass
class Member:
    account: Any  # the provider slot (National Compute: the account number)
    ticket: str
    shard: str
    state: str = PROVISIONING
    lease_until: float | None = None
    replaces: str | None = None  # ticket of the member this one replaces
    replaced_by: str | None = None
    created: float = 0.0
    data_slot: str | None = None  # "I/N": a replacement keeps the slot of the member it replaces
    replaces_shard: str | None = None  # its sampler continues that shard's prompts (continues_shard)
    # -- elastic bookkeeping
    granted: float | None = None  # first tick the ticket read active
    charged_to: float | None = None  # spend counted up to here
    hourly_ucr: int | None = None
    started: float | None = None  # sampler start (or the last in-place restart)
    joined: bool = False
    restarts: int = 0
    unreachable_since: float | None = None
    status_fail_since: float | None = None
    alerted: list[str] = field(default_factory=list)  # one-shot events already sent
    predecessor_lost: bool = False  # the shard it continues has no live sampler to wait for
    handed_off: bool = False  # a successor that continues this shard started (its sampler stops)
    end_reason: str | None = None
    drain_error: str | None = None
    config_error: str | None = None  # its sampler refused the config: kept as is until pool-retry
    tokens_seen: float | None = None  # the engine's generation-token counter at the last probe
    tokens_moved_at: float | None = None  # last probe it moved, or the engine was idle
    stall_failures: int = 0  # stall restarts in a row with no token progress between them
    metrics_misses: int = 0  # probes in a row of a running sampler whose engine /metrics did not read
    # scale down in "drain" mode: since when its sampler claims no new task (None: not retiring)
    retiring_since: float | None = None
    drain_seq: str | None = None  # the drain request its sampler answers (set by ops.request_drain)
    paused: bool = False  # pool-pause: no automatic action on it until pool-resume (manual)


class PoolOps(Protocol):
    def order(self, account: Any) -> str: ...  # ticket id
    def poll(self, member: Member) -> Any: ...  # el.LeaseStatus, or ("active"/"waiting"/"ended", lease_until)
    def start(self, member: Member, shards: list[str]) -> None: ...  # bootstrap + sampler
    def ready(self, member: Member) -> bool: ...  # used when ops has no ``probe``
    def drain(self, member: Member) -> None: ...  # stop sampler, release node, delete own ticket, confirm ended
    def balance(self) -> int | None: ...
    def account_free(self, account: Any) -> bool: ...
    # optional (``getattr``): probe(member) -> el.Probe; restart(member, shards); fence(member);
    # fence_incarnation(member) (lease only, no handoff: a stalled engine restarted in place);
    # run_tickets() -> [(account, ticket)] open tickets of this run in the ledger;
    # scale down in "drain" mode: request_drain(member) (its sampler claims no new task),
    # drained(member) -> bool (it holds no task claim any more), cancel_drain(member)


@dataclass
class PoolConfig:
    target: int
    accounts: list[Any]  # the provider slots (one machine per slot at a time)
    shard_base: str
    lease_s: float = LEASE_HOURS * 3600
    replace_lead_s: float = 60 * 60
    end_margin_s: float = 10 * 60
    stop_below_ucr: int = 20_000_000
    min_balance_to_order_ucr: int | None = None  # default: one hour of a node + the stop line
    shard_limit: int | None = None  # shard indexes 0..N-1 only (buffer shards, tunnel ports)
    data_slots: int | None = None  # N: members split the prompts "I/N" (None: every member reads all)
    elastic: el.ElasticConfig = field(default_factory=el.ElasticConfig)
    # an open ticket of this run that no member owns is adopted; True deletes it instead
    release_orphans: bool = False
    # one node hour in the provider's cost unit, when the provider reports no rate
    # (budget gate, order floor); the National Compute config uses the MI355X rate
    hourly_cost: float = 0

    def hourly(self) -> float:
        return self.hourly_cost

    def order_floor(self) -> float:
        if self.min_balance_to_order_ucr is not None:
            return self.min_balance_to_order_ucr
        return self.stop_below_ucr + self.hourly()


def is_config_error(error: BaseException) -> bool:
    """A start failure that fails the same way on any node: the agentic setup
    (``SetupError``: image pull, keys, Harbor), or an error that says so
    (``config_error = True``). The node is kept for ``pool-retry``."""
    return type(error).__name__ == "SetupError" or bool(getattr(error, "config_error", False))


@dataclass
class IslandPool:
    config: PoolConfig
    ops: PoolOps
    clock: Callable[[], float] = time.time
    log: Callable[[str], None] = print
    events: el.EventLog | None = None
    members: list[Member] = field(default_factory=list)
    next_shard: int = 0
    stopped: str | None = None
    finished: str | None = None  # a sampler ended normally (data ended): nothing is ordered again
    halted: str | None = None  # a configuration error: no order, restart or fence until retry()
    pending: list[dict[str, Any]] = field(default_factory=list)  # handoffs the next orders continue
    breaker: el.Breaker = field(default_factory=el.Breaker)
    budget: el.Budget = field(default_factory=el.Budget)
    _refused: str | None = None
    _starts: dict[str, Any] = field(default_factory=dict)  # ticket -> Future of a background start
    _executor: Any = None
    state_path: Path | None = None  # saved right after every order
    _recheck: set[str] = field(default_factory=set)  # restored STARTING members: probe before a new start
    # ``pool-scale``: the control file the running loop reads every tick (target, retire tickets)
    scale_path: Path | None = None
    scale_run: str | None = None  # the run id the scale file must carry (None: not checked)
    scale_seq: int = 0  # the last scale request whose ``retire`` list was applied
    shrink: int = 0  # members a lowered target still has to release
    _retire_req: list[str] = field(default_factory=list)
    scale_mode: str | None = None  # the last scale request's scale-down mode (None: elastic.scale_down)
    _no_account: str | None = None  # the last "no free account" refusal reported
    # the control file's manual entries: tickets a person released (``nc down``) and whether
    # the target was lowered for them already; paused tickets and those holding no tasks
    manual_released: dict[str, Any] = field(default_factory=dict)
    paused_req: set[str] = field(default_factory=set)
    hold_req: set[str] = field(default_factory=set)
    _warned: set[str] = field(default_factory=set)
    # () -> why no order may be placed now (None: allowed): the controller's run budget
    gate: Callable[[], str | None] | None = None

    def __post_init__(self) -> None:
        if self.events is None:
            self.events = el.EventLog(log=self.log, clock=self.clock)

    @property
    def ecfg(self) -> el.ElasticConfig:
        return self.config.elastic

    def _event(self, name: str, **detail: Any) -> None:
        assert self.events is not None
        self.events(name, **detail)

    # -- helpers -------------------------------------------------------------
    def live(self) -> list[Member]:
        return [m for m in self.members if m.state != GONE]

    def shards(self) -> list[str]:
        return [m.shard for m in self.members if m.state != GONE]

    def _free_slot(self) -> str | None:
        """The lowest data slot no live member holds (None without ``data_slots``)."""
        count = self.config.data_slots
        if not count:
            return None
        held = {m.data_slot for m in self.live() if m.replaced_by is None}
        held |= {p.get("data_slot") for p in self.pending}
        for index in range(int(count)):
            if f"{index}/{count}" not in held:
                return f"{index}/{count}"
        raise RuntimeError(f"pool: every one of the {count} data slots is held; target exceeds data_slots")

    def _free_account(self) -> int | None:
        busy = {m.account for m in self.live()}
        for account in self.config.accounts:
            if account not in busy and self.ops.account_free(account):
                return account
        return None

    def _gate(self, now: float) -> str | None:
        """Why no order may be placed now (breaker, budget), None when one may."""
        if self.breaker.tripped:
            return "breaker tripped"
        if not self.breaker.allows(now):
            return f"breaker open for {self.breaker.open_until - now:.0f}s"
        billed = len(self.live())
        if self.ecfg.max_nodes is None:  # the default cap (2 x target) leaves out nodes a scale down drains
            billed -= len(self.retiring())
        refused = self.budget.refusal(self.ecfg, billed_nodes=billed, target=self.config.target,
                                      hourly_ucr=self.config.hourly())
        if refused is None and self.gate is not None:
            try:
                refused = self.gate()
            except Exception as error:  # noqa: BLE001 - an unreadable budget refuses, never crashes the pool
                refused = f"run budget unreadable: {error!r}"[:300]
        return refused

    def _order(self, replaces: Member | None = None) -> Member | None:
        now = self.clock()
        balance = self.ops.balance()
        if balance is not None and balance < self.config.order_floor():
            self.log(f"pool: balance {balance} < order floor {self.config.order_floor()}: not ordering")
            return None
        refused = self._gate(now)
        if refused:
            if refused != self._refused and not refused.startswith("breaker open"):
                self._event("order_refused", why=refused)
            self._refused = refused
            return None
        self._refused = None
        account = self._free_account()
        if account is None:
            refused = (f"no free National Compute account: {len(self.live())} of {len(self.config.accounts)} "
                       f"accounts hold a node of this pool (one node per account), target {self.config.target}")
            if refused != self._no_account:  # once per change, not every tick
                self._event("order_refused", why=refused)
            self._no_account = refused
            return None
        self._no_account = None
        handoff = self.pending[0] if replaces is None and self.pending else None
        if self.config.shard_limit is not None and self.next_shard >= self.config.shard_limit:
            refused = (f"shard {self.config.shard_base}-{self.next_shard} is past shard_limit "
                       f"{self.config.shard_limit} (the trainer's buffer shards / the tunnel ports)")
            if refused != self._refused:
                self.log(f"pool: {refused}: not ordering")
                self._event("order_refused", why=refused)
            self._refused = refused
            return None
        shard = f"{self.config.shard_base}-{self.next_shard}"
        try:
            ticket = self.ops.order(account)
        except Exception as error:  # noqa: BLE001 - an order failure is counted, never fatal
            self._event("order_failed", account=account, error=repr(error)[:300])
            self._fail(now, f"order on account {account}")
            return None
        self.next_shard += 1
        self.budget.orders += 1
        if handoff is not None:
            self.pending.pop(0)
        if replaces is not None:
            slot, continues, lost = replaces.data_slot, replaces.shard, False
        elif handoff is not None:
            slot, continues, lost = handoff.get("data_slot"), handoff.get("shard"), bool(handoff.get("lost"))
        else:
            slot, continues, lost = self._free_slot(), None, False
        member = Member(account, ticket, shard, created=now, replaces=replaces.ticket if replaces else None,
                        data_slot=slot, replaces_shard=continues, predecessor_lost=lost)
        if replaces:
            replaces.replaced_by = ticket
        self.members.append(member)
        self._save_now()
        self.log(f"pool: ordered {ticket} on account {account} shard {shard}"
                 + (f" (replaces {replaces.ticket})" if replaces else "")
                 + (f" (continues {continues})" if continues and not replaces else ""))
        self._event("ordered", ticket=ticket, account=account, shard=shard, data_slot=slot,
                    continues_shard=continues, replaces=replaces.ticket if replaces else None)
        return member

    def _save_now(self) -> None:
        """Persist the state now (after an order): a crash before the tick's save must
        not leave a billed ticket that no state file knows."""
        if self.state_path is None:
            return
        try:
            self.save(self.state_path)
        except Exception as error:  # noqa: BLE001 - the patrol adopts the ticket anyway
            self.log(f"pool: could not save {self.state_path} after an order: {error!r}")

    def _fail(self, now: float, what: str) -> None:
        outcome = self.breaker.failure(now, self.ecfg)
        if outcome == "open":
            self._event("breaker_open", after=what, trips=self.breaker.trips,
                        cooldown_s=round(self.breaker.open_until - now))
        elif outcome == "tripped":
            self._event("breaker_tripped", after=what, trips=self.breaker.trips)

    def _charge(self, m: Member, now: float) -> None:
        if m.granted is None:
            return
        start = m.charged_to if m.charged_to is not None else m.granted
        self.budget.charge(now - start, m.hourly_ucr or self.config.hourly())
        m.charged_to = now

    def _drain(self, member: Member, why: str) -> bool:
        """Stop, release, delete; True once the ticket is confirmed gone. A failure
        keeps the member DRAINING (the next tick tries again: a ticket left open
        keeps billing)."""
        if member.state == GONE:
            return True
        if member.state != DRAINING:
            self.log(f"pool: draining {member.ticket} ({why})")
            member.state = DRAINING
        try:
            self.ops.drain(member)
        except Exception as error:  # noqa: BLE001
            member.drain_error = repr(error)[:300]
            self._event("drain_failed", ticket=member.ticket, account=member.account, why=why,
                        error=member.drain_error)
            return False
        self._charge(member, self.clock())
        member.state = GONE
        member.drain_error = None
        self._event("released", ticket=member.ticket, account=member.account, why=why,
                    spent_ucr=round(self.budget.spent_ucr))
        for m in self.members:  # a replacement that lost its predecessor is a plain member now
            if m.replaces == member.ticket:
                m.replaces = None
        return True

    def _fence(self, m: Member, why: str) -> None:
        fence = getattr(self.ops, "fence", None)
        if fence is None or m.started is None:
            return
        try:
            fence(m)
            self._event("shard_fenced", ticket=m.ticket, shard=m.shard, why=why)
        except Exception as error:  # noqa: BLE001 - the trainer's age filter is the second line
            self._event("fence_failed", ticket=m.ticket, shard=m.shard, error=repr(error)[:300])

    def _retire(self, m: Member, why: str, *, lost: bool) -> None:
        """A member that will not sample again: fence, release, hand its slot on."""
        if m.retiring_since is not None:  # a scale-down victim: nothing is handed on
            self._finish_retire(m, self.clock(), outcome=why)
            return
        self._fence(m, why)
        successor = next((s for s in self.members if s.state != GONE and s.replaces == m.ticket), None)
        if successor is not None:  # its replacement is on the way: it continues m's shard
            successor.predecessor_lost = successor.predecessor_lost or lost or m.state != READY
        else:
            if m.joined:
                continues = m.shard
            else:  # wrote nothing: continue what it was meant to continue
                continues = m.replaces_shard
            handoff = {"data_slot": m.data_slot, "shard": continues, "lost": True, "from": m.ticket}
            if m.replaces:  # its predecessor is still live: re-plan that member's replacement
                for p in self.members:
                    if p.ticket == m.replaces and p.state != GONE:
                        p.replaced_by = None
                        handoff = None
            if handoff is not None and (handoff["data_slot"] is not None or handoff["shard"] is not None):
                self.pending.append(handoff)
        self._drain(m, why)

    def _once(self, m: Member, key: str, name: str, **detail: Any) -> None:
        if key not in m.alerted:
            m.alerted.append(key)
            self._event(name, ticket=m.ticket, account=m.account, shard=m.shard, **detail)

    # -- scale (pool-scale) ----------------------------------------------------
    def apply_scale(self) -> None:
        """Read the ``pool-scale`` control file: its target replaces the config's (it
        persists across restarts); its ``retire`` tickets are applied once (by ``seq``)."""
        if self.scale_path is None or not self.scale_path.exists():
            return
        try:
            doc = json.loads(self.scale_path.read_text())
            target = int(doc["target"])
            seq = int(doc.get("seq", 0))
            mode = doc.get("mode")
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.log(f"pool: unreadable scale file {self.scale_path}: {error!r}; target unchanged")
            return
        if self.scale_run is not None and doc.get("run") != self.scale_run:
            if "scale_run" not in self._warned:
                self.log(f"pool: scale file {self.scale_path} is for run {doc.get('run')!r}, not "
                         f"{self.scale_run!r}: ignored")
                self._warned.add("scale_run")
            return
        if target < 0:
            return
        source = str(doc.get("source") or "manual")
        if target != self.config.target:
            if target < self.config.target:
                self.shrink = max(self.shrink, len(self.counted()) - target)
            self._event("target_changed", old=self.config.target, new=target, seq=seq, source=source)
            if source == "manual":
                self._event("manual_scale", old=self.config.target, new=target, seq=seq,
                            retire=list(doc.get("retire") or ()))
            self.log(f"pool: target {self.config.target} -> {target} (pool-scale, {source})")
            self.config.target = target
        if seq > self.scale_seq:
            self.scale_seq = seq
            self._retire_req = [str(t) for t in doc.get("retire") or ()]
            self.scale_mode = str(mode) if mode in el.SCALE_DOWN_MODES else None
        self.manual_released = dict(doc.get("manual_released") or {})
        self._apply_pause({str(t) for t in doc.get("paused") or ()}, {str(t) for t in doc.get("hold_tasks") or ()})

    def _apply_pause(self, paused: set[str], hold: set[str]) -> None:
        """``pool-pause`` / ``pool-resume``: a paused member gets no automatic action (no
        probe verdict, restart, retire, lease-end drain, replacement or scale-down pick)
        and keeps counting toward the target; ``hold`` ones also claim no new task."""
        self.paused_req, self.hold_req = paused, hold
        for m in self.live():
            if m.ticket in paused and not m.paused:
                m.paused = True
                self._event("manual_pause", ticket=m.ticket, account=m.account, shard=m.shard,
                            hold_tasks=m.ticket in hold)
                if m.ticket in hold and getattr(self.ops, "request_drain", None) is not None:
                    try:
                        self.ops.request_drain(m)
                    except Exception as error:  # noqa: BLE001 - the pause holds anyway
                        self.log(f"pool: task hold of {m.ticket} failed: {error!r}"[:300])
            elif m.paused and m.ticket not in paused:
                m.paused = False
                m.unreachable_since = m.status_fail_since = None
                m.tokens_moved_at = None  # no stall verdict from the paused time
                self._event("manual_resume", ticket=m.ticket, account=m.account, shard=m.shard)
                if m.drain_seq is not None and m.retiring_since is None \
                        and getattr(self.ops, "cancel_drain", None) is not None:
                    try:
                        self.ops.cancel_drain(m)
                    except Exception as error:  # noqa: BLE001
                        self.log(f"pool: task hold release of {m.ticket} failed: {error!r}"[:300])

    def record_target(self, target: int, *, source: str, **extra: Any) -> None:
        """The pool lowers its own target (a manual release it detected): written to the
        control file under the scale lock, so the target outlives a restart and a later
        ``pool-scale`` starts from it."""
        if self.scale_path is None:
            return
        import fcntl  # noqa: PLC0415

        self.scale_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.scale_path.with_suffix(".lock"), "a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                old = json.loads(self.scale_path.read_text()) if self.scale_path.exists() else {}
            except (OSError, ValueError):
                old = {}
            if self.scale_run is not None and old.get("run") not in (None, self.scale_run):
                old = {}
            seq = max(int(old.get("seq", 0)), self.scale_seq) + 1
            new = {**{k: v for k, v in old.items() if k in ("paused", "hold_tasks", "manual_released")},
                   "run": self.scale_run if self.scale_run is not None else old.get("run"), "target": int(target),
                   "seq": seq, "retire": [], "mode": None, "source": source, "requested": self.clock(),
                   "previous": self.config.target, **extra}
            tmp = self.scale_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(new, indent=1))
            tmp.replace(self.scale_path)
        self.scale_seq = seq
        self.config.target = int(target)

    def counted(self) -> list[Member]:
        """The members that count against the target, as the fill step counts them (a
        member whose replacement is ready does not; one draining does not)."""
        return [m for m in self.live() if m.state != DRAINING and m.retiring_since is None
                and not (m.replaced_by and m.state == READY)]

    def retiring(self) -> list[Member]:
        """Scale-down victims still finishing their trials in flight ("drain" mode)."""
        return [m for m in self.live() if m.retiring_since is not None and m.state != DRAINING]

    def _scale_victims(self, excess: int) -> list[Member]:
        counted = self.counted()
        chosen = [m for t in self._retire_req for m in counted if m.ticket == t][:excess]
        for m in chosen:  # a person named it: the pause ends with it
            m.paused = False
        rest = sorted((m for m in counted if m not in chosen and not m.paused),
                      key=lambda m: (m.state == READY, -(m.created or 0.0), m.hourly_ucr or self.config.hourly(),
                                     m.lease_until if m.lease_until is not None else float("inf")))
        return chosen + rest[:excess - len(chosen)]

    def _linked(self, m: Member) -> list[Member]:
        """``m`` with its live predecessor / replacement (one sampler slot together)."""
        out = [m]
        for x in self.live():
            if x is not m and x.state != DRAINING and (x.ticket in (m.replaces, m.replaced_by)
                                                       or m.ticket in (x.replaces, x.replaced_by)):
                out.append(x)
        return out

    def _scale_down(self, now: float) -> None:
        """Above target: choose the victims. "drain" mode (``elastic.scale_down``, or the
        request's ``--fence`` / ``--drain``): a ready victim's sampler is asked to claim no
        new task; it stays billed until its trials in flight are written (``drained``) or
        ``scale_drain_timeout_s`` passes, then goes as in "fence" mode. "fence" mode: fence
        at once (its dispense claims go back at once), drain (sampler stop, release, delete,
        ended). No replacement is ordered for a victim, nothing is handed on (with static
        ``data_slots`` its slot waits in ``pending`` for the next scale up)."""
        self._revive()
        excess = min(self.shrink, len(self.counted()) - self.config.target)
        if excess <= 0:
            self.shrink = 0
            if self._retire_req:
                self._event("retire_ignored", tickets=self._retire_req, why="not above target")
            self._retire_req = []
        else:
            victims = self._scale_victims(excess)
            self._retire_req = []
            self.shrink -= len(victims)
            mode = self.scale_mode or self.ecfg.scale_down
            for m in victims:
                if m.state in (GONE, DRAINING):
                    continue
                if mode == "drain" and m.state != READY:
                    # a replacement not ready yet: it goes now, the ready node it replaces drains
                    ready = [x for x in self._linked(m)[1:] if x.state == READY]
                    if ready and self._request_drain(ready[0], now):
                        if m.state not in (GONE, DRAINING):
                            self._finish_retire(m, now, outcome="linked to a draining node", alone=True)
                        continue
                if mode == "drain" and self._request_drain(m, now):
                    continue
                self._finish_retire(m, now, outcome="fenced" if mode == "fence" else "fenced (not draining)")
        for m in self.retiring():
            if m.state == GONE or m.retiring_since is None:
                continue
            if now - m.retiring_since >= float(self.ecfg.scale_drain_timeout_s):
                self._finish_retire(m, now, outcome="drain timeout")
            elif self._drained(m):
                self._finish_retire(m, now, outcome="drained")

    def _request_drain(self, m: Member, now: float) -> bool:
        """Start the "drain" scale down of ``m``: False when it cannot drain (not ready:
        nothing in flight; static data slots; no ``request_drain``; the request failed)."""
        request = getattr(self.ops, "request_drain", None)
        if m.state != READY or request is None or self.config.data_slots is not None:
            return False
        try:
            request(m)
        except Exception as error:  # noqa: BLE001 - fall back to a fence
            self._event("scale_drain_failed", ticket=m.ticket, shard=m.shard, error=repr(error)[:300])
            return False
        partners = self._linked(m)[1:]
        m.retiring_since = now
        m.replaced_by = None
        for x in partners:  # its make-before-break partner goes with it
            x.replaced_by = None
            if x.state != READY:  # nothing in flight yet: it goes now, alone
                self._finish_retire(x, now, outcome="linked to a draining node", alone=True)
                continue
            x.retiring_since = now
            try:
                request(x)
            except Exception:  # noqa: BLE001 - it times out and is fenced then
                pass
        self._event("scale_draining", ticket=m.ticket, account=m.account, shard=m.shard,
                    timeout_s=float(self.ecfg.scale_drain_timeout_s), target=self.config.target,
                    with_linked=[x.ticket for x in partners])
        return True

    def _drained(self, m: Member) -> bool:
        if m.state != READY:  # restarting / starting: nothing of it is in flight
            return True
        drained = getattr(self.ops, "drained", None)
        if drained is None:
            return False
        try:
            return bool(drained(m))
        except Exception as error:  # noqa: BLE001 - read again next tick; the timeout bounds it
            self.log(f"pool: drain check of {m.ticket} failed: {error!r}"[:300])
            return False

    def _revive(self) -> None:
        """A raised target while victims still drain: keep them instead of ordering."""
        cancel = getattr(self.ops, "cancel_drain", None)
        if cancel is None:
            return
        for m in sorted(self.retiring(), key=lambda x: -(x.retiring_since or 0.0)):
            if len(self.counted()) >= self.config.target:
                return
            if m.state != READY:
                continue
            try:
                cancel(m)
            except Exception as error:  # noqa: BLE001 - it finishes its drain; a new node is ordered
                self.log(f"pool: cancel drain of {m.ticket} failed: {error!r}"[:300])
                continue
            m.retiring_since = None
            self._event("scale_drain_cancelled", ticket=m.ticket, shard=m.shard, target=self.config.target)

    def _finish_retire(self, m: Member, now: float, *, outcome: str, alone: bool = False) -> None:
        """Fence and drain (release, delete, ended) ``m`` and its linked members that are not
        draining on their own (``alone``: ``m`` only); event ``scaled_down``."""
        waited = None if m.retiring_since is None else round(now - m.retiring_since)
        group = [m] if alone else [m] + [x for x in self._linked(m)[1:] if x.retiring_since is None]
        if self.config.data_slots is not None and (m.data_slot is not None or m.joined):
            # static slots: no dispenser takes the slot's prompts; it waits for the next scale up
            self.pending.append({"data_slot": m.data_slot, "shard": m.shard if m.joined else m.replaces_shard,
                                 "lost": False, "from": m.ticket, "paused": True})
            self._event("slot_paused", ticket=m.ticket, shard=m.shard, data_slot=m.data_slot)
        for x in group:
            x.retiring_since = None
            self._fence(x, "scale down")
            x.replaced_by = None
            x.end_reason = "scaled down"
            self._drain(x, "scale down")
        ended = None
        if m.state == GONE:
            try:
                status = self.ops.poll(m)
                ended = (status.state if hasattr(status, "state") else status[0]) == "ended"
            except Exception:  # noqa: BLE001 - the drain already confirmed the delete
                ended = None
        self._event("scaled_down", ticket=m.ticket, account=m.account, shard=m.shard,
                    target=self.config.target, released=m.state == GONE, ended=ended, outcome=outcome,
                    waited_s=waited, with_linked=[x.ticket for x in group[1:]])

    # -- one control step ----------------------------------------------------
    def tick(self) -> None:
        now = self.clock()
        if self.stopped:
            return
        self.apply_scale()
        balance = self.ops.balance()
        if balance is not None and balance < self.config.stop_below_ucr:
            self.stop(f"balance {balance} < {self.config.stop_below_ucr}")
            return

        for m in [m for m in self.members if m.state == DRAINING]:
            self._drain(m, "retry")

        for m in self.live():
            if m.paused and self._watch_paused(m, now):
                continue
            if m.config_error is not None:  # kept for pool-retry: only its ticket is read
                self._watch_kept(m, now)
            elif m.state in (PROVISIONING, STARTING, READY):
                self._watch(m, now)
                if self.stopped:
                    return

        # the predecessor of a ready replacement goes; anything at its lease end goes
        by_ticket = {m.ticket: m for m in self.members}
        for m in self.live():
            if m.state == READY and m.replaces and by_ticket.get(m.replaces) \
                    and by_ticket[m.replaces].state not in (GONE, DRAINING):
                self._drain(by_ticket[m.replaces], f"replacement {m.ticket} ready")
                m.replaces = None
        for m in self.live():
            if m.state != DRAINING and not m.paused and m.lease_until is not None \
                    and now >= m.lease_until - self.config.end_margin_s:
                if m.retiring_since is not None:
                    self._finish_retire(m, now, outcome="lease end")
                    continue
                if m.replaced_by is None or by_ticket.get(m.replaced_by) is None \
                        or by_ticket[m.replaced_by].state != READY:
                    self._handoff_slot(m)
                self._drain(m, "lease end")

        self._scale_down(now)
        if self.halted:  # a configuration error: nothing new until pool-retry (lease ends still apply)
            self._patrol(now)
            return
        if self.finished:  # the data ended: members finish or go at their lease end, none is added
            self._patrol(now)
            if not self.live():
                self.stop(self.finished)
            return

        # adopt this run's unowned tickets before ordering (a restarted pool must not order
        # a second node for a target its adopted node already fills)
        self._patrol(now)

        # replacements for members close to their lease end
        for m in self.live():
            if (m.state == READY and not m.replaced_by and m.retiring_since is None and not m.paused
                    and m.lease_until is not None and now >= m.lease_until - self.config.replace_lead_s):
                self._order(replaces=m)

        # fill up to target (members waiting for a replacement count until they go)
        while len([m for m in self.live() if m.retiring_since is None
                   and not (m.replaced_by and m.state == READY)]) < self.config.target:
            if self._order() is None:
                break
        if self.breaker.tripped and not self.live():
            self.stop("breaker tripped and no member left")

    def _watch_paused(self, m: Member, now: float) -> bool:
        """A paused member: only its ticket is read (spend, lease end). True: nothing else
        is done this tick; False: its ticket ended, so the pause is over and the member is
        watched as usual (an eviction is replaced, a release by a person is respected)."""
        status = self._status(m, now)
        if status.state == "active":
            if m.granted is None:
                m.granted = now
            self._charge(m, now)
            end = el.effective_end(status)
            if end is not None:
                m.lease_until = end
            if m.lease_until is not None and now >= m.lease_until - self.config.end_margin_s:
                self._once(m, "paused_lease_end", "manual_paused_past_lease_end", lease_until=m.lease_until)
            return True
        if status.state == "ended":
            m.paused = False
            self._event("manual_pause_ended", ticket=m.ticket, shard=m.shard, end_reason=status.end_reason)
            return False
        return True

    def _manual_release(self, m: Member, now: float, *, by_console: bool) -> None:
        """A person released ``m`` (``nc down`` or the NC console): lower the target by one
        (unless ``nc down`` did already, or a replacement in flight takes its place), order
        no replacement, hand nothing on."""
        entry = self.manual_released.get(m.ticket) or {}
        successor = next((s for s in self.members if s.state != GONE and s.replaces == m.ticket), None)
        old = self.config.target
        new = old
        if successor is None and not entry.get("target_lowered"):
            new = max(0, old - 1)
        self._event("manual_release_detected", ticket=m.ticket, account=m.account, shard=m.shard,
                    old=old, new=new, by="console" if by_console else "nc down",
                    detail=("a person released this node: the pool lowers its target and orders no replacement "
                            "(elastic.external_release replace: order one as before)"))
        if new != old:
            self.record_target(new, source="manual_release",
                               manual_released={**self.manual_released, m.ticket: {"target_lowered": True}})
            self.manual_released[m.ticket] = {"target_lowered": True}
        if m.replaces:  # it was a replacement: its predecessor is not replaced again by it
            for p in self.members:
                if p.ticket == m.replaces and p.state != GONE:
                    p.replaced_by = None
        m.retiring_since = m.retiring_since or now
        self._finish_retire(m, now, outcome="released by a person", alone=True)

    def _watch_kept(self, m: Member, now: float) -> None:
        """A member kept after a configuration error: its lease end and its ticket's
        end still count (an ended or overdue ticket is released); the node is not probed."""
        status = self._status(m, now)
        if status.state == "active":
            self._charge(m, now)
            end = el.effective_end(status)
            if end is not None:
                m.lease_until = end
        elif status.state == "ended":
            m.end_reason = status.end_reason
            self._charge(m, now)
            self._drain(m, "ended while halted")
        if status.state == "active" and m.lease_until is not None \
                and now > m.lease_until + self.ecfg.overdue_grace_s:
            self._drain(m, "overdue while halted")

    def _handoff_slot(self, m: Member) -> None:
        """A member drained at its lease end with no ready replacement: its slot and
        shard go to the replacement in flight, else to the next order."""
        successor = next((s for s in self.members if s.state != GONE and s.replaces == m.ticket), None)
        if successor is not None:
            return
        if m.joined or m.replaces_shard is not None or m.data_slot is not None:
            self.pending.append({"data_slot": m.data_slot, "shard": m.shard if m.joined else m.replaces_shard,
                                 "lost": False, "from": m.ticket})

    def _status(self, m: Member, now: float) -> el.LeaseStatus:
        try:
            status = el.LeaseStatus.of(self.ops.poll(m))
        except Exception as error:  # noqa: BLE001 - the control plane blinked; nothing is decided on it
            if m.status_fail_since is None:
                m.status_fail_since = now
            self.log(f"pool: status of {m.ticket} failed: {error!r}"[:300])
            return el.LeaseStatus("unknown")
        if m.status_fail_since is not None:
            m.status_fail_since = None
            if "control_plane" in m.alerted:
                m.alerted.remove("control_plane")
                self._event("control_plane_recovered", ticket=m.ticket)
        return status

    def _probe(self, m: Member) -> el.Probe | None:
        probe = getattr(self.ops, "probe", None)
        if probe is None or m.state not in (STARTING, READY) or self._starting(m):
            return None
        try:
            return probe(m)
        except Exception as error:  # noqa: BLE001 - a probe that cannot run is an unreachable node
            self.log(f"pool: probe of {m.ticket} failed: {error!r}"[:300])
            return el.Probe(node=el.PROBE_UNREACHABLE)

    def _watch(self, m: Member, now: float) -> None:
        cfg = self.ecfg
        if m.ticket in self._recheck:  # restored mid-start: keep it when its sampler runs
            self._recheck.discard(m.ticket)
            probe = self._probe(m)
            if probe is not None and probe.node != el.PROBE_UNREACHABLE and probe.sampler == "running":
                self._event("start_resumed", ticket=m.ticket, shard=m.shard, joined=bool(probe.joined))
            else:
                m.state = PROVISIONING  # every start step is idempotent: run it again
        future = self._starts.get(m.ticket)
        if future is not None and future.done():  # a background start finished
            self._starts.pop(m.ticket)
            if future.exception() is not None:
                self._start_failed(m, now, future.exception())
                return
        status = self._status(m, now)
        if status.state == "active":
            if m.granted is None:
                m.granted = now
                if status.gpus and status.rate_ucr_per_gpu_hour:
                    m.hourly_ucr = int(status.gpus) * int(status.rate_ucr_per_gpu_hour)
            self._charge(m, now)
            end = el.effective_end(status)
            if end is not None:
                m.lease_until = end
        probe = self._probe(m) if status.state == "active" else None
        if probe is not None:
            if probe.node == el.PROBE_UNREACHABLE:
                m.unreachable_since = m.unreachable_since if m.unreachable_since is not None else now
            else:
                if m.unreachable_since is not None:
                    self._event("node_reachable_again", ticket=m.ticket, after_s=round(now - m.unreachable_since))
                    if "blip" in m.alerted:
                        m.alerted.remove("blip")
                m.unreachable_since = None
            if probe.joined and not m.joined:
                m.joined = True
            self._metrics_seen(m, probe)
        verdict = el.classify(status, probe, now=now, unreachable_since=m.unreachable_since, cfg=cfg,
                              joined=m.joined if probe is None else (m.joined or probe.joined))
        kind = verdict.kind

        if kind == el.CONTROL_BLIP:
            if m.status_fail_since is not None and now - m.status_fail_since >= cfg.control_plane_alert_s:
                self._once(m, "control_plane", "control_plane_unreachable",
                           for_s=round(now - m.status_fail_since))
            return
        if kind in (el.EVICTED, el.EXPIRED, el.RELEASED, el.BALANCE):
            m.end_reason = status.end_reason
            self._charge(m, now)
            ours = kind == el.RELEASED and bool(getattr(self.ops, "released_by_us", lambda t: False)(m.ticket))
            if kind == el.RELEASED and (not ours or m.ticket in self.manual_released) \
                    and self.ecfg.external_release == "respect":
                self._manual_release(m, now, by_console=not ours)
                return
            self._event({el.EVICTED: "evicted", el.EXPIRED: "expired",
                         el.RELEASED: "released_by_earlier_pool" if ours else "released_externally",
                         el.BALANCE: "ended_for_balance"}[kind], ticket=m.ticket, account=m.account,
                        shard=m.shard, end_reason=status.end_reason, state_was=m.state)
            if m.state == PROVISIONING:  # never granted (e.g. lost in provisioning)
                self._fail(now, f"ticket {m.ticket} ended before a grant")
            self._retire(m, kind, lost=True)
            if kind == el.BALANCE:
                self.stop("a ticket ended for the balance")
            return
        if kind == el.OVERDUE:
            self._event("lease_overdue", ticket=m.ticket, account=m.account, shard=m.shard, detail=verdict.detail)
            self._retire(m, "overdue", lost=False)
            return
        if kind == el.LOST:
            self._event("node_lost", ticket=m.ticket, account=m.account, shard=m.shard, detail=verdict.detail)
            if not m.joined:
                self._fail(now, f"node of {m.ticket} lost before its sampler joined")
            self._retire(m, "node lost", lost=True)
            return
        if kind == el.NETWORK_BLIP:
            self._once(m, "blip", "network_blip", detail=verdict.detail)
            return
        if kind == el.PREEMPT_NOTICE:
            self._once(m, "preempt", "preempt_notice", detail=verdict.detail, preempt_at=status.preempt_at)
        if kind == el.FINISHED and m.handed_off:  # it stopped for its successor (sampler_resume handoff)
            if not any(s.state != GONE and s.replaces == m.ticket for s in self.members):
                self._once(m, "handed_off", "sampler_handed_off")
                self._retire(m, "handed off", lost=False)
            return
        if kind == el.FINISHED:
            self._event("sampler_finished", ticket=m.ticket, shard=m.shard)
            # a normal end (the data ended): no restart, no replacement node, no new order
            self.finished = self.finished or f"sampler {m.shard} finished (exit 0)"
            self.pending.clear()
            self._drain(m, "sampler finished")
            if self.ecfg.stop_on_finish:
                self.stop(f"sampler {m.shard} finished (exit 0)")
            return
        if kind == el.CRASHED:
            self._crashed(m, now, verdict.detail)
            return
        if kind == el.CONFIG_ERROR:
            self._config_error(m, verdict.detail)
            return
        if kind == el.HEALTHY and m.state == READY and probe is not None and self._engine_stalled(m, probe, now):
            return

        if m.state == PROVISIONING:
            if status.state == "active":  # a provider without leases (sky) has no lease_until
                m.state = STARTING
                m.started = now
                for p in self.members:  # its sampler asks the one of that shard to stop
                    if m.replaces_shard and p.shard == m.replaces_shard and p.state != GONE:
                        p.handed_off = True
                if getattr(self.ops, "start_in_background", False):
                    # node setup + model pull take tens of minutes: the other members
                    # stay watched meanwhile
                    if self._executor is None:
                        from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

                        self._executor = ThreadPoolExecutor(max_workers=max(2, len(self.config.accounts)),
                                                            thread_name_prefix="nc-start")
                    self._starts[m.ticket] = self._executor.submit(self.ops.start, m, self.shards())
                    return
                try:
                    self.ops.start(m, self.shards())
                except Exception as error:  # noqa: BLE001
                    self._start_failed(m, now, error)
                    return
            elif now - m.created > cfg.start_timeout_s:
                self._event("grant_timeout", ticket=m.ticket, account=m.account, state=status.state,
                            waited_s=round(now - m.created))
                self._fail(now, f"ticket {m.ticket} not granted")
                self._retire(m, "not granted", lost=False)
                return
        elif m.state == STARTING:
            ready = (m.joined if getattr(self.ops, "probe", None) is not None else self.ops.ready(m))
            if ready:
                m.state, m.joined = READY, True
                self.breaker.success()
                self.log(f"pool: {m.ticket} ready (shard {m.shard})")
                self._event("member_ready", ticket=m.ticket, account=m.account, shard=m.shard,
                            order_to_ready_s=round(now - m.created),
                            grant_to_ready_s=None if m.granted is None else round(now - m.granted),
                            restarts=m.restarts, continues_shard=m.replaces_shard)
            elif m.started is not None and now - m.created > cfg.start_timeout_s:
                self._event("start_timeout", ticket=m.ticket, shard=m.shard, waited_s=round(now - m.created))
                self._fail(now, f"sampler on {m.ticket} not up")
                self._retire(m, "start timeout", lost=False)

    def _config_error(self, m: Member, detail: str) -> None:
        """A deterministic refusal (the same config fails on every node): keep the node
        and its shard (no fence, no drain, no restart, no replacement), halt the pool
        until :meth:`retry`. Lease ends, evictions and the patrol still apply."""
        if m.config_error is None:
            m.config_error = detail
            self.halted = f"configuration error on {m.ticket} (shard {m.shard}): {detail}"
            self.log(f"pool: CONFIGURATION ERROR: {self.halted}; fix the config and run pool-retry")
            self._event("sampler_config_error", ticket=m.ticket, account=m.account, shard=m.shard, detail=detail)

    def retry(self, check: Callable[[], None] | None = None) -> list[str]:
        """Restart the sampler in place on every member halted by a configuration
        error (``check()`` raises when the fixed config is still wrong); returns
        their tickets."""
        if check is not None:
            check()
        restart = getattr(self.ops, "restart", None)
        done = []
        for m in self.live():
            if m.config_error is None:
                continue
            if restart is None:
                raise RuntimeError("these pool ops cannot restart a sampler in place")
            restart(m, self.shards())
            now = self.clock()
            m.config_error = None
            m.state, m.started, m.joined, m.created = STARTING, now, False, now
            self._event("sampler_retry", ticket=m.ticket, shard=m.shard)
            done.append(m.ticket)
        self.halted = None
        return done

    def _start_failed(self, m: Member, now: float, error: BaseException) -> None:
        if is_config_error(error):  # deterministic (setup, image, spec): keep the node, halt until pool-retry
            self._event("start_failed", ticket=m.ticket, shard=m.shard, error=repr(error)[:300], kept=True)
            self._config_error(m, repr(error)[:300])
            return
        self._event("start_failed", ticket=m.ticket, shard=m.shard, error=repr(error)[:300])
        self._fail(now, f"start on {m.ticket}")
        self._retire(m, "start failed", lost=False)

    def _starting(self, m: Member) -> bool:
        """A background start of ``m`` still runs (its node is not probed for a sampler yet)."""
        future = self._starts.get(m.ticket)
        return future is not None and not future.done()

    def _crashed(self, m: Member, now: float, detail: str) -> None:
        if m.retiring_since is not None:  # scaled down: not restarted
            self._finish_retire(m, now, outcome="sampler exited")
            return
        restart = getattr(self.ops, "restart", None)
        if restart is not None and m.restarts < self.ecfg.max_sampler_restarts:
            m.restarts += 1
            self._event("sampler_restart", ticket=m.ticket, shard=m.shard, attempt=m.restarts, detail=detail)
            try:
                restart(m, self.shards())
            except Exception as error:  # noqa: BLE001
                self._event("start_failed", ticket=m.ticket, shard=m.shard, error=repr(error)[:300])
                if is_config_error(error):
                    self._config_error(m, repr(error)[:300])
                    return
            m.state, m.started, m.joined = STARTING, now, False
            m.created = now  # the start timeout counts from the restart
            return
        self._event("sampler_crash_loop", ticket=m.ticket, shard=m.shard, restarts=m.restarts, detail=detail)
        self._fail(now, f"sampler on {m.ticket} crash loop")
        self._retire(m, "sampler crash loop", lost=False)

    def _metrics_seen(self, m: Member, probe: el.Probe) -> None:
        """Count probes in a row of a running sampler with no engine ``/metrics`` (wrong port,
        engine down): the stall detection is blind then; an event at the third."""
        if not getattr(self.ops, "reads_metrics", False) or probe.node == el.PROBE_UNREACHABLE \
                or probe.sampler != "running" or m.state != READY:
            return
        if probe.gen_tokens is not None:
            if m.metrics_misses >= METRICS_MISS_ALERT:
                self._event("engine_metrics_recovered", ticket=m.ticket, shard=m.shard, after=m.metrics_misses)
            m.metrics_misses = 0
            return
        m.metrics_misses += 1
        if m.metrics_misses == METRICS_MISS_ALERT:
            self._event("engine_metrics_unreadable", ticket=m.ticket, account=m.account, shard=m.shard,
                        misses=m.metrics_misses, port=getattr(self.ops, "metrics_port", None))

    def _engine_stalled(self, m: Member, probe: el.Probe, now: float) -> bool:
        """Track the engine's token counter; True when the member was handled as stalled
        (requests running / queued, no generated token for ``engine_stall_s``)."""
        limit = self.ecfg.engine_stall_s
        if limit is None or probe.gen_tokens is None or probe.sampler != "running":
            return False
        busy = (probe.running or 0) + (probe.queued or 0) > 0
        if m.tokens_seen is None or probe.gen_tokens != m.tokens_seen:
            if m.tokens_seen is not None and probe.gen_tokens > m.tokens_seen and m.stall_failures:
                self._event("engine_progress_resumed", ticket=m.ticket, shard=m.shard,
                            after_restarts=m.stall_failures)
                m.stall_failures = 0
            m.tokens_seen, m.tokens_moved_at = probe.gen_tokens, now
            return False
        if not busy or m.tokens_moved_at is None:  # idle (waiting for the trainer): not a stall
            m.tokens_moved_at = now
            return False
        if now - m.tokens_moved_at < limit:
            return False
        detail = (f"{probe.running} running, {probe.queued} queued, generation tokens flat at "
                  f"{probe.gen_tokens:.0f} for {now - m.tokens_moved_at:.0f}s")
        self._event("engine_stalled", ticket=m.ticket, shard=m.shard, detail=detail,
                    trainer_wait=probe.trainer_wait, failures=m.stall_failures)
        if m.stall_failures >= self.ecfg.max_stall_restart_failures:
            self._event("engine_stall_loop", ticket=m.ticket, shard=m.shard, restarts=m.stall_failures)
            self._fail(now, f"engine on {m.ticket} stalls after {m.stall_failures} restarts")
            self._retire(m, "engine stall loop", lost=False)
            return True
        restart = getattr(self.ops, "restart", None)
        if restart is None:
            return False
        fence = getattr(self.ops, "fence_incarnation", None)  # lease only: the shard keeps sampling
        if fence is not None:  # the stuck incarnation's claims go back at once
            try:
                fence(m)
                self._event("incarnation_fenced", ticket=m.ticket, shard=m.shard, why="engine stalled")
            except Exception as error:  # noqa: BLE001 - the restart supersedes it at its join anyway
                self._event("fence_failed", ticket=m.ticket, shard=m.shard, error=repr(error)[:300])
        m.stall_failures += 1
        self._event("sampler_restart", ticket=m.ticket, shard=m.shard, attempt=m.stall_failures,
                    detail=f"engine stalled: {detail}")
        try:
            restart(m, self.shards())
        except Exception as error:  # noqa: BLE001 - counts as a failed restart; probed again next tick
            self._event("start_failed", ticket=m.ticket, shard=m.shard, error=repr(error)[:300])
            if is_config_error(error):
                self._config_error(m, repr(error)[:300])
                return True
        m.state, m.started, m.joined, m.created = STARTING, now, False, now
        m.tokens_seen = m.tokens_moved_at = None
        return True

    def _patrol(self, now: float) -> None:
        """Open tickets of this run (ledger, by id) that no live member owns are adopted
        as members (module doc); a stopping pool or ``release_orphans`` deletes them."""
        run_tickets = getattr(self.ops, "run_tickets", None)
        if run_tickets is None:
            return
        owned = {m.ticket for m in self.members if m.state != GONE}
        released = {m.ticket for m in self.members if m.state == GONE}  # drained and confirmed by us
        try:
            rows = list(run_tickets())
        except Exception as error:  # noqa: BLE001
            self.log(f"pool: patrol could not read the ledger: {error!r}")
            return
        for account, ticket in rows:
            account = _slot(account)
            if ticket in owned:
                continue
            if not self.stopped and not self.config.release_orphans and ticket not in released:
                self._adopt(account, str(ticket), now)
                owned.add(str(ticket))
                continue
            stray = Member(account, str(ticket), shard="", state=DRAINING, created=now)
            self._event("orphan_lease", ticket=ticket, account=account)
            try:
                self.ops.drain(stray)
                self._event("released", ticket=ticket, account=account, why="orphan")
            except Exception as error:  # noqa: BLE001
                self._event("drain_failed", ticket=ticket, account=account, why="orphan", error=repr(error)[:300])

    def _adopt(self, account: Any, ticket: str, now: float) -> Member:
        """A ticket of this run no member owns becomes a member (PROVISIONING: the next
        ticks read it, start its sampler once it is active, watch it like any other).

        A ticket the pool has no room for is drained instead (released, never started): its
        shard would pass ``shard_limit`` (no buffer shard / tunnel port for it), every data
        slot is held (it would read every prompt), or the pool already counts ``target``
        members (an extra node billed for nothing)."""
        why = None
        slot = None
        if self.config.shard_limit is not None and self.next_shard >= self.config.shard_limit:
            why = (f"shard {self.config.shard_base}-{self.next_shard} is past shard_limit "
                   f"{self.config.shard_limit}")
        elif len(self.counted()) >= self.config.target:
            why = f"the pool already has {len(self.counted())} member(s) for target {self.config.target}"
        else:
            try:
                slot = self._free_slot()
            except RuntimeError:
                why = f"every one of the {self.config.data_slots} data slots is held"
        if why is not None:
            self.log(f"pool: WARNING: open ticket {ticket} of this run on account {account} not adopted: "
                     f"{why}; draining it")
            self._event("adopt_refused", ticket=ticket, account=account, why=why)
            extra = Member(account, ticket, shard="", state=DRAINING, created=now, end_reason=why)
            self.members.append(extra)  # a failed drain is retried every tick like any other
            self._drain(extra, f"not adopted: {why}")
            self._save_now()
            return extra
        shard = f"{self.config.shard_base}-{self.next_shard}"
        self.next_shard += 1
        member = Member(account, ticket, shard, created=now, data_slot=slot)
        self.members.append(member)
        self.log(f"pool: adopted {ticket} on account {account} as shard {shard} (a ticket of this run "
                 "no member owned; it is never deleted as an orphan)")
        self._event("adopted", ticket=ticket, account=account, shard=shard, data_slot=slot)
        self._save_now()
        return member

    def stop(self, why: str) -> None:
        self.stopped = why
        self.log(f"pool: stopping: {why}")
        self._event("pool_stopped", why=why)
        for m in self.live():
            self._drain(m, "stop")

    def drain_all(self, why: str, *, attempts: int = 3, wait_s: float = 10.0,
                  sleep: Callable[[float], None] | None = None) -> list[Member]:
        """Stop the pool (drain every member) and retry failed drains ``attempts - 1``
        more times; the patrol then releases this run's machines no member owns.
        Returns the members left (still billed: the caller prints the banner)."""
        sleep = sleep or time.sleep
        self.stop(why)
        for _ in range(attempts - 1):
            left = self.live()
            if not left:
                break
            sleep(wait_s)
            for m in left:
                self._drain(m, "stop retry")
        self._patrol(self.clock())
        return self.live()

    def snapshot(self) -> dict:
        return {"stopped": self.stopped, "finished": self.finished, "halted": self.halted, "next_shard": self.next_shard,
                "target": self.config.target, "scale_seq": self.scale_seq, "shrink": self.shrink,
                "members": [asdict(m) for m in self.members], "pending": list(self.pending),
                "breaker": asdict(self.breaker), "budget": asdict(self.budget)}

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.snapshot(), indent=1))
        tmp.replace(path)

    @classmethod
    def restore(cls, path: Path, config: PoolConfig, ops: PoolOps, **kw) -> IslandPool:
        pool = cls(config, ops, **kw)
        pool.state_path = path
        if path.exists():
            doc = json.loads(path.read_text())
            pool.members = [Member(**m) for m in doc.get("members", [])]
            pool.next_shard = doc.get("next_shard", 0)
            pool.stopped = doc.get("stopped")
            pool.finished = doc.get("finished")
            pool.halted = doc.get("halted")
            pool.pending = list(doc.get("pending", []))
            pool.breaker = el.Breaker(**doc.get("breaker", {}))
            pool.budget = el.Budget(**doc.get("budget", {}))
            pool.scale_seq = int(doc.get("scale_seq", 0))
            pool.shrink = int(doc.get("shrink", 0))
            for m in pool.members:  # a start cut off by the restart: probed first (``_watch``)
                if m.state == STARTING and not m.joined:
                    if getattr(ops, "probe", None) is not None:
                        pool._recheck.add(m.ticket)
                    else:  # no probe: run the start again (every step is idempotent)
                        m.state = PROVISIONING
        return pool


def _slot(account: Any) -> Any:
    """A slot as a provider listed it: an int when it reads as one (National Compute
    accounts, Daytona / sky slots), else as given (``"<provider>:<slot>"``)."""
    if isinstance(account, int):
        return account
    try:
        return int(account)
    except (TypeError, ValueError):
        return account
