"""Capacity broker: how many sampler nodes each pool / offer should hold (6.3, 6.5).

The broker is a slow loop (minutes) over a pure decision function. It decides
counts only: ``{"<pool>/<offer>": nodes}`` and, per pool, which offer a refill
order should use. Fencing, handoff, drain and replacement stay with the pools,
which reach their targets by their own scale rules.

**Score** (lower is better): cost per unit of work = price / (throughput x
availability), availability = E[life] / (E[life] + cold start). It is computed by
:func:`yeto.cloud.replace.score` (price / effective compute + prep cost / expected
stay, with prep = cold start, stay = E[life]): price/tput x (1 + cold/life) is
exactly price / (tput x life / (life + cold)).

**Order of use**: own machines, prepaid leases with quota left (National
Compute), spot, then on-demand as a fallback. Within a tier, best score first.

**Rules** (all hold at once):

* each island's floor first (in island order), then each island up to its want;
* the run budget: an hourly ceiling (``budget.usd_per_h``, and the money left over
  the hours left) is never passed by a node the broker adds;
* at most ``spread_cap`` nodes per (provider, region, gpu) of the spread tiers
  (spot by default) and at most ``spot_max_frac`` of an island's want on spot;
* hysteresis: a running node keeps its place unless it is at a natural boundary
  (lease end, a replacement ready, its island ended): a better score elsewhere
  never moves it; at a boundary its own offer wins unless another beats it by
  more than ``hysteresis`` (fraction of its score);
* on-demand only covers a spot gap (spot wanted but not orderable: unavailable or
  full) that has lasted ``on_demand_after_s``; a shortfall the spot caps (spread,
  share) leave goes to on-demand at once; nodes already on on-demand stay until
  their boundary.

The broker is off unless a run config's ``broker`` section sets ``enabled: true``
(:func:`config_from_run`), so a single-pool run (the customer path) is never touched.

Phase 2 (:func:`plan_migrations`, off by default: ``migrate: true``): a node an
island no longer needs moves to an island short of nodes on the same offer
without a new order: drain the old island's lane, load the new island's release,
start a lane there (:func:`run_migration` over a :class:`LaneMover`).

The price table, the offer ledger and the lifetime model are read through narrow
protocols (:class:`PriceSource`, :class:`OfferLedger`, :class:`Lifetimes`) so the
broker does not depend on how they are stored.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from yeto.cloud import replace as cloud_replace

OWN, PREPAID, SPOT, ON_DEMAND = "own", "prepaid", "spot", "on_demand"
TIERS = (OWN, PREPAID, SPOT, ON_DEMAND)
DEFAULT_INTERVAL_S = 300.0
DEFAULT_HYSTERESIS = 0.10
DEFAULT_ON_DEMAND_AFTER_S = 1800.0
DEFAULT_LIFE_S = 3600.0  # E[life] when neither the ledger nor the lifetime model knows
EVENT = "broker_decision"


# -- what the broker reads (S3: price table, offer ledger, lifetime model) -------


class PriceSource(Protocol):
    def node_usd_h(self, offer: Offer) -> float | None: ...  # whole node, USD/h; None: unknown


@dataclass(frozen=True)
class OfferStats:
    """Measured facts of one offer (the offer ledger): cold start from order to a ready
    member, order success rate, reclaims per node hour, tokens/s per node for the model."""

    cold_start_p50_s: float | None = None
    cold_start_p90_s: float | None = None
    success_rate: float | None = None
    reclaim_per_h: float | None = None
    tokens_per_s: float | None = None
    samples: int = 0


class OfferLedger(Protocol):
    def stats(self, offer: Offer, model: str | None = None) -> OfferStats | None: ...


class Lifetimes(Protocol):
    def expected_life_s(self, offer: Offer, now: float) -> float | None: ...  # E[life] of a new node


# -- inputs and outputs ------------------------------------------------------------


@dataclass(frozen=True)
class Offer:
    """One way a pool gets a node. ``capacity``: nodes it can hold at most (inventory
    size, free NC accounts, a quota share; None: unbounded); ``available``: False while
    the pool cannot order it (breaker, cooldown, sold out). ``usd_h`` / ``tokens_per_s``
    / ``cold_start_s`` / ``life_s`` are stated fallbacks for the protocols."""

    pool: str
    name: str
    island: str
    tier: str
    provider: str
    region: str = ""
    gpu: str = ""
    capacity: int | None = None
    available: bool = True
    usd_h: float | None = None
    tokens_per_s: float | None = None
    cold_start_s: float | None = None
    life_s: float | None = None
    model: str | None = None

    def __post_init__(self) -> None:
        if self.tier not in TIERS:
            raise ValueError(f"broker: offer {self.key}: tier {self.tier!r} is not one of {TIERS}")

    @property
    def key(self) -> str:
        return f"{self.pool}/{self.name}"

    @property
    def spread_key(self) -> tuple[str, str, str]:
        return (self.provider, self.region, self.gpu)


@dataclass(frozen=True)
class Demand:
    island: str
    want: int
    floor: int = 0


@dataclass(frozen=True)
class Policy:
    spread_cap: int | None = None
    spread_tiers: tuple[str, ...] = (SPOT,)
    spot_max_frac: float = 1.0
    hysteresis: float = DEFAULT_HYSTERESIS
    on_demand_after_s: float = DEFAULT_ON_DEMAND_AFTER_S
    usd_per_h: float | None = None  # hourly ceiling of the run's sampler nodes


@dataclass(frozen=True)
class Budget:
    usd_total: float | None = None  # run budget; None: no cap
    spent_usd: float = 0.0
    hours_left: float | None = None  # wall clock left; None: unknown

    def hourly_ceiling(self, policy_cap: float | None) -> float | None:
        caps = [c for c in (policy_cap,) if c is not None]
        if self.usd_total is not None:
            left = max(0.0, self.usd_total - self.spent_usd)
            caps.append(left / self.hours_left if self.hours_left else left)
        return min(caps) if caps else None


@dataclass(frozen=True)
class Scored:
    offer: Offer
    usd_h: float | None
    score: float | None
    availability: float | None
    terms: dict[str, Any]
    risks: tuple[str, ...]


@dataclass
class State:
    """What the broker carries between ticks: since when each island's spot gap lasts."""

    spot_gap_since: dict[str, float] = field(default_factory=dict)


@dataclass
class Decision:
    targets: dict[str, int]  # "<pool>/<offer>" -> nodes
    refill: dict[str, str]  # pool -> offer key a refill order uses
    pools: dict[str, int]  # pool -> nodes (sum of its offers)
    islands: dict[str, int]
    short: dict[str, int]  # island -> nodes below its want
    floor_unmet: dict[str, int]
    usd_h: float
    ceiling_usd_h: float | None
    state: State
    scores: dict[str, float | None]
    notes: list[str] = field(default_factory=list)

    def record(self) -> dict[str, Any]:
        return {"targets": dict(self.targets), "refill": dict(self.refill), "pools": dict(self.pools),
                "islands": dict(self.islands), "short": dict(self.short), "floor_unmet": dict(self.floor_unmet),
                "usd_h": round(self.usd_h, 4), "ceiling_usd_h": self.ceiling_usd_h,
                "scores": dict(self.scores), "notes": list(self.notes)}


# -- scoring -----------------------------------------------------------------------


def score_offer(offer: Offer, *, prices: PriceSource | None = None, ledger: OfferLedger | None = None,
                lifetimes: Lifetimes | None = None, now: float = 0.0, per_token: bool = True) -> Scored:
    """Cost per unit of work of one offer. :func:`yeto.cloud.replace.score` with the price,
    prep = cold start and stay = E[life] gives price x (1 + cold / life) = price /
    availability (USD per useful node hour); divided by tokens/s it is USD per unit of work.
    ``per_token`` False (some offer of the decision has no throughput) keeps USD per
    useful node hour, so every offer is ranked in one unit. An own machine costs nothing
    (score 0). Unknown price: no score (the offer is not used)."""
    stats = ledger.stats(offer, offer.model) if ledger is not None else None
    usd_h = prices.node_usd_h(offer) if prices is not None else None
    if usd_h is None:
        usd_h = offer.usd_h
    if offer.tier == OWN and usd_h is None:
        usd_h = 0.0
    tput = (stats.tokens_per_s if stats and stats.tokens_per_s else None) or offer.tokens_per_s
    measured_cold = stats is not None and stats.cold_start_p50_s is not None
    cold = stats.cold_start_p50_s if measured_cold else offer.cold_start_s  # type: ignore[union-attr]
    life = lifetimes.expected_life_s(offer, now) if lifetimes is not None else None
    if life is None and stats and stats.reclaim_per_h:
        life = 3600.0 / stats.reclaim_per_h
    life = offer.life_s if life is None else life
    life = DEFAULT_LIFE_S if life is None else float(life)
    cold = 0.0 if cold is None else float(cold)
    availability = life / (life + cold) if life + cold > 0 else 1.0
    if offer.tier == OWN:
        return Scored(offer, usd_h, 0.0, availability, {"price": {"value": 0.0, "source": "own machine"}}, ())
    c = cloud_replace.score(
        cloud_replace.Offer(cloud=offer.provider, region=offer.region, gpu=offer.gpu, gpus=0,
                            price_per_hour=usd_h, has_weights=True),
        prep_s=cold, prep_source="offer ledger cold start p50" if measured_cold else "stated cold start",
        expected_stay_s=life)
    risks = [r for r in c.risks if r not in ("reclaim rate unknown", "effective compute unknown")]
    terms = dict(c.terms)
    value = c.score
    if value is not None and per_token and tput:
        value = value / tput
        terms["throughput"] = {"value": tput, "source": "offer ledger" if stats and stats.tokens_per_s
                               else "stated tokens_per_s"}
    elif value is not None:
        risks.append("throughput unknown: ranked per useful node hour" if not tput
                     else "another offer has no throughput: ranked per useful node hour")
    if stats is not None and stats.success_rate is not None and stats.success_rate < 0.5:
        risks.append(f"order success rate {stats.success_rate:.0%}")
    return Scored(offer, usd_h, value, availability, terms, tuple(risks))


def _throughput_known(offer: Offer, ledger: OfferLedger | None) -> bool:
    stats = ledger.stats(offer, offer.model) if ledger is not None else None
    return bool((stats and stats.tokens_per_s) or offer.tokens_per_s)


# -- the decision ------------------------------------------------------------------


def decide(offers: Iterable[Offer], demands: Iterable[Demand], *, running: Mapping[str, int] | None = None,
           at_boundary: Mapping[str, int] | None = None, policy: Policy = Policy(), budget: Budget = Budget(),
           state: State | None = None, now: float = 0.0, prices: PriceSource | None = None,
           ledger: OfferLedger | None = None, lifetimes: Lifetimes | None = None) -> Decision:
    """The node count of every offer. ``running``: nodes each offer holds now (ordered
    and alive); ``at_boundary``: how many of those are at a natural boundary and may be
    placed anew. ``state``: the previous tick's (spot gap clocks); the decision carries
    the next one."""
    running = {k: max(0, int(v)) for k, v in (running or {}).items()}
    at_boundary = {k: max(0, min(int(v), running.get(k, 0))) for k, v in (at_boundary or {}).items()}
    state = State(dict(state.spot_gap_since)) if state is not None else State()
    offers = list(offers)
    per_token = all(_throughput_known(o, ledger) for o in offers if o.tier != OWN)
    scored = {o.key: score_offer(o, prices=prices, ledger=ledger, lifetimes=lifetimes, now=now, per_token=per_token)
              for o in offers}
    demand = {d.island: d for d in demands}
    ceiling = budget.hourly_ceiling(policy.usd_per_h)
    notes: list[str] = []
    targets = {k: 0 for k in scored}
    spread: dict[tuple[str, str, str], int] = {}
    usd_h = 0.0

    def rank_key(s: Scored, held: bool) -> tuple[int, float, str]:
        value = s.score if s.score is not None else math.inf
        if held:  # hysteresis: its own offer wins unless beaten by more than the threshold
            value *= (1.0 - policy.hysteresis)
        return (TIERS.index(s.offer.tier), value, s.offer.key)

    def room(s: Scored) -> int:
        o = s.offer
        free = (o.capacity if o.capacity is not None else 10**9) - targets[o.key]
        if o.tier in policy.spread_tiers and policy.spread_cap is not None:
            free = min(free, policy.spread_cap - spread.get(o.spread_key, 0))
        return max(0, free)

    def place(s: Scored) -> None:
        nonlocal usd_h
        targets[s.offer.key] += 1
        spread[s.offer.spread_key] = spread.get(s.offer.spread_key, 0) + 1
        usd_h += s.usd_h or 0.0

    def island_count(island: str, tier: str | None = None) -> int:
        return sum(n for k, n in targets.items() if scored[k].offer.island == island
                   and (tier is None or scored[k].offer.tier == tier))

    # 1. nodes not at a boundary keep their place (up to their island's want, worst score shed first)
    for island, d in demand.items():
        kept = sorted((s for s in scored.values() if s.offer.island == island),
                      key=lambda s: rank_key(s, False))
        left = max(0, d.want)
        for s in kept:
            n = min(running.get(s.offer.key, 0) - at_boundary.get(s.offer.key, 0), left)
            for _ in range(max(0, n)):
                place(s)
            left -= max(0, n)
    for k, n in running.items():
        if k in scored and scored[k].offer.island not in demand and n:
            notes.append(f"{k}: island {scored[k].offer.island} has no demand: its {n} node(s) go at their boundary")

    if ceiling is not None and usd_h > ceiling + 1e-9:
        notes.append(f"running nodes cost {usd_h:.2f} USD/h, above the hourly ceiling {ceiling:.2f}: none is "
                     "added; the pools' budget breakers decide what stops")

    # 2. floors, then wants: fill each island tier by tier, best score first
    for goal in ("floor", "want"):
        for island, d in demand.items():
            need = max(0, getattr(d, goal)) - island_count(island)
            if need <= 0:
                if goal == "want":
                    state.spot_gap_since.pop(island, None)
                continue
            spot_cap = math.floor(policy.spot_max_frac * max(0, d.want) + 1e-9)
            cands = sorted((s for s in scored.values() if s.offer.island == island and s.offer.available
                            and (s.score is not None or s.offer.tier == OWN)),
                           key=lambda s: rank_key(s, at_boundary.get(s.offer.key, 0) > 0))
            capped = False  # spot held back by a policy cap (share, spread), not by the market
            for s in cands:
                if need <= 0:
                    break
                if s.offer.tier == ON_DEMAND:
                    continue  # after the spot gap check below
                while need > 0:
                    if room(s) <= 0:
                        o = s.offer
                        capped |= o.tier == SPOT and (o.capacity is None or targets[o.key] < o.capacity)
                        break
                    if s.offer.tier == SPOT and island_count(island, SPOT) >= spot_cap:
                        capped = True
                        break
                    if ceiling is not None and (s.usd_h or 0.0) > 0 and usd_h + (s.usd_h or 0.0) > ceiling + 1e-9:
                        notes.append(f"{s.offer.key}: hourly ceiling {ceiling:.2f} USD/h reached")
                        break
                    place(s)
                    need -= 1
            if goal != "want":
                continue
            if need <= 0:
                state.spot_gap_since.pop(island, None)
                continue
            # 3. spot gap -> on-demand after on_demand_after_s
            spot_wanted = any(s.offer.island == island and s.offer.tier == SPOT for s in scored.values())
            od = [s for s in cands if s.offer.tier == ON_DEMAND]
            if not od:
                continue
            if spot_wanted and not capped:
                since = state.spot_gap_since.setdefault(island, now)
                if now - since < policy.on_demand_after_s:
                    notes.append(f"island {island}: spot gap {need} node(s) for {now - since:.0f} s; on-demand "
                                 f"after {policy.on_demand_after_s:.0f} s")
                    continue
            for s in od:
                while need > 0 and room(s) > 0:
                    if ceiling is not None and usd_h + (s.usd_h or 0.0) > ceiling + 1e-9:
                        notes.append(f"{s.offer.key}: hourly ceiling {ceiling:.2f} USD/h reached")
                        break
                    place(s)
                    need -= 1
    for island in list(state.spot_gap_since):
        if island not in demand:
            state.spot_gap_since.pop(island)

    islands = {i: island_count(i) for i in demand}
    pools: dict[str, int] = {}
    for k, n in targets.items():
        pools[scored[k].offer.pool] = pools.get(scored[k].offer.pool, 0) + n
    return Decision(
        targets=targets, refill=refill_choices(scored.values(), running, policy), pools=pools, islands=islands,
        short={i: d.want - islands[i] for i, d in demand.items() if d.want > islands[i]},
        floor_unmet={i: d.floor - islands[i] for i, d in demand.items() if d.floor > islands[i]},
        usd_h=usd_h, ceiling_usd_h=ceiling, state=state, scores={k: s.score for k, s in scored.items()},
        notes=list(dict.fromkeys(notes)))


def refill_choices(scored: Iterable[Scored], running: Mapping[str, int], policy: Policy) -> dict[str, str]:
    """Per pool, the offer its next order uses: the best available one by tier and score,
    except that an offer the pool runs nodes on now stays unless another of the same or a
    better tier beats its score by more than ``policy.hysteresis``."""
    by_pool: dict[str, list[Scored]] = {}
    for s in scored:
        if s.offer.available and (s.score is not None or s.offer.tier == OWN):
            by_pool.setdefault(s.offer.pool, []).append(s)
    out = {}
    for pool, rows in by_pool.items():
        best = min(rows, key=lambda s: (TIERS.index(s.offer.tier), s.score or 0.0, s.offer.key))
        held = [s for s in rows if running.get(s.offer.key, 0) > 0]
        if held:
            cur = min(held, key=lambda s: (TIERS.index(s.offer.tier), s.score or 0.0, s.offer.key))
            if TIERS.index(cur.offer.tier) <= TIERS.index(best.offer.tier) and \
                    (best.score or 0.0) >= (cur.score or 0.0) * (1.0 - policy.hysteresis):
                best = cur
        out[pool] = best.offer.key
    return out


# -- run config ---------------------------------------------------------------------


@dataclass(frozen=True)
class BrokerConfig:
    interval_s: float = DEFAULT_INTERVAL_S
    policy: Policy = Policy()
    floors: Mapping[str, int] = field(default_factory=dict)
    migrate: bool = False
    migrate_min_lease_s: float = 1800.0


BROKER_KEYS = frozenset({"enabled", "interval_s", "spread_cap", "spread_tiers", "spot_max_frac", "hysteresis",
                         "on_demand_after_s", "floors", "migrate", "migrate_min_lease_s"})


def config_from_run(cfg: Mapping[str, Any]) -> BrokerConfig | None:
    """The run config's ``broker`` section; None (the broker does not run) when it is
    absent or ``enabled`` is not true. Raises ValueError on a bad section."""
    section = cfg.get("broker")
    if section is None:
        return None
    if not isinstance(section, Mapping):
        raise ValueError("run config: broker must be an object")
    unknown = set(section) - BROKER_KEYS
    if unknown:
        raise ValueError(f"run config: unknown broker keys {sorted(unknown)} (known: {sorted(BROKER_KEYS)})")
    if section.get("enabled") is not True:
        return None

    def num(key: str, default: float, lo: float, hi: float = math.inf) -> float:
        v = section.get(key, default)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not lo <= v <= hi:
            raise ValueError(f"run config: broker.{key} must be a number in [{lo}, {hi}] (got {v!r})")
        return float(v)

    cap = section.get("spread_cap")
    if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or cap < 1):
        raise ValueError(f"run config: broker.spread_cap must be a whole number >= 1 or null (got {cap!r})")
    tiers = tuple(section.get("spread_tiers", (SPOT,)))
    if not set(tiers) <= set(TIERS):
        raise ValueError(f"run config: broker.spread_tiers {list(tiers)} must be among {list(TIERS)}")
    floors = section.get("floors") or {}
    if not isinstance(floors, Mapping) or any(isinstance(v, bool) or not isinstance(v, int) or v < 0
                                              for v in floors.values()):
        raise ValueError("run config: broker.floors must map an island to a whole number >= 0")
    budget = cfg.get("budget") if isinstance(cfg.get("budget"), Mapping) else {}
    hourly = budget.get("usd_per_h")
    policy = Policy(spread_cap=cap, spread_tiers=tiers, spot_max_frac=num("spot_max_frac", 1.0, 0.0, 1.0),
                    hysteresis=num("hysteresis", DEFAULT_HYSTERESIS, 0.0, 0.9),
                    on_demand_after_s=num("on_demand_after_s", DEFAULT_ON_DEMAND_AFTER_S, 0.0),
                    usd_per_h=float(hourly) if isinstance(hourly, (int, float)) and not isinstance(hourly, bool)
                    else None)
    return BrokerConfig(interval_s=num("interval_s", DEFAULT_INTERVAL_S, 10.0), policy=policy,
                        floors={str(k): int(v) for k, v in floors.items()},
                        migrate=section.get("migrate") is True,
                        migrate_min_lease_s=num("migrate_min_lease_s", 1800.0, 0.0))


# -- the slow loop ------------------------------------------------------------------


class TargetSink(Protocol):
    """Where the broker's counts go: the pools' scale control (``pool-scale`` with source
    "auto", which yields to a manual change) or the fleet's desired state."""

    def set_target(self, pool: str, nodes: int, *, offers: Mapping[str, int], refill: str | None) -> bool: ...


@dataclass
class Snapshot:
    offers: list[Offer]
    demands: list[Demand]
    running: dict[str, int] = field(default_factory=dict)
    at_boundary: dict[str, int] = field(default_factory=dict)
    budget: Budget = Budget()
    nodes: list[Node] = field(default_factory=list)  # phase 2: the nodes a move may take
    ended: tuple[str, ...] = ()  # islands whose trainer ended (their nodes may move)


@dataclass
class Broker:
    """The slow loop: every ``config.interval_s`` read a :class:`Snapshot`, decide, and
    write each pool's count to the sink when it changed (a sink refusal, e.g. a manual
    hold, is retried at the next tick).

    Phase 2 (``config.migrate`` and a ``mover``): the moves of :func:`plan_migrations`
    run before any count is sent, so neither pool orders or releases for a moved node:
    the source pool still holds its old (higher) target and the destination its old
    (lower) one while the node changes lanes; the counts sent afterwards already
    include it on its new island."""

    config: BrokerConfig
    snapshot: Callable[[], Snapshot]
    sink: TargetSink
    emit: Callable[..., Any] = lambda *a, **k: None
    prices: PriceSource | None = None
    ledger: OfferLedger | None = None
    lifetimes: Lifetimes | None = None
    clock: Callable[[], float] = time.time
    mover: LaneMover | None = None
    compatible: Callable[[Offer, Offer], bool] | None = None
    state: State = field(default_factory=State)
    sent: dict[str, int] = field(default_factory=dict)
    last: Decision | None = None
    _next_at: float = 0.0

    def due(self) -> bool:
        return self.clock() >= self._next_at

    def tick(self, *, force: bool = False) -> Decision | None:
        now = self.clock()
        if not force and now < self._next_at:
            return None
        self._next_at = now + self.config.interval_s
        snap = self.snapshot()
        demands = [replace(d, floor=max(d.floor, int(self.config.floors.get(d.island, 0)))) for d in snap.demands]
        dec = decide(snap.offers, demands, running=snap.running, at_boundary=snap.at_boundary,
                     policy=self.config.policy, budget=snap.budget, state=self.state, now=now,
                     prices=self.prices, ledger=self.ledger, lifetimes=self.lifetimes)
        self.state = dec.state
        self.last = dec
        if self.config.migrate and self.mover is not None:
            moves = plan_migrations(snap.nodes, snap.offers, dec, running=snap.running, enabled=True, now=now,
                                    min_lease_s=self.config.migrate_min_lease_s, compatible=self.compatible,
                                    ended=snap.ended)
            for move in moves:
                run_migration(move, self.mover, self.emit)
        for pool, nodes in sorted(dec.pools.items()):
            offers = {k: n for k, n in dec.targets.items() if k.split("/", 1)[0] == pool}
            if self.sent.get(pool) == nodes:
                continue
            if self.sink.set_target(pool, nodes, offers=offers, refill=dec.refill.get(pool)):
                self.sent[pool] = nodes
        self.emit(EVENT, **dec.record())
        return dec


class ScaleFileSink:
    """A :class:`TargetSink` over each pool's ``pool-scale`` control file
    (``yeto.launch.nc.pool_run.scale_pool`` with ``source="auto"``: refused for
    ``elastic.manual_hold_s`` after a manual change, so a person's scale wins)."""

    def __init__(self, pools: Mapping[str, tuple[dict[str, Any], str]], *,
                 scale: Callable[..., int] | None = None) -> None:
        self.pools = dict(pools)  # pool -> (pool config, config path)
        self._scale = scale

    def set_target(self, pool: str, nodes: int, *, offers: Mapping[str, int], refill: str | None) -> bool:
        if pool not in self.pools:
            return False
        scale = self._scale
        if scale is None:
            from yeto.launch.nc.pool_run import scale_pool  # noqa: PLC0415

            scale = scale_pool
        cfg, path = self.pools[pool]
        return scale(cfg, path, nodes, source="auto",
                     extra={"broker": {"offers": dict(offers), "refill": refill}}) == 0


# -- phase 2: moving a node between islands ----------------------------------------


@dataclass(frozen=True)
class Node:
    """A running node the broker may move: its pool's offer, its island, when its lease
    ends (None: no hard end) and whether its pool marks it migratable (ready, not paused,
    not retiring)."""

    ticket: str
    offer: str  # "<pool>/<offer>"
    island: str
    lease_end: float | None = None
    movable: bool = True


@dataclass(frozen=True)
class Migration:
    ticket: str
    from_island: str
    to_island: str
    from_offer: str
    to_offer: str
    reason: str


def plan_migrations(nodes: Iterable[Node], offers: Iterable[Offer], decision: Decision, *,
                    running: Mapping[str, int], enabled: bool = False, now: float = 0.0,
                    min_lease_s: float = 1800.0, compatible: Callable[[Offer, Offer], bool] | None = None,
                    ended: Iterable[str] = ()) -> list[Migration]:
    """Nodes to move from an island over its target (or ended: ``ended``) to an island
    short of nodes, instead of releasing one and ordering another. A move needs a target
    offer on the destination island with the same provider, region and gpu (and
    ``compatible`` when given: same node profile, model fits), room under that offer's
    target, and a lease that does not end within ``min_lease_s``. Off unless ``enabled``."""
    if not enabled:
        return []
    by_key = {o.key: o for o in offers}
    gone = set(ended)
    surplus: dict[str, int] = {}
    for k, n in running.items():
        if k in by_key:
            over = n - decision.targets.get(k, 0)
            if by_key[k].island in gone:
                over = n
            if over > 0:
                surplus[k] = over
    room = {k: decision.targets.get(k, 0) - running.get(k, 0) for k in by_key}
    room = {k: v for k, v in room.items() if v > 0 and by_key[k].island not in gone}
    out: list[Migration] = []
    for node in sorted(nodes, key=lambda n: (-(n.lease_end or math.inf), n.ticket)):
        src = by_key.get(node.offer)
        if src is None or not node.movable or surplus.get(node.offer, 0) <= 0:
            continue
        if node.lease_end is not None and node.lease_end - now < min_lease_s:
            continue
        for k in sorted(room, key=lambda k: (decision.scores.get(k) or 0.0, k)):
            dst = by_key[k]
            if room[k] <= 0 or dst.island == src.island or dst.spread_key != src.spread_key:
                continue
            if compatible is not None and not compatible(src, dst):
                continue
            out.append(Migration(node.ticket, src.island, dst.island, src.key, dst.key,
                                 "island ended" if src.island in gone else "island over target"))
            room[k] -= 1
            surplus[node.offer] -= 1
            break
    return out


class LaneMover(Protocol):
    """What a move asks of the pools (the NodePool / Lane split): stop the node's lane on
    its island (fence, return claims, drain), load the new island's release on the
    running engine, then start a lane of the new island on the same machine."""

    def drain_lane(self, ticket: str, island: str) -> None: ...
    def load_release(self, ticket: str, island: str) -> None: ...
    def start_lane(self, ticket: str, island: str) -> None: ...


def run_migration(move: Migration, mover: LaneMover, emit: Callable[..., Any] = lambda *a, **k: None) -> bool:
    """One move, step by step; no machine is ordered or released here. A failed step stops
    the move (event ``broker_migrate_failed`` with the step) and leaves the node to its
    pool: after a failed drain it still serves the old island; after a failed load or
    start the old lane is gone and the pool's own rules retire or restart it."""
    detail = {"ticket": move.ticket, "from_island": move.from_island, "to_island": move.to_island,
              "from_offer": move.from_offer, "to_offer": move.to_offer, "reason": move.reason}
    emit("broker_migrate", **detail)
    for step, call, island in (("drain_lane", mover.drain_lane, move.from_island),
                               ("load_release", mover.load_release, move.to_island),
                               ("start_lane", mover.start_lane, move.to_island)):
        try:
            call(move.ticket, island)
        except Exception as error:  # noqa: BLE001 - reported; the pool owns the node
            emit("broker_migrate_failed", step=step, error=repr(error)[:300], **detail)
            return False
    emit("broker_migrated", **detail)
    return True
