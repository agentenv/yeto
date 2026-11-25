"""Breakers of an island fleet: a run config with ``islands`` (one trainer and its own pools each).

A fleet adds scope, not semantics. The ``run`` and ``trainer`` layers are instantiated once
per island: each island resolves them over its own trainer argv (the shared
``fleet.trainer_argv`` plus the island's ``trainer.argv``) and its own pools, the dependency
rules run once per island over that island's facts, and a trip stops that island only.
``run.budget`` and ``run.wall_clock`` span the fleet: one budget over every pool, every
island's trainer and the syncer, one wall clock. The ``launch`` layer is the fleet's one
``yeto launch``; ``pool`` / ``sampler`` / ``harness`` entries stay per pool, as for one trainer.

This module holds what every caller shares: which entries span the fleet
(:func:`breaker_scope`), the single-trainer run config each island is read as
(:func:`island_run_config`, :func:`fleet_run_config`), the fleet's stop record
(:class:`FleetStops`) and its spend (:func:`fleet_spend`). It reads the raw ``islands``
section only (``islands[*].name``, ``islands[*].trainer``, ``islands[*].samplers``) and
never the compiled fleet plan. Standard library only, like the rest of the package.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

FLEET_ENTRIES: tuple[str, ...] = ("run.budget", "run.wall_clock")  # one budget, one wall clock
ISLAND_LAYERS: tuple[str, ...] = ("run", "trainer")  # instantiated once per island
FLEET_LAYERS: tuple[str, ...] = ("launch",)  # the fleet's one yeto launch
# the one-budget rules (D11, D12): checked once over the whole fleet; every other rule per island
FLEET_RULES = frozenset({"D11", "D12"})
# flags the fleet derives for each island's trainer (they never come from fleet.trainer_argv)
ISLAND_TRAINER_FLAGS = ("--rl-sample-role", "trainer", "--rl-sampler-follow", "trainer")
# a trainer flag an island may set for itself over the fleet's (the compile's --rl-island-sample-wait-s)
SAMPLE_WAIT_FLAG = "--rl-sample-wait-s"
# the island flags that override a breakers.trainer param of the run config: (entry, param)
ISLAND_SECTION_FLAGS = {SAMPLE_WAIT_FLAG: ("sample_wait", "timeout_s"),
                        "--rl-max-policy-age": ("policy_age", "max_versions")}

Scope = Literal["fleet", "island", "component"]


@dataclass(frozen=True)
class Island:
    """One island of a fleet: its index (the learner id and ``trainer<k>`` sub-store) and name."""

    index: int
    name: str

    @property
    def label(self) -> str:
        return f"island {self.index} ({self.name})"

    def scope(self) -> dict[str, str]:
        """The ``scope`` of a finding or an ``rl_breaker`` event of this island (``island`` as
        the trainer's events spell it: the index as a string)."""
        return {"island": str(self.index), "island_name": self.name}


def is_fleet(run_config: Any) -> bool:
    """Whether a run config is a fleet (it has an ``islands`` list)."""
    return isinstance(run_config, Mapping) and isinstance(run_config.get("islands"), list)


def _entries(run_config: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [e if isinstance(e, Mapping) else {} for e in (run_config.get("islands") or ())]


def islands(run_config: Mapping[str, Any]) -> tuple[Island, ...]:
    """The fleet's islands in order; an island without a ``name`` is ``island<k>``."""
    return tuple(Island(k, str(e.get("name") or f"island{k}")) for k, e in enumerate(_entries(run_config)))


def breaker_scope(name: str) -> Scope:
    """Where entry ``name`` acts in a fleet: ``fleet`` (run.budget, run.wall_clock and the
    launch layer: one for the whole fleet), ``island`` (every other run / trainer entry: one
    per island) or ``component`` (pool, sampler, harness, monitor: per pool or process,
    as for one trainer)."""
    layer = name.split(".", 1)[0]
    if name in FLEET_ENTRIES or layer in FLEET_LAYERS:
        return "fleet"
    if layer in ISLAND_LAYERS:
        return "island"
    return "component"


def pool_name(island: int, entry: Mapping[str, Any], i: int) -> str:
    """A pool's name in the fleet, ``i<k>-<name>`` (the name the compiled fleet gives it, and
    the key of its ``breakers.pools`` overrides); ``<name>`` as :func:`rules_facts.run_pools`
    names a pool (its name, else its provider, else ``samplers[i]``)."""
    provider = entry.get("provider")
    base = entry.get("name") or (provider if isinstance(provider, str) else None) or f"samplers[{i}]"
    return f"i{island}-{base}"


def _strings(value: Any) -> list[str]:
    return [str(a) for a in value] if isinstance(value, list) else []


def fleet_trainer_argv(run_config: Mapping[str, Any]) -> list[str]:
    """``fleet.trainer_argv``: the trainer arguments every island shares ([] when unset)."""
    fleet = run_config.get("fleet")
    return _strings(fleet.get("trainer_argv")) if isinstance(fleet, Mapping) else []


def _sets(argv: Sequence[str], flag: str) -> bool:
    return any(a == flag or a.startswith(f"{flag}=") for a in argv)


def _without(argv: Sequence[str], flag: str) -> list[str]:
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
        elif a == flag:
            skip = True
        elif not a.startswith(f"{flag}="):
            out.append(a)
    return out


def _island_own_argv(run_config: Mapping[str, Any], index: int) -> list[str]:
    entry = _entries(run_config)[index]
    trainer = entry.get("trainer") if isinstance(entry.get("trainer"), Mapping) else {}
    return _strings(trainer.get("argv"))


def island_trainer_argv(run_config: Mapping[str, Any], index: int) -> list[str]:
    """Island ``index``'s trainer as one ``yeto launch`` argv: the shared
    ``fleet.trainer_argv``, the island's GPU, the fleet's buffer-trainer role, then the
    island's own ``trainer.argv`` (its release cadence, policy age and sample wait; its
    ``--rl-sample-wait-s`` replaces the fleet's)."""
    entry = _entries(run_config)[index]
    trainer = entry.get("trainer") if isinstance(entry.get("trainer"), Mapping) else {}
    argv = fleet_trainer_argv(run_config)
    if _sets(_island_own_argv(run_config, index), SAMPLE_WAIT_FLAG):
        argv = _without(argv, SAMPLE_WAIT_FLAG)
    if trainer.get("gpu"):
        argv += ["--gpu", str(trainer["gpu"])]
    return [*argv, *ISLAND_TRAINER_FLAGS, *_strings(trainer.get("argv"))]


def _base(run_config: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in run_config.items() if k not in ("islands", "fleet", "trainer", "pool", "samplers")}


def is_miles_sampler(entry: Any) -> bool:
    """Whether an island's samplers entry is a Miles sampler island of the fleet's launch
    (``"kind": "miles"``, ``yeto.rl.fleet``): no pool, no pool breakers; it bills with
    its island (:func:`island_price`)."""
    return isinstance(entry, Mapping) and entry.get("kind") == "miles"


def _island_samplers(run_config: Mapping[str, Any], index: int) -> list[dict[str, Any]]:
    entry = _entries(run_config)[index]
    out = []
    for i, pool in enumerate(entry.get("samplers") or ()):
        if isinstance(pool, Mapping) and not is_miles_sampler(pool):
            out.append({**pool, "name": pool_name(index, pool, i)})
    return _with_auto_backlog(run_config, index, out)


def _with_auto_backlog(run_config: Mapping[str, Any], index: int, pools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``pools`` of island ``index`` with the ``"auto"`` backlog cap the fleet compile gives
    them (``auto_backlog``, :func:`yeto.rl.fleet.island_auto_backlog`), so a view of the raw
    run config reads the cap the samplers run with; unchanged when it cannot be derived."""
    try:
        from yeto.launch.rollout.pools import entry_doc  # noqa: PLC0415
        from yeto.rl import up_consistency  # noqa: PLC0415

        docs = [entry_doc(p, p["name"])[0] for p in pools]
        auto = up_consistency.island_auto_backlog(island_trainer_argv(run_config, index), docs)
        if auto is None:
            return pools
        name = _entries(run_config)[index].get("name")
        cap = {"groups": int(auto["groups"]), "why": f"island {index} ({name}): {auto['why']}"}
        return [{**p, "auto_backlog": cap}
                if "auto_backlog" not in p and up_consistency.is_auto_backlog(up_consistency.effective_spec(d)) else p
                for p, d in zip(pools, docs)]
    except (OSError, ValueError, TypeError, KeyError):
        return pools  # a pool that does not load: the pool dry-run reports it


def island_price(run_config: Mapping[str, Any], index: int) -> float:
    """Island ``index``'s ``usd_per_h`` in the launch: its trainer's plus its Miles samplers'
    (0 when unset or null)."""
    entry = _entries(run_config)[index]
    trainer = entry.get("trainer")
    miles = sum(_price(s.get("usd_per_h")) for s in entry.get("samplers") or () if is_miles_sampler(s))
    return _price(trainer.get("usd_per_h") if isinstance(trainer, Mapping) else None) + miles


def island_prices(run_config: Mapping[str, Any]) -> tuple[float, ...]:
    return tuple(island_price(run_config, k) for k in range(len(_entries(run_config))))


def sandbox_price(run_config: Mapping[str, Any]) -> float:
    """``fleet.sandbox_usd_per_h``: the hourly price of the fleet's sandboxes (its agentic
    pools' boat sandboxes, docker hosts); 0 when unset."""
    section = run_config.get("fleet")
    return _price(section.get("sandbox_usd_per_h") if isinstance(section, Mapping) else None)


# ``yeto launch`` defaults the syncer placement reads (pinned against the parser by a test)
DEFAULT_CONTROLLER = "head"
DEFAULT_SYNCER_CLOUD = "aws"
DEFAULT_SYNCER_MEMORY_GIB = 32
HEAD_VM = "controller head VM"


@dataclass(frozen=True)
class SyncerPlacement:
    """Where a fleet's syncer runs: ``head`` (on the controller head VM it shares with the fleet
    controller, --controller head), ``vm`` (its own syncer VM, --controller local) or ``none`` (no
    syncer). Either machine is one on-demand sky VM on ``provider`` (the --syncer-region cloud) with
    cpus 8+ and ``memory_gib`` (--syncer-memory)."""

    where: Literal["head", "vm", "none"]
    provider: str | None = None
    memory_gib: float | None = field(default=None, compare=False)

    def to_json(self) -> dict[str, Any]:
        return {"where": self.where, "provider": self.provider}


def _last_value(argv: Sequence[str], flag: str) -> str | None:
    value = None
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            value = argv[i + 1]
        elif a.startswith(f"{flag}="):
            value = a.split("=", 1)[1]
    return value


def syncer_placement(argv: Sequence[str]) -> SyncerPlacement:
    """Where the trainer launch ``argv`` runs its syncer (``--controller``, ``--syncer-region``
    as ``region`` (AWS) or ``cloud/region``, ``--rl-single-island-no-sync``)."""
    argv = [str(a) for a in argv]
    if _sets(argv, "--rl-single-island-no-sync"):
        return SyncerPlacement("none")
    where = "head" if (_last_value(argv, "--controller") or DEFAULT_CONTROLLER) == "head" else "vm"
    region = _last_value(argv, "--syncer-region") or ""
    try:
        memory = float(_last_value(argv, "--syncer-memory") or DEFAULT_SYNCER_MEMORY_GIB)
    except ValueError:
        memory = None
    return SyncerPlacement(where, region.split("/", 1)[0].lower() if "/" in region else DEFAULT_SYNCER_CLOUD,
                           memory)


def run_syncer_placement(run_config: Mapping[str, Any]) -> SyncerPlacement:
    """The syncer placement of a fleet run config: its compiled ``trainer.argv`` when it has
    one, else ``fleet.trainer_argv``."""
    trainer = run_config.get("trainer")
    argv = _strings(trainer.get("argv")) if isinstance(trainer, Mapping) else []
    return syncer_placement(argv or fleet_trainer_argv(run_config))


def syncer_rate(usd_per_h: Any, placement: SyncerPlacement | None) -> tuple[float | None, str]:
    """The syncer's USD per hour and where it comes from: ``syncer.usd_per_h`` when set (0: it
    bills nothing); else the machine it runs on (the controller head VM or its own syncer VM) at
    the price table's ``services.syncer`` row of that VM's provider; no syncer: $0. None:
    unpriced (no row for the provider, or the placement is unknown)."""
    if usd_per_h is not None and not isinstance(usd_per_h, bool):
        return _price(usd_per_h), "run config syncer.usd_per_h"
    if placement is None:
        return None, "the syncer's placement is unknown"
    if placement.where == "none":
        return 0.0, "no syncer (--rl-single-island-no-sync)"
    from yeto.cost import prices  # noqa: PLC0415

    machine = (f"the {HEAD_VM} on {placement.provider} (it hosts the syncer and the fleet controller)"
               if placement.where == "head" else f"its own syncer VM on {placement.provider}")
    row = prices.service(prices.SYNCER, placement.provider or "")
    if row is None:
        return None, f"{machine}: no price table row for it"
    return row.usd_h, f"{machine}: price table {row.cited()}"


def syncer_price(run_config: Mapping[str, Any]) -> float:
    """The syncer's USD per hour for the fleet spend (:func:`syncer_rate` of the run config;
    0 when unpriced)."""
    syncer = run_config.get("syncer")
    value = syncer.get("usd_per_h") if isinstance(syncer, Mapping) else None
    rate, _source = syncer_rate(value, run_syncer_placement(run_config))
    return 0.0 if rate is None else rate


def _price(value: Any) -> float:
    if value is None or isinstance(value, bool):
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) and number > 0 else 0.0


def island_run_config(run_config: Mapping[str, Any], index: int) -> dict[str, Any]:
    """Island ``index`` read as a single-trainer run config: the run's shared sections
    (``budget``, ``breakers``, ``env``, ...), the island's trainer (:func:`island_trainer_argv`,
    its ``usd_per_h``) and the island's pools as ``samplers`` named ``i<k>-<name>``. Every
    single-trainer reader (the resolver, the dependency-rule facts) reads it unchanged."""
    entry = _entries(run_config)[index]
    trainer: dict[str, Any] = {"argv": island_trainer_argv(run_config, index)}
    raw = entry.get("trainer") if isinstance(entry.get("trainer"), Mapping) else {}
    if raw.get("usd_per_h") is not None:
        trainer["usd_per_h"] = raw["usd_per_h"]
    base = _base(run_config)
    own = _island_own_argv(run_config, index)
    for flag, (name, param) in ISLAND_SECTION_FLAGS.items():
        if _sets(own, flag):  # the island's own sample wait / policy age wins over the run's
            base = _without_section_param(base, name, param)
    return {**base, "trainer": trainer, "samplers": _island_samplers(run_config, index)}


def _without_section_param(base: dict[str, Any], name: str, param: str) -> dict[str, Any]:
    """``base`` without ``breakers.trainer.<name>.<param>`` (copied where changed)."""
    section = base.get("breakers")
    trainer = section.get("trainer") if isinstance(section, Mapping) else None
    entry = trainer.get(name) if isinstance(trainer, Mapping) else None
    if not isinstance(entry, Mapping) or param not in entry:
        return base
    entry = {k: v for k, v in entry.items() if k != param}
    trainer = {**trainer, name: entry} if entry else {k: v for k, v in trainer.items() if k != name}
    return {**base, "breakers": {**section, "trainer": trainer}}


def fleet_run_config(run_config: Mapping[str, Any]) -> dict[str, Any]:
    """The fleet read as one run config, for its fleet-wide entries: the shared sections,
    ``fleet.trainer_argv`` as the trainer (the launch layer's knobs), the islands' summed
    ``usd_per_h`` and every pool of every island (``i<k>-<name>``)."""
    samplers = [p for k in range(len(_entries(run_config))) for p in _island_samplers(run_config, k)]
    trainer = {"argv": fleet_trainer_argv(run_config), "usd_per_h": sum(island_prices(run_config))}
    return {**_base(run_config), "trainer": trainer, "samplers": samplers}


def resolve_islands(run_config: Mapping[str, Any], *, strict: bool = False) -> list[tuple[Island, Any]]:
    """``(island, ResolvedBreakers)`` of each island: its run and trainer entries as that
    island's trainer and monitor apply them (:func:`island_run_config`)."""
    from yeto.breakers.resolve import Sources, resolve  # noqa: PLC0415

    return [(island, resolve(Sources.from_run_config(island_run_config(run_config, island.index)), strict=strict))
            for island in islands(run_config)]


# --- stops ------------------------------------------------------------------------------------------


class FleetStops:
    """Which islands a fleet's trips stopped, and why.

    An island-scoped trip (a ``run`` / ``trainer`` entry with a stop reason) stops its own
    island only; a fleet-scoped one (``run.budget``, ``run.wall_clock``, the launch layer)
    stops every island; a pool, sampler, harness or monitor trip stops no island. The first
    reason of each island wins, as :func:`yeto.breakers.events.record_stop_reason` keeps the
    first reason of a process. A run / trainer trip that names no island stops every island
    (the fleet's one launch reported it)."""

    def __init__(self, islands_: Iterable[Island]):
        self.islands = tuple(islands_)
        self._reasons: dict[int, str] = {}
        self._fleet: str | None = None

    def record(self, event: Mapping[str, Any], *, island: int | None = None) -> tuple[int, ...]:
        """Record one ``rl_breaker`` event (``island``: the island whose tape it came from,
        else its ``scope.island``); the islands it stops (() for a warn, a clear, a trip
        without a stop reason or a component trip)."""
        if event.get("event", "rl_breaker") != "rl_breaker" or event.get("state") != "trip":
            return ()
        reason = event.get("stop_reason")
        name = str(event.get("name") or "")
        if not reason or not name:
            return ()
        scope = breaker_scope(name)
        if scope == "component":
            return ()
        known = tuple(i.index for i in self.islands)
        target = island if island is not None else _island_of(event)
        if scope == "fleet" or target is None:
            hit = known
            if scope == "fleet" and self._fleet is None:
                self._fleet = str(reason)
        elif target in known:
            hit = (target,)
        else:
            return ()
        for k in hit:
            self._reasons.setdefault(k, str(reason))
        return hit

    def stop(self, island: int, reason: str) -> bool:
        """Island ``island`` stopped by an operator (``yeto island stop``), not by a trip: it
        stops that island only (True when it ran)."""
        if island not in {i.index for i in self.islands} or island in self._reasons:
            return False
        self._reasons[island] = str(reason)
        return True

    def stop_reason(self, island: int) -> str | None:
        """Why island ``island`` stopped (None: it runs)."""
        return self._reasons.get(island)

    def running(self) -> tuple[int, ...]:
        return tuple(i.index for i in self.islands if i.index not in self._reasons)

    @property
    def fleet_stop_reason(self) -> str | None:
        """The fleet's stop reason: a fleet-scoped trip's, else (every island stopped) the
        first island's; None while an island runs."""
        if self._fleet is not None:
            return self._fleet
        if self.islands and not self.running():
            return self._reasons[self.islands[0].index]
        return None

    def to_json(self) -> dict[str, Any]:
        return {"stop_reason": self.fleet_stop_reason,
                "islands": [{"index": i.index, "name": i.name, "stop_reason": self._reasons.get(i.index)}
                            for i in self.islands]}


def _island_of(event: Mapping[str, Any]) -> int | None:
    scope = event.get("scope")
    value = scope.get("island") if isinstance(scope, Mapping) else None
    try:
        return None if value is None or isinstance(value, bool) else int(value)
    except (TypeError, ValueError):
        return None


# --- spend ------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FleetSpend:
    """A fleet's spend: its pools' node hours, each island's trainer and the syncer (USD)."""

    pools_usd: float
    islands_usd: tuple[float, ...]
    syncer_usd: float
    sandbox_usd: float = 0.0  # the agentic pools' sandboxes (fleet.sandbox_usd_per_h)

    @property
    def trainers_usd(self) -> float:
        return sum(self.islands_usd)

    @property
    def total_usd(self) -> float:
        return self.pools_usd + self.trainers_usd + self.syncer_usd + self.sandbox_usd

    def to_json(self) -> dict[str, Any]:
        out = {"usd": round(self.total_usd, 2), "pools_usd": round(self.pools_usd, 2),
               "islands_usd": [round(u, 2) for u in self.islands_usd], "syncer_usd": round(self.syncer_usd, 2)}
        if self.sandbox_usd:  # a fleet without priced sandboxes keeps its record as it was
            out["sandbox_usd"] = round(self.sandbox_usd, 2)
        return out


def fleet_spend(*, pool_node_hours: Iterable[tuple[float, float]], island_hours: Sequence[float],
                island_usd_per_h: Sequence[float], syncer_hours: float = 0.0,
                syncer_usd_per_h: float = 0.0, sandbox_hours: float = 0.0,
                sandbox_usd_per_h: float = 0.0) -> FleetSpend:
    """The fleet's spend: Σ pool node hours × that pool's node price, plus Σ island trainer
    hours × its ``usd_per_h``, plus syncer hours × ``syncer.usd_per_h``, plus the sandboxes'
    hours (while the agentic pools run) × ``fleet.sandbox_usd_per_h`` (:func:`sandbox_price`).

    ``pool_node_hours``: ``(billed node hours, usd per node hour)`` of each pool (``run up``'s
    ``node_hours(i)`` and ``node_usd_per_h(i)``); ``island_hours``: each island trainer's billed
    hours (an island that finished bills up to its finish), with ``island_usd_per_h`` in the
    same order (:func:`island_prices`). A negative or non-finite hour or price counts 0.
    Raises ValueError when the two island lists differ in length."""
    if len(island_hours) != len(island_usd_per_h):
        raise ValueError(f"fleet spend: {len(island_hours)} island hours for {len(island_usd_per_h)} island prices")
    pools = sum(_amount(hours) * _amount(price) for hours, price in pool_node_hours)
    trainers = tuple(_amount(h) * _amount(p) for h, p in zip(island_hours, island_usd_per_h))
    return FleetSpend(pools, trainers, _amount(syncer_hours) * _amount(syncer_usd_per_h),
                      _amount(sandbox_hours) * _amount(sandbox_usd_per_h))


def _amount(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) and number > 0 else 0.0


def budget_tripped(spend_usd: float, cap_usd: float | None) -> bool:
    """Whether the fleet's spend passed ``run.budget.usd`` (null: no cap), as ``run up``'s
    breach judges one trainer's run."""
    from yeto.rl.up_consistency import NO_CAP_USD  # noqa: PLC0415

    return spend_usd > (NO_CAP_USD if cap_usd is None else float(cap_usd))


def budget_trip(spend: FleetSpend | float, cap_usd: float | None, *, emit: Any = None,
                **fields: Any) -> dict[str, Any] | None:
    """The fleet's ``run.budget`` trip (an ``rl_breaker`` event whose ``stop_reason`` is
    ``breaker:run.budget``, recorded for the process and passed to ``emit``) when the spend
    passed the cap, else None. :meth:`FleetStops.record` of it stops every island."""
    from yeto.breakers.events import trip  # noqa: PLC0415

    total = spend.total_usd if isinstance(spend, FleetSpend) else float(spend)
    if not budget_tripped(total, cap_usd):
        return None
    detail = fields.pop("detail", None) or f"fleet spend ${total:.2f} > budget ${float(cap_usd or 0):.2f}"
    return trip("run.budget", round(total, 2), cap_usd, emit=emit, detail=detail, **fields)
