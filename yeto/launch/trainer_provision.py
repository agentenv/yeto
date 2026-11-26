"""Trainer islands' first launch: one adapter per machine source.

The launcher walks the islands of ``--gpu`` and hands each to the adapter of its source
(:meth:`TrainerProvision.source`); it holds no per-source launch branch. An adapter
renders what its source launches, orders it and says what the run's resource ledger
records of it:

* ``sky`` (every sky cloud, the default): ``sky.launch`` submitted during the walk, waited
  on in a thread (``sky.stream_and_get``), the Nebius reclaim hook after it;
* ``verda``: ordered ``any_of`` in the wait thread (:func:`yeto.launch.verda.launch_verda_island`);
* ``modal``: the island's function config; every Modal island is defined on one app,
  deployed and spawned after the walk;
* ``ssh`` (resident inventory islands, :mod:`yeto.launch.ssh_resident`): the SSH trainer
  spec, started (or reattached to) over SSH after the walk.

A new source is one more adapter (:func:`register`, ahead of the ``sky`` default); the
launcher does not change. The held trainer island (the trainer scheduler's), the mixed
islands of :mod:`yeto.launch.engine_cluster` and adopted islands stay with the launcher
(design D5: Verda and engine_cluster keep their own paths until they get a provider).

The render, the order of the calls and every line printed are those of the launcher
before the adapters (``tests/test_trainer_launch_render.py`` holds them byte for byte).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..gpu_spec import ClusterSpec


@dataclass
class Island:
    """One island the provision launches: ``task`` is what its source launches (a sky
    task, a Modal island config, an SSH trainer spec)."""

    name: str
    m: int
    spec: ClusterSpec
    role: str | None
    task: Any
    source: str = "sky"


@dataclass
class TrainerProvision:
    """The first launch of a run's trainer islands. ``on_order(name)``: the island record
    before an order (raises: nothing ordered); ``on_call(name, call_id)``: a Modal call id
    once spawned; ``adopted(name)``: the job / call id an adopted island keeps (None: launch
    it); ``guard_order(name)``: the records lock is still ours (raises: nothing ordered)."""

    args: Any
    sky: Any
    specs: list[ClusterSpec]
    names: list[str]
    ssh_names: set[str] = field(default_factory=set)
    syncer_addr: str = "none"
    no_sync: bool = False
    engine_args: Callable[[int], Any] = lambda m: None
    verda: dict | None = None
    on_order: Callable[[str], None] = lambda name: None
    on_call: Callable[[str, str], None] | None = None
    adopted: Callable[[str], Any] = lambda name: None
    guard_order: Callable[[str], None] | None = None
    ledger: Any = None
    events_dir: Path | None = None
    collectors: dict = field(default_factory=dict)
    results: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    islands: dict[str, Island] = field(default_factory=dict)
    held: Island | None = None
    modal_addr: str | None = None

    def __post_init__(self) -> None:
        for adapter in _SOURCES:
            adapter.setup(self)

    # -- the walk
    def source(self, name: str, spec: ClusterSpec) -> Source:
        return next(a for a in _SOURCES if a.claims(self, name, spec))

    def island(self, name: str, m: int, spec: ClusterSpec, role: str | None, task: Any) -> Island:
        """The island ``name`` as its source launches it (an adapter may print its route)."""
        adapter = self.source(name, spec)
        island = Island(name, m, spec, role, task, adapter.kind)
        island.task = adapter.render(self, island)
        return island

    def hold(self, island: Island) -> tuple[Any, Any]:
        """``island`` is the trainer scheduler's: ``(modal config, sky task)`` it starts it from."""
        self.held = island
        return _adapter(island.source).hold(self, island)

    def add(self, island: Island) -> None:
        self.islands[island.name] = island

    def announce(self, island: Island) -> None:
        _adapter(island.source).announce(self, island)

    def ordered(self, island: Island) -> bool:
        """True: the source orders the island during the walk (and :meth:`wait` it after)."""
        return _adapter(island.source).ordered

    def order(self, island: Island) -> Any:
        return _adapter(island.source).order(self, island)

    def wait(self, island: Island, handle: Any) -> Any:
        return _adapter(island.source).wait(self, island, handle)

    def ledger_attrs(self, island: Island) -> dict[str, Any]:
        return _adapter(island.source).ledger_attrs(self, island)

    # -- the batch sources, after the walk
    def batch(self, kind: str) -> dict[str, Island]:
        return {n: i for n, i in self.islands.items() if i.source == kind}

    def modal_app(self):
        """The Modal app with every Modal island (and a held one) defined; None: no Modal
        island. The caller keeps it before :meth:`spawn_modal`, so a deploy that fails is
        still stopped."""
        return _adapter("modal").app(self)

    def spawn_modal(self, modal_ops):
        """Deploy ``modal_ops`` and spawn every Modal island; their ops."""
        return _adapter("modal").spawn(self, modal_ops)

    def start_ssh(self):
        """Start (or reattach to) every resident ``ssh:`` island; their ops (None: none)."""
        return _adapter("ssh").start(self)


class Source:
    """A source adapter. ``ordered``: one order per island during the walk, waited on in
    a thread after it; otherwise the source starts its islands in one batch after the walk."""

    kind = ""
    ordered = False

    def setup(self, prov: TrainerProvision) -> None:  # once, before the walk
        pass

    def claims(self, prov: TrainerProvision, name: str, spec: ClusterSpec) -> bool:
        return spec.cloud == self.kind

    def render(self, prov: TrainerProvision, island: Island) -> Any:
        return island.task

    def hold(self, prov: TrainerProvision, island: Island) -> tuple[Any, Any]:
        return None, island.task

    def announce(self, prov: TrainerProvision, island: Island) -> None:
        pass

    def order(self, prov: TrainerProvision, island: Island) -> Any:
        raise NotImplementedError(f"{self.kind}: islands start in one batch")

    def wait(self, prov: TrainerProvision, island: Island, handle: Any) -> Any:
        raise NotImplementedError(f"{self.kind}: islands start in one batch")

    def ledger_attrs(self, prov: TrainerProvision, island: Island) -> dict[str, Any]:
        return {}


class SkySource(Source):
    """Every sky cloud: one cluster per island, submitted during the walk."""

    kind = "sky"
    ordered = True

    def claims(self, prov, name, spec):
        return True  # the default: any cloud no other adapter claims is sky's

    def announce(self, prov, island):
        from .syncer import island_syncer_addrs
        from .topology import rl_island_args, rl_island_spec

        args, spec, m, name = prov.args, island.spec, island.m, island.name
        print(f"[launcher] {name} dials {island_syncer_addrs(getattr(island.task, 'envs', None))}", flush=True)
        alloc = (rl_island_spec(rl_island_args(args, m), spec)
                 if getattr(args, "training_mode", "sft") == "rl" else spec)
        print(f"[launcher] launching learner {m} on {spec} as {name}"
              + (f" (physical {spec.gpus_per_node} GPUs/node billed, island allocated "
                 f"{alloc.gpus_per_node}/node = GPUs 0..{alloc.gpus_per_node - 1})"
                 if alloc is not spec else ""))

    def order(self, prov, island):
        from .sky_ops import submit_island_launch

        prov.on_order(island.name)
        return submit_island_launch(prov.sky, prov.args, island.spec, island.task, island.name)

    def wait(self, prov, island, handle):
        from .sky_ops import maybe_install_nebius_reclaim_hook

        result = prov.sky.stream_and_get(handle)
        maybe_install_nebius_reclaim_hook(island.task, result, island.name)
        return result

    def ledger_attrs(self, prov, island):
        """What prices the cluster: ``--trainer-usd-per-hour`` x its nodes when stated
        (``usd_per_hour``, billing's first choice), else the unified price table's row of
        its cloud, card, region and spotness (:func:`yeto.cost.prices.node_usd_h`, read by
        billing), as for a sky sampler. A machine without a table row records the sky
        catalog's on-demand price of its ``--learner-instance-type`` (else of the dearest
        instance for its cards) x its nodes (:func:`yeto.cost.prices.catalog_node_usd_h`),
        so the cost guard and ``run.budget`` count it instead of reporting it unpriced."""
        spec = island.spec
        attrs: dict[str, Any] = {"role": "trainer", "cloud": spec.cloud, "gpu": spec.gpu,
                                 "gpus": int(spec.gpus_per_node) * int(spec.num_nodes),
                                 "num_nodes": int(spec.num_nodes)}
        if island.role:
            attrs["island_role"] = island.role  # anchor / droppable / sampler (an island of the trainer fleet)
        if getattr(spec, "region", None):
            attrs["region"] = spec.region
        spot = _task_spot(island.task)
        if spot is None:
            spot = getattr(prov.args, "spot", None)
        if spot is not None:
            attrs["use_spot"] = bool(spot)
        stated = getattr(prov.args, "trainer_usd_per_hour", None)
        if stated is not None:
            attrs["usd_per_hour"] = float(stated) * int(spec.num_nodes)
        elif _table_unpriced(attrs):
            from yeto.cost import prices  # noqa: PLC0415

            rate, note = prices.catalog_node_usd_h(spec.cloud, spec.gpu, spec.gpus_per_node, region=attrs.get("region"),
                                                   instance_type=getattr(prov.args, "learner_instance_type", None))
            if rate is not None:
                attrs["usd_per_hour"] = rate * int(spec.num_nodes)
                attrs["price_note"] = note + " (no price table row" + ("; spot billed at on-demand)"
                                                                        if attrs.get("use_spot") else ")")
            else:
                print(f"[launcher] WARNING: trainer cluster {island.name}: no price table row and {note}: "
                      "the cost guard reports it unpriced", flush=True)
        return attrs


class VerdaSource(SkySource):
    """Verda: ordered any_of on live stock in the wait thread (refresh + bounded backoff)."""

    kind = "verda"

    def claims(self, prov, name, spec):
        return spec.cloud == self.kind

    def order(self, prov, island):
        prov.on_order(island.name)
        return island.spec  # launched in wait(): live stock -> ordered any_of

    def wait(self, prov, island, handle):
        from ..cloud.droppable import role_args
        from .verda import launch_verda_island

        result = launch_verda_island(prov.sky, island.task, island.name, handle, role_args(prov.args, island.role))
        prov.verda["guard"].record(island.name, getattr(result[1], "cluster_name_on_cloud", None))
        return result

    def ledger_attrs(self, prov, island):
        return {}  # priced by the Verda instance guard's own records


class ModalSource(Source):
    """Modal: one function per island on one app, deployed and spawned after the walk."""

    kind = "modal"

    def setup(self, prov):
        if prov.no_sync:
            prov.modal_addr = prov.syncer_addr  # nothing to reach
        elif any(spec.cloud == self.kind for spec in prov.specs):
            from ..modal_runner import resolve_syncer_for_modal

            # Fails BEFORE any Modal container starts when the syncer is
            # not reachable from Modal's network.
            prov.modal_addr = resolve_syncer_for_modal(
                prov.syncer_addr, getattr(prov.args, "syncer_public_addr", None)
            )

    def render(self, prov, island):
        from ..cloud.droppable import role_args
        from .modal_islands import build_modal_island_config
        from .syncer import island_syncer_addrs

        cfg = build_modal_island_config(role_args(prov.args, island.role), island.spec, island.m, island.task,
                                        prov.modal_addr, engine_args=prov.engine_args(island.m))
        print(f"[launcher] {island.name} dials {island_syncer_addrs(cfg.envs)}", flush=True)
        return cfg

    def hold(self, prov, island):
        return island.task, None  # defined on the app; the scheduler spawns it

    def app(self, prov):
        cfgs = [i.task for i in prov.batch(self.kind).values()]
        held = prov.held.task if prov.held is not None and prov.held.source == self.kind else None
        if not cfgs and held is None:
            return None
        from ..modal_runner import ModalOps, modal_app_name

        modal_ops = ModalOps(modal_app_name(prov.args.cluster_prefix))
        modal_ops.ledger = prov.ledger
        # which of the trainer's tape records count as egress out of Modal (yeto.cost.egress),
        # and whose: the island ids its records carry (a launch's other islands are elsewhere)
        from ..cost import egress as cost_egress

        islands = [i.m for i in prov.batch(self.kind).values()]
        if held is not None:
            islands.append(prov.held.m)
        modal_ops.ledger_egress = {"egress_events": cost_egress.egress_events(prov.args),
                                   "egress_islands": cost_egress.tape_island_ids(prov.args, islands),
                                   "egress_fallback_bytes": cost_egress.fallback_bytes(prov.args)}
        for cfg in [*cfgs, *([] if held is None else [held])]:
            modal_ops.define(cfg)
        print(f"[launcher] deploying Modal app {modal_ops.app_name} "
              f"({len(cfgs) + (held is not None)} island(s))")
        return modal_ops

    def spawn(self, prov, modal_ops):
        from ..modal_runner import ModalIslandOps

        if prov.guard_order is not None:
            prov.guard_order(modal_ops.app_name)  # the lock still ours (raises: nothing deployed)
        modal_ops.deploy()
        island_ops = ModalIslandOps(modal_ops)
        if prov.on_call is not None:  # the call id is in the island's record from its spawn on
            island_ops.on_call = prov.on_call
        for name, island in prov.batch(self.kind).items():
            cfg = island.task
            kept = prov.adopted(name)
            if kept is not None:
                call_id = str(kept[0])
                island_ops.calls[name] = call_id
                prov.results[name] = (call_id, None)
                print(f"[launcher] learner {cfg.learner_id} on Modal as {name}: adopted call {call_id}")
                continue
            print(f"[launcher] launching learner {cfg.learner_id} on Modal as {name}")
            prov.on_order(name)
            call_id = island_ops.relaunch(cfg, name)
            if call_id is None:
                prov.errors[name] = RuntimeError("Modal refused to start the island")
            else:
                prov.results[name] = (call_id, None)
        return island_ops


class SshSource(Source):
    """Resident ``ssh:`` trainer islands on the node inventory, started over SSH."""

    kind = "ssh"

    def claims(self, prov, name, spec):
        return name in prov.ssh_names

    def render(self, prov, island):
        spec = ssh_trainer_spec(prov.args, island.spec, island.task)
        print(f"[launcher] {island.name} is a resident ssh: trainer island on {island.spec.num_nodes} inventory "
              "node(s): the launcher starts it over SSH and supervises it for the run", flush=True)
        return spec

    def start(self, prov):
        names = set(prov.batch(self.kind))
        if not names:
            return None
        tasks = {n: prov.islands[n].task for n in names}
        if prov.guard_order is not None:  # a start that is no reattach orders the island's processes
            prov.guard_order(",".join(sorted(names)))
        return start_resident_ssh(prov.args, tasks, names, prov.results, prov.errors, prov.events_dir,
                                  prov.collectors)


_SOURCES: list[Source] = [SshSource(), ModalSource(), VerdaSource(), SkySource()]


def register(adapter: Source) -> None:
    """Add a source adapter, ahead of the ``sky`` default (a kind registered again replaces)."""
    _SOURCES[:] = [a for a in _SOURCES if a.kind != adapter.kind]
    _SOURCES.insert(len(_SOURCES) - 1, adapter)


def _adapter(kind: str) -> Source:
    return next(a for a in _SOURCES if a.kind == kind)


def _table_unpriced(attrs: dict[str, Any]) -> bool:
    """Whether billing finds no unified price table row for a sky trainer cluster of ``attrs``
    (the lookup billing makes for a ``sky_cluster`` without ``usd_per_hour``)."""
    from yeto.cost import prices  # noqa: PLC0415

    return prices.node_usd_h(attrs["cloud"], attrs["gpu"], attrs["gpus"], spot=attrs.get("use_spot"),
                             region=attrs.get("region")) is None


def _task_spot(task: Any) -> bool | None:
    """``use_spot`` of the resources a sky task orders (the island's own, which a role or
    an island override may set apart from ``--spot``); None: not readable."""
    resources = getattr(task, "resources", None)
    if resources is None:
        return None
    if not isinstance(resources, dict):
        try:
            resources = next(iter(resources), None)
        except TypeError:
            pass
    spot = resources.get("use_spot") if isinstance(resources, dict) else getattr(resources, "use_spot", None)
    return None if spot is None else bool(spot)


# -- resident ssh islands (moved from the launcher) --------------------------------

def resident_ssh_names(args, names: list[str], specs: list[ClusterSpec]) -> set[str]:
    """The ``--gpu`` islands that are resident ``ssh:`` trainer islands (ssh_resident)."""
    from .ssh_resident import is_resident

    return {n for m, (n, s) in enumerate(zip(names, specs)) if is_resident(args, s, m, len(specs))}


def start_resident_ssh(args, tasks, names, results, errors, events_dir, collectors):
    """Start (or reattach to) every resident ``ssh:`` island; returns their ops. Node 0's
    lines go to the launcher log and the island's echo tape collector."""
    from .. import runs
    from ..rl.event_echo import TapeCollector as EventCollector
    from .ssh_resident import RESIDENT_DIR, SshIslandOps

    def sink_for(name: str):
        collector = collectors.get(name)
        if collector is None and events_dir is not None:
            collector = collectors[name] = EventCollector(Path(events_dir) / f"{name}.jsonl", fresh=False)

        def sink(line: str) -> None:
            print(f"[{name}] {str(line).rstrip()}", flush=True)
            if collector is not None:
                collector.feed(line)
        return sink

    ops = SshIslandOps(run_id=args.cluster_prefix, journal_dir=runs.run_dir(args.cluster_prefix) / RESIDENT_DIR,
                       events_dir=events_dir, sink_for=sink_for)
    for name in sorted(names):
        ops.add(name, tasks[name])
        try:
            results[name] = (ops.start(name), None)
        except Exception as e:  # noqa: BLE001 - reported with the other launch errors
            errors[name] = e
    return ops


def ssh_trainer_spec(args, spec: ClusterSpec, task):
    """The SSH spec of an ``ssh:`` trainer island (None on other clouds)."""
    if spec.cloud != "ssh":
        return None
    from .ssh_trainer_island import island_spec_from_inventory

    return island_spec_from_inventory(args, spec, task)


# -- a budget needs a price ---------------------------------------------------------

# sky clouds of the operator's own machines: no cloud bill to price
OWN_MACHINE_CLOUDS = frozenset({"ssh", "kubernetes"})


def sky_price_problems(args, specs: list[ClusterSpec], names: list[str] | None = None) -> list[str]:
    """The sky islands this launch orders that no price covers: neither
    ``--trainer-usd-per-hour`` nor a unified price table row of their cloud, card, region
    and spotness (the row a sky sampler's offer is priced from). Their spend is invisible to
    ``--budget-usd``. The trainer scheduler's held island is one of them when it is on a
    sky cloud (each incarnation is a sky cluster priced the same way). Modal (its own app
    billing), Verda (its instance guard), resident ``ssh:`` islands and the operator's own
    machines (``ssh`` / ``kubernetes`` sky clouds) are not this check's."""
    if getattr(args, "trainer_usd_per_hour", None) is not None:
        return []
    from ..cost import prices
    from .topology import rl_island_args

    rl = getattr(args, "training_mode", "sft") == "rl"
    names = names or [f"island {m}" for m in range(len(specs))]
    ssh = resident_ssh_names(args, names, specs) if rl else set()
    out = []
    for m, spec in enumerate(specs):
        if names[m] in ssh or spec.cloud in OWN_MACHINE_CLOUDS:
            continue
        if next(a for a in _SOURCES if a.claims(_Probe(ssh), names[m], spec)).kind != SkySource.kind:
            continue
        spot = bool(getattr(rl_island_args(args, m) if rl else args, "spot", False))
        if prices.node_usd_h(spec.cloud, spec.gpu, spec.gpus_per_node, spot=spot, region=spec.region) is None:
            out.append(f"--gpu island {m} ({spec}, {'spot' if spot else 'on-demand'}): no unified price table row "
                       f"for {spec.cloud} {spec.gpu}" + (f" in {spec.region}" if spec.region else ""))
    return out


@dataclass
class _Probe:
    """What an adapter's ``claims`` reads before a provision exists."""

    ssh_names: set[str]


def check_sky_prices(args, specs: list[ClusterSpec], names: list[str] | None = None) -> None:
    """With ``--budget-usd``, refuse (ConfigError) a sky island no price covers, as a capped
    run refuses an unpriced sampler offer; without a budget, nothing changes."""
    if getattr(args, "budget_usd", None) is None:
        return
    problems = sky_price_problems(args, specs, names)
    if problems:
        from ..config_errors import ConfigError

        raise ConfigError(
            f"--budget-usd {args.budget_usd:g} cannot see the spend of {len(problems)} sky island(s): "
            + "; ".join(problems) + ". State the price with --trainer-usd-per-hour USD (per node-hour; its "
            "clusters' ledger rows then carry usd_per_hour), use a priced region / card, or drop --budget-usd")
