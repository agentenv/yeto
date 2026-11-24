"""Run facts the dependency rules read besides the resolved breakers.

The rules (:mod:`yeto.breakers.rules`) compare breaker thresholds with values
that are not breakers: the trainer's release cadence and groups per step
(``trainer.argv`` through the ``yeto launch`` parser), each pool's sampler
count and backlog cap, its group size and agent timeout (semantic entries, read
where they are configured: the sampler spec), the dispense passes and the
training rows of a local ``prompts_file``. Everything here is read from local
files only, and a value that cannot be read is None (the rule then skips).

Each pool gets its own resolve (``pool=<name>``, with its spec, elastic and
dispense sections and its agentic env file) so per-pool rules see the values
that pool's samplers run with.

A fleet (a run config with ``islands``, :mod:`yeto.breakers.fleet`) has one trainer per
island: :func:`collect` then gives one :class:`IslandFacts` per island (that island's
resolve, trainer and pools), and the rules run once per island over them; the fleet's own
facts keep every pool and no trainer (the one-budget rules read them).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from yeto.breakers.fleet import Island
    from yeto.breakers.resolve import ResolvedBreakers, Sources

AUTO = "auto"  # max_backlog_groups "auto": run up derives the cap from the release cadence (#628)
# advantage estimators whose baseline is the group's own mean: a group needs 2 samples
GROUP_BASELINE_ESTIMATORS = frozenset({"grpo", "gspo", "reinforce_plus_plus_baseline"})


@dataclass(frozen=True)
class TrainerFacts:
    """The trainer of ``trainer.argv`` (a ``yeto launch`` argv), as far as the rules need it."""
    groups_per_step: int | None          # --rollout-batch-size: groups of one training round
    group_size: int | None               # --n-samples-per-prompt
    steps: int | None                    # --total-steps: rounds the trainer trains
    release_every: int = 1               # --rl-release-every
    record_rounds: int = 0               # m of sample_roles.release_backlog_need
    backlog_need: int | None = None      # groups the samplers must hold between two releases
    single_buffer_trainer: bool = False  # --rl-sample-role trainer --rl-single-island-no-sync
    # a Miles island reads the sample buffer (role trainer or convertible): it heartbeats while it waits
    buffer_trainer: bool = False
    max_policy_staleness: int | None = None  # the algorithm spec's execution.max_policy_staleness
    estimator: str | None = None         # the algorithm spec's advantage estimator
    backend: str | None = None           # --rl-backend
    modal: bool = False                  # --gpu modal:...: a Modal function (launch.provider_timeout)


@dataclass(frozen=True)
class Backlog:
    """A pool's backlog shape (:func:`yeto.rl.up_consistency.pool_backlog`); ``cap``
    is ``"auto"`` when the pool sizes it itself."""
    cap: int | str
    target: int
    most: int
    over: int


@dataclass(frozen=True)
class Dispense:
    """A pool's store dispenser: its namespace, passes and the training rows of the
    pool's local ``prompts_file`` minus the hold-out (None: not readable here)."""
    namespace: str
    passes: int
    rows: int | None


@dataclass(frozen=True)
class PoolFacts:
    name: str
    resolved: ResolvedBreakers
    data_path: str = "r2"
    sampling_n: int | None = None
    agentic: bool = False
    agent_timeout_s: float | None = None
    provider: str | None = None          # agentic.task_defaults.provider
    drain: bool = False                  # an agentic sampler whose weight switch is a drain
    compaction: bool = False             # agentic.compaction is set
    backlog: Backlog | None = None
    dispense: Dispense | None = None


@dataclass(frozen=True)
class IslandFacts:
    """One island of a fleet: its resolve (the run and trainer entries over its own trainer
    argv) and its facts (its trainer, its pools)."""
    island: Island
    resolved: ResolvedBreakers
    facts: Facts


@dataclass(frozen=True)
class Facts:
    trainer: TrainerFacts | None
    pools: tuple[PoolFacts, ...] = ()
    outage_tolerance_s: float | None = None
    run_up: bool = False                 # a whole run config (yeto run up): the run-layer rules apply
    islands: tuple[IslandFacts, ...] = ()  # a fleet: the rules run once per island (trainer is None)


def collect(resolved: ResolvedBreakers, *, run_config: Mapping[str, Any] | None = None,
            sources: Sources | None = None,
            pools: Iterable[tuple[str, Mapping[str, Any], str]] | None = None) -> Facts:
    """The facts of one run (see :func:`yeto.breakers.rules.check` for the arguments); for a
    fleet run config (and no ``pools``), :func:`collect_fleet`."""
    from yeto.breakers import fleet  # noqa: PLC0415

    if pools is None and fleet.is_fleet(run_config):
        return collect_fleet(resolved, run_config)
    deps = getattr(resolved, "deps", None)
    tolerance = getattr(deps, "outage_tolerance_s", None)
    argv = _argv(run_config, sources)
    args = _trainer_args(argv)
    trainer = trainer_facts(args, argv) if args is not None and argv is not None else None
    if pools is None and run_config is not None:
        pools = run_pools(run_config)
    found: list[PoolFacts] = []
    if pools is not None:
        for name, cfg, data_path in pools:
            found.append(pool_facts(name, cfg, data_path, run_config=run_config, argv=argv))
    elif sources is not None and (getattr(sources, "spec", None) is not None or getattr(sources, "pool_config", None)):
        cfg = dict(getattr(sources, "pool_config", None) or {})  # one pool's process (the pool dry-run)
        spec = dict(sources.spec) if sources.spec is not None else _effective_spec(cfg)
        dispense = getattr(sources, "pool_dispense", None) or (_pool_dispense(cfg) if cfg else None)
        found.append(_facts(getattr(resolved, "pool", None) or "pool", resolved, spec, cfg,
                            str(cfg.get("data_path", "r2")), dispense=dispense))
    return Facts(trainer=trainer, pools=tuple(found), outage_tolerance_s=tolerance,
                 run_up=isinstance((run_config or {}).get("trainer"), Mapping))


def collect_fleet(resolved: ResolvedBreakers, run_config: Mapping[str, Any], *,
                  island_pools: Mapping[int, Iterable[tuple[str, Mapping[str, Any], str]]] | None = None,
                  island_resolved: Mapping[int, ResolvedBreakers] | None = None) -> Facts:
    """The facts of a fleet run config: one :class:`IslandFacts` per island, each collected
    from :func:`yeto.breakers.fleet.island_run_config` like one trainer's run, and the fleet's
    own facts (every pool, no trainer) for the fleet-wide rules. ``island_pools`` /
    ``island_resolved``: an island's loaded pools / its resolve when the caller has them
    (``yeto run up``), else they are read from the run config."""
    from yeto.breakers import fleet  # noqa: PLC0415
    from yeto.breakers.resolve import Sources, resolve  # noqa: PLC0415

    found = []
    for island in fleet.islands(run_config):
        view = fleet.island_run_config(run_config, island.index)
        rb = (island_resolved or {}).get(island.index)
        if rb is None:
            rb = resolve(Sources.from_run_config(view), strict=False)
        mine = (island_pools or {}).get(island.index)
        found.append(IslandFacts(island, rb, collect(rb, run_config=view, pools=mine)))
    pools = tuple(p for i in found for p in i.facts.pools)
    deps = getattr(resolved, "deps", None)
    return Facts(trainer=None, pools=pools, outage_tolerance_s=getattr(deps, "outage_tolerance_s", None),
                 run_up=True, islands=tuple(found))


def _argv(run_config: Mapping[str, Any] | None, sources: Sources | None) -> list[str] | None:
    if sources is not None and getattr(sources, "argv", None) is not None:
        return [str(a) for a in sources.argv]
    trainer = (run_config or {}).get("trainer")
    if isinstance(trainer, Mapping) and isinstance(trainer.get("argv"), list):
        return [str(a) for a in trainer["argv"]]
    return None


def _trainer_args(argv: list[str] | None) -> Any | None:
    if argv is None:
        return None
    from yeto.rl.up_consistency import trainer_args  # noqa: PLC0415

    return trainer_args(argv)


def positive_int(value: Any) -> int | None:
    """``value`` as an int > 0 (a bool, a ``<placeholder>`` or anything else: None)."""
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def trainer_facts(args: Any, argv: list[str]) -> TrainerFacts:
    """:class:`TrainerFacts` of a parsed ``yeto launch`` namespace (RL) and its argv;
    the release cadence is :func:`yeto.rl.up_consistency.release_cadence` (a single
    buffer trainer only)."""
    from yeto.rl.up_consistency import release_cadence  # noqa: PLC0415

    cadence = release_cadence(argv) or {}
    algorithm = _algorithm(args)
    return TrainerFacts(
        groups_per_step=positive_int(getattr(args, "rollout_batch_size", None)),
        group_size=positive_int(getattr(args, "n_samples_per_prompt", None)),
        steps=positive_int(getattr(args, "total_steps", None)),
        release_every=cadence.get("release_every") or positive_int(getattr(args, "rl_release_every", None)) or 1,
        record_rounds=cadence.get("m", 0), backlog_need=cadence.get("need"),
        single_buffer_trainer=bool(cadence), buffer_trainer=buffer_trainer(args),
        max_policy_staleness=None if algorithm is None else algorithm[0],
        estimator=None if algorithm is None else algorithm[1],
        backend=getattr(args, "rl_backend", None),
        modal=str(getattr(args, "gpu", "") or "").startswith("modal:"))


def buffer_trainer(args: Any) -> bool:
    """Whether a Miles island of the launch reads the sample buffer (role ``trainer`` or
    ``convertible``): the islands whose reader heartbeats while it waits (the launcher gives
    them the interval, ``launcher._wait_heartbeat_env``; the verl buffer feed does not beat)."""
    from yeto.launch.topology import rl_island_count  # noqa: PLC0415
    from yeto.rl.engine import sample_roles  # noqa: PLC0415

    if (getattr(args, "rl_backend", None) or "miles") != "miles":
        return False

    readers = (sample_roles.TRAINER, sample_roles.CONVERTIBLE)
    try:
        roles = set(sample_roles.island_roles(args, rl_island_count(args)).values())
        roles.add(sample_roles.sample_role(args))
    except (TypeError, ValueError):
        return False  # the launch's own checks name the bad role
    return bool(roles & set(readers))


def _algorithm(args: Any) -> tuple[int, str] | None:
    """``(max_policy_staleness, advantage estimator)`` of the trainer's algorithm spec
    (None: it cannot be read here, e.g. a spec file on another host)."""
    from yeto.launch.harness_env import _launch_algorithm  # noqa: PLC0415

    raw = _launch_algorithm(args)
    if raw is None:
        return None
    try:
        return policy_staleness(raw), estimator_of(raw)
    except (TypeError, ValueError):
        return None


def policy_staleness(algorithm: Mapping[str, Any]) -> int:
    """``execution.max_policy_staleness`` of an algorithm spec dict (v1 specs: 0)."""
    from yeto.rl.engine.algorithm import AlgorithmSpec  # noqa: PLC0415
    from yeto.rl.engine.execution_profile import algorithm_max_policy_staleness  # noqa: PLC0415

    return int(algorithm_max_policy_staleness(AlgorithmSpec.from_dict(dict(algorithm))))


def estimator_of(algorithm: Mapping[str, Any]) -> str:
    advantage = algorithm.get("advantage") or {}  # v2; a v1 spec names only the estimator
    return str(advantage.get("estimator") or algorithm.get("advantage_estimator") or "grpo")


# -------------------------------------------------------------------- pools

def run_pools(run_config: Mapping[str, Any], *, strict: bool = False) -> list[tuple[str, dict[str, Any], str]]:
    """``(name, pool config, data path)`` of a run config's pools (``pool`` or
    ``samplers``; a fleet: every island's pools, named ``i<k>-<name>``), read like
    :func:`yeto.launch.rollout.pools.resolve` without writing the merged inline configs; a
    pool whose file does not load is left out (the pool dry-run reports it), or with
    ``strict`` raises ValueError naming the file."""
    from yeto.breakers import fleet  # noqa: PLC0415
    from yeto.launch.rollout.pools import ENTRY_KEYS  # noqa: PLC0415

    if fleet.is_fleet(run_config) and run_config.get("pool") is None and run_config.get("samplers") is None:
        return run_pools(fleet.fleet_run_config(run_config), strict=strict)

    def load(path: Any) -> dict[str, Any] | None:
        doc = _load_json(path)
        if doc is None and strict:
            raise ValueError(f"pool config {path}: cannot be read as a JSON object")
        return doc

    if isinstance(run_config.get("pool"), Mapping):
        cfg = load(run_config["pool"].get("config"))
        return [] if cfg is None else [("pool", cfg, str(cfg.get("data_path", "r2")))]
    out = []
    for i, entry in enumerate(run_config.get("samplers") or ()):
        if not isinstance(entry, Mapping):
            continue
        base = load(entry.get("config")) if entry.get("config") else {}
        if base is None:
            continue
        cfg = {**base, **{k: v for k, v in entry.items() if k not in ENTRY_KEYS}}
        name = str(entry.get("name") or (entry.get("provider") if isinstance(entry.get("provider"), str) else None)
                   or f"samplers[{i}]")
        out.append((name, cfg, str(entry.get("data_path", cfg.get("data_path", "r2")))))
    return out


def _load_json(path: Any) -> dict[str, Any] | None:
    if not path:
        return None
    try:
        doc = json.loads(Path(os.path.expanduser(str(path))).read_text())
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def pool_sources(cfg: Mapping[str, Any], *, run_config: Mapping[str, Any] | None = None,
                 argv: list[str] | None = None, spec: Mapping[str, Any] | None = None) -> Sources:
    """The resolver sources of one pool's samplers: the pool-merged spec (the pool's
    ``sampler`` knobs laid over it; its ``sampler.dispense`` replaces ``prompts.dispense``
    wholesale, so the raw spec's dispense is never read beside the pool's), the pool
    file, its elastic and dispense sections (``{}`` when absent: write-back targets),
    and the env they see (the run config's ``env`` and the pool's ``agentic_env_file``).
    ``spec``: this spec instead of the pool's spec file (the pool's knobs still laid over it)."""
    from yeto.breakers.resolve import Sources  # noqa: PLC0415

    cfg = dict(cfg)
    env = {**{str(k): str(v) for k, v in ((run_config or {}).get("env") or {}).items()}, **_env_file(cfg)}
    merged = _effective_spec(cfg) if spec is None else _laid_over(cfg, spec)
    return Sources(run_config=run_config, argv=argv, spec=merged or None, pool_config=cfg,
                   pool_elastic=dict(cfg.get("elastic") or {}),
                   pool_dispense=dict((cfg.get("sampler") or {}).get("dispense") or {}), env=env)


def launch_budget_inherited_only(section: Any) -> bool:
    """``breakers.launch.budget.usd`` is not written, so ``launch.budget.usd`` takes
    ``run.budget.usd`` by inheritance alone. ``yeto run up`` does not pass that value to its
    ``yeto launch``: the launcher's cost guard prices the run's shared resource ledger (the
    pools' leases included) its own way, so the same limit there could stop the run before
    run up's ``run.budget`` does. ``run.budget`` enforces it."""
    if not isinstance(section, Mapping):
        return False
    budget = (section.get("launch") or {}).get("budget") if isinstance(section.get("launch"), Mapping) else None
    return not (isinstance(budget, Mapping) and "usd" in budget)


def mark_run_enforced(data: dict[str, Any], run_config: Mapping[str, Any] | None) -> dict[str, Any]:
    """A run config's view (``show --config``, run up's ``breakers.resolved.json``):
    ``launch.budget`` on with its ``usd`` inherited from ``run.budget.usd`` alone gets
    ``enforced_by: "run"`` (run up's ``run.budget`` stops the run at it; the launch is not
    given it, :func:`launch_budget_inherited_only`)."""
    section = (run_config or {}).get("breakers")
    if not launch_budget_inherited_only(section):
        return data
    for entry in data.get("entries") or ():
        if (entry.get("name") == "launch.budget" and entry.get("enabled")  # off (cost_guard off): no claim to mark
                and (entry.get("sources") or {}).get("usd", {}).get("kind") == "inherit"):
            entry["enforced_by"] = "run"
    return data


@dataclass(frozen=True)
class RunView:
    """The sources of a run config's one-page breaker view and the pool whose knobs it reads."""
    sources: Sources
    pool: str | None  # the pool (and its samplers) the view reads; None: the run config alone
    pools: tuple[str, ...]  # every pool of the run


def run_view_sources(run_config: Mapping[str, Any], *, pool: str | None = None,
                     spec: Mapping[str, Any] | None = None) -> RunView:
    """The sources of a run config's one-page view (``yeto breaker show --config``, run up's
    ``breakers.resolved.json``). The run config holds no pool, sampler or harness knob: each
    pool resolves its own (:func:`pool_sources`), so the view reads one pool's, the run's only
    pool or ``pool`` by name. With several pools and no name it is the run config alone (those
    rows then show defaults), as is a run without pools. ``spec`` (``show --spec``) replaces
    the pool's spec file; the pool's knobs and env file are still read. Raises ValueError on a
    pool file that does not load and on a ``pool`` the run does not have."""
    from yeto.breakers.resolve import Sources  # noqa: PLC0415

    pools = run_pools(run_config, strict=True)
    names = tuple(name for name, _, _ in pools)
    if pool is not None and names and pool not in names:
        raise ValueError(f"pool {pool!r}: the run config has no such pool (its pools: {', '.join(names)})")
    chosen = [p for p in pools if p[0] == pool] if pool is not None else pools if len(pools) == 1 else []
    if not chosen:
        return RunView(Sources.from_run_config(run_config, spec=spec), pool, names)
    name, cfg, _ = chosen[0]
    argv = _argv(run_config, None)
    sources = pool_sources(cfg, run_config=run_config, argv=argv, spec=spec)
    from yeto.rl import up_consistency  # noqa: PLC0415

    if argv is not None and up_consistency.is_auto_backlog(sources.spec):
        auto = up_consistency.auto_backlog(argv, [(c, path) for _n, c, path in pools])
        if auto is not None:  # run up hands this cap to the pool's samplers (sample_roles.AUTO_BACKLOG_ENV)
            sources.spec = {**sources.spec, "max_backlog_groups": int(auto["groups"])}
    return RunView(sources, name, names)


def pool_facts(name: str, cfg: Mapping[str, Any], data_path: str = "r2", *,
               run_config: Mapping[str, Any] | None = None, argv: list[str] | None = None) -> PoolFacts:
    """One pool's facts, with its own resolve (``pool=name``, :func:`pool_sources`)."""
    from yeto.breakers.resolve import resolve  # noqa: PLC0415

    sources = pool_sources(cfg, run_config=run_config, argv=argv)
    resolved = resolve(sources, pool=name, strict=False)
    return _facts(name, resolved, dict(sources.spec or {}), dict(cfg), data_path, dispense=_pool_dispense(cfg))


def _laid_over(cfg: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, Any]:
    """``spec`` with the pool's ``sampler`` knobs laid over it, as the pool renders it
    (``spec`` as it is when the knobs do not apply: the pool dry-run names why)."""
    from yeto.launch.nc import cli as nc_cli  # noqa: PLC0415
    from yeto.launch.nc import tasks  # noqa: PLC0415

    try:
        return tasks.apply_sampler_overrides(spec, **nc_cli._overrides(dict(cfg.get("sampler") or {})))
    except (ValueError, TypeError):
        return dict(spec)


def _effective_spec(cfg: Mapping[str, Any]) -> dict[str, Any]:
    from yeto.rl.up_consistency import effective_spec  # noqa: PLC0415

    return effective_spec(dict(cfg)) or {}


def _facts(name: str, resolved: ResolvedBreakers, spec: dict[str, Any], cfg: Mapping[str, Any], data_path: str,
           *, dispense: Mapping[str, Any] | None = None) -> PoolFacts:
    agentic = spec.get("agentic") if isinstance(spec.get("agentic"), Mapping) else None
    defaults = (agentic or {}).get("task_defaults")
    defaults = defaults if isinstance(defaults, Mapping) else {}
    timeout = defaults.get("agent_timeout_s")
    return PoolFacts(
        name=name, resolved=resolved, data_path=data_path,
        sampling_n=positive_int((spec.get("sampling") or {}).get("n")),
        agentic=agentic is not None,
        agent_timeout_s=float(timeout) if isinstance(timeout, (int, float)) and not isinstance(timeout, bool) else None,
        provider=None if defaults.get("provider") is None else str(defaults["provider"]),
        drain=agentic is not None and _drains(spec),
        compaction=agentic is not None and agentic.get("compaction") is not None,
        backlog=_backlog(cfg, spec) if cfg else None,
        dispense=_dispense(cfg, spec, dispense) if dispense is not None else None)


def _drains(spec: Mapping[str, Any]) -> bool:
    """Whether an agentic sampler switches weights by a drain (the drain cut applies)."""
    from yeto.rl.engine import agentic_round as ar  # noqa: PLC0415

    switch = str((spec.get("agentic") or {}).get("weight_switch", "auto"))
    return not (switch == "live" or (switch == "auto" and ar.spec_live_engine(spec)))


def _env_file(cfg: Mapping[str, Any]) -> dict[str, str]:
    """The pool's ``agentic_env_file`` as its samplers' harness sees it (docker
    ``--env-file`` semantics: :func:`yeto.launch.agentic_setup.read_docker_env_file`);
    {} when it cannot be read, which :func:`env_file_problems` names."""
    path = cfg.get("agentic_env_file")
    if not path:
        return {}
    from yeto.launch import agentic_setup  # noqa: PLC0415

    try:
        return agentic_setup.read_docker_env_file(os.path.expanduser(str(path)))
    except (OSError, ValueError):
        return {}


def env_file_problems(cfg: Mapping[str, Any]) -> list[str]:
    """Why the pool's ``agentic_env_file`` would not give its samplers what the breaker
    views show: it cannot be read (or docker refuses a line of it), or a variable a
    breaker reads has a quoted value (docker keeps the quotes, so the harness reads
    ``"6"`` and refuses it). [] when there is no env file or it is fine. ``yeto run up``
    and the pool dry run refuse these; ``yeto breaker show`` / ``check`` print them."""
    path = cfg.get("agentic_env_file")
    if not path:
        return []
    from yeto.breakers.registry import default_registry  # noqa: PLC0415
    from yeto.launch import agentic_setup  # noqa: PLC0415

    try:
        env = agentic_setup.read_docker_env_file(os.path.expanduser(str(path)))
    except OSError as error:
        return [f"agentic_env_file {path} unreadable ({error.strerror or error}): the samplers cannot start "
                "with it, and the harness values shown are defaults"]
    except ValueError as error:
        return [f"agentic_env_file {error}: the samplers cannot start with it, and the harness values shown "
                "are defaults"]
    names = {a.path for b in default_registry() for a in b.aliases if a.surface == "env"}
    return [f"agentic_env_file {path}: {name}={value} is quoted: docker --env-file keeps the quotes, so the "
            f"sampler reads {value} (quotes included) and refuses it; write {name}={value[1:-1]}"
            for name, value in env.items() if name in names and agentic_setup.quoted(value)]


def _pool_dispense(cfg: Mapping[str, Any]) -> dict[str, Any] | None:
    from yeto.launch.nc.cli import pool_dispense  # noqa: PLC0415

    try:
        return pool_dispense(dict(cfg))
    except (TypeError, ValueError):
        return None


def _backlog(cfg: Mapping[str, Any], spec: Mapping[str, Any]) -> Backlog | None:
    from yeto.rl.up_consistency import is_auto_backlog, pool_backlog  # noqa: PLC0415

    target = positive_int(cfg.get("target"))
    if is_auto_backlog(dict(spec)) and not cfg.get("backlog"):  # a pool backlog section wins over the spec
        return None if target is None else Backlog(AUTO, target, target, 0)

    shape = pool_backlog(dict(cfg))
    return None if shape is None else Backlog(shape["cap"], shape["target"], shape["most"], shape["over"])


def _dispense(cfg: Mapping[str, Any], spec: Mapping[str, Any], dispense: Mapping[str, Any]) -> Dispense:
    passes = positive_int(dispense.get("passes", 1)) or 1
    return Dispense(namespace=str(dispense.get("namespace") or ""), passes=passes,
                    rows=training_rows(cfg, spec, dispense))


def training_rows(cfg: Mapping[str, Any], spec: Mapping[str, Any], dispense: Mapping[str, Any]) -> int | None:
    """Rows of the pool's local ``prompts_file`` minus the hold-out tasks (the rows the
    dispenser plans slots for); None without a readable local file or hold-out."""
    path = cfg.get("prompts_file")
    if not path:
        return None
    try:
        lines = [line for line in Path(os.path.expanduser(str(path))).read_text().splitlines() if line.strip()]
    except (OSError, UnicodeDecodeError):
        return None
    holdout = _holdout(dispense)
    if holdout is None:
        return None
    if not holdout:
        return len(lines)
    from yeto.rl.engine.store_dispense import row_task_id  # noqa: PLC0415

    task_key = (spec.get("prompts") or {}).get("task_key", "metadata")
    kept = 0
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            return None
        kept += not (isinstance(row, Mapping) and row_task_id(row, task_key) in holdout)
    return kept


def _holdout(dispense: Mapping[str, Any]) -> set[str] | None:
    """The hold-out task ids ({} without one; None: a hold-out file not readable here)."""
    if dispense.get("holdout_ids") is not None:
        return {str(i) for i in dispense["holdout_ids"]}
    if not dispense.get("holdout"):
        return set()
    from yeto.rl.engine.store_dispense import load_holdout  # noqa: PLC0415

    try:
        return load_holdout(os.path.expanduser(str(dispense["holdout"])))[0]
    except (OSError, ValueError, TypeError):
        return None
