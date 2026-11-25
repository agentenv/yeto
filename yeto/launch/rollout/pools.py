"""The rollout pools of a run config (``yeto run up`` / ``yeto rollout``).

A run declares its sampler pools either the old way, one pool config file::

    "pool": {"config": "<pool.json>", "interval_s": 30}

or as a list, one entry per pool (any number of providers)::

    "samplers": [
      {"name": "nc", "provider": "nc", "config": "<pool.json>", "target": 2},
      {"name": "modal", "provider": {"kind": "modal", "image": "...", "secret": "...", "bucket": "..."},
       "data_path": "r2", "target": 1, "slots": [1, 2], "shard_base": "k31-modal",
       "spec": "<sampler-spec.json>", "run_prefix": "<run prefix>", "r2_env_file": "<store env>"}
    ]

An entry names a pool config file (``config``) and/or carries pool config keys
inline (``target``, ``spec``, ``slots``, ``shard_base``, ``elastic``, ...: the
``pool.json`` format of docs/NC_POOL.md, unchanged); inline keys override the
file's. ``provider`` is a registered kind (``nc`` | ``daytona`` | ``sky`` | ``modal`` |
``ssh``, :func:`provider_kinds`: the provider registry of :mod:`yeto.launch.pool.provider`)
or a full provider section. Entry-only keys: ``name`` (default: the provider kind),
``interval_s`` (the pool loop's tick, default 30), ``usd_per_h`` (one node's price
for the run's spend estimate; default ``budget.nc_usd_per_h``), ``data_path``
(``r2``, the default: buffer shards and policy releases in the run store; or
``syncer``: weights from the syncer broadcast and samples through the relay, the
sampler spec says how). An entry that changes nothing of its file uses the file
as it is; any other is written to ``<run state dir>/pools/<name>.json`` and the
pool commands run on that file. A run config with a ``breakers:`` section has the
pool-layer values written into each pool's ``elastic`` (and ``sampler.dispense``, the
lease keys) the same way (:func:`with_breakers`).

Every pool keeps its own state (``pool-<shard_base>.json``), so ``shard_base`` is
unique across the pools of a run; all pools of a run use one data path (a trainer
fed by both R2 buffer shards and the relay is not verified).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

DATA_PATHS = ("r2", "syncer")
ENTRY_KEYS = frozenset({"name", "provider", "config", "interval_s", "usd_per_h", "data_path", "required"})


def provider_kinds() -> tuple[str, ...]:
    """The provider kinds a pool entry may name: every kind of the provider registry
    (:func:`yeto.launch.pool.provider.kinds`; the registry imports no cloud code)."""
    from yeto.launch.pool import provider as registry  # noqa: PLC0415

    return tuple(registry.kinds())


PROVIDER_KINDS = provider_kinds()  # the kinds registered at import (the --provider choices)


@dataclass(frozen=True)
class PoolRef:
    name: str
    config: str  # the pool config file the pool commands run on
    interval_s: float = 30.0
    usd_per_h: float | None = None
    data_path: str = "r2"
    spec_carried: tuple[str, ...] = ()  # breakers: settings the pool's spec carries to the sampler (with_breakers)
    # ``required: false`` (a samplers entry): a pool whose preflight fails is left out (degraded) and
    # the run starts with the others; true (the default, and the one pool of ``pool``): it refuses the run
    required: bool = True

    def load(self) -> dict[str, Any]:
        return json.loads(Path(self.config).read_text())

    def kind(self) -> str:
        from yeto.launch.nc.provider import provider_kind  # noqa: PLC0415

        try:
            return provider_kind(self.load())
        except (OSError, ValueError):
            return "?"


def _provider_section(value: Any, base: dict[str, Any] | None) -> dict[str, Any]:
    """The pool's provider section: a full section as it is; a kind keeps the pool
    file's section (``base``: the file's config, None without a file) and must match it."""
    if isinstance(value, dict):
        return dict(value)
    kind = str(value)
    kinds = provider_kinds()
    if kind not in kinds:
        raise ValueError(f"samplers: provider {kind!r} is not one of {', '.join(kinds)} "
                         "(or a provider section)")
    if base is not None:
        from yeto.launch.nc.provider import provider_kind  # noqa: PLC0415

        have = provider_kind(base)
        if have != kind:
            raise ValueError(f"samplers: provider {kind!r} but the pool config's provider is {have!r}")
        return dict(base.get("provider") or {})
    return {} if kind == "nc" else {"kind": kind}


def resolve(cfg: dict[str, Any], state_dir: Path) -> list[PoolRef]:
    """The pools of a run config, in order. Raises ValueError on a bad declaration."""
    if "samplers" in cfg and "pool" in cfg:
        raise ValueError("run config: set 'samplers' (a list of pools) or 'pool' (one pool config), not both")
    if "pool" in cfg:
        pool = cfg["pool"] or {}
        if "config" not in pool:
            raise ValueError("run config: pool.config (the pool config file) is required")
        ref = PoolRef("pool", os.path.expanduser(str(pool["config"])), _number(pool, "pool", "interval_s", 30))
        return with_boat_shares(cfg, [with_breakers(cfg, ref, state_dir)], state_dir)
    entries = cfg.get("samplers")
    if not isinstance(entries, list) or not entries:
        raise ValueError("run config: 'samplers' must be a non-empty list of pools (or use 'pool')")
    refs: list[PoolRef] = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"samplers[{i}] must be an object")
        refs.append(_entry(cfg, entry, i, state_dir))
    names = [r.name for r in refs]
    if len(set(names)) != len(names):
        raise ValueError(f"samplers: pool names repeat {names}; set 'name' on each")
    refs = [with_breakers(cfg, ref, state_dir) for ref in refs]
    refs = with_boat_shares(cfg, refs, state_dir)
    paths = {r.data_path for r in refs}
    if len(paths) > 1:
        raise ValueError(f"samplers: one data path per run (got {sorted(paths)}): a trainer that reads R2 buffer "
                         "shards and the syncer relay at once is not verified")
    bases: dict[str, str] = {}
    for ref in refs:
        try:
            base = str(ref.load().get("shard_base"))
        except (OSError, ValueError):
            continue
        if base in bases:
            raise ValueError(f"samplers: pools {bases[base]!r} and {ref.name!r} share shard_base {base!r} "
                             "(each pool keeps its state and shards under its own shard_base)")
        bases[base] = ref.name
    return refs


def entry_doc(entry: dict[str, Any], where: str) -> tuple[dict[str, Any], dict[str, Any], str | None]:
    """``(doc, base, path)`` of one pool entry: the pool config the entry declares (its
    file's keys, the provider section, the inline keys over them), the file's own keys
    and the file's path (None: no file). ``where`` names the entry in an error."""
    inline = {k: v for k, v in entry.items() if k not in ENTRY_KEYS}
    path = os.path.expanduser(str(entry["config"])) if entry.get("config") else None
    base = json.loads(Path(path).read_text()) if path else {}
    doc = dict(base)
    if "provider" in entry:
        doc["provider"] = _provider_section(entry["provider"], base if path else None)
        if doc["provider"] == {} and "provider" not in base:
            doc.pop("provider")
    doc.update(inline)
    if not doc:
        raise ValueError(f"{where}: no pool config (set 'config' or the pool keys inline)")
    return doc, base, path


def _entry(cfg: dict[str, Any], entry: dict[str, Any], i: int, state_dir: Path) -> PoolRef:
    doc, base, path = entry_doc(entry, f"samplers[{i}]")
    from yeto.launch.nc.provider import provider_kind  # noqa: PLC0415

    kind = provider_kind(doc)
    name = str(entry.get("name") or kind)
    data_path = str(entry.get("data_path", doc.get("data_path", "r2")))
    if data_path not in DATA_PATHS:
        raise ValueError(f"samplers[{i}]: data_path {data_path!r} is not one of {DATA_PATHS}")
    if data_path != "r2":
        doc["data_path"] = data_path
    if data_path == "syncer":
        if "syncer" not in doc and cfg.get("syncer"):
            doc["syncer"] = dict(cfg["syncer"])  # the run's syncer (address, key_env) for every pool
        if not (doc.get("syncer") or {}).get("address"):
            raise ValueError(f"samplers[{i}]: data_path syncer needs syncer.address (on the pool or the run)")
    if doc == base and path:
        target = path
    else:
        out = state_dir / "pools" / f"{name}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=1, sort_keys=True))
        target = str(out)
    interval = _number(entry, f"samplers[{i}]", "interval_s", 30)
    required = entry.get("required", True)
    if not isinstance(required, bool):
        raise ValueError(f"samplers[{i}]: required must be true or false")
    return PoolRef(name, target, interval, _number(entry, f"samplers[{i}]", "usd_per_h", None), data_path,
                   required=required)


POOL_LAYERS = ("pool", "sampler")  # the layers whose knobs a pool config holds
SPEC_LAYERS = ("sampler", "harness")  # the layers a sampler resolves from its spec (sampler_breakers)


def with_breakers(cfg: dict[str, Any], ref: PoolRef, state_dir: Path) -> PoolRef:
    """The pool with the run's ``breakers:`` section written into its native knobs (``elastic``,
    ``sampler.dispense``, the lease keys; ``breakers.pools.<name>`` before ``breakers.pool``):
    a copy at ``<run state dir>/pools/<name>.json`` when that changes anything, else the ref as
    it is (always without a section). A section that disagrees with the pool config, its
    spec or its env files is refused.

    The section's ``sampler`` and ``harness`` entries run in the sampler process, which
    reads only its spec: they go into the pool's ``sampler.breakers`` knob, which every
    rendered spec carries as its ``breakers`` key (``tasks.apply_sampler_overrides``) and
    the sampler resolves (``sampler_breakers``). ``ref.spec_carried`` names every setting
    that goes that way, so none is dropped without a word."""
    section = cfg.get("breakers")
    if not section:
        return ref
    from yeto import breakers  # noqa: PLC0415
    from yeto.breakers.rules_facts import pool_sources  # noqa: PLC0415

    doc = ref.load()
    sources = pool_sources(doc, run_config=cfg)  # the spec as the pool renders it (read only)
    try:
        resolved = breakers.resolve(sources, pool=ref.name, layers=(*POOL_LAYERS, "harness"))
        out = breakers.materialize(resolved, sources, surfaces=("pool_config", "pool_elastic", "pool_dispense"))
    except breakers.BreakerConfigError as error:
        raise ValueError(f"run config: pool {ref.name!r}: {error}") from None
    new = dict(out.pool_config if out.pool_config is not None else doc)
    if out.pool_elastic is not None and out.pool_elastic != sources.pool_elastic:
        new["elastic"] = out.pool_elastic
    if out.pool_dispense is not None and out.pool_dispense != sources.pool_dispense:
        new["sampler"] = dict(doc.get("sampler") or {}, dispense=out.pool_dispense)
    carried = spec_carried(resolved)
    if carried:
        new["sampler"] = dict(new.get("sampler") or {}, breakers=_spec_section(new, resolved, carried, ref.name))
    if new == doc:
        return ref
    path = state_dir / "pools" / f"{ref.name}.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(new, indent=1, sort_keys=True))
    except OSError as error:
        raise ValueError(f"run config: pool {ref.name!r}: cannot write {path} ({error})") from None
    return replace(ref, config=str(path), spec_carried=carried)


# harness.sandbox_caps.<param> of a boat sampler process -> the sandbox.boat limit it shares
BOAT_SHARES = {"boat_max_active_n": "max_active_n", "boat_starts_per_minute_n": "starts_per_minute_n",
               "boat_starts_per_hour_n": "starts_per_hour_n"}
# their env aliases (yeto.breakers.layers.harness), as a pool may set them in a Modal provider's env
BOAT_ENV = {"boat_max_active_n": "YETO_BOAT_MAX_ACTIVE", "boat_starts_per_minute_n": "YETO_BOAT_STARTS_PER_MINUTE",
            "boat_starts_per_hour_n": "YETO_BOAT_STARTS_PER_HOUR"}


def boat_pool(cfg: dict[str, Any], doc: dict[str, Any], name: str) -> dict[str, Any] | None:
    """A pool whose samplers run boat sandboxes: ``{"processes": samplers at the elastic
    maximum, "max_in_flight", "explicit": {param: value the pool sets itself}}``; None
    for any other pool (no agentic spec, another provider)."""
    from yeto import breakers  # noqa: PLC0415
    from yeto.breakers.rules_facts import pool_sources  # noqa: PLC0415

    sources = pool_sources(doc, run_config=cfg)
    agentic = (sources.spec or {}).get("agentic")
    if not isinstance(agentic, dict):
        return None
    defaults = agentic.get("task_defaults") if isinstance(agentic.get("task_defaults"), dict) else {}
    provider = defaults.get("provider") or str((sources.env or {}).get("HARBOR_ENV_TYPE") or "").strip().lower()
    if provider != "boat":
        return None
    try:
        resolved = breakers.resolve(sources, pool=name, layers=("harness",))
    except breakers.BreakerConfigError as error:
        raise ValueError(f"run config: pool {name!r}: {error}") from None
    entry = resolved.entry("harness.sandbox_caps")
    explicit = {p: entry.params[p] for p in BOAT_SHARES if entry.sources[p].kind != "default"}
    provider_env = (doc.get("provider") or {}).get("env") if isinstance(doc.get("provider"), dict) else None
    for param, env_name in BOAT_ENV.items():  # a Modal pool's provider.env reaches the sampler too
        value = (provider_env or {}).get(env_name)
        if value is not None and str(value).strip().isdigit():
            explicit.setdefault(param, int(str(value).strip()))
    target = doc.get("target") if isinstance(doc.get("target"), int) else 1
    elastic = doc.get("elastic") if isinstance(doc.get("elastic"), dict) else {}
    nodes = elastic.get("max_nodes") if isinstance(elastic.get("max_nodes"), int) else target
    mif = agentic.get("max_in_flight")
    return {"processes": max(1, int(nodes), int(target)), "explicit": explicit,
            "max_in_flight": int(mif) if isinstance(mif, int) and not isinstance(mif, bool) else 1}


def boat_shares(account: dict[str, Any], pools: dict[str, dict[str, Any]]) -> dict[str, dict[str, int]]:
    """``{pool: {param: per-process value}}``: the boat account's limits (``account``,
    :func:`yeto.rl.fleet_sandbox.boat_account`) split over every boat sampler process of
    the run (``pools``: :func:`boat_pool` of each). A value a pool sets itself is kept and
    counted; the rest of the limit goes to the other pools' processes: the active
    sandboxes in proportion to their ``max_in_flight`` (each gets at least that, or the
    run is refused), the start rates evenly (at least one each, or refused). Raises
    ValueError naming the pools."""
    out: dict[str, dict[str, int]] = {name: {} for name in pools}
    for param, limit_key in BOAT_SHARES.items():
        limit = account.get(limit_key)
        if limit is None:
            continue  # not held (starts_per_hour_n null)
        fixed = {n: p for n, p in pools.items() if param in p["explicit"]}
        free = {n: p for n, p in pools.items() if param not in p["explicit"]}
        used = sum(int(p["explicit"][param]) * p["processes"] for p in fixed.values())
        left = int(limit) - used
        listed = ", ".join(f"{n} ({p['processes']} sampler(s) x max_in_flight {p['max_in_flight']})"
                           for n, p in pools.items())
        if param == "boat_max_active_n":
            need = sum(p["processes"] * p["max_in_flight"] for p in free.values())
            if left < need or left < 0:
                raise ValueError(f"run config: the boat account holds {limit} active sandboxes (sandbox.boat."
                                 f"{limit_key}) and the run's boat samplers need {need + used} at their elastic "
                                 f"maximum ({listed}): lower agentic.max_in_flight, the pools' targets / "
                                 "elastic.max_nodes, or raise sandbox.boat.max_active_n to the account's plan")
            for n, p in free.items():
                out[n][param] = max(p["max_in_flight"], left * p["max_in_flight"] // max(1, need))
        else:
            count = sum(p["processes"] for p in free.values())
            if not count:
                continue
            share = left // count
            if share < 1:
                raise ValueError(f"run config: the boat account starts {limit} sandboxes "
                                 f"{'a minute' if 'minute' in param else 'an hour'} (sandbox.boat.{limit_key}) and the "
                                 f"run has {count + len(fixed)} boat sampler process(es) ({listed}): each needs at "
                                 "least one start; run fewer boat samplers or raise the limit to the account's plan")
            for n in free:
                out[n][param] = share
    return out


def with_boat_shares(cfg: dict[str, Any], refs: list[PoolRef], state_dir: Path) -> list[PoolRef]:
    """The pools with each boat sampler process's share of the boat account (:func:`boat_shares`)
    in their ``sampler.breakers`` knob (``harness.sandbox_caps.boat_*``: the samplers write
    them to ``YETO_BOAT_*``, ``sampler_breakers.apply_harness_env``): each process holds its
    quota alone, so their sum is the account's. Unchanged when the run has at most one
    boat sampler process (the plan's limits are its own) or none."""
    from yeto.launch.nc import tasks  # noqa: PLC0415
    from yeto.rl.fleet_sandbox import boat_account  # noqa: PLC0415

    found = {}
    for ref in refs:
        try:
            doc = ref.load()
        except (OSError, ValueError):
            continue  # the pool dry-run reports an unreadable pool config
        row = boat_pool(cfg, doc, ref.name)
        if row is not None:
            found[ref.name] = row
    if sum(p["processes"] for p in found.values()) <= 1:
        return refs
    shares = boat_shares(boat_account(cfg.get("sandbox")), found)
    out = []
    for ref in refs:
        share = shares.get(ref.name)
        if not share:
            out.append(ref)
            continue
        doc = ref.load()
        sampler = dict(doc.get("sampler") or {})
        try:
            sampler["breakers"] = tasks.merge_breakers(sampler.get("breakers") or {},
                                                       {"harness": {"sandbox_caps": share}})
        except ValueError as error:
            raise ValueError(f"run config: pool {ref.name!r}: boat share: {error}") from None
        path = state_dir / "pools" / f"{ref.name}.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(dict(doc, sampler=sampler), indent=1, sort_keys=True))
        except OSError as error:
            raise ValueError(f"run config: pool {ref.name!r}: cannot write {path} ({error})") from None
        carried = tuple(dict.fromkeys(ref.spec_carried + tuple(f"harness.sandbox_caps.{p}" for p in share)))
        out.append(replace(ref, config=str(path), spec_carried=carried))
    return out


def spec_carried(resolved: Any) -> tuple[str, ...]:
    """``<entry>.<key>`` of every sampler / harness setting the ``breakers:`` section sets
    (directly, by inheritance or by a switch): the settings only the sampler's spec can carry."""
    return tuple(f"{entry.name}.{key}" for layer in SPEC_LAYERS for entry in resolved.entries(layer)
                 for key, source in entry.sources.items() if source.kind in ("breakers", "inherit", "switch"))


def _spec_section(doc: dict[str, Any], resolved: Any, carried: tuple[str, ...], name: str) -> dict[str, Any]:
    """The pool's ``sampler.breakers`` knob with the ``carried`` settings at their resolved
    values (inherited and switched ones included). The pool has to have a spec that loads; the
    spec's own ``breakers`` key and an earlier ``sampler.breakers`` may repeat a value, never
    contradict one; and the spec it renders has to resolve every carried setting to the same
    value in the sampler (``sampler_breakers.spec_breakers``). Raises ValueError."""
    from yeto.launch.nc import cli as nc_cli  # noqa: PLC0415
    from yeto.launch.nc import tasks  # noqa: PLC0415
    from yeto.rl.engine.sampler_breakers import spec_breakers  # noqa: PLC0415

    carry: dict[str, Any] = {}
    for path in carried:
        layer, entry, key = path.split(".")
        carry.setdefault(layer, {}).setdefault(entry, {})[key] = resolved.get(path)
    where = f"run config: pool {name!r}: breakers.{'/'.join(carry)}"
    if not doc.get("spec"):
        raise ValueError(f"{where} run in the sampler, which reads them from its spec, but the pool has no spec")
    source = Path(os.path.expanduser(str(doc["spec"])))
    try:
        text = source.read_text().replace("__RUN_PREFIX__", str(doc.get("run_prefix") or "").strip("/"))
        spec = json.loads(text)
        if not isinstance(spec, dict):
            raise ValueError("not a JSON object")
    except (OSError, ValueError) as error:
        raise ValueError(f"{where}: the pool's spec {source} does not load ({error})") from None
    try:
        knob = tasks.merge_breakers((doc.get("sampler") or {}).get("breakers") or {}, carry)
        sampler = nc_cli._overrides(dict(doc.get("sampler") or {}, breakers=knob))
        seen = spec_breakers(tasks.apply_sampler_overrides(spec, **sampler))
    except ValueError as error:  # BreakerConfigError is one
        raise ValueError(f"{where}: {error}") from None
    differ = [f"{p} {seen.get(p)!r} (the run resolves {resolved.get(p)!r})" for p in carried
              if seen.get(p) != resolved.get(p)]
    if differ:
        raise ValueError(f"{where}: the sampler would resolve {', '.join(differ)}")
    return knob


def _number(entry: dict[str, Any], where: str, key: str, default: float | None) -> float | None:
    """``interval_s`` (a number > 0) or ``usd_per_h`` (a number >= 0, null: the provider's
    price) of a pool entry; a bool, a string, nan or inf is refused (a negative or nan node
    price makes the run's spend estimate fall or never compare, so budget.usd never trips)."""
    value = entry.get(key, default)
    if value is None and key == "usd_per_h":
        return None
    ok = (not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
          and (value > 0 if key == "interval_s" else value >= 0))
    if not ok:
        raise ValueError(f"run config: {where}.{key} must be a finite number "
                         f"{'> 0' if key == 'interval_s' else '>= 0'} (got {value!r})")
    return float(value)


def select(refs: list[PoolRef], *, pool: str | None = None, provider: str | None = None) -> list[PoolRef]:
    """The pools named ``pool`` and / or of provider kind ``provider``."""
    out = [r for r in refs if pool is None or r.name == pool]
    if provider is not None:
        out = [r for r in out if r.kind() == provider]
    return out
