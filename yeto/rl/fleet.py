"""An island fleet in one run config: several async GRPO islands, one syncer.

A run config with an ``islands`` section declares one trainer per island and the
sampler pools that feed it, and a ``fleet`` section with what the trainers share::

    {"run": "<run>", "store": "s3://<bucket>/runs/<run>", "budget": {...}, ...,
     "fleet": {"trainer_argv": ["--training-mode", "rl", "--tuning", "full",
                                "--rl-sync-preset", "decoupled", ...,
                                "--rl-checkpoint-store", "s3://<bucket>/runs/<run>"],
               "dispense": "fleet"},              # "fleet" (default) | "island"
     "islands": [
       {"name": "east", "trainer": {"gpu": "modal:2x8xb300", "usd_per_h": 96.0,
                                    "argv": ["--rl-release-every", "2"]},
        "samplers": [<pool entries, the format of the run config's samplers list>]},
       ...]}

``islands`` excludes ``trainer``, ``pool`` and the top-level ``samplers``. Island k
(its position in the list) is the learner id of its trainer and its store
sub-directory ``trainer<k>``: its trainer releases its policy there and reads its
samples there, and its pools write there.

:func:`compile_fleet` turns the run config into the single-trainer form the rest of
``yeto run up`` drives (``trainer`` is the one synced-group ``yeto launch`` of every
island's trainer and the syncer; ``samplers`` lists every island's pools) plus a
:class:`FleetPlan` (the island table). The compiled config carries the plan under
:data:`PLAN_KEY`. For pool ``<name>`` of island k the compile:

* renames it ``i<k>-<name>`` (a pool config can serve several islands);
* moves its store to ``<store>/trainer<k>``: its ``run_prefix`` gets the segment
  ``trainer<k>`` (the samplers write ``s3://<bucket>/runs/<run_prefix>``, and a
  spec's ``__RUN_PREFIX__`` follows);
* renames its shards: ``shard_base`` becomes ``<shard_base>-i<k>``, so the shards
  are ``<shard_base>-i<k>-<n>`` for n below the pool's ``shard_limit`` (required);
* sets the pool's sampler knobs ``island`` (k), ``until_trainers_end`` (1: the
  sampler stops after its own trainer ended) and, with ``dispense: fleet``,
  ``dispense_store`` (``<store>/fleet/dispense``: one task ledger for the fleet).

The trainer argv is ``fleet.trainer_argv`` with ``--gpu`` (every island's cluster,
in island order), ``--rl-sample-role trainer --rl-sampler-follow trainer``, one
``--rl-island-buffer-shards k:<shards>`` per island (the shards of its pools) and
the island's ``--rl-island-release-every`` / ``--rl-island-max-policy-age``. Its
``usd_per_h`` is the sum of the islands'.

Supported training forms: full parameters or partial experts on Miles, synchronized
by asynchronous full-parameter rounds (``--rl-sync-preset decoupled`` routes there);
:func:`form_problem` refuses the rest (LoRA, verl) in the preflight.

``yeto run up`` writes :meth:`FleetPlan.to_json` to ``<store>/fleet/plan.json``.

Import-light: no torch, ray or miles.
"""

from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

PLAN_VERSION = 1
PLAN_KEY = "fleet_plan"  # the compiled run config's key holding the FleetPlan
FLEET_DIR = "fleet"
PLAN_OBJECT = f"{FLEET_DIR}/plan.json"  # the plan in the run store
DISPENSE_DIR = f"{FLEET_DIR}/dispense"  # the fleet's task ledger root (dispense: fleet)
DISPENSE_MODES = ("fleet", "island")
FLEET_KEYS = frozenset({"trainer_argv", "dispense"})
ISLAND_KEYS = frozenset({"name", "trainer", "samplers"})
ISLAND_TRAINER_KEYS = frozenset({"gpu", "usd_per_h", "argv"})
# the only trainer flags an island sets for itself (each becomes --rl-island-<flag> k:N)
ISLAND_FLAGS = {"--rl-release-every": "release_every", "--rl-max-policy-age": "max_policy_age"}
# flags the compile writes; fleet.trainer_argv must not set them
GENERATED_FLAGS = ("--gpu", "--rl-sample-role", "--rl-sampler-follow", "--rl-buffer-shards")
GENERATED_PREFIX = "--rl-island-"
EXCLUSIVE_KEYS = ("trainer", "pool", "samplers")
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def substore(k: int) -> str:
    """The store sub-directory of island k (``sample_roles`` names it so)."""
    return f"trainer{int(k)}"


def substore_uri(store: str, name: str) -> str:
    """``yeto.rl.engine.sample_roles.substore_uri`` (one more path segment)."""
    from yeto.rl.engine.sample_roles import substore_uri as _substore_uri  # noqa: PLC0415

    return _substore_uri(store, name)


@dataclass(frozen=True)
class FleetPool:
    name: str  # the compiled pool name, i<k>-<source_name>
    source_name: str  # the pool's name in the island's samplers list
    provider: str  # the provider kind
    shard_base: str  # <source shard_base>-i<k>
    shard_limit: int
    run_prefix: str  # <source run_prefix>/trainer<k>
    engine: str | None = None  # vllm / sglang (the spec's engine adapter), None: unknown
    rollout_quant: Any = None  # the spec's rollout_quant tier, None: none / unknown
    target: int | None = None

    def shards(self) -> list[str]:
        return [f"{self.shard_base}-{n}" for n in range(self.shard_limit)]

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "source_name": self.source_name, "provider": self.provider,
                "shard_base": self.shard_base, "shard_limit": self.shard_limit, "run_prefix": self.run_prefix,
                "target": self.target, "engine": self.engine, "rollout_quant": self.rollout_quant}


@dataclass(frozen=True)
class FleetIsland:
    index: int
    name: str
    gpu: str
    usd_per_h: float | None
    store: str  # <store>/trainer<k>
    release_every: int | None = None
    max_policy_age: int | None = None
    pools: tuple[FleetPool, ...] = ()

    @property
    def learner(self) -> int:
        return self.index  # no sampler island in the fleet's launch: island = learner id

    @property
    def substore(self) -> str:
        return substore(self.index)

    def buffer_shards(self) -> list[str]:
        return [s for pool in self.pools for s in pool.shards()]

    def to_json(self) -> dict[str, Any]:
        return {"index": self.index, "name": self.name, "learner": self.learner, "substore": self.substore,
                "store": self.store, "gpu": self.gpu, "usd_per_h": self.usd_per_h,
                "release_every": self.release_every, "max_policy_age": self.max_policy_age,
                "buffer_shards": self.buffer_shards(), "pools": [p.to_json() for p in self.pools]}


@dataclass(frozen=True)
class FleetPlan:
    run: str
    store: str
    dispense: str
    islands: tuple[FleetIsland, ...]
    dispense_store: str | None = None  # the fleet's ledger root (dispense: fleet), None: per island
    sync: dict[str, Any] = field(default_factory=dict)  # the trainers' sync route (training_form)
    version: int = PLAN_VERSION

    def to_json(self) -> dict[str, Any]:
        return {"version": self.version, "run": self.run, "store": self.store, "dispense": self.dispense,
                "dispense_store": self.dispense_store, "sync": dict(self.sync),
                "islands": [i.to_json() for i in self.islands]}

    def dumps(self) -> str:
        return json.dumps(self.to_json(), indent=1, sort_keys=True)

    def island_of_pool(self, pool: str) -> FleetIsland | None:
        """The island whose pools include the compiled pool ``pool`` (None: none)."""
        return next((i for i in self.islands if any(p.name == pool for p in i.pools)), None)

    def lines(self) -> list[str]:
        """The plan as the preflight prints it: one line per island and one per pool."""
        sync = self.sync or {}
        route = (f"{sync.get('requested')} -> {sync.get('preset')} ({sync.get('rounds')} rounds, "
                 f"{sync.get('form')})" if sync.get("preset") else "unknown (the trainer argv does not parse)")
        out = [f"fleet: {len(self.islands)} island(s), one synced-group trainer launch and its syncer; "
               f"sync {route}; dispense {self.dispense}"
               + (f" (ledger {self.dispense_store})" if self.dispense_store else " (one ledger per island)")
               + f"; plan {substore_uri(self.store, PLAN_OBJECT)}"]
        for i in self.islands:
            price = "?" if i.usd_per_h is None else f"${i.usd_per_h:g}/h"
            extra = "".join(f", {flag} {value}" for flag, value in (("release_every", i.release_every),
                                                                    ("max_policy_age", i.max_policy_age))
                            if value is not None)
            out.append(f"fleet: island {i.index} {i.name}: learner {i.learner}, store {i.substore}/ ({i.store}), "
                       f"trainer {i.gpu} at {price}{extra}, {len(i.buffer_shards())} buffer shard(s), "
                       f"pools {', '.join(p.name for p in i.pools)}")
            for p in i.pools:
                out.append(f"fleet:   pool {p.name} (from {p.source_name}): {p.provider}"
                           f"{', ' + p.engine if p.engine else ''}"
                           f"{', rollout_quant ' + str(p.rollout_quant) if p.rollout_quant is not None else ''}, "
                           f"shards {p.shard_base}-0..{p.shard_limit - 1}, run_prefix {p.run_prefix}")
        return out

    @classmethod
    def from_json(cls, doc: dict[str, Any]) -> FleetPlan:
        islands = tuple(
            FleetIsland(index=int(i["index"]), name=str(i["name"]), gpu=str(i["gpu"]), usd_per_h=i.get("usd_per_h"),
                        store=str(i["store"]), release_every=i.get("release_every"),
                        max_policy_age=i.get("max_policy_age"),
                        pools=tuple(FleetPool(name=str(p["name"]), source_name=str(p["source_name"]),
                                              provider=str(p["provider"]), shard_base=str(p["shard_base"]),
                                              shard_limit=int(p["shard_limit"]), run_prefix=str(p["run_prefix"]),
                                              engine=p.get("engine"), rollout_quant=p.get("rollout_quant"),
                                              target=p.get("target"))
                                    for p in i.get("pools") or ()))
            for i in doc.get("islands") or ())
        return cls(run=str(doc["run"]), store=str(doc["store"]), dispense=str(doc["dispense"]), islands=islands,
                   dispense_store=doc.get("dispense_store"), sync=dict(doc.get("sync") or {}),
                   version=int(doc.get("version", PLAN_VERSION)))


def plan_of(cfg: dict[str, Any]) -> FleetPlan | None:
    """The FleetPlan of a compiled run config (None: a single-trainer run)."""
    plan = cfg.get(PLAN_KEY)
    return plan if isinstance(plan, FleetPlan) else None


# ---------------------------------------------------------------------- checks

def _err(msg: str) -> ValueError:
    return ValueError(f"run config: {msg}")


def _flag(arg: str) -> str:
    return arg.split("=", 1)[0]


def _flag_values(argv: list[str], where: str) -> dict[str, int]:
    """The island trainer argv: ``--rl-release-every N`` / ``--rl-max-policy-age N``
    (or ``--flag=N``), each at most once, N a whole number >= 1."""
    out: dict[str, int] = {}
    i = 0
    while i < len(argv):
        arg = argv[i]
        flag = _flag(arg)
        if flag not in ISLAND_FLAGS:
            raise _err(f"{where}: {arg!r} is not an island trainer flag (an island sets only "
                       f"{' / '.join(ISLAND_FLAGS)}; the rest goes in fleet.trainer_argv)")
        if "=" in arg:
            raw, i = arg.split("=", 1)[1], i + 1
        elif i + 1 < len(argv):
            raw, i = argv[i + 1], i + 2
        else:
            raise _err(f"{where}: {flag} needs a value")
        if not raw.isdigit() or int(raw) < 1:
            raise _err(f"{where}: {flag} must be a whole number >= 1 (got {raw!r})")
        key = ISLAND_FLAGS[flag]
        if key in out:
            raise _err(f"{where}: {flag} repeats")
        out[key] = int(raw)
    return out


def _string_list(value: Any, where: str) -> list[str]:
    if not (isinstance(value, list) and all(isinstance(a, str) for a in value)):
        raise _err(f"{where} must be a list of strings (got {value!r})")
    return list(value)


def _norm_store(uri: str) -> str:
    from yeto.rl.up_consistency import _norm_store as norm  # noqa: PLC0415

    return norm(uri)


def check_exclusive(cfg: dict[str, Any]) -> None:
    """``islands`` excludes ``trainer``, ``pool`` and ``samplers``; ``fleet`` needs ``islands``."""
    if "islands" not in cfg:
        if "fleet" in cfg:
            raise _err("'fleet' (the trainers' shared settings) needs an 'islands' section")
        return
    both = [k for k in EXCLUSIVE_KEYS if k in cfg]
    if both:
        raise _err(f"'islands' and {' / '.join(repr(k) for k in both)} are mutually exclusive: a fleet declares "
                   "each island's trainer and sampler pools in 'islands' and the trainers' shared argv in "
                   "'fleet.trainer_argv'")


def _fleet_section(cfg: dict[str, Any]) -> tuple[list[str], str]:
    section = cfg.get("fleet")
    if not isinstance(section, dict):
        raise _err("an island fleet needs a 'fleet' object with trainer_argv (the trainers' shared yeto launch "
                   "arguments)")
    unknown = set(section) - FLEET_KEYS
    if unknown:
        raise _err(f"unknown fleet keys {sorted(unknown)} (known: {sorted(FLEET_KEYS)})")
    if "trainer_argv" not in section:
        raise _err("fleet.trainer_argv (the trainers' shared yeto launch arguments) is required")
    argv = _string_list(section["trainer_argv"], "fleet.trainer_argv")
    for arg in argv:
        flag = _flag(arg)
        if flag in GENERATED_FLAGS or flag.startswith(GENERATED_PREFIX):
            raise _err(f"fleet.trainer_argv: {flag} is written by the fleet compile (from each island's trainer "
                       "and pools); drop it")
        if flag == "--rl-single-island-no-sync":
            raise _err("fleet.trainer_argv: --rl-single-island-no-sync is a single buffer trainer; a fleet's "
                       "trainers merge through the syncer")
    dispense = section.get("dispense", "fleet")
    if dispense not in DISPENSE_MODES:
        raise _err(f"fleet.dispense must be one of {list(DISPENSE_MODES)} (got {dispense!r})")
    return argv, str(dispense)


def _argv_value(argv: list[str], flag: str) -> str | None:
    value = None
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            value = argv[i + 1]
        elif a.startswith(flag + "="):
            value = a.split("=", 1)[1]
    return value


def _island_trainer(island: dict[str, Any], where: str) -> tuple[str, float | None, dict[str, int]]:
    if "trainer" not in island:
        raise _err(f"{where}: missing 'trainer' (every island runs one trainer: gpu, usd_per_h, argv)")
    tr = island["trainer"]
    if not isinstance(tr, dict):
        raise _err(f"{where}.trainer must be an object")
    unknown = set(tr) - ISLAND_TRAINER_KEYS
    if unknown:
        raise _err(f"unknown {where}.trainer keys {sorted(unknown)} (known: {sorted(ISLAND_TRAINER_KEYS)})")
    gpu = tr.get("gpu")
    if not isinstance(gpu, str) or not gpu.strip() or "," in gpu:
        raise _err(f"{where}.trainer.gpu must be one --gpu cluster, e.g. modal:2x8xh100@<region> (got {gpu!r})")
    price = tr.get("usd_per_h")
    if price is not None and (isinstance(price, bool) or not isinstance(price, (int, float))
                              or not math.isfinite(price) or price < 0):
        raise _err(f"{where}.trainer.usd_per_h must be a number >= 0 (got {price!r})")
    flags = _flag_values(_string_list(tr.get("argv", []), f"{where}.trainer.argv"), f"{where}.trainer.argv")
    return gpu.strip(), None if price is None else float(price), flags


def _spec_facts(doc: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None, Any]:
    """``(spec, engine, rollout_quant)`` of a pool config's rendered spec (None: unknown)."""
    from yeto.rl.up_consistency import effective_spec  # noqa: PLC0415

    spec = effective_spec(doc)
    if spec is None:
        return None, None, None
    adapter = (spec.get("engine") or {}).get("adapter") if isinstance(spec.get("engine"), dict) else None
    engine = str(adapter).removesuffix("-server") if adapter else None
    quant = spec.get("rollout_quant")
    if isinstance(quant, dict):
        quant = quant.get("tier")
    return spec, engine, quant


def _pool(entry: Any, k: int, j: int, island_where: str, dispense_store: str | None
          ) -> tuple[dict[str, Any], FleetPool]:
    """The compiled pool entry of pool j of island k, and its plan row."""
    from yeto.launch.nc import cli as nc_cli  # noqa: PLC0415
    from yeto.launch.nc.provider import provider_kind  # noqa: PLC0415
    from yeto.launch.rollout import pools  # noqa: PLC0415

    where = f"{island_where}.samplers[{j}]"
    if not isinstance(entry, dict):
        raise _err(f"{where} must be an object (a pool entry)")
    try:
        doc, _base, _path = pools.entry_doc(entry, where)
    except (OSError, ValueError) as error:
        raise _err(f"{where}: {error}") from None
    source = str(entry.get("name") or provider_kind(doc))
    data_path = str(entry.get("data_path", doc.get("data_path", "r2")))
    if data_path != "r2":
        raise _err(f"{where} ({source}): data_path {data_path!r}: a fleet's pools write to the run store "
                   "(data_path r2), each under its island's trainer<k>")
    for key in ("shard_base", "run_prefix"):
        if not str(doc.get(key) or "").strip("/"):
            raise _err(f"{where} ({source}): a fleet pool needs {key} (the compile moves it to island {k}'s "
                       f"{substore(k)})")
    if doc.get("shard_limit") is None:
        raise _err(f"{where} ({source}): a fleet pool needs shard_limit (island {k}'s trainer reads exactly "
                   "the shards below it)")
    sampler = doc.get("sampler") or {}
    if not isinstance(sampler, dict):
        raise _err(f"{where} ({source}): sampler must be an object")
    for key in ("island", "until_trainers_end", "dispense_store"):
        if key in sampler:
            raise _err(f"{where} ({source}): sampler.{key} is written by the fleet compile; drop it")
    if isinstance(sampler.get("dispense"), dict) and sampler["dispense"].get("store"):
        raise _err(f"{where} ({source}): sampler.dispense.store: fleet.dispense decides where the task ledger "
                   "lives; drop it")
    base = str(doc["shard_base"])
    prefix = f"{str(doc['run_prefix']).strip('/')}/{substore(k)}"
    knobs = dict(sampler, island=k, until_trainers_end=1)
    if dispense_store:
        knobs["dispense_store"] = dispense_store
    out = dict(entry)  # its entry keys and inline pool keys, with the island's overrides
    out.update({"name": f"i{k}-{source}", "run_prefix": prefix, "shard_base": f"{base}-i{k}", "sampler": knobs})
    compiled = dict(doc, run_prefix=prefix, shard_base=f"{base}-i{k}", sampler=knobs)
    try:
        limit = nc_cli.pool_shard_limit(compiled)
        nc_cli._overrides(knobs)
    except (TypeError, ValueError) as error:
        raise _err(f"{where} ({source}): {error}") from None
    if limit is None or limit < 1:
        raise _err(f"{where} ({source}): shard_limit must be a whole number >= 1 (got {doc.get('shard_limit')!r})")
    _spec, engine, quant = _spec_facts(compiled)
    target = doc.get("target")
    row = FleetPool(name=out["name"], source_name=source, provider=provider_kind(doc), shard_base=f"{base}-i{k}",
                    shard_limit=int(limit), run_prefix=prefix, engine=engine, rollout_quant=quant,
                    target=target if isinstance(target, int) and not isinstance(target, bool) else None)
    return out, row


def training_form(trainer_argv: list[str]) -> dict[str, Any]:
    """The trainers' training form and sync route, read from ``fleet.trainer_argv``
    through the ``yeto launch`` parser and the launcher's own routing
    (``sync_route.apply_sync_route``, then ``syncer.rl_full_parameter_async``):
    ``backend``, ``tuning``, ``expert_selection``, ``requested`` / ``preset`` /
    ``rounds`` (the route), ``form`` and ``full_parameter_async``. Empty when the
    argv does not parse as an RL launch (``yeto launch --dry-run`` names why)."""
    from yeto.rl.up_consistency import trainer_args  # noqa: PLC0415

    args = trainer_args([str(a) for a in trainer_argv])
    if args is None:
        return {}
    from yeto.launch.syncer import rl_full_parameter_async  # noqa: PLC0415
    from yeto.rl.engine.sync_route import SyncRouteError, apply_sync_route  # noqa: PLC0415

    out: dict[str, Any] = {"backend": str(getattr(args, "rl_backend", None) or "miles"),
                           "tuning": str(getattr(args, "tuning", "lora")),
                           "expert_full_count": int(getattr(args, "expert_full_count", 0) or 0),
                           "expert_selection": bool(getattr(args, "rl_expert_selection", None))}
    try:
        route = apply_sync_route(args)
    except SyncRouteError as error:
        return {**out, "error": str(error)}
    if route is None:
        return out
    out.update({"requested": route.requested, "preset": route.preset, "transport": route.transport,
                "rounds": getattr(args, "rl_sync_full_parameter_rounds", None),
                "full_parameter_async": bool(rl_full_parameter_async(args))})
    out["form"] = ("partial experts" if out["tuning"] == "full" and out["expert_selection"]
                   else "full parameters" if out["tuning"] == "full" else "LoRA")
    return out


def form_problem(form: dict[str, Any]) -> str | None:
    """Why a fleet with this :func:`training_form` cannot run (None: it can).

    A fleet's samplers are external engine pools that load each release whole and
    in place (vLLM ``reload_weights`` / SGLang ``update_weights_from_disk``); LoRA
    releases to external engines are out of scope, and the synced-group launch of
    external pools is Miles only."""
    if not form:
        return ("fleet.trainer_argv does not parse as an RL yeto launch argv (--training-mode rl ...); "
                "`yeto launch --dry-run` names the argument")
    if form.get("error"):
        return f"fleet.trainer_argv: {form['error']}"
    if form["backend"] != "miles":
        return (f"a fleet trainer runs on Miles (got --rl-backend {form['backend']}): the synced-group launch "
                "of external sampler pools is not built for other backends")
    if form["tuning"] == "lora" or form.get("expert_full_count"):
        return ("a LoRA fleet (--tuning lora or --expert-full-count) is not supported: the islands' samplers are "
                "external vLLM / SGLang pools, which load full-parameter releases in place; LoRA releases to "
                "external engines wait for upstream support. Train full parameters or partial experts "
                "(--tuning full [--rl-expert-selection FILE]) with --rl-sync-preset decoupled")
    if not form.get("full_parameter_async"):
        return (f"a fleet merges its trainers with asynchronous full-parameter rounds: --rl-sync-preset "
                f"{form.get('requested')} runs {form.get('preset')}"
                + (f" with {form.get('rounds')} rounds" if form.get("rounds") else "")
                + "; use --rl-sync-preset decoupled (routed to full-parameter async rounds)")
    return None


# ---------------------------------------------------------------------- compile

def compile_fleet(cfg: dict[str, Any]) -> tuple[dict[str, Any], FleetPlan]:
    """The single-trainer run config of an island fleet and its plan (module doc).
    Raises ValueError (``run config: ...``) on a bad ``islands`` / ``fleet`` section.
    The input is not changed; the output keeps ``islands`` and ``fleet`` as they were
    and adds ``trainer``, ``samplers`` and :data:`PLAN_KEY`."""
    check_exclusive(cfg)
    if "islands" not in cfg:
        raise _err("compile_fleet: no 'islands' section")
    store = str(cfg["store"])
    trainer_argv, dispense = _fleet_section(cfg)
    named = _argv_value(trainer_argv, "--rl-checkpoint-store")
    if named is None:
        raise _err("fleet.trainer_argv needs --rl-checkpoint-store <store>: each trainer releases to and reads "
                   "from <store>/trainer<k> of it")
    if _norm_store(named) != _norm_store(store):
        raise _err(f"fleet.trainer_argv: --rl-checkpoint-store {named} != the run config's store {store}: the "
                   "islands' pools write under the run store")
    islands = cfg["islands"]
    if not isinstance(islands, list) or not islands:
        raise _err("'islands' must be a non-empty list (one entry per island: name, trainer, samplers)")
    dispense_store = substore_uri(store, DISPENSE_DIR) if dispense == "fleet" else None
    rows: list[FleetIsland] = []
    samplers: list[dict[str, Any]] = []
    names: set[str] = set()
    for k, island in enumerate(islands):
        where = f"islands[{k}]"
        if not isinstance(island, dict):
            raise _err(f"{where} must be an object (name, trainer, samplers)")
        name = island.get("name")
        if not isinstance(name, str) or not NAME.fullmatch(name):
            raise _err(f"{where}: name must be a word of letters, digits, '.', '_' or '-' (got {name!r})")
        where = f"islands[{k}] ({name})"
        if name in names:
            raise _err(f"{where}: island names repeat ({name!r})")
        names.add(name)
        unknown = set(island) - ISLAND_KEYS
        if unknown:
            raise _err(f"unknown {where} keys {sorted(unknown)} (known: {sorted(ISLAND_KEYS)})")
        gpu, price, flags = _island_trainer(island, where)
        entries = island.get("samplers")
        if not isinstance(entries, list) or not entries:
            raise _err(f"{where}: samplers must be a non-empty list of pools (every island has its own)")
        pools = []
        for j, entry in enumerate(entries):
            out, row = _pool(entry, k, j, where, dispense_store)
            samplers.append(out)
            pools.append(row)
        seen = [p.name for p in pools]
        if len(set(seen)) != len(seen):
            raise _err(f"{where}: pool names repeat {seen}; set 'name' on each")
        rows.append(FleetIsland(index=k, name=name, gpu=gpu, usd_per_h=price, store=substore_uri(store, substore(k)),
                                release_every=flags.get("release_every"),
                                max_policy_age=flags.get("max_policy_age"), pools=tuple(pools)))
    argv = [*trainer_argv, "--gpu", ",".join(i.gpu for i in rows),
            "--rl-sample-role", "trainer", "--rl-sampler-follow", "trainer"]
    for i in rows:
        argv += ["--rl-island-buffer-shards", f"{i.index}:{','.join(i.buffer_shards())}"]
    for i in rows:
        if i.release_every is not None:
            argv += ["--rl-island-release-every", f"{i.index}:{i.release_every}"]
        if i.max_policy_age is not None:
            argv += ["--rl-island-max-policy-age", f"{i.index}:{i.max_policy_age}"]
    prices = [i.usd_per_h for i in rows]
    plan = FleetPlan(run=str(cfg["run"]), store=store, dispense=dispense, islands=tuple(rows),
                     dispense_store=dispense_store, sync=training_form(trainer_argv))
    out = copy.deepcopy({k: v for k, v in cfg.items() if k != PLAN_KEY})
    out["trainer"] = {"argv": argv, "usd_per_h": None if all(p is None for p in prices)
                      else float(sum(p or 0.0 for p in prices))}
    out["samplers"] = samplers
    out[PLAN_KEY] = plan
    return out, plan
