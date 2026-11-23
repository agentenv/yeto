"""Effective breaker settings: the ``breakers:`` section merged with every legacy knob.

Per param, in order (the first that is set wins): the ``breakers:`` section
(``pools.<pool>`` first), the legacy aliases (run config sections, trainer argv, the
sampler spec, pool elastic / dispense / CLI, env), the ``inherit_from`` parent, the
registry default. Two sources that disagree are a conflict, never a silent override.

Without a ``breakers:`` section nothing changes: no value is inherited and
:func:`materialize` returns its inputs unchanged.
"""

from __future__ import annotations

import copy
import importlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from yeto.breakers.registry import Registry, default_registry
from yeto.breakers.spec import (
    FLAG_SURFACES,
    INHERIT,
    LAYERS,
    Alias,
    Breaker,
    BreakerConfigError,
    Finding,
    Param,
)

RESOLVED_SCHEMA = "yeto-breakers-resolved-v1"
RULES_MODULE = "yeto.breakers.rules"
SECTION_KEYS = ("pools", "switches", "deps")
DEPS_KEYS = ("outage_tolerance_s", "overrides")
OVERRIDE_LEVELS = ("error", "warn", "info", "off")
# show: the most specific source of an entry names the entry's source column
SOURCE_ORDER = ("breakers", "switch", "legacy", "inherit", "default")
MISSING: Any = object()
_ABSENT: Any = object()


@dataclass(frozen=True)
class Source:
    kind: str  # breakers / legacy / switch / inherit / default
    location: str

    def to_json(self) -> dict[str, str]:
        return {"kind": self.kind, "location": self.location}


@dataclass(frozen=True)
class Conflict:
    name: str
    message: str
    locations: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "message": self.message, "locations": list(self.locations)}


@dataclass(frozen=True)
class DepsConfig:
    outage_tolerance_s: float | None = None  # a declared design target, not a breaker
    overrides: Mapping[str, str] = field(default_factory=dict)  # rule id -> error / warn / info / off

    def to_json(self) -> dict[str, Any]:
        return {"outage_tolerance_s": self.outage_tolerance_s, "overrides": dict(self.overrides)}


@dataclass(frozen=True)
class Section:
    """A validated ``breakers:`` section."""

    entries: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)  # "<layer>.<entry>" -> keys
    pools: Mapping[str, Mapping[str, Mapping[str, Any]]] = field(default_factory=dict)
    switches: Mapping[str, bool] = field(default_factory=dict)
    deps: DepsConfig = field(default_factory=DepsConfig)


@dataclass
class Sources:
    """The places one process can see. None: that surface is not known here."""

    run_config: Mapping[str, Any] | None = None
    argv: Sequence[str] | None = None  # trainer argv (the yeto launch arguments)
    spec: Mapping[str, Any] | None = None
    pool_config: Mapping[str, Any] | None = None  # the pool config file's top level
    pool_elastic: Mapping[str, Any] | None = None
    pool_dispense: Mapping[str, Any] | None = None
    pool_cli: Sequence[str] | None = None
    env: Mapping[str, str] | None = None
    section: Mapping[str, Any] | None = None  # a bare breakers section (else run_config["breakers"])

    @classmethod
    def from_run_config(cls, cfg: Mapping[str, Any], **surfaces: Any) -> Sources:
        trainer = cfg.get("trainer") if isinstance(cfg.get("trainer"), Mapping) else {}
        surfaces.setdefault("argv", trainer.get("argv"))
        surfaces.setdefault("env", cfg.get("env"))
        return cls(run_config=cfg, **surfaces)

    def breakers_section(self) -> Mapping[str, Any] | None:
        if self.section is not None:
            return self.section
        if self.run_config is not None:
            return self.run_config.get("breakers")
        return None

    def surface(self, surface: str) -> Any:
        return getattr(self, _SURFACE_FIELD[surface])


_SURFACE_FIELD = {"run_config": "run_config", "trainer_flag": "argv", "launch_flag": "argv", "spec": "spec",
                  "pool_config": "pool_config", "pool_elastic": "pool_elastic", "pool_dispense": "pool_dispense", "pool_cli_flag": "pool_cli",
                  "env": "env"}


@dataclass(frozen=True)
class ResolvedEntry:
    spec: Breaker
    enabled: bool
    action: str | None
    params: Mapping[str, Any]
    sources: Mapping[str, Source]  # per param, plus "enabled" and "action"
    redundant: tuple[str, ...] = ()
    switch_groups: tuple[str, ...] = ()
    deprecations: tuple[str, ...] = ()  # one line per deprecated legacy knob this entry was read from

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def source(self) -> Source:
        """The most specific source over enabled, action and params."""
        return min(self.sources.values(), key=lambda s: SOURCE_ORDER.index(s.kind),
                   default=Source("default", "default"))

    @property
    def implicit(self) -> bool:
        """Active only by a default the config never wrote (``run.wall_clock`` without ``wall_h``)."""
        return self.spec.implicit and self.enabled and all(s.kind == "default" for s in self.sources.values())

    def to_json(self) -> dict[str, Any]:
        b = self.spec
        return {"name": b.name, "layer": b.layer, "kind": b.kind, "enabled": self.enabled, "action": self.action,
                "can_disable": b.can_disable, "implicit": self.implicit,
                "params": {k: _thaw(v) for k, v in self.params.items()},
                "units": {p.name: p.unit for p in b.params},
                "sources": {k: s.to_json() for k, s in self.sources.items()},
                "redundant": list(self.redundant), "switch_groups": list(self.switch_groups),
                "depends_on": list(b.depends_on), "gauge": b.gauge, "direction": b.direction,
                "owner": b.owner, **({"native": b.native} if b.native else {}),
                **({"deprecations": list(self.deprecations)} if self.deprecations else {})}


class ResolvedBreakers:
    """Every entry's effective setting, with where it came from."""

    def __init__(self, entries: Iterable[ResolvedEntry], *, pool: str | None = None,
                 deps: DepsConfig | None = None, conflicts: Iterable[Conflict] = (), has_section: bool = False,
                 switches: Mapping[str, dict[str, Any]] | None = None):
        self._entries = {e.name: e for e in entries}
        self.pool = pool
        self.deps = deps or DepsConfig()
        self.conflicts = tuple(conflicts)
        self.has_section = has_section
        self._switches = dict(switches or {})

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def entry(self, name: str) -> ResolvedEntry:
        try:
            return self._entries[name]
        except KeyError:
            raise KeyError(f"no breaker {name!r} in the resolved settings") from None

    def entries(self, layer: str | None = None) -> list[ResolvedEntry]:
        return [e for e in self._entries.values() if layer is None or e.spec.layer == layer]

    def enabled(self, name: str) -> bool:
        return self.entry(name).enabled

    def get(self, path: str, default: Any = MISSING) -> Any:
        """``"sampler.drain.timeout_s"`` -> its value; also ``<name>.enabled`` / ``<name>.action``."""
        layer, _, rest = path.partition(".")
        entry, _, key = rest.partition(".")
        name = f"{layer}.{entry}"
        if name not in self._entries or not key:
            if default is not MISSING:
                return default
            raise KeyError(f"no breaker setting {path!r} (expected '<layer>.<entry>.<param>')")
        e = self._entries[name]
        if key == "enabled":
            return e.enabled
        if key == "action":
            return e.action
        if key in e.params:
            return e.params[key]
        if default is not MISSING:
            return default
        raise KeyError(f"{name} has no param {key!r} (params: {sorted(e.params)})")

    def implicit(self) -> list[ResolvedEntry]:
        return [e for e in self._entries.values() if e.implicit]

    def deprecations(self) -> list[str]:
        """Deprecated legacy knobs this config uses (decision 9). Printed once by the user-facing
        call sites (run up, launch, the pool CLI, ``show``), never by components: an env alias is
        also the write-back channel, so a component cannot tell a user's knob from a written-back one."""
        return [line for e in self._entries.values() for line in e.deprecations]

    def switches(self) -> dict[str, bool]:
        return {name: bool(s["on"]) for name, s in self._switches.items()}

    def switch_source(self, group: str) -> str:
        return str(self._switches[group]["source"])

    def to_json(self, findings: Iterable[Finding] = ()) -> dict[str, Any]:
        return {"schema": RESOLVED_SCHEMA, "pool": self.pool, "has_section": self.has_section,
                "entries": [e.to_json() for e in self._entries.values()],
                "switches": copy.deepcopy(self._switches), "deps": self.deps.to_json(),
                "findings": [f.to_json() for f in findings],
                "conflicts": [c.to_json() for c in self.conflicts], "deprecations": self.deprecations()}


def load_resolved(path: str | Path, registry: Registry | None = None) -> ResolvedBreakers:
    """A ``breakers.resolved.json`` (``ResolvedBreakers.to_json``) read back."""
    data = json.loads(Path(path).read_text())
    return from_json(data, registry)


def from_json(data: Mapping[str, Any], registry: Registry | None = None) -> ResolvedBreakers:
    if data.get("schema") != RESOLVED_SCHEMA:
        raise ValueError(f"not a resolved breakers file (schema {data.get('schema')!r}, want {RESOLVED_SCHEMA})")
    reg = registry if registry is not None else default_registry()
    entries = []
    for row in data.get("entries", ()):
        name = row["name"]
        spec = reg.get(name) if name in reg else Breaker(name=name, layer=row.get("layer", name.split(".")[0]),
                                                         kind=row.get("kind", "breaker"))
        entries.append(ResolvedEntry(
            spec=spec, enabled=bool(row["enabled"]), action=row.get("action"),
            params={k: _freeze(v) for k, v in row.get("params", {}).items()},
            sources={k: Source(v["kind"], v["location"]) for k, v in row.get("sources", {}).items()},
            redundant=tuple(row.get("redundant", ())), switch_groups=tuple(row.get("switch_groups", ())),
            deprecations=tuple(row.get("deprecations", ()))))
    deps = data.get("deps") or {}
    return ResolvedBreakers(
        entries, pool=data.get("pool"), has_section=bool(data.get("has_section")),
        deps=DepsConfig(deps.get("outage_tolerance_s"), dict(deps.get("overrides") or {})),
        conflicts=[Conflict(c["name"], c["message"], tuple(c.get("locations", ()))) for c in data.get("conflicts", ())],
        switches=data.get("switches"))


# --- the breakers: section ---------------------------------------------------------------------


def parse_section(section: Any, registry: Registry | None = None) -> Section:
    """The section validated for structure (no aliases); every problem in one error."""
    reg = registry if registry is not None else default_registry()
    if section is None:
        return Section()
    if not isinstance(section, Mapping):
        raise BreakerConfigError("breakers: must be an object")
    problems: list[str] = []
    entries: dict[str, dict[str, Any]] = {}
    pools: dict[str, dict[str, dict[str, Any]]] = {}
    switches: dict[str, bool] = {}
    deps = DepsConfig()
    for key, value in section.items():
        where = f"breakers.{key}"
        if key in LAYERS:
            if not isinstance(value, Mapping):
                problems.append(f"{where}: must be an object of entries")
                continue
            for entry, keys in value.items():
                name = f"{key}.{entry}"
                if _check_entry(reg, name, keys, f"breakers.{name}", problems):
                    entries[name] = dict(keys)
        elif key == "pools":
            if not isinstance(value, Mapping):
                problems.append(f"{where}: must be an object of pool names")
                continue
            for pool, items in value.items():
                if not isinstance(items, Mapping):
                    problems.append(f"{where}.{pool}: must be an object of pool entries")
                    continue
                for entry, keys in items.items():
                    name = f"pool.{entry}"
                    if _check_entry(reg, name, keys, f"{where}.{pool}.{entry}", problems, pool_only=True):
                        pools.setdefault(str(pool), {})[name] = dict(keys)
        elif key == "switches":
            if not isinstance(value, Mapping):
                problems.append(f"{where}: must be an object of switch groups")
                continue
            known = sorted(g.name for g in reg.switch_groups)
            for group, on in value.items():
                if group not in known:
                    problems.append(f"{where}.{group}: unknown switch group (known: {known})")
                elif not isinstance(on, bool):
                    problems.append(f"{where}.{group}: must be true or false (got {on!r})")
                else:
                    switches[group] = on
        elif key == "deps":
            deps = _parse_deps(value, problems)
        else:
            problems.append(f"{where}: unknown key (known: {[*LAYERS, *SECTION_KEYS]})")
    if problems:
        raise BreakerConfigError(problems)
    return Section(entries, pools, switches, deps)


def _check_entry(reg: Registry, name: str, keys: Any, where: str, problems: list[str], *,
                 pool_only: bool = False) -> bool:
    if name not in reg:
        layer = name.split(".")[0]
        known = sorted(b.entry for b in reg.layer(layer))
        problems.append(f"{where}: unknown {layer} entry (known: {known})")
        return False
    b = reg.get(name)
    if not b.configurable:
        if isinstance(keys, Mapping) and keys.get("enabled") is False and not b.can_disable:
            problems.append(f"{where}.enabled: {name} cannot be disabled: a data-correctness check "
                            f"(configured at {b.native})")
        else:
            problems.append(f"{where}: a {b.kind} entry is not configured in breakers: (configured at {b.native})")
        return False
    if not isinstance(keys, Mapping):
        problems.append(f"{where}: must be an object ({{\"enabled\", \"action\", <params>}})")
        return False
    ok = True
    for key, value in keys.items():
        at = f"{where}.{key}"
        if key == "enabled":
            if not isinstance(value, bool):
                problems.append(f"{at}: must be true or false (got {value!r})")
                ok = False
            elif not value and not b.can_disable:
                problems.append(f"{at}: {name} cannot be disabled"
                                + (f" (configured at {b.native})" if b.native else ""))
                ok = False
        elif key == "action":
            if not b.actions:
                problems.append(f"{at}: {name} has no action choice")
                ok = False
            elif value not in b.actions:
                problems.append(f"{at}: {value!r} is not an action of {name} (choices: {list(b.actions)})")
                ok = False
        else:
            try:
                p = b.param(key)
            except KeyError:
                problems.append(f"{at}: unknown key (known: {['enabled', 'action', *(p.name for p in b.params)]})")
                ok = False
                continue
            error = _value_error(p, value, allow_inherit=True)
            if error:
                problems.append(f"{at}: {error}")
                ok = False
    return ok


def _parse_deps(value: Any, problems: list[str]) -> DepsConfig:
    if not isinstance(value, Mapping):
        problems.append("breakers.deps: must be an object")
        return DepsConfig()
    for key in value:
        if key not in DEPS_KEYS:
            problems.append(f"breakers.deps.{key}: unknown key (known: {list(DEPS_KEYS)})")
    tolerance = value.get("outage_tolerance_s")
    if tolerance is not None and not _is_number(tolerance, lo=0):
        problems.append(f"breakers.deps.outage_tolerance_s: must be a number >= 0 (got {tolerance!r})")
        tolerance = None
    overrides = value.get("overrides") or {}
    if not isinstance(overrides, Mapping):
        problems.append("breakers.deps.overrides: must be an object of rule id -> level")
        overrides = {}
    rule_ids = _rule_ids()
    for rule, level in overrides.items():
        if level not in OVERRIDE_LEVELS:
            problems.append(f"breakers.deps.overrides.{rule}: {level!r} is not one of {list(OVERRIDE_LEVELS)}")
        base = str(rule).split(".", 1)[0]
        if rule_ids is not None and base not in rule_ids and rule not in rule_ids:
            problems.append(f"breakers.deps.overrides.{rule}: unknown rule (rules: {list(rule_ids)})")
    return DepsConfig(None if tolerance is None else float(tolerance), dict(overrides))


def load_rules() -> Any:
    """``yeto.breakers.rules`` (the dependency rules), None until that module exists."""
    try:
        return importlib.import_module(RULES_MODULE)
    except ModuleNotFoundError as error:
        if error.name == RULES_MODULE:
            return None
        raise


def _rule_ids() -> tuple[str, ...] | None:
    rules = load_rules()
    if rules is None:
        return None
    ids = getattr(rules, "RULE_IDS", None)
    return tuple(ids) if ids is not None else None


# --- values ----------------------------------------------------------------------------------------


def _is_number(value: Any, *, lo: float | None = None, hi: float | None = None) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or math.isnan(value) or math.isinf(value):
        return False
    return (lo is None or value >= lo) and (hi is None or value <= hi)


def _value_error(p: Param, value: Any, *, allow_inherit: bool = False) -> str | None:
    """Why ``value`` is not a valid canonical value of ``p`` (None: it is)."""
    if value == INHERIT and isinstance(value, str) and p.type is not str:
        if allow_inherit and p.inherit_from:
            return None
        return f"{p.name} cannot be 'inherit' (it inherits from nothing)"
    if value is None:
        return None if p.nullable else "null is not allowed (only enabled: false turns an entry off)"
    if p.type is bool:
        return None if isinstance(value, bool) else f"must be true or false (got {value!r})"
    if p.type is dict:  # the declaring layer checks the contents (in its alias's to_canonical)
        return None if isinstance(value, Mapping) else f"must be an object (got {value!r})"
    if p.type is str:
        if not isinstance(value, str):
            return f"must be a string (got {value!r})"
        return None if not p.choices or value in p.choices else f"{value!r} is not one of {list(p.choices)}"
    if p.type is int and not (isinstance(value, int) and not isinstance(value, bool)):
        return f"must be an integer (got {value!r})"
    if not _is_number(value):
        return f"must be a finite number (got {value!r})"
    if not _is_number(value, lo=p.lo, hi=p.hi):
        bounds = " and ".join(s for s in (f">= {p.lo}" if p.lo is not None else "",
                                          f"<= {p.hi}" if p.hi is not None else "") if s)
        return f"must be {bounds} (got {value!r})"
    if p.choices and value not in p.choices:
        return f"{value!r} is not one of {list(p.choices)}"
    return None


def _freeze(value: Any) -> Any:
    """A mapping value as a read-only view: a resolved setting cannot be changed by its reader."""
    return MappingProxyType(dict(value)) if isinstance(value, Mapping) else value


def _thaw(value: Any) -> Any:
    return dict(value) if isinstance(value, Mapping) else value


def _same(a: Any, b: Any) -> bool:
    if _is_number(a) and _is_number(b):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def _matches(raw: Any, values: Iterable[Any]) -> bool:
    """``raw`` equals one of the legacy spellings; strings (flags, env) compare by text or number."""
    for v in values:
        if isinstance(raw, bool) or isinstance(v, bool):
            if isinstance(raw, bool) and isinstance(v, bool) and raw is v:
                return True
            if isinstance(raw, str) and isinstance(v, bool) and raw.strip().lower() == str(v).lower():
                return True
            continue
        if raw is None or v is None:
            if raw is None and v is None:
                return True
            continue
        if isinstance(raw, str):
            text = raw.strip().lower()
            if text == str(v).strip().lower():
                return True
            try:
                if _is_number(v) and math.isclose(float(text), float(v)):
                    return True
            except ValueError:
                pass
        elif raw == v:
            return True
    return False


def _coerce(p: Param, raw: Any, alias: Alias) -> Any:
    """A legacy value in canonical units (``ValueError`` when it does not parse)."""
    if alias.to_canonical is not None:
        return alias.to_canonical(raw)
    value = raw
    if p.type is dict:
        value = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(value, Mapping):
            raise ValueError(f"{raw!r} is not an object")
        return value
    if isinstance(raw, str) and p.type is not str:
        text = raw.strip()
        if p.type is bool:
            lowered = text.lower()
            if lowered in ("1", "true", "yes", "on"):
                return True
            if lowered in ("0", "false", "no", "off", ""):
                return False
            raise ValueError(f"{raw!r} is not a boolean")
        value = float(text)
        if p.type is int and value.is_integer() and alias.scale == 1:
            value = int(value)
    if alias.scale != 1 and _is_number(value):
        value = value * alias.scale
        if p.type is int and float(value).is_integer():
            value = int(value)
    elif p.type is float and _is_number(value):
        value = float(value) if not isinstance(value, int) else value
    return value


def _to_legacy(p: Param, value: Any, alias: Alias) -> Any:
    if alias.from_canonical is not None:
        return alias.from_canonical(_thaw(value))
    if isinstance(value, Mapping):
        return json.dumps(dict(value), sort_keys=True) if alias.surface in (*FLAG_SURFACES, "env") else dict(value)
    if alias.scale != 1 and _is_number(value):
        value = value / alias.scale
        if float(value).is_integer():
            value = int(value)
    return value


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Mapping):
        return json.dumps(dict(value), sort_keys=True)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


# --- reading and writing native surfaces -------------------------------------------------------------


def _dict_get(data: Mapping[str, Any] | None, path: str) -> Any:
    node: Any = data
    for part in path.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return _ABSENT
        node = node[part]
    return node


def _dict_set(data: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    node = data
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = node[part] = {}
        node = child
    node[parts[-1]] = value


def _argv_get(argv: Sequence[str] | None, flag: str, nargs: int) -> Any:
    if not argv:
        return _ABSENT
    value: Any = _ABSENT
    for i, arg in enumerate(argv):
        if arg == flag:
            if nargs == 0:
                value = True
            elif i + 1 < len(argv):
                value = argv[i + 1]
        elif nargs == 1 and arg.startswith(flag + "="):
            value = arg.split("=", 1)[1]
    return value


def _argv_set(argv: list[str], flag: str, value: str) -> None:
    found = False
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            argv[i + 1] = value
            found = True
        elif arg.startswith(flag + "="):
            argv[i] = f"{flag}={value}"
            found = True
    if not found:
        argv.extend([flag, value])


def _read(sources: Sources, alias: Alias) -> Any:
    surface = sources.surface(alias.surface)
    if surface is None:
        return _ABSENT
    if alias.surface in FLAG_SURFACES:
        return _argv_get(surface, alias.path, alias.nargs)
    if alias.surface == "env":
        return surface.get(alias.path, _ABSENT)
    return _dict_get(surface, alias.path)


# --- resolution ------------------------------------------------------------------------------------


def resolve(sources: Sources, *, pool: str | None = None, registry: Registry | None = None,
            strict: bool = True, layers: Iterable[str] | None = None) -> ResolvedBreakers:
    """Every entry's effective setting for one process (``pool``: one pool's overrides).

    ``strict``: a conflict raises :class:`BreakerConfigError`; else it is kept in ``conflicts``.
    ``layers``: only these layers' entries and conflicts (a process judging its own layer is
    not stopped by a malformed knob of another; inherited parents are still read)."""
    reg = registry if registry is not None else default_registry()
    raw_section = sources.breakers_section()
    section = parse_section(raw_section, reg)
    keep = tuple(layers) if layers is not None else None
    if keep is not None and set(keep) - set(LAYERS):
        raise ValueError(f"unknown layers {sorted(set(keep) - set(LAYERS))} (layers: {list(LAYERS)})")
    return _Resolver(reg, sources, section, pool, has_section=raw_section is not None).run(strict, keep)


def resolved(config: Mapping[str, Any] | Sources, *, pool: str | None = None, registry: Registry | None = None,
             strict: bool = True, layers: Iterable[str] | None = None, **surfaces: Any) -> ResolvedBreakers:
    """The resolved settings of a run config (or of explicit :class:`Sources`)."""
    sources = config if isinstance(config, Sources) else Sources.from_run_config(config, **surfaces)
    return resolve(sources, pool=pool, registry=registry, strict=strict, layers=layers)


@dataclass
class _Value:
    value: Any
    source: Source
    from_section: bool = False  # set (or inherited) from the breakers: section: children may inherit it


class _Resolver:
    def __init__(self, reg: Registry, sources: Sources, section: Section, pool: str | None, *, has_section: bool):
        self.reg, self.sources, self.section, self.pool = reg, sources, section, pool
        self.has_section = has_section
        self.conflicts: list[Conflict] = []
        self.done: dict[str, ResolvedEntry] = {}
        self.values: dict[str, dict[str, _Value]] = {}
        self.busy: set[str] = set()
        self.groups = {g.name: self._group_state(g) for g in reg.switch_groups}

    def run(self, strict: bool, layers: tuple[str, ...] | None = None) -> ResolvedBreakers:
        chosen = [b for b in self.reg if layers is None or b.layer in layers]
        for b in chosen:
            self._entry(b)
        if layers is not None:
            self.conflicts = [c for c in self.conflicts if c.name.split(".", 1)[0] in layers]
        if strict and self.conflicts:
            raise BreakerConfigError([f"{c.name}: {c.message}" for c in self.conflicts])
        switches = {name: {"on": on, "members": list(self.reg.group(name).members), "source": src.location}
                    for name, (on, src) in self.groups.items()}
        return ResolvedBreakers((self.done[b.name] for b in chosen), pool=self.pool, deps=self.section.deps,
                                conflicts=self.conflicts, has_section=self.has_section, switches=switches)

    def conflict(self, name: str, message: str, *locations: str) -> None:
        self.conflicts.append(Conflict(name, message, tuple(locations)))

    def _group_state(self, g: Any) -> tuple[bool, Source]:
        legacy_off = None
        for alias in g.aliases:
            raw = _read(self.sources, alias)
            if raw is _ABSENT:
                continue
            if _matches(raw, alias.off_values) or (g.off_when_present and not _matches(raw, alias.on_values)):
                legacy_off = Source("switch", f"switch {g.name} ({alias.location})")
                break
        if g.name in self.section.switches:
            on = self.section.switches[g.name]
            location = f"breakers.switches.{g.name}"
            if on and legacy_off is not None:
                self.conflict(g.members[0], f"switch {g.name}: {location} = true but {legacy_off.location} turns "
                                            f"it off", location, legacy_off.location)
            return on, Source("switch", location)
        return (False, legacy_off) if legacy_off is not None else (True, Source("default", "default"))

    def _section_keys(self, b: Breaker) -> dict[str, tuple[Any, str]]:
        """``key -> (value, location)`` from the section, one pool's overrides on top."""
        out = {k: (v, f"breakers.{b.name}.{k}") for k, v in self.section.entries.get(b.name, {}).items()}
        if self.pool is not None and b.layer == "pool":
            for k, v in self.section.pools.get(self.pool, {}).get(b.name, {}).items():
                out[k] = (v, f"breakers.pools.{self.pool}.{b.entry}.{k}")
        return out

    def _entry(self, b: Breaker) -> ResolvedEntry:
        if b.name in self.done:
            return self.done[b.name]
        if b.name in self.busy:
            raise BreakerConfigError(f"{b.name}: inherit_from forms a cycle")
        self.busy.add(b.name)
        sec = self._section_keys(b) if b.configurable else {}
        legacy_params: dict[str, list[tuple[Any, Source]]] = {}
        legacy_enabled: list[tuple[bool, Source]] = []
        legacy_action: list[tuple[Any, Source]] = []
        for alias in b.aliases:
            raw = _read(self.sources, alias)
            if raw is _ABSENT:
                continue
            src = Source("legacy", alias.location)
            if _matches(raw, alias.off_values):
                legacy_enabled.append((False, src))
                continue
            if alias.param is None:
                if _matches(raw, alias.on_values):
                    legacy_enabled.append((True, src))
                continue  # an unknown spelling of an enable-only knob means nothing today
            if alias.param == "action":
                if alias.to_canonical is not None:
                    try:
                        raw = alias.to_canonical(raw)
                    except (TypeError, ValueError) as error:
                        self.conflict(b.name, f"{alias.location} cannot be read as the action ({error})",
                                      alias.location)
                        continue
                    if raw is None:
                        continue
                if b.actions and raw not in b.actions:
                    self.conflict(b.name, f"{alias.location} = {raw!r} is not an action of {b.name} "
                                          f"(choices: {list(b.actions)})", alias.location)
                    continue
                legacy_action.append((raw, src))
                continue
            p = b.param(alias.param)
            if raw is None:
                if not p.nullable or alias.null_means_default:
                    continue  # a legacy null reads as "the default"
                legacy_params.setdefault(p.name, []).append((None, src))  # null: no limit
                continue
            try:
                value = _coerce(p, raw, alias)
            except (TypeError, ValueError) as error:
                self.conflict(b.name, f"{alias.location} = {raw!r} cannot be read as {p.name} ({error})",
                              alias.location)
                continue
            error = _value_error(p, value)
            if error:
                self.conflict(b.name, f"{alias.location} = {raw!r}: {p.name} {error}", alias.location)
                continue
            legacy_params.setdefault(p.name, []).append((value, src))

        redundant: list[str] = []
        values: dict[str, _Value] = {}
        for p in b.params:
            values[p.name] = self._param(b, p, sec.get(p.name), legacy_params.get(p.name, []), redundant)
        self.values[b.name] = values
        enabled = self._enabled(b, sec.get("enabled"), legacy_enabled, legacy_params, values)
        action = self._action(b, sec.get("action"), legacy_action, redundant)
        sources = {"enabled": enabled.source, "action": action.source,
                   **{name: v.source for name, v in values.items()}}
        if b.implicit and all(s.kind == "default" for s in sources.values()):
            sources = {k: Source("default", "default (implicit)") for k in sources}
        entry = ResolvedEntry(b, bool(enabled.value), action.value, {k: _freeze(v.value) for k, v in values.items()},
                              sources, tuple(redundant), self.reg.groups_of(b.name),
                              _deprecations(b, sources, redundant))
        self.busy.discard(b.name)
        self.done[b.name] = entry
        return entry

    def _agree(self, b: Breaker, key: str, chosen: tuple[Any, str] | None,
               legacy: list[tuple[Any, Source]], redundant: list[str]) -> None:
        """Conflicts between the section value (if any) and the legacy values, or among the latter."""
        if chosen is not None:
            value, location = chosen
            for v, src in legacy:
                if _same(v, value):
                    redundant.append(src.location)
                else:
                    self.conflict(b.name, f"{key}: {location} = {_fmt(value)} but {src.location} = {_fmt(v)}",
                                  location, src.location)
        elif legacy:
            first, src0 = legacy[0]
            for v, src in legacy[1:]:
                if not _same(v, first):
                    self.conflict(b.name, f"{key}: {src0.location} = {_fmt(first)} but {src.location} = {_fmt(v)}",
                                  src0.location, src.location)

    def _param(self, b: Breaker, p: Param, sec: tuple[Any, str] | None,
               legacy: list[tuple[Any, Source]], redundant: list[str]) -> _Value:
        if sec is not None:
            value, location = sec
            if isinstance(value, str) and value == INHERIT and p.type is not str:
                inherited = self._inherited(p, explicit=True)
                self._agree(b, p.name, (inherited.value, location), legacy, redundant)
                return inherited
            self._agree(b, p.name, sec, legacy, redundant)
            return _Value(value, Source("breakers", location), from_section=True)
        self._agree(b, p.name, None, legacy, redundant)
        if legacy:
            return _Value(legacy[0][0], legacy[0][1])
        if p.inherit_from:
            inherited = self._inherited(p, explicit=False)
            if inherited is not None:
                return inherited
        return _Value(p.default, Source("default", "default"))

    def _inherited(self, p: Param, *, explicit: bool) -> Any:
        """The parent's value; None (not inherited) unless explicit or the parent came from the section."""
        parent_name, _, pname = p.inherit_from.rpartition(".")
        parent = self._entry(self.reg.get(parent_name))
        pv = self.values[parent_name][pname]
        if not explicit and not pv.from_section:
            return None
        value = pv.value if parent.enabled else None
        return _Value(value, Source("inherit", f"inherit {p.inherit_from}"), from_section=pv.from_section)

    def _enabled(self, b: Breaker, sec: tuple[Any, str] | None, legacy: list[tuple[bool, Source]],
                 legacy_params: dict[str, list[tuple[Any, Source]]], values: dict[str, _Value]) -> _Value:
        group_off = [(g, self.groups[g][1]) for g in self.reg.groups_of(b.name) if not self.groups[g][0]]
        if sec is not None:
            on, location = sec
            for _, src in legacy:
                if _ != on:
                    self.conflict(b.name, f"enabled: {location} = {_fmt(on)} but {src.location} says "
                                          f"{'on' if _ else 'off'}", location, src.location)
            if on:
                for g, src in group_off:
                    self.conflict(b.name, f"enabled: {location} = true but switch group {g} is off ({src.location})",
                                  location, src.location)
            else:
                self._off_against_values(b, location, legacy_params)
            return _Value(on, Source("breakers", location))
        section_switch = [(g, src) for g, src in group_off if src.location.startswith("breakers.")]
        if section_switch:
            self._off_against_values(b, section_switch[0][1].location, legacy_params)
            return _Value(False, section_switch[0][1])
        offs = [src for on, src in legacy if not on]
        ons = [src for on, src in legacy if on]
        for src in ons:
            if offs and src.location != offs[0].location:
                self.conflict(b.name, f"enabled: {offs[0].location} turns it off but {src.location} turns it on",
                              offs[0].location, src.location)
        if offs:
            return _Value(False, offs[0])
        if group_off:
            return _Value(False, group_off[0][1])
        if b.off_when_null and all(values[n].value is None for n in b.off_when_null):
            set_null = [values[n].source for n in b.off_when_null if values[n].source.kind != "default"]
            if set_null:  # every verdict turned off by the config: the entry is off (starvation 0 / 0)
                return _Value(False, set_null[-1])
        if ons:
            return _Value(True, ons[0])
        if not b.default_enabled:
            for name in b.enabled_by:
                v = values[name]
                if v.value is not None and v.source.kind != "default":
                    return _Value(True, v.source)
        return _Value(b.default_enabled, Source("default", "default"))

    def _off_against_values(self, b: Breaker, location: str, legacy_params: dict[str, list[tuple[Any, Source]]]) -> None:
        for name, rows in legacy_params.items():
            for v, src in rows:
                self.conflict(b.name, f"enabled: {location} turns it off but {src.location} sets {name} = {_fmt(v)}",
                              location, src.location)

    def _action(self, b: Breaker, sec: tuple[Any, str] | None, legacy: list[tuple[Any, Source]],
                redundant: list[str]) -> _Value:
        self._agree(b, "action", sec, legacy, redundant)
        if sec is not None:
            return _Value(sec[0], Source("breakers", sec[1]))
        if legacy:
            return _Value(legacy[0][0], legacy[0][1])
        return _Value(b.default_action, Source("default", "default"))


def _deprecations(b: Breaker, sources: Mapping[str, Source], redundant: Iterable[str]) -> tuple[str, ...]:
    deprecated = {a.location: a for a in b.aliases if a.deprecated}
    if not deprecated:
        return ()
    used: dict[str, str] = {}
    for key, src in sources.items():
        if src.kind == "legacy" and src.location in deprecated:
            used.setdefault(src.location, key)
    for location in redundant:
        if location in deprecated:
            alias = deprecated[location]
            used.setdefault(location, alias.param or "enabled")
    return tuple(f"{location} is deprecated (removed two minor versions after it was deprecated): "
                 f"write breakers.{b.name}.{key}" for location, key in used.items())


# --- write-back --------------------------------------------------------------------------------------


@dataclass
class Materialized:
    """The native surfaces with the section's values written back (copies; inputs are never mutated)."""

    run_config: dict[str, Any] | None = None
    argv: list[str] | None = None
    spec: dict[str, Any] | None = None
    pool_config: dict[str, Any] | None = None
    pool_elastic: dict[str, Any] | None = None
    pool_dispense: dict[str, Any] | None = None
    pool_cli: list[str] | None = None
    env: dict[str, str] | None = None
    changes: list[str] = field(default_factory=list)  # "trainer.argv --rl-sample-wait-s 7200"
    unapplied: list[str] = field(default_factory=list)  # settings no native knob here can carry

    def surface(self, surface: str) -> Any:
        return getattr(self, _SURFACE_FIELD[surface])


def materialize(resolved_: ResolvedBreakers, sources: Sources, surfaces: Iterable[str] | None = None, *,
                registry: Registry | None = None, strict: bool = True) -> Materialized:
    """Write the effective values that came from the section (or inheritance / a section switch)
    back into the native knobs of ``sources``. Without a section the copies equal the inputs.

    A setting whose only native knobs are read-only (or absent) cannot reach its component
    before phase 4: ``strict`` raises :class:`BreakerConfigError`, else it is listed in
    ``unapplied``. A knob of another process (a surface not in ``sources``) is skipped."""
    out = Materialized(
        run_config=copy.deepcopy(dict(sources.run_config)) if sources.run_config is not None else None,
        argv=list(sources.argv) if sources.argv is not None else None,
        spec=copy.deepcopy(dict(sources.spec)) if sources.spec is not None else None,
        pool_config=copy.deepcopy(dict(sources.pool_config)) if sources.pool_config is not None else None,
        pool_elastic=copy.deepcopy(dict(sources.pool_elastic)) if sources.pool_elastic is not None else None,
        pool_dispense=copy.deepcopy(dict(sources.pool_dispense)) if sources.pool_dispense is not None else None,
        pool_cli=list(sources.pool_cli) if sources.pool_cli is not None else None,
        env=dict(sources.env) if sources.env is not None else None)
    if not resolved_.has_section:
        return out
    reg = registry if registry is not None else default_registry()
    writer = _Writer(out, sources, set(surfaces) if surfaces is not None else set(_SURFACE_FIELD))
    switched = {g.name for g in reg.switch_groups
                if g.name in resolved_.switches() and not resolved_.switches()[g.name]
                and resolved_.switch_source(g.name).startswith("breakers.") and writer.switch_off(g)}
    for e in resolved_.entries():
        writer.entry(e, switched)
    if strict and out.unapplied:
        raise BreakerConfigError([f"{u}: no native knob in this process carries it (cannot be applied)"
                                  for u in out.unapplied])
    return out


class _Writer:
    def __init__(self, out: Materialized, sources: Sources, wanted: set[str]):
        self.out, self.sources, self.wanted = out, sources, wanted

    def _available(self, alias: Alias) -> bool:
        return alias.materialize and alias.surface in self.wanted and self.out.surface(alias.surface) is not None

    def _write(self, alias: Alias, legacy: Any) -> bool:
        target = self.out.surface(alias.surface)
        if alias.surface in FLAG_SURFACES:
            if alias.nargs == 0:
                if legacy is True and alias.path not in target:
                    target.append(alias.path)
                elif legacy is False:
                    target[:] = [a for a in target if a != alias.path]
                else:
                    return legacy is True
                self.out.changes.append(f"{alias.location}{'' if legacy else ' (removed)'}")
                return True
            if legacy is None:
                return False
            _argv_set(target, alias.path, _fmt(legacy))
        elif alias.surface == "env":
            if legacy is None:
                return False
            target[alias.path] = _fmt(legacy)
        else:
            _dict_set(target, alias.path, legacy)
        self.out.changes.append(f"{alias.location} {_fmt(legacy) if legacy is not None else 'null'}")
        return True

    def switch_off(self, group: Any) -> bool:
        for alias in group.aliases:
            if _read(self.sources, alias) is not _ABSENT:
                return True  # already off natively
        for alias in group.aliases:
            if self._available(alias):
                off = alias.off_values[0] if alias.off_values else True
                if self._write(alias, off):
                    return True
        return False

    def _here(self, b: Breaker) -> bool:
        """Some native knob of ``b`` lives in this process. An entry with no alias at all is applied
        by components that read the resolved settings, and one whose knobs all live in another
        process is applied there: neither is reported as unapplied."""
        writable = [a for a in b.aliases if a.materialize]
        return any(a.surface in self.wanted and self.sources.surface(a.surface) is not None
                   for a in (writable or b.aliases))

    def entry(self, e: ResolvedEntry, switched: set[str]) -> None:
        b = e.spec
        if not b.configurable:
            return
        en = e.sources["enabled"]
        if not e.enabled:
            if en.kind == "switch" and any(g in switched for g in e.switch_groups):
                return
            if en.kind in ("breakers", "switch"):
                off = [a for a in b.aliases if a.off_values]
                already = any(_matches(_read(self.sources, a), a.off_values) for a in off
                              if _read(self.sources, a) is not _ABSENT)
                if not already and not self._first(off, lambda a: a.off_values[0], check_present=False):
                    self._unapplied(b, "enabled = false")
            return
        if en.kind == "breakers" and not b.default_enabled:
            on = [a for a in b.aliases if a.param is None and a.on_values]
            self._first(on, lambda a: a.on_values[0])
        for p in b.params:
            if e.sources[p.name].kind in ("breakers", "inherit"):
                value = e.params[p.name]
                aliases = [a for a in b.aliases if a.param == p.name]
                if not self._first(aliases, lambda a, p=p, value=value: _to_legacy(p, value, a)):
                    self._unapplied(b, f"{p.name} = {_fmt(value) if value is not None else 'null'}")
        if e.sources["action"].kind == "breakers" and e.action != b.default_action:
            aliases = [a for a in b.aliases if a.param == "action"]
            if not self._first(aliases, lambda a: e.action):
                self._unapplied(b, f"action = {e.action}")

    def _unapplied(self, b: Breaker, what: str) -> None:
        if self._here(b):
            self.out.unapplied.append(f"{b.name}.{what}")

    def _first(self, aliases: list[Alias], legacy: Any, *, check_present: bool = True) -> bool:
        """Write ``legacy(alias)`` to the first writable alias; True when written or already set natively."""
        if check_present and any(_read(self.sources, a) is not _ABSENT for a in aliases):
            return True  # already set natively (the resolver checked it agrees)
        for a in aliases:
            if self._available(a) and self._write(a, legacy(a)):
                return True
        return False


# --- dependency rules hook ---------------------------------------------------------------------------


def preflight(resolved_: ResolvedBreakers, *, run_config: Mapping[str, Any] | None = None,
              sources: Sources | None = None) -> list[Finding]:
    """The dependency rules' findings (``yeto.breakers.rules.check``); [] until that module exists."""
    rules = load_rules()
    if rules is None:
        return []
    return list(rules.check(resolved_, run_config=run_config, sources=sources))
