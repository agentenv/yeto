"""The breaker registry: every entry of every layer, validated once when it loads.

Entries are declared in ``yeto/breakers/layers/<layer>.py`` as module-level tuples
``BREAKERS`` (and optionally ``SWITCH_GROUPS``). :func:`default_registry` imports those
modules on first use, never at ``import yeto.breakers``; a layer module that does not
exist yet is skipped.
"""

from __future__ import annotations

import importlib
import math
import re
from collections.abc import Iterable, Iterator

from yeto.breakers.spec import (
    ACTIONS,
    GAUGES,
    KINDS,
    LAYERS,
    SURFACES,
    Breaker,
    RegistryError,
    SwitchGroup,
)

_ENTRY = re.compile(r"[a-z][a-z0-9_]*")
LAYER_PACKAGE = "yeto.breakers.layers"


class Registry:
    """Breakers by name, and the switch groups over them."""

    def __init__(self, breakers: Iterable[Breaker] = (), switch_groups: Iterable[SwitchGroup] = ()):
        self._breakers: dict[str, Breaker] = {}
        for b in breakers:
            if b.name in self._breakers:
                raise RegistryError(f"breaker {b.name!r} is declared twice")
            self._breakers[b.name] = b
        self._groups: dict[str, SwitchGroup] = {}
        for g in switch_groups:
            if g.name in self._groups:
                raise RegistryError(f"switch group {g.name!r} is declared twice")
            self._groups[g.name] = g
        problems = [p for b in self._breakers.values() for p in _check_breaker(b, self._breakers)]
        problems += [p for g in self._groups.values() for p in _check_group(g, self._breakers)]
        problems += _check_alias_claims(self._breakers.values())
        if problems:
            raise RegistryError("breaker registry: " + "; ".join(problems))

    def __contains__(self, name: object) -> bool:
        return name in self._breakers

    def __iter__(self) -> Iterator[Breaker]:
        return iter(self._breakers.values())

    def __len__(self) -> int:
        return len(self._breakers)

    def get(self, name: str) -> Breaker:
        try:
            return self._breakers[name]
        except KeyError:
            raise KeyError(f"no breaker {name!r} in the registry") from None

    def layer(self, layer: str) -> list[Breaker]:
        return [b for b in self._breakers.values() if b.layer == layer]

    @property
    def switch_groups(self) -> list[SwitchGroup]:
        return list(self._groups.values())

    def group(self, name: str) -> SwitchGroup:
        try:
            return self._groups[name]
        except KeyError:
            raise KeyError(f"no switch group {name!r} (groups: {sorted(self._groups)})") from None

    def groups_of(self, name: str) -> tuple[str, ...]:
        """The switch groups an entry belongs to."""
        return tuple(g.name for g in self._groups.values() if name in g.members)


def _check_breaker(b: Breaker, by_name: dict[str, Breaker]) -> list[str]:
    out = []
    layer, _, entry = b.name.partition(".")
    if b.layer not in LAYERS:
        out.append(f"{b.name}: layer {b.layer!r} is not one of {LAYERS}")
    if layer != b.layer or not _ENTRY.fullmatch(entry):
        out.append(f"{b.name}: the name must be '{b.layer}.<entry>' (lowercase, digits, _)")
    if b.kind not in KINDS:
        out.append(f"{b.name}: kind {b.kind!r} is not one of {KINDS}")
    if b.kind == "invariant" and b.can_disable:
        out.append(f"{b.name}: an invariant cannot be disabled (declare can_disable=False)")
    if b.kind in ("invariant", "semantic") and not b.native:
        out.append(f"{b.name}: a {b.kind} entry names its native location (native=...)")
    bad = [a for a in b.actions if a not in ACTIONS]
    if bad:
        out.append(f"{b.name}: actions {bad} are not in {ACTIONS}")
    if b.default_action is not None and b.actions and b.default_action not in b.actions:
        out.append(f"{b.name}: default_action {b.default_action!r} is not in its actions {b.actions}")
    if b.gauge not in GAUGES:
        out.append(f"{b.name}: gauge {b.gauge!r} is not one of {GAUGES}")
    if b.direction not in ("max", "min"):
        out.append(f"{b.name}: direction must be 'max' or 'min'")
    names = [p.name for p in b.params]
    if len(set(names)) != len(names):
        out.append(f"{b.name}: a param is declared twice ({names})")
    reserved = {"enabled", "action"} & set(names)
    if reserved:
        out.append(f"{b.name}: params cannot be named {sorted(reserved)}")
    for p in b.params:
        if p.default is None and not p.nullable:
            out.append(f"{b.name}.{p.name}: default None needs nullable=True")
        elif p.default is not None and not _in_bounds(p.default, p.lo, p.hi):
            out.append(f"{b.name}.{p.name}: default {p.default!r} is outside [{p.lo}, {p.hi}]")
        if p.choices and p.default is not None and p.default not in p.choices:
            out.append(f"{b.name}.{p.name}: default {p.default!r} is not one of {p.choices}")
        if p.inherit_from is not None:
            parent, _, pname = p.inherit_from.rpartition(".")
            if parent not in by_name or pname not in {q.name for q in by_name[parent].params}:
                out.append(f"{b.name}.{p.name}: inherit_from {p.inherit_from!r} names no param")
            elif parent == b.name:
                out.append(f"{b.name}.{p.name}: cannot inherit from its own entry")
    for name in b.enabled_by:
        if name not in names:
            out.append(f"{b.name}: enabled_by names no param {name!r}")
    nullable = {p.name for p in b.params if p.nullable}
    for name in b.off_when_null:
        if name not in nullable:
            out.append(f"{b.name}: off_when_null names no nullable param {name!r}")
    for a in b.aliases:
        if a.surface not in SURFACES:
            out.append(f"{b.name}: alias surface {a.surface!r} is not one of {SURFACES}")
        if a.param == "action":
            if not b.actions:
                out.append(f"{b.name}: alias {a.path} maps the action, but the entry has no actions")
        elif a.param is not None and a.param not in names:
            out.append(f"{b.name}: alias {a.path} maps no param {a.param!r}")
        if a.param is None and not (a.off_values or a.on_values):
            out.append(f"{b.name}: alias {a.path} maps 'enabled' but has no off_values / on_values")
        if (a.off_values or (a.param is None and a.nargs == 0)) and not b.can_disable:
            out.append(f"{b.name}: alias {a.path} turns off an entry that cannot be disabled")
        if a.nargs not in (0, 1) or (a.nargs == 0 and not a.surface.endswith("flag")):
            out.append(f"{b.name}: alias {a.path}: nargs 0 is for flags only (nargs 0 or 1)")
    return out


def _check_group(g: SwitchGroup, by_name: dict[str, Breaker]) -> list[str]:
    out = []
    if not _ENTRY.fullmatch(g.name):
        out.append(f"switch group {g.name!r}: lowercase, digits, _")
    if not g.members:
        out.append(f"switch group {g.name}: no members")
    for m in g.members:
        if m not in by_name:
            out.append(f"switch group {g.name}: member {m!r} is not a breaker")
        elif not by_name[m].can_disable:
            out.append(f"switch group {g.name}: member {m} cannot be disabled")
    for a in g.aliases:
        if a.param is not None:
            out.append(f"switch group {g.name}: alias {a.path} maps 'enabled' only (param=None)")
    return out


def _check_alias_claims(breakers: Iterable[Breaker]) -> list[str]:
    """Two entries writing back to the same native knob would fight; read-only aliases may share."""
    owners: dict[tuple[str, str], str] = {}
    out = []
    for b in breakers:
        for a in b.aliases:
            if not a.materialize:
                continue
            key = (a.surface, a.path)
            if key in owners and owners[key] != b.name:
                out.append(f"{a.location} is written back by both {owners[key]} and {b.name}")
            owners.setdefault(key, b.name)
    return out


def _in_bounds(value: object, lo: float | None, hi: float | None) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return True  # bounds apply to numbers only
    if math.isnan(value):
        return False
    return (lo is None or value >= lo) and (hi is None or value <= hi)


_DEFAULT: Registry | None = None


def load_layers(layers: Iterable[str] = LAYERS) -> Registry:
    """A registry of every ``yeto.breakers.layers.<layer>`` module that exists."""
    breakers: list[Breaker] = []
    groups: list[SwitchGroup] = []
    for layer in layers:
        module_name = f"{LAYER_PACKAGE}.{layer}"
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            if error.name in (module_name, LAYER_PACKAGE):
                continue  # not declared yet
            raise
        for b in getattr(module, "BREAKERS", ()):
            if b.layer != layer:
                raise RegistryError(f"{b.name} is declared in layers/{layer}.py but has layer {b.layer!r}")
            breakers.append(b)
        groups.extend(getattr(module, "SWITCH_GROUPS", ()))
    return Registry(breakers, groups)


def default_registry() -> Registry:
    """The registry of every declared layer (loaded once per process)."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = load_layers()
    return _DEFAULT
