"""Declarations of the breaker registry: what a breaker is, where it is configured today.

Standard library only: the trainer, the sampler, the pool and the dashboard import this
without pulling in any training backend. The registry describes; components still judge.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

Layer = Literal["run", "launch", "trainer", "sampler", "harness", "pool", "monitor"]
Kind = Literal["breaker", "guard", "retry_budget", "alert", "invariant", "semantic"]
Action = Literal["warn", "drop", "quarantine", "downweight", "pause", "replace", "halt", "stop"]
Surface = Literal["run_config", "trainer_flag", "launch_flag", "trainer_env", "spec", "pool_config",
                  "pool_elastic", "pool_dispense", "pool_cli_flag", "env"]
Gauge = Literal["elapsed", "level", "count", "ratio", "none"]

LAYERS: tuple[str, ...] = ("run", "launch", "trainer", "sampler", "harness", "pool", "monitor")
KINDS: tuple[str, ...] = ("breaker", "guard", "retry_budget", "alert", "invariant", "semantic")
ACTIONS: tuple[str, ...] = ("warn", "drop", "quarantine", "downweight", "pause", "replace", "halt", "stop")
SURFACES: tuple[str, ...] = ("run_config", "trainer_flag", "launch_flag", "trainer_env", "spec", "pool_config",
                             "pool_elastic", "pool_dispense", "pool_cli_flag", "env")
# the trainer island's env: ``env``, then the trainer argv's ``--island-env K=V``, then ``--modal-env K=V``
ENV_PAIR_FLAGS: tuple[str, ...] = ("--island-env", "--modal-env")
GAUGES: tuple[str, ...] = ("elapsed", "level", "count", "ratio", "none")
# only these kinds are written in ``breakers:``; invariants and semantic entries stay at their native place
CONFIGURABLE_KINDS: tuple[str, ...] = ("breaker", "guard", "retry_budget", "alert")
STOP_ACTIONS: tuple[str, ...] = ("halt", "stop")  # a trip with one of these sets the run's stop_reason
INHERIT = "inherit"  # the section literal of an inherited param ("usd": "inherit")
FLAG_SURFACES: tuple[str, ...] = ("trainer_flag", "launch_flag", "pool_cli_flag")


@dataclass(frozen=True)
class Param:
    """One number (or flag) of a breaker, in canonical units."""

    name: str  # "timeout_s": unit suffix _s / _usd / _n / _ratio
    type: type
    default: Any  # None only with nullable
    unit: str = ""
    lo: float | None = None  # inclusive bounds
    hi: float | None = None
    nullable: bool = False  # null: no limit
    inherit_from: str | None = None  # "run.budget.usd"
    choices: tuple[Any, ...] = ()
    doc: str = ""


@dataclass(frozen=True)
class Alias:
    """A place a breaker is configured today (a flag, a spec field, an env var).

    ``param=None`` maps ``enabled`` only (through ``off_values`` / ``on_values``);
    ``param="action"`` maps the entry's action (a value in ``Breaker.actions``)."""

    surface: Surface
    path: str  # "--rl-sample-wait-s" / "budget.usd" / "agentic.min_samples" / "HARBOR_SETUP_RETRIES"
    param: str | None
    off_values: tuple[Any, ...] = ()  # legacy spellings of "off"; env and flag values compare as strings
    on_values: tuple[Any, ...] = ()
    scale: float = 1.0  # canonical = legacy * scale (minutes -> seconds: 60)
    to_canonical: Callable[[Any], Any] | None = None  # replaces scale (e.g. "k=v" strings)
    from_canonical: Callable[[Any], Any] | None = None
    nargs: int = 1  # flags: 0 for a bare switch (present: True)
    materialize: bool = True  # write-back target
    null_means_default: bool = False  # a nullable param's legacy null falls through (else: null = no limit)
    shadowed_by: tuple[str, ...] = ()  # sibling alias paths, or surfaces: when one is present, this alias
    #                                    is not read (the launcher derives it from that knob)
    deprecated: bool = False

    @property
    def location(self) -> str:
        """Where the alias lives, as ``show`` and the conflict messages name it."""
        prefix = {"run_config": "run config", "trainer_flag": "trainer.argv", "launch_flag": "trainer.argv",
                  "trainer_env": "trainer env",
                  "spec": "spec", "pool_config": "pool config", "pool_elastic": "pool elastic", "pool_dispense": "pool sampler.dispense",
                  "pool_cli_flag": "pool argv", "env": "env"}[self.surface]
        return f"{prefix} {self.path}"


@dataclass(frozen=True)
class Breaker:
    """One registry entry: an automatic stop, drop, pause or refusal (or a registered invariant)."""

    name: str  # "<layer>.<entry>"
    layer: Layer
    kind: Kind
    params: tuple[Param, ...] = ()
    actions: tuple[Action, ...] = ()  # implemented actions only
    default_action: Action | None = None
    default_enabled: bool = True
    can_disable: bool = True  # False for every invariant
    aliases: tuple[Alias, ...] = ()
    enabled_by: tuple[str, ...] = ()  # params whose non-null value turns a default-off entry on
    off_when_null: tuple[str, ...] = ()  # nullable params that, all set to null, turn the entry off
    implicit: bool = False  # active although not written (shown as ``default (implicit)``)
    native: str = ""  # invariant / semantic: where it is configured
    depends_on: tuple[str, ...] = ()  # rule ids
    legacy_events: tuple[str, ...] = ()
    gauge: Gauge = "none"
    direction: Literal["max", "min"] = "max"
    owner: str = ""
    doc: str = ""

    @property
    def entry(self) -> str:
        """The name without its layer: ``sample_wait`` of ``trainer.sample_wait``."""
        return self.name.split(".", 1)[1]

    @property
    def configurable(self) -> bool:
        return self.kind in CONFIGURABLE_KINDS

    def param(self, name: str) -> Param:
        for p in self.params:
            if p.name == name:
                return p
        raise KeyError(f"{self.name} has no param {name!r} (params: {[p.name for p in self.params]})")


@dataclass(frozen=True)
class SwitchGroup:
    """One switch over several entries (``--no-cost-guard``: ``launch.budget`` and ``launch.idle``).

    The group is off when its section key is false or one of its aliases is present
    (``off_when_present``) or holds an off value; then every member is off."""

    name: str
    members: tuple[str, ...]
    aliases: tuple[Alias, ...] = ()
    off_when_present: bool = True
    doc: str = ""


@dataclass(frozen=True)
class Finding:
    """One dependency-rule result (``yeto.breakers.rules``), rendered by ``show``."""

    rule: str  # "D1"
    level: Literal["error", "warn", "info"]
    message: str
    items: tuple[str, ...] = ()
    values: Mapping[str, Any] = field(default_factory=dict)
    suggested: Mapping[str, Any] = field(default_factory=dict)  # "<entry>.<param>" -> a value that passes
    overridden: str | None = None  # the deps.overrides value that changed the level

    def to_json(self) -> dict[str, Any]:
        out = {"rule": self.rule, "level": self.level, "message": self.message,
               "items": list(self.items), "values": dict(self.values)}
        if self.suggested:
            out["suggested"] = dict(self.suggested)
        if self.overridden is not None:
            out["overridden"] = self.overridden
        return out


class RegistryError(ValueError):
    """A breaker declaration is malformed (raised when the registry loads)."""


class BreakerConfigError(ValueError):
    """A ``breakers:`` section or its aliases cannot be used; one problem per line."""

    def __init__(self, problems: list[str] | str):
        self.problems = [problems] if isinstance(problems, str) else list(problems)
        super().__init__("\n".join(self.problems))
