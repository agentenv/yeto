"""``AlgorithmSpec``: yeto-owned algorithm description with a canonical SHA256.

Pure Python; no torch/ray/miles imports. The spec is translated to engine
arguments only by adapters; ``to_legacy_argv`` documents the legacy mapping
used by ``yeto.rl.learner.build_miles_argv``.

Structure (change ``rl-algorithm-capabilities``, design D1/D2/D7):

* grouped frozen dataclasses ``AdvantageSpec`` / ``LossSpec`` / ``KlSpec`` /
  ``CorrectionSpec`` / ``SamplingSpec`` / ``ExecutionSpec`` plus
  ``entropy_coef`` and ``plugins`` (``PluginRef`` = dotted path + source
  SHA256);
* canonical JSON keeps the byte-identical R0 v1 form whenever the spec is
  v1-expressible (so every R0 hash is unchanged) and switches to the
  ``yeto-rl-algorithm-spec-v2`` form otherwise;
* ``required_mechanisms()`` drives the capability handshake,
  ``rejections()`` is the pre-GPU rejection matrix and ``expects_gradient()``
  the per-algorithm zero-gradient invariant.

Extension points for the follow-up algorithm changes (a new module plus one
line in :data:`yeto.rl.algos.EXTENSION_MODULES`): :func:`register_field`,
:func:`register_mechanism`, :func:`register_rejection` (spec-only rejection
matrix), :func:`register_launch_check` (needs run configuration values),
:func:`register_island_check` (needs this island's identity, before outer
sync), :func:`register_runtime_attrs` (Miles namespace attributes, e.g. plugin
configuration) and :func:`register_gradient_rule` (may only relax the
zero-gradient expectation). Registered fields
enter the canonical v2 JSON only when they differ from their default, so a new
field never changes an existing hash.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, fields
from typing import Any

ALGORITHM_SPEC_SCHEMA = "yeto-rl-algorithm-spec-v1"
ALGORITHM_SPEC_SCHEMA_V2 = "yeto-rl-algorithm-spec-v2"
ALGORITHM_SPEC_SCHEMAS = (ALGORITHM_SPEC_SCHEMA, ALGORITHM_SPEC_SCHEMA_V2)
BOUNDED_NONZERO_STD_FILTER = "yeto.rl.filters.bounded_nonzero_reward_std"
STOCK_NONZERO_STD_FILTER = (
    "miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std"
)
# v1 (R0) vocabulary: what the flat v1 fields can express.
SUPPORTED_ADVANTAGE_ESTIMATORS = frozenset({"grpo", "ppo"})  # ppo: rl-algo-critic-family 2.2
SUPPORTED_LOSSES = frozenset({"policy_loss"})
SUPPORTED_FILTERS = frozenset({BOUNDED_NONZERO_STD_FILTER})

# v2 vocabulary: what the structured spec can *express* (upstream Miles
# ``arguments.py`` choices at MILES_NEXT_COMMIT). Expressible is not
# supported: support is the engine's capability declaration.
ADVANTAGE_ESTIMATORS = (
    "grpo",
    "gspo",
    "reinforce_plus_plus",
    "reinforce_plus_plus_baseline",
    "ppo",
)
CRITIC_ESTIMATORS = frozenset({"ppo"})
# rl-algo-critic-family (design D1/D9): critic vocabulary. Expressible is not
# supported -- what the pinned Miles cannot run is refused by the rejection
# matrix (``critic_not_at_pin``) until the fork's GAE extension point (group 6).
GAE_VARIANTS = ("vanilla", "decoupled", "cross_segment")
LAMBD_MODES = ("fixed", "length_adaptive")
CRITIC_VALUE_LOSSES = ("mse", "hl_gauss")
CRITIC_INITS = ("copy_actor_backbone", "load")
CRITIC_PARAM_MODES = ("full", "lora")
# Concrete values a critic spec carries when the user leaves them unset (Miles
# c35702e defaults: --gamma/--lambd 1.0 arguments.py:1729-1730, --value-clip
# 0.2 :1648); they enter the hash explicitly so two islands agree on them.
CRITIC_ADVANTAGE_DEFAULTS = (
    ("lambd", 1.0), ("lambd_mode", "fixed"), ("gae_variant", "vanilla"),
)
LENGTH_ADAPTIVE_ALPHA = 1.5
HL_GAUSS_BINS = 51
# Execution requirements a single-island G1 smoke may allow with
# --rl-allow-unverified-mechanism (they are not registered mechanisms).
EXECUTION_ALLOWANCES = frozenset({"execution:critic"})
# Estimators whose advantage drops a reward-side KL (Miles loss_hub/advantages.py).
REWARD_KL_DROPPING_ESTIMATORS = frozenset({"grpo", "gspo"})
# Estimators that define a sequence-level ratio: clip range must be explicit.
SEQUENCE_RATIO_ESTIMATORS = {"gspo"}
# Upstream Miles refuses these estimators without --normalize-advantages.
WHITEN_REQUIRED_ESTIMATORS = frozenset({"reinforce_plus_plus", "reinforce_plus_plus_baseline"})
LOSS_VARIANTS = ["policy_loss", "custom_loss"]
LOSS_AGGREGATIONS = ["default", "token", "constant"]
KL_PLACEMENTS = ("none", "reward", "loss")
KL_ESTIMATORS = ("k1", "k2", "k3", "low_var_kl")
CORRECTION_METHODS = ["none", "tis", "opsm", "custom"]
DYNAMIC_SAMPLING_FILTERS = {BOUNDED_NONZERO_STD_FILTER, STOCK_NONZERO_STD_FILTER}
# Plugins may only live in yeto or in Miles' own (built-in) namespace (D7).
PLUGIN_NAMESPACES = ["yeto.", "miles."]

# Mechanism dimensions (design D4). ``EngineCapabilities`` has one declared
# set per dimension; ``required_mechanisms`` yields (dimension, name) pairs.
MECHANISM_DIMENSIONS = (
    "advantage_estimators",
    "losses",
    "loss_aggregations",
    "kl_placements",
    "corrections",
    "reward_postprocessors",
    "dynamic_sampling_filters",
    "features",
)


def valid_masked_fraction(value: Any) -> float | None:
    """A reported masked fraction, or None unless a real number in [0, 1]."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and 0.0 <= value <= 1.0 else None


# Settings an estimator requires (by the rejection matrix or upstream Miles)
# are part of that estimator's mechanism, not separately declared features:
# GSPO's explicit clip range (sequence_ratio_without_clip) and the rpp
# family's advantage normalization (rpp_requires_whiten). Decision: main
# agent, rl-infra-spec alignment §7b (may be overridden by the user).
ESTIMATOR_COMPANIONS: dict[str, frozenset[tuple[str, str]]] = {
    "gspo": frozenset({("features", "eps_clip"), ("features", "clip_higher")}),
    "reinforce_plus_plus": frozenset({("features", "whiten_advantages")}),
    "reinforce_plus_plus_baseline": frozenset({("features", "whiten_advantages")}),
}


# Same principle for corrections: mechanisms that make Miles set use_tis
# already produce the mismatch metrics (losses.py:233/386: get_mismatch_metrics
# or use_tis), so --get-mismatch-metrics changes nothing there and is claimed
# by them. Main-agent decision, alignment §7b (may be overridden by the user).
CORRECTION_COMPANIONS: dict[tuple[str, str], frozenset[tuple[str, str]]] = {
    ("corrections", name): frozenset({("features", "mismatch_metrics")})
    for name in ("tis", "icepop", "mis_mask", "mismatch_observe")
}


class AlgorithmSpecError(ValueError):
    """An algorithm description is malformed or unsupported."""


class PluginIdentityError(AlgorithmSpecError):
    """A plugin cannot be imported or its source hash does not match."""


# --------------------------------------------------------------------------
# field validation helpers (every error names the field)
# --------------------------------------------------------------------------


def _number(path: str, value: Any, *, low: float | None = None, low_open: bool = False,
            optional: bool = True) -> float | None:
    if value is None:
        if optional:
            return None
        raise AlgorithmSpecError(f"{path} is required")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AlgorithmSpecError(f"{path} must be a number, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise AlgorithmSpecError(f"{path} must be finite, got {value!r}")
    if low is not None and (value <= low if low_open else value < low):
        bound = f"> {low}" if low_open else f">= {low}"
        raise AlgorithmSpecError(f"{path} must be {bound}, got {value!r}")
    return value


def _integer(path: str, value: Any, *, low: int = 0, optional: bool = True) -> int | None:
    if value is None:
        if optional:
            return None
        raise AlgorithmSpecError(f"{path} is required")
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int):
        raise AlgorithmSpecError(f"{path} must be an integer, got {value!r}")
    if value < low:
        raise AlgorithmSpecError(f"{path} must be >= {low}, got {value!r}")
    return value


def _boolean(path: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise AlgorithmSpecError(f"{path} must be a boolean, got {value!r}")
    return value


def _choice(path: str, value: Any, allowed: Iterable[str]) -> str:
    allowed = list(allowed)
    if not isinstance(value, str) or value.lower() not in allowed:
        raise AlgorithmSpecError(f"{path} must be one of {sorted(allowed)}, got {value!r}")
    return value.lower()


def _plugin(path: str, value: Any) -> "PluginRef | None":
    if value is None or isinstance(value, PluginRef):
        return value
    if isinstance(value, Mapping):
        return PluginRef.from_dict(value, field_path=path)
    raise AlgorithmSpecError(f"{path} must be a plugin reference {{path, sha256}}, got {value!r}")


# --------------------------------------------------------------------------
# PluginRef (design D7)
# --------------------------------------------------------------------------


def plugin_module_file(path: str) -> str:
    """Source file of the module owning dotted callable ``path`` (no import)."""

    module, _, _name = path.rpartition(".")
    if not module:
        raise PluginIdentityError(f"plugin {path!r} is not a dotted 'module.callable' path")
    try:
        found = importlib.util.find_spec(module)
    except (ImportError, ValueError) as exc:
        raise PluginIdentityError(f"plugin {path!r}: module {module!r} not importable: {exc}")
    if found is None or not found.origin or not found.has_location:
        raise PluginIdentityError(f"plugin {path!r}: module {module!r} not importable")
    return found.origin


def plugin_source_sha256(path: str) -> str:
    with open(plugin_module_file(path), "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


@dataclass(frozen=True)
class PluginRef:
    """A plugin callable identified by dotted path + its module source SHA256."""

    path: str
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path or " " in self.path:
            raise AlgorithmSpecError(f"plugin path must be a dotted name, got {self.path!r}")
        if not any(self.path.startswith(prefix) for prefix in PLUGIN_NAMESPACES):
            raise AlgorithmSpecError(
                f"plugin {self.path!r} is outside the allowed namespaces "
                f"{sorted(PLUGIN_NAMESPACES)}"
            )
        digest = self.sha256
        if not isinstance(digest, str) or len(digest) != 64 or any(
            c not in "0123456789abcdef" for c in digest.lower()
        ):
            raise AlgorithmSpecError(f"plugin {self.path!r}: sha256 must be 64 hex chars")
        object.__setattr__(self, "sha256", digest.lower())

    @classmethod
    def from_path(cls, path: str) -> "PluginRef":
        """Pin ``path`` to the source currently on ``sys.path``."""

        if not any(path.startswith(prefix) for prefix in PLUGIN_NAMESPACES):
            raise AlgorithmSpecError(
                f"plugin {path!r} is outside the allowed namespaces {sorted(PLUGIN_NAMESPACES)}"
            )
        return cls(path=path, sha256=plugin_source_sha256(path))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, field_path: str = "plugin") -> "PluginRef":
        if not isinstance(payload, Mapping) or set(payload) != {"path", "sha256"}:
            raise AlgorithmSpecError(f"{field_path} must be an object with exactly path and sha256")
        return cls(path=payload["path"], sha256=payload["sha256"])

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "sha256": self.sha256}

    def verify(self, *, import_callable: bool = True) -> None:
        """Startup check: importable and the source hash matches (else reject)."""

        actual = plugin_source_sha256(self.path)
        if actual != self.sha256:
            raise PluginIdentityError(
                f"plugin {self.path!r} source hash mismatch: spec {self.sha256}, "
                f"runtime {actual}"
            )
        if import_callable:
            module, _, name = self.path.rpartition(".")
            try:
                loaded = importlib.import_module(module)
            except Exception as exc:  # noqa: BLE001 - any import failure rejects
                raise PluginIdentityError(f"plugin {self.path!r}: import failed: {exc}") from exc
            if not callable(getattr(loaded, name, None)):
                raise PluginIdentityError(f"plugin {self.path!r}: {name!r} is not a callable")


# --------------------------------------------------------------------------
# extension registry (follow-up changes: new file + one registration line)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldDef:
    """A registered extension field of one spec group."""

    group: str
    name: str
    default: Any
    parse: Callable[[str, Any], Any]  # (field path, raw) -> normalized value
    to_json: Callable[[Any], Any] = lambda value: value
    # When it returns True for the group, the value is kept in the canonical
    # JSON even if it equals the default (a semantic choice that must be
    # explicit in the identity whenever it applies, e.g. OPSM's pi_old source).
    always_emit: Callable[[Any], bool] | None = None


@dataclass(frozen=True)
class MechanismDef:
    """A named mechanism: how to detect it on a spec and what it implies."""

    dimension: str
    name: str
    detect: Callable[["AlgorithmSpec"], bool]
    # Declares that it can legitimately mask every token/sequence of a round.
    masks_tokens: bool = False
    requires_binary_reward: bool = False


_FIELDS: dict[str, dict[str, FieldDef]] = {}
_MECHANISMS: dict[tuple[str, str], MechanismDef] = {}
_REJECTIONS: dict[str, Callable[["AlgorithmSpec"], str | None]] = {}
# correction.function paths that have their own mechanism name (see
# register_named_correction_function); they no longer require 'custom'.
# path -> the corrections mechanisms that claim it (the path's own detectors)
NAMED_CORRECTION_FUNCTIONS: dict[str, frozenset[str]] = {}


def register_field(group: str, name: str, *, default: Any,
                   parse: Callable[[str, Any], Any],
                   to_json: Callable[[Any], Any] | None = None,
                   always_emit: Callable[[Any], bool] | None = None) -> FieldDef:
    if group not in _GROUPS:
        raise ValueError(f"unknown spec group {group!r}; one of {sorted(_GROUPS)}")
    core = {f.name for f in fields(_GROUPS[group])}
    if name in core or name in _FIELDS.get(group, {}):
        raise ValueError(f"field {group}.{name} already exists")
    try:
        hash(default)
    except TypeError as exc:
        raise ValueError(f"field {group}.{name}: default must be hashable") from exc
    definition = FieldDef(group, name, default, parse, to_json or (lambda value: value),
                          always_emit)
    _FIELDS.setdefault(group, {})[name] = definition
    return definition


def register_mechanism(dimension: str, name: str, detect: Callable[["AlgorithmSpec"], bool], *,
                       masks_tokens: bool = False,
                       requires_binary_reward: bool = False) -> MechanismDef:
    if dimension not in MECHANISM_DIMENSIONS:
        raise ValueError(f"unknown mechanism dimension {dimension!r}")
    key = (dimension, name)
    if key in _MECHANISMS:
        raise ValueError(f"mechanism {dimension}:{name} already registered")
    definition = MechanismDef(dimension, name, detect, masks_tokens, requires_binary_reward)
    _MECHANISMS[key] = definition
    return definition


_PIPELINE_PLUGIN_MODULES: set[str] = set()


def register_pipeline_plugin_module(module: str) -> None:
    """Declare ``module`` an extension-owned plugin module.

    A ``spec.plugins`` entry whose callable lives in such a module (or in a
    module listed in ``yeto.rl.algos.EXTENSION_MODULES``) still enters the
    hash and is re-hashed at startup, but does not require the
    ``features:plugins`` mechanism.
    """

    if not module.startswith("yeto."):
        raise ValueError(f"pipeline plugin module {module!r} must be in the yeto namespace")
    _PIPELINE_PLUGIN_MODULES.add(module)


def _owned_plugin(path: str) -> bool:
    load_extensions()
    from yeto.rl.algos import EXTENSION_MODULES

    module = path.rpartition(".")[0]
    return module in _PIPELINE_PLUGIN_MODULES or module in EXTENSION_MODULES


def register_named_correction_function(path: str, *, mechanisms: Iterable[str]) -> None:
    """``path`` is declared by its own correction mechanisms, not 'custom'.

    The exemption from ``corrections:custom`` applies only while one of the
    path's OWN mechanisms (``mechanisms``, corrections dimension) detects the
    spec, so a named path can never escape the capability check through
    another mechanism's detector; its source identity stays covered by the
    PluginRef hash.
    """

    mechanisms = frozenset(mechanisms)
    if not mechanisms:
        raise ValueError(f"named correction function {path!r} needs its mechanism name(s)")

    if not any(path.startswith(prefix) for prefix in PLUGIN_NAMESPACES):
        raise ValueError(
            f"named correction function {path!r} must be in {sorted(PLUGIN_NAMESPACES)} "
            "(Miles built-ins such as icepop_function are named too; the source "
            "hash of the PluginRef still pins them)"
        )
    NAMED_CORRECTION_FUNCTIONS[path] = NAMED_CORRECTION_FUNCTIONS.get(path, frozenset()) | mechanisms


# reducer path -> (the "dimension:name" mechanisms that claim it, pinned
# source sha256 or None)
NAMED_REDUCERS: dict[str, tuple[frozenset[str], str | None]] = {}


def register_named_reducer(path: str, *, mechanisms: Iterable[str],
                           sha256: str | None = None) -> None:
    """``path`` (a pg_loss reducer) is claimed by its own mechanisms.

    While one of ``mechanisms`` detects a spec using this reducer -- and, when
    ``sha256`` is given, the spec's PluginRef pins exactly that source (the
    one the declaration's evidence ran) -- the generic
    ``features:custom_pg_loss_reducer`` is not required. Any other reducer, or
    another source of this one, still requires it.
    """

    mechanisms = frozenset(mechanisms)
    if not mechanisms or any(":" not in m for m in mechanisms):
        raise ValueError(f"named reducer {path!r} needs 'dimension:name' mechanism(s)")
    if not any(path.startswith(prefix) for prefix in PLUGIN_NAMESPACES):
        raise ValueError(f"named reducer {path!r} must be in {sorted(PLUGIN_NAMESPACES)}")
    old, pinned = NAMED_REDUCERS.get(path, (frozenset(), None))
    if pinned is not None and sha256 is not None and pinned != sha256:
        raise ValueError(f"named reducer {path!r} already pinned to {pinned}")
    NAMED_REDUCERS[path] = (old | mechanisms, sha256 or pinned)


def _named_reducer_claimed(spec: "AlgorithmSpec") -> bool:
    owners, pinned = NAMED_REDUCERS.get(spec.loss.reducer.path, (frozenset(), None))
    if pinned is not None and spec.loss.reducer.sha256 != pinned:
        return False
    return any(
        f"{m.dimension}:{m.name}" in owners and m.detect(spec) for m in registered_mechanisms()
    )


def register_rejection(name: str, check: Callable[["AlgorithmSpec"], str | None]) -> None:
    """``check(spec)`` returns a problem (with the viable alternative) or None."""

    if name in _REJECTIONS:
        raise ValueError(f"rejection rule {name!r} already registered")
    _REJECTIONS[name] = check


_RUNTIME_ATTRS: dict[str, Callable[["AlgorithmSpec"], Mapping[str, Any]]] = {}
_LAUNCH_CHECKS: dict[str, Callable[["AlgorithmSpec", Mapping[str, Any]], Iterable[str]]] = {}
_ISLAND_CHECKS: dict[str, Callable[["AlgorithmSpec", Mapping[str, Any]], Iterable[str]]] = {}
# name -> (mechanism "dimension:name", rule)
_GRADIENT_RULES: dict[str, tuple[str, Callable[["AlgorithmSpec", Any, Any], "bool | None"]]] = {}


def register_runtime_attrs(name: str, derive: Callable[["AlgorithmSpec"], Mapping[str, Any]]) -> None:
    """``derive(spec)`` -> extra Miles namespace attributes (``{}`` when unused).

    Merged into :meth:`AlgorithmSpec.to_legacy_runtime_attrs`; a default spec
    must derive ``{}`` so the R0 attributes stay unchanged.
    """

    if name in _RUNTIME_ATTRS:
        raise ValueError(f"runtime attrs {name!r} already registered")
    _RUNTIME_ATTRS[name] = derive


def register_launch_check(
    name: str, check: Callable[["AlgorithmSpec", Mapping[str, Any]], Iterable[str]]
) -> None:
    """``check(spec, run_values)`` -> problems needing the run configuration.

    ``run_values`` (adapter-provided): ``rollout_batch_size``,
    ``rollout_max_response_len``, ``context_parallel_size``, ``multi_lora``.
    """

    if name in _LAUNCH_CHECKS:
        raise ValueError(f"launch check {name!r} already registered")
    _LAUNCH_CHECKS[name] = check


def register_island_check(
    name: str, check: Callable[["AlgorithmSpec", Mapping[str, Any]], Iterable[str]]
) -> None:
    """``check(spec, island)`` -> problems against this island's identity.

    ``island``: ``base_model_revision`` (learner, before joining outer sync).
    """

    if name in _ISLAND_CHECKS:
        raise ValueError(f"island check {name!r} already registered")
    _ISLAND_CHECKS[name] = check


def register_gradient_rule(
    name: str,
    rule: Callable[["AlgorithmSpec", Any, Any], "bool | None"],
    *,
    mechanism: str,
) -> None:
    """``rule(spec, batch_summary, step_metrics)``: ``False`` lifts the gradient
    expectation for this round, ``True`` requires one where the R0 rule (some
    group with non-zero reward std) expects none, ``None`` abstains.

    The rule is bound to ``mechanism`` (``"dimension:name"``) and is consulted
    only when the spec requires that mechanism, so a registered rule can never
    change the default GRPO judgement. Tightening (``True``) is allowed by
    design D6 (each mechanism's change supplies its rule) and is the safer
    direction; a non-finite grad norm is checked by the driver regardless."""

    if name in _GRADIENT_RULES:
        raise ValueError(f"gradient rule {name!r} already registered")
    dimension, sep, mech = mechanism.partition(":")
    if not sep or dimension not in MECHANISM_DIMENSIONS or not mech:
        raise ValueError(f"gradient rule {name!r}: mechanism must be 'dimension:name'")
    _GRADIENT_RULES[name] = (mechanism, rule)


def launch_problems(spec: "AlgorithmSpec", run_values: Mapping[str, Any]) -> list[str]:
    load_extensions()
    return [f"[{name}] {p}" for name, check in _LAUNCH_CHECKS.items() for p in check(spec, run_values)]


def island_problems(spec: "AlgorithmSpec", island: Mapping[str, Any]) -> list[str]:
    load_extensions()
    return [f"[{name}] {p}" for name, check in _ISLAND_CHECKS.items() for p in check(spec, island)]


def unregister(*, field: tuple[str, str] | None = None,
               mechanism: tuple[str, str] | None = None,
               rejection: str | None = None,
               runtime_attrs: str | None = None,
               launch_check: str | None = None,
               island_check: str | None = None,
               gradient_rule: str | None = None) -> None:
    """Test helper: undo a registration."""

    if field is not None:
        _FIELDS.get(field[0], {}).pop(field[1], None)
    if mechanism is not None:
        _MECHANISMS.pop(mechanism, None)
    for registry, name in (
        (_REJECTIONS, rejection), (_RUNTIME_ATTRS, runtime_attrs),
        (_LAUNCH_CHECKS, launch_check), (_ISLAND_CHECKS, island_check),
        (_GRADIENT_RULES, gradient_rule),
    ):
        if name is not None:
            registry.pop(name, None)


def registered_mechanisms() -> tuple[MechanismDef, ...]:
    load_extensions()
    return tuple(_MECHANISMS.values())


def mechanism_names() -> frozenset[str]:
    """Qualified ``dimension:name`` of every registered mechanism."""

    return frozenset(f"{m.dimension}:{m.name}" for m in registered_mechanisms())


def allowance_names() -> frozenset[str]:
    """Names --rl-allow-unverified-mechanism accepts: mechanisms plus
    :data:`EXECUTION_ALLOWANCES` (``execution:critic``)."""

    return mechanism_names() | EXECUTION_ALLOWANCES


_EXTENSIONS_LOADED = False


def load_extensions() -> None:
    """Import every follow-up change's registration module once."""

    global _EXTENSIONS_LOADED
    if _EXTENSIONS_LOADED:
        return
    from yeto.rl.algos import EXTENSION_MODULES

    for module in EXTENSION_MODULES:
        importlib.import_module(module)  # a failure leaves the flag unset
    _EXTENSIONS_LOADED = True


# --------------------------------------------------------------------------
# groups
# --------------------------------------------------------------------------


class _Group:
    """Base of the frozen group dataclasses: typed core fields + ``ext``.

    ``ext`` holds registered extension fields as a sorted tuple of
    ``(name, value)``; unset extension fields read as their default.
    """

    GROUP = ""
    ext: tuple[tuple[str, Any], ...]

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        definitions = _FIELDS.get(type(self).GROUP, {})
        if name in definitions:
            for key, value in object.__getattribute__(self, "ext"):
                if key == name:
                    return value
            return definitions[name].default
        raise AttributeError(f"{type(self).__name__} has no field {name!r}")

    def _normalize_ext(self) -> None:
        definitions = _FIELDS.get(self.GROUP, {})
        raw = dict(self.ext or ())
        unknown = sorted(set(raw) - set(definitions))
        if unknown:
            raise AlgorithmSpecError(
                f"unknown {self.GROUP} fields: {[f'{self.GROUP}.{u}' for u in unknown]}"
            )
        normalized = []
        for name in sorted(set(raw) | set(definitions)):
            definition = definitions[name]
            value = definition.parse(f"{self.GROUP}.{name}", raw[name]) if name in raw \
                else definition.default
            try:
                hash(value)
            except TypeError:
                raise AlgorithmSpecError(
                    f"{self.GROUP}.{name}: the registered parser returned an unhashable "
                    f"{type(value).__name__} (return tuples / frozen values)"
                ) from None
            forced = definition.always_emit is not None and definition.always_emit(self)
            if value != definition.default or forced:
                normalized.append((name, value))
        object.__setattr__(self, "ext", tuple(normalized))

    def with_ext(self, **values: Any):
        from dataclasses import replace

        merged = dict(self.ext)
        merged.update(values)
        return replace(self, ext=tuple(merged.items()))

    def core_items(self) -> list[tuple[str, Any]]:
        return [(f.name, getattr(self, f.name)) for f in fields(self) if f.name != "ext"]

    def is_default(self) -> bool:
        return self == type(self)()

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, value in self.core_items():
            out[name] = value.to_dict() if isinstance(value, PluginRef) else value
        definitions = _FIELDS.get(self.GROUP, {})
        for name, value in self.ext:
            out[name] = definitions[name].to_json(value)
        return out

    @classmethod
    def from_dict(cls, payload: Any):
        if not isinstance(payload, Mapping):
            raise AlgorithmSpecError(f"{cls.GROUP} must be an object")
        core = {f.name for f in fields(cls)} - {"ext"}
        kwargs = {k: v for k, v in payload.items() if k in core}
        ext = tuple((k, v) for k, v in payload.items() if k not in core)
        return cls(**kwargs, ext=ext)


@dataclass(frozen=True)
class AdvantageSpec(_Group):
    GROUP = "advantage"
    estimator: str = "grpo"
    std_normalization: bool = True  # --disable-grpo-std-normalization when False
    rewards_normalization: bool = True  # --disable-rewards-normalization when False
    whiten: bool = False  # --normalize-advantages (island-local DP all-reduce)
    reward_postprocess: PluginRef | None = None  # --custom-reward-post-process-path
    reward_binary: bool = False  # the reward function is declared binary {0,1}
    ext: tuple = ()

    def __post_init__(self) -> None:
        s = object.__setattr__
        s(self, "estimator", _choice("advantage.estimator", self.estimator, ADVANTAGE_ESTIMATORS))
        for name in ("std_normalization", "rewards_normalization", "whiten", "reward_binary"):
            _boolean(f"advantage.{name}", getattr(self, name))
        s(self, "reward_postprocess",
          _plugin("advantage.reward_postprocess", self.reward_postprocess))
        self._normalize_ext()


@dataclass(frozen=True)
class LossSpec(_Group):
    GROUP = "loss"
    variant: str = "policy_loss"  # --loss-type
    eps_clip: float | None = None  # None: engine default, not emitted
    eps_clip_high: float | None = None
    eps_clip_c: float | None = None  # dual-clip
    aggregation: str = "default"  # default | token (--calculate-per-token-loss) | constant
    reducer: PluginRef | None = None  # --custom-pg-loss-reducer-function-path
    custom_loss: PluginRef | None = None  # --custom-loss-function-path
    ext: tuple = ()

    def __post_init__(self) -> None:
        s = object.__setattr__
        s(self, "variant", _choice("loss.variant", self.variant, LOSS_VARIANTS))
        s(self, "eps_clip", _number("loss.eps_clip", self.eps_clip, low=0.0, low_open=True))
        s(self, "eps_clip_high",
          _number("loss.eps_clip_high", self.eps_clip_high, low=0.0, low_open=True))
        s(self, "eps_clip_c", _number("loss.eps_clip_c", self.eps_clip_c, low=1.0, low_open=True))
        s(self, "aggregation", _choice("loss.aggregation", self.aggregation, LOSS_AGGREGATIONS))
        s(self, "reducer", _plugin("loss.reducer", self.reducer))
        s(self, "custom_loss", _plugin("loss.custom_loss", self.custom_loss))
        if (self.variant == "custom_loss") != (self.custom_loss is not None):
            raise AlgorithmSpecError(
                "loss.custom_loss is required exactly when loss.variant is 'custom_loss'"
            )
        self._normalize_ext()


@dataclass(frozen=True)
class KlSpec(_Group):
    GROUP = "kl"
    placement: str = "none"  # none | reward (--kl-coef) | loss (--use-kl-loss ...)
    coef: float | None = None
    estimator: str | None = None  # --kl-loss-type (loss placement only)
    unbiased: bool = False  # --use-unbiased-kl (loss placement only)
    ext: tuple = ()

    def __post_init__(self) -> None:
        s = object.__setattr__
        s(self, "placement", _choice("kl.placement", self.placement, KL_PLACEMENTS))
        s(self, "coef", _number("kl.coef", self.coef, low=0.0))
        _boolean("kl.unbiased", self.unbiased)
        if self.estimator is not None:
            s(self, "estimator", _choice("kl.estimator", self.estimator, KL_ESTIMATORS))
        if self.placement == "none":
            if self.coef is not None or self.estimator is not None or self.unbiased:
                raise AlgorithmSpecError(
                    "kl.coef/kl.estimator/kl.unbiased require kl.placement reward or loss"
                )
        elif self.coef is None:
            raise AlgorithmSpecError(f"kl.coef is required for kl.placement={self.placement!r}")
        if self.placement == "reward" and (self.estimator is not None or self.unbiased):
            raise AlgorithmSpecError(
                "kl.estimator/kl.unbiased apply to kl.placement='loss' only"
            )
        if self.placement == "loss" and self.estimator is None:
            raise AlgorithmSpecError(
                f"kl.estimator is required for kl.placement='loss' (one of {list(KL_ESTIMATORS)})"
            )
        self._normalize_ext()


@dataclass(frozen=True)
class CorrectionSpec(_Group):
    GROUP = "correction"
    method: str = "none"  # none | tis | opsm | custom (custom tis function)
    tis_clip: float | None = None
    tis_clip_low: float | None = None
    opsm_delta: float | None = None
    function: PluginRef | None = None  # --custom-tis-function-path
    use_rollout_logprobs: bool = False
    mismatch_metrics: bool = False  # --get-mismatch-metrics
    ext: tuple = ()

    def __post_init__(self) -> None:
        s = object.__setattr__
        s(self, "method", _choice("correction.method", self.method, CORRECTION_METHODS))
        s(self, "tis_clip", _number("correction.tis_clip", self.tis_clip, low=0.0, low_open=True))
        s(self, "tis_clip_low", _number("correction.tis_clip_low", self.tis_clip_low, low=0.0))
        s(self, "opsm_delta",
          _number("correction.opsm_delta", self.opsm_delta, low=0.0, low_open=True))
        s(self, "function", _plugin("correction.function", self.function))
        _boolean("correction.use_rollout_logprobs", self.use_rollout_logprobs)
        _boolean("correction.mismatch_metrics", self.mismatch_metrics)
        if self.method == "custom" and self.function is None:
            raise AlgorithmSpecError("correction.function is required for correction.method='custom'")
        if self.method != "custom" and self.function is not None:
            raise AlgorithmSpecError("correction.function requires correction.method='custom'")
        if self.method not in ("tis", "custom") and (
            self.tis_clip is not None or self.tis_clip_low is not None
        ):
            raise AlgorithmSpecError(
                "correction.tis_clip/tis_clip_low require correction.method tis or custom"
            )
        if self.method == "none" and self.opsm_delta is not None:
            raise AlgorithmSpecError(
                "correction.opsm_delta requires correction.method 'opsm' (OPSM alone) or "
                "'tis'/'custom' (OPSM combined with an importance-weighting correction)"
            )
        if self.method == "opsm" and self.opsm_delta is None:
            raise AlgorithmSpecError("correction.opsm_delta is required for correction.method='opsm'")
        if self.method == "tis" and (self.tis_clip is None or self.tis_clip_low is None):
            raise AlgorithmSpecError(
                "correction.tis_clip and correction.tis_clip_low are required for "
                "correction.method='tis' (no engine default is inherited)"
            )
        if self.mismatch_metrics and self.method not in ("tis", "custom"):
            raise AlgorithmSpecError(
                "correction.mismatch_metrics needs a tis/custom correction function"
            )
        self._normalize_ext()


@dataclass(frozen=True)
class SamplingSpec(_Group):
    GROUP = "sampling"
    filter: str | None = None  # --dynamic-sampling-filter-path
    max_replacements: int | None = None  # runtime attr of the bounded filter
    over_sampling_batch_size: int | None = None
    overlong_filter: bool = False
    ext: tuple = ()

    def __post_init__(self) -> None:
        s = object.__setattr__
        if self.filter is not None and self.filter not in DYNAMIC_SAMPLING_FILTERS:
            raise AlgorithmSpecError(
                f"unsupported dynamic sampling filter {self.filter!r} (sampling.filter; "
                f"one of {sorted(DYNAMIC_SAMPLING_FILTERS)})"
            )
        s(self, "max_replacements", _integer("sampling.max_replacements", self.max_replacements))
        if self.max_replacements is not None and self.filter is None:
            raise AlgorithmSpecError(
                "sampling.max_replacements requires a dynamic sampling filter"
            )
        s(self, "over_sampling_batch_size",
          _integer("sampling.over_sampling_batch_size", self.over_sampling_batch_size, low=1))
        _boolean("sampling.overlong_filter", self.overlong_filter)
        self._normalize_ext()


@dataclass(frozen=True)
class ExecutionSpec(_Group):
    GROUP = "execution"
    needs_critic: bool = False
    # Largest policy age the algorithm tolerates. Fixed at 0 for every
    # algorithm of this change family (rl-infra-spec alignment A1/A6); a
    # non-zero value is expressible but rejected by the rejection matrix.
    max_policy_staleness: int = 0
    group_required: bool = True
    needs_rollout_logprobs: bool = False
    ext: tuple = ()

    def __post_init__(self) -> None:
        for name in ("needs_critic", "group_required", "needs_rollout_logprobs"):
            _boolean(f"execution.{name}", getattr(self, name))
        object.__setattr__(
            self, "max_policy_staleness",
            _integer("execution.max_policy_staleness", self.max_policy_staleness, optional=False),
        )
        self._normalize_ext()


def _optional_choice(path: str, value: Any, allowed: Iterable[str]) -> str | None:
    return None if value is None else _choice(path, value, allowed)


@dataclass(frozen=True)
class CriticSpec(_Group):
    """rl-algo-critic-family D1/D9: the critic role of a critic algorithm.

    Every field defaults to None ("not given"). A spec with
    ``execution.needs_critic`` gets the concrete defaults filled in
    (:meth:`with_defaults`), so the group enters the canonical JSON only for
    critic algorithms; on any other algorithm a non-None field is rejected.
    ``critic_lr=None`` on a critic spec means "inherit the actor lr" (Miles
    arguments.py:3609-3610). ``param_mode='lora'`` and the ``lora_*`` fields
    are the reserved D9 interface (refused in this round).
    """

    GROUP = "critic"
    value_clip: float | None = None
    critic_lr: float | None = None
    critic_lr_warmup: int | None = None
    critic_updates_per_step: int | None = None
    value_loss: str | None = None
    hl_gauss_bins: int | None = None
    init: str | None = None
    load: str | None = None
    warmup_steps: int | None = None
    param_mode: str | None = None
    lora_rank: int | None = None
    lora_alpha: float | None = None
    lora_target_modules: tuple[str, ...] | None = None
    ext: tuple = ()

    def __post_init__(self) -> None:
        s = object.__setattr__
        s(self, "value_clip", _number("critic.value_clip", self.value_clip, low=0.0, low_open=True))
        s(self, "critic_lr", _number("critic.critic_lr", self.critic_lr, low=0.0, low_open=True))
        s(self, "lora_alpha", _number("critic.lora_alpha", self.lora_alpha, low=0.0, low_open=True))
        s(self, "critic_lr_warmup", _integer("critic.critic_lr_warmup", self.critic_lr_warmup))
        s(self, "critic_updates_per_step",
          _integer("critic.critic_updates_per_step", self.critic_updates_per_step, low=1))
        s(self, "hl_gauss_bins", _integer("critic.hl_gauss_bins", self.hl_gauss_bins, low=2))
        s(self, "warmup_steps", _integer("critic.warmup_steps", self.warmup_steps))
        s(self, "lora_rank", _integer("critic.lora_rank", self.lora_rank, low=1))
        s(self, "value_loss", _optional_choice("critic.value_loss", self.value_loss,
                                               CRITIC_VALUE_LOSSES))
        s(self, "init", _optional_choice("critic.init", self.init, CRITIC_INITS))
        s(self, "param_mode", _optional_choice("critic.param_mode", self.param_mode,
                                               CRITIC_PARAM_MODES))
        if self.load is not None and (not isinstance(self.load, str) or not self.load):
            raise AlgorithmSpecError("critic.load must be a non-empty path string")
        modules = self.lora_target_modules
        if modules is not None:
            if isinstance(modules, str) or not all(isinstance(m, str) and m for m in modules):
                raise AlgorithmSpecError("critic.lora_target_modules must be a list of names")
            s(self, "lora_target_modules", tuple(modules))
        self._normalize_ext()

    def with_defaults(self) -> "CriticSpec":
        from dataclasses import replace

        def pick(value, default):
            return default if value is None else value

        value_loss = pick(self.value_loss, "mse")
        return replace(
            self,
            value_clip=pick(self.value_clip, 0.2),
            critic_updates_per_step=pick(self.critic_updates_per_step, 1),
            value_loss=value_loss,
            hl_gauss_bins=(pick(self.hl_gauss_bins, HL_GAUSS_BINS)
                           if value_loss == "hl_gauss" else self.hl_gauss_bins),
            init=pick(self.init, "load" if self.load is not None else "copy_actor_backbone"),
            warmup_steps=pick(self.warmup_steps, 0),
            param_mode=pick(self.param_mode, "full"),
        )


_GROUPS: dict[str, type] = {
    "advantage": AdvantageSpec,
    "loss": LossSpec,
    "kl": KlSpec,
    "correction": CorrectionSpec,
    "sampling": SamplingSpec,
    "execution": ExecutionSpec,
    "critic": CriticSpec,
}


def _critic_unit(path: str, value: Any) -> float | None:
    value = _number(path, value, low=0.0)
    if value is not None and value > 1.0:
        raise AlgorithmSpecError(f"{path} must be in [0, 1], got {value!r}")
    return value


def _critic_alpha(path: str, value: Any) -> float | None:
    return _number(path, value, low=0.0, low_open=True)


# rl-algo-critic-family D1: advantage-side critic fields (registered extension
# fields: absent from the canonical JSON while None, i.e. on every non-critic
# spec; a critic spec fills CRITIC_ADVANTAGE_DEFAULTS).
register_field("advantage", "lambd", default=None, parse=_critic_unit)
register_field("advantage", "lambd_mode", default=None,
               parse=lambda path, v: _optional_choice(path, v, LAMBD_MODES))
register_field("advantage", "alpha", default=None, parse=_critic_alpha)
register_field("advantage", "gae_variant", default=None,
               parse=lambda path, v: _optional_choice(path, v, GAE_VARIANTS))
# ``advantage.gamma`` is shared with REINFORCE++ and owned by
# ``yeto.rl.algos.seq_adv`` (default 1.0, not emitted); a critic spec uses it.
CRITIC_ADVANTAGE_FIELDS = ("lambd", "lambd_mode", "alpha", "gae_variant")

_V1_FIELDS = (
    "advantage_estimator",
    "loss",
    "kl_coef",
    "dynamic_sampling_filter",
    "dynamic_sampling_max_replacements",
)


# --------------------------------------------------------------------------
# AlgorithmSpec
# --------------------------------------------------------------------------


@dataclass(frozen=True, init=False)
class AlgorithmSpec:
    """Structured algorithm description (v2) with v1-compatible identity.

    Accepts either the v1 flat keyword arguments (``advantage_estimator``,
    ``loss`` as a string, ``kl_coef``, ``dynamic_sampling_filter``,
    ``dynamic_sampling_max_replacements``; R0 validation) or the v2 groups.
    """

    advantage: AdvantageSpec = field(default_factory=AdvantageSpec)
    loss: LossSpec = field(default_factory=LossSpec)
    kl: KlSpec = field(default_factory=KlSpec)
    correction: CorrectionSpec = field(default_factory=CorrectionSpec)
    sampling: SamplingSpec = field(default_factory=SamplingSpec)
    execution: ExecutionSpec = field(default_factory=ExecutionSpec)
    entropy_coef: float = 0.0
    plugins: tuple[PluginRef, ...] = ()
    critic: CriticSpec = field(default_factory=CriticSpec)

    def __init__(
        self,
        advantage: AdvantageSpec | Mapping | None = None,
        loss: LossSpec | Mapping | str | None = None,
        kl: KlSpec | Mapping | None = None,
        correction: CorrectionSpec | Mapping | None = None,
        sampling: SamplingSpec | Mapping | None = None,
        execution: ExecutionSpec | Mapping | None = None,
        entropy_coef: float = 0.0,
        plugins: Iterable[PluginRef | Mapping] = (),
        critic: CriticSpec | Mapping | None = None,
        *,
        advantage_estimator: str | None = None,
        kl_coef: float | None = None,
        dynamic_sampling_filter: str | None = None,
        dynamic_sampling_max_replacements: int | None = None,
    ) -> None:
        s = object.__setattr__
        v1_loss = loss if isinstance(loss, str) else None
        # ---- v1 keyword arguments keep R0 validation and messages ----------
        if advantage_estimator is not None:
            if advantage is not None:
                raise AlgorithmSpecError("give advantage_estimator (v1) or advantage (v2), not both")
            if advantage_estimator not in SUPPORTED_ADVANTAGE_ESTIMATORS:
                raise AlgorithmSpecError(
                    f"unsupported advantage estimator {advantage_estimator!r}; "
                    f"supported: {sorted(SUPPORTED_ADVANTAGE_ESTIMATORS)} (v1 field "
                    "advantage_estimator; other estimators use advantage.estimator)"
                )
            advantage = AdvantageSpec(estimator=advantage_estimator)
            if advantage_estimator in CRITIC_ESTIMATORS and execution is None:
                # v1 ppo (rl-algo-critic-family 2.2): Miles derives use_critic
                # from the estimator (arguments.py:3591); the spec says it.
                execution = ExecutionSpec(needs_critic=True)
        if v1_loss is not None:
            if v1_loss not in SUPPORTED_LOSSES:
                raise AlgorithmSpecError(
                    f"unsupported loss {v1_loss!r}; supported: {sorted(SUPPORTED_LOSSES)}"
                )
            loss = LossSpec(variant=v1_loss)
        if kl_coef is not None:
            if kl is not None:
                raise AlgorithmSpecError("give kl_coef (v1) or kl (v2), not both")
            if isinstance(kl_coef, bool) or not isinstance(kl_coef, (int, float)):
                raise AlgorithmSpecError("kl_coef must be a number")
            if not math.isfinite(kl_coef) or kl_coef < 0:
                raise AlgorithmSpecError("kl_coef must be finite and non-negative")
            kl = KlSpec(placement="reward", coef=float(kl_coef))  # design D5
        if dynamic_sampling_filter is not None or dynamic_sampling_max_replacements is not None:
            if sampling is not None:
                raise AlgorithmSpecError("give v1 dynamic_sampling_* or sampling (v2), not both")
            if (
                dynamic_sampling_filter is not None
                and dynamic_sampling_filter not in SUPPORTED_FILTERS
            ):
                raise AlgorithmSpecError(
                    f"unsupported dynamic sampling filter {dynamic_sampling_filter!r}"
                )
            limit = dynamic_sampling_max_replacements
            if limit is not None:
                if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
                    raise AlgorithmSpecError(
                        "dynamic_sampling_max_replacements must be a non-negative int"
                    )
                if dynamic_sampling_filter is None:
                    raise AlgorithmSpecError(
                        "dynamic_sampling_max_replacements requires a dynamic sampling filter"
                    )
            sampling = SamplingSpec(filter=dynamic_sampling_filter, max_replacements=limit)
        # ---- v2 groups ------------------------------------------------------
        for name, value in (
            ("advantage", advantage), ("loss", loss), ("kl", kl), ("correction", correction),
            ("sampling", sampling), ("execution", execution), ("critic", critic),
        ):
            group = _GROUPS[name]
            if value is None:
                value = group()
            elif isinstance(value, Mapping):
                value = group.from_dict(value)
            elif not isinstance(value, group):
                raise AlgorithmSpecError(f"{name} must be a {group.__name__} or an object")
            s(self, name, value)
        s(self, "entropy_coef", _number("entropy_coef", entropy_coef, optional=False))
        if isinstance(plugins, (str, bytes)) or isinstance(plugins, Mapping):
            raise AlgorithmSpecError("plugins must be a list of plugin references")
        refs = [_plugin(f"plugins[{i}]", p) for i, p in enumerate(plugins)]
        if len({r.path for r in refs}) != len(refs):
            raise AlgorithmSpecError("plugins lists the same path twice")
        s(self, "plugins", tuple(sorted(refs, key=lambda r: r.path)))
        if self.execution.needs_critic:
            load_extensions()  # advantage.gamma is an extension field (seq_adv)
            # rl-algo-critic-family D1: critic fields enter the identity only
            # for critic algorithms, always with explicit values.
            filled = {name: default for name, default in CRITIC_ADVANTAGE_DEFAULTS
                      if getattr(self.advantage, name) is None}
            mode = filled.get("lambd_mode", self.advantage.lambd_mode)
            if mode == "length_adaptive" and self.advantage.alpha is None:
                filled["alpha"] = LENGTH_ADAPTIVE_ALPHA
            if filled:
                s(self, "advantage", self.advantage.with_ext(**filled))
            s(self, "critic", self.critic.with_defaults())

    # -- v1 compatible accessors ------------------------------------------
    @property
    def advantage_estimator(self) -> str:
        return self.advantage.estimator

    @property
    def kl_coef(self) -> float | None:
        return self.kl.coef if self.kl.placement == "reward" else None

    @property
    def dynamic_sampling_filter(self) -> str | None:
        return self.sampling.filter

    @property
    def dynamic_sampling_max_replacements(self) -> int | None:
        return self.sampling.max_replacements

    # -- identity (design D2) -----------------------------------------------
    def is_v1_expressible(self) -> bool:
        return (
            self.advantage.estimator in SUPPORTED_ADVANTAGE_ESTIMATORS
            and self.advantage == AdvantageSpec(estimator=self.advantage.estimator)
            and self.loss.variant in SUPPORTED_LOSSES
            and self.loss == LossSpec(variant=self.loss.variant)
            and self.kl.placement in ("none", "reward")
            and not self.kl.ext
            and self.correction.is_default()
            and self.sampling == SamplingSpec(
                filter=self.sampling.filter, max_replacements=self.sampling.max_replacements
            )
            and (self.sampling.filter is None or self.sampling.filter in SUPPORTED_FILTERS)
            and self.execution.is_default()
            and self.critic.is_default()
            and self.entropy_coef == 0.0
            and not self.plugins
        )

    def to_dict(self) -> dict[str, Any]:
        if self.is_v1_expressible():
            return {
                "schema": ALGORITHM_SPEC_SCHEMA,
                "advantage_estimator": self.advantage.estimator,
                "dynamic_sampling_filter": self.sampling.filter,
                "dynamic_sampling_max_replacements": self.sampling.max_replacements,
                "kl_coef": self.kl_coef,
                "loss": self.loss.variant,
            }
        return self.structured_dict()

    def structured_dict(self) -> dict[str, Any]:
        """Full v2 form (also for v1-expressible specs); ``from_dict`` accepts it."""

        return {
            "schema": ALGORITHM_SPEC_SCHEMA_V2,
            **{name: getattr(self, name).to_dict() for name in _GROUPS if name != "critic"},
            # rl-algo-critic-family D1: absent unless the algorithm has a critic
            # (or a stray critic field the rejection matrix will refuse), so
            # every pre-existing v2 hash is unchanged.
            **({"critic": self.critic.to_dict()}
               if self.execution.needs_critic or not self.critic.is_default() else {}),
            "entropy_coef": self.entropy_coef,
            "plugins": [p.to_dict() for p in self.plugins],
        }

    def get_path(self, path: str) -> Any:
        """Value at dotted spec path (``loss.eps_clip``, ``entropy_coef``)."""

        head, _, tail = path.partition(".")
        value = getattr(self, head)
        return getattr(value, tail) if tail else value

    @staticmethod
    def default_at(path: str) -> Any:
        head, _, tail = path.partition(".")
        if not tail:
            return {"entropy_coef": 0.0, "plugins": ()}[head]
        group = _GROUPS[head]
        core = {f.name: f.default for f in fields(group)}
        if tail in core:
            return core[tail]
        return _FIELDS[head][tail].default

    def effective_default_at(self, path: str) -> Any:
        """:meth:`default_at`, except the critic defaults a critic spec fills in
        (rl-algo-critic-family D1): a filled value is not a user choice."""

        if self.execution.needs_critic:
            head, _, tail = path.partition(".")
            filled = AlgorithmSpec(advantage={"estimator": self.advantage.estimator},
                                   execution={"needs_critic": True})
            if head == "critic" or (head == "advantage" and tail in CRITIC_ADVANTAGE_FIELDS):
                return filled.get_path(path)
        return self.default_at(path)

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @property
    def schema(self) -> str:
        return ALGORITHM_SPEC_SCHEMA if self.is_v1_expressible() else ALGORITHM_SPEC_SCHEMA_V2

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AlgorithmSpec":
        if not isinstance(payload, Mapping):
            raise AlgorithmSpecError("algorithm spec must be an object")
        load_extensions()
        data = dict(payload)
        schema = data.pop("schema", ALGORITHM_SPEC_SCHEMA)
        if schema == ALGORITHM_SPEC_SCHEMA:
            unknown = sorted(set(data) - set(_V1_FIELDS))
            if unknown:
                raise AlgorithmSpecError(f"unknown algorithm spec fields: {unknown}")
            return cls(**data)
        if schema != ALGORITHM_SPEC_SCHEMA_V2:
            raise AlgorithmSpecError(f"unknown algorithm spec schema {schema!r}")
        known = set(_GROUPS) | {"entropy_coef", "plugins"}
        unknown = sorted(set(data) - known)
        if unknown:
            raise AlgorithmSpecError(f"unknown algorithm spec fields: {unknown}")
        for name in _GROUPS:
            if name in data and isinstance(data[name], (str, type(None))):
                raise AlgorithmSpecError(f"{name} must be an object in a v2 spec")
        return cls(**data)

    @classmethod
    def from_json_file(cls, path: str) -> "AlgorithmSpec":
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise AlgorithmSpecError(f"cannot read algorithm spec {path!r}: {exc}") from exc
        return cls.from_dict(payload)

    def replace(self, **changes: Any) -> "AlgorithmSpec":
        values = {name: getattr(self, name) for name in (*_GROUPS, "entropy_coef", "plugins")}
        values.update(changes)
        return AlgorithmSpec(**values)

    # -- plugins (design D7) -------------------------------------------------
    def referenced_plugins(self) -> tuple[PluginRef, ...]:
        refs = {p.path: p for p in self.plugins}
        for group in (self.advantage, self.loss, self.correction):
            for _, value in group.core_items():
                if isinstance(value, PluginRef):
                    refs.setdefault(value.path, value)
        return tuple(refs[k] for k in sorted(refs))

    def verify_plugins(self, *, import_callable: bool = True) -> None:
        for ref in self.referenced_plugins():
            ref.verify(import_callable=import_callable)

    # -- derived (design D4/D6) ----------------------------------------------
    def required_mechanisms(self) -> frozenset[tuple[str, str]]:
        """(dimension, mechanism) pairs the engine must declare.

        Settings an estimator mandates (:data:`ESTIMATOR_COMPANIONS`) are
        claimed by that estimator's mechanism in this combination only; the
        same setting under another estimator is its own mechanism.
        """

        required = {(m.dimension, m.name) for m in registered_mechanisms() if m.detect(self)}
        required -= ESTIMATOR_COMPANIONS.get(self.advantage.estimator, frozenset())
        for claimant, companions in CORRECTION_COMPANIONS.items():
            if claimant in required:
                required -= companions
        return frozenset(required)

    def _mechanism_defs(self) -> list[MechanismDef]:
        return [m for m in registered_mechanisms() if m.detect(self)]

    def rejections(self) -> list[str]:
        """Rejection matrix: combinations refused before any GPU process."""

        load_extensions()
        problems = []
        for name, check in _REJECTIONS.items():
            problem = check(self)
            if problem:
                problems.append(f"[{name}] {problem}")
        return problems

    def expects_gradient(self, batch_summary: Any, step_metrics: Any = None) -> bool:
        """Whether this round must produce a non-zero gradient.

        Default (every algorithm): some group has non-zero reward variance --
        exactly the R0 GRPO rule. A mechanism that can legitimately mask every
        token/sequence lifts the expectation only when the trainer reports
        ``masked_fraction == 1.0``; an unknown mask fraction (None) keeps the
        stricter R0 rule.
        """

        return self.gradient_expectation(batch_summary, step_metrics)[0]

    def gradient_expectation(
        self, batch_summary: Any, step_metrics: Any = None
    ) -> tuple[bool, str | None]:
        """``(expects_gradient, source)``; ``source`` names the deciding rule.

        ``source`` is None when the R0 rule decides; otherwise
        ``"gradient_rule:<name> (<mechanism>)"`` or ``"masked:<mechanism>"``,
        for the driver's event.

        Precedence (fixed): a tightening rule wins. If any rule of a required
        mechanism returns a truthy verdict the round expects a gradient
        (source ``"tightened:gradient_rule:..."``); only otherwise can a
        falsy verdict (``False``, ``np.bool_(False)``; ``None`` abstains) or
        a full mask relax the R0 rule.
        """

        groups = getattr(batch_summary, "groups", batch_summary)
        required = {f"{d}:{n}" for d, n in self.required_mechanisms()}
        verdicts = []
        for name, (mechanism, rule) in _GRADIENT_RULES.items():
            if mechanism in required:
                verdict = rule(self, batch_summary, step_metrics)
                if verdict is not None:
                    verdicts.append((name, mechanism, bool(verdict)))
        for name, mechanism, verdict in verdicts:
            if verdict:
                # e.g. GDPO reward vectors / REINFORCE++ group means: a gradient
                # the scalar reward-variance rule cannot see.
                return True, f"tightened:gradient_rule:{name} ({mechanism})"
        if not any(g.reward_std > 0 for g in groups):
            return False, None
        for name, mechanism, verdict in verdicts:
            if not verdict:
                return False, f"gradient_rule:{name} ({mechanism})"
        masked = valid_masked_fraction(getattr(step_metrics, "masked_fraction", None))
        if masked is not None and masked >= 1.0:
            maskers = sorted(
                f"{m.dimension}:{m.name}" for m in self._mechanism_defs() if m.masks_tokens
            )
            if maskers:
                return False, "masked:" + ",".join(maskers)
        return True, None

    # -- legacy mapping ---------------------------------------------------
    @classmethod
    def from_legacy_args(cls, args: Any) -> "AlgorithmSpec":
        """Build the spec equivalent to a legacy learner/launcher namespace.

        Legacy always uses GRPO (``--advantage-estimator grpo``); the stock
        nonzero-std filter is normalized to the bounded one exactly as
        ``yeto.launcher`` does when a replacement bound is set.
        """

        flt = getattr(args, "dynamic_sampling_filter_path", None)
        limit = getattr(args, "dynamic_sampling_max_replacements", None)
        if flt == STOCK_NONZERO_STD_FILTER and limit is not None:
            flt = BOUNDED_NONZERO_STD_FILTER
        return cls(
            advantage_estimator="grpo",
            kl_coef=getattr(args, "kl_coef", None),
            dynamic_sampling_filter=flt,
            dynamic_sampling_max_replacements=limit,
        )

    def to_legacy_argv(self) -> list[str]:
        """Miles argv fragment emitted by legacy ``build_miles_argv`` for this spec."""

        argv = ["--advantage-estimator", self.advantage_estimator]
        if self.kl_coef is not None:
            argv += ["--kl-coef", str(self.kl_coef)]
        if self.dynamic_sampling_filter is not None:
            argv += ["--dynamic-sampling-filter-path", self.dynamic_sampling_filter]
        return argv

    def to_legacy_runtime_attrs(self) -> dict[str, Any]:
        """Attributes legacy sets on the Miles namespace (read by the filter)."""

        attrs: dict[str, Any] = {
            "yeto_rl_dynamic_sampling_max_replacements": (
                self.dynamic_sampling_max_replacements
            )
        }
        load_extensions()
        for name, derive in _RUNTIME_ATTRS.items():
            extra = dict(derive(self))
            clash = sorted(set(extra) & set(attrs))
            if clash:
                raise AlgorithmSpecError(f"runtime attrs {name!r} redefines {clash}")
            attrs.update(extra)
        return attrs


# --------------------------------------------------------------------------
# built-in mechanisms (design D4) and rejection matrix (spec: 不兼容组合)
# --------------------------------------------------------------------------


def _builtin_mechanisms() -> None:
    for estimator in ADVANTAGE_ESTIMATORS:
        register_mechanism(
            "advantage_estimators", estimator,
            lambda s, e=estimator: s.advantage.estimator == e,
        )
    for variant in LOSS_VARIANTS:
        register_mechanism("losses", variant, lambda s, v=variant: s.loss.variant == v)
    for aggregation in LOSS_AGGREGATIONS:
        register_mechanism(
            "loss_aggregations", aggregation, lambda s, a=aggregation: s.loss.aggregation == a
        )
    for placement in KL_PLACEMENTS:
        register_mechanism("kl_placements", placement, lambda s, p=placement: s.kl.placement == p)
    for method in CORRECTION_METHODS:
        if method in ("opsm", "custom"):
            continue
        register_mechanism(
            "corrections", method, lambda s, m=method: s.correction.method == m
        )
    # OPSM alone (method='opsm') or combined with tis/custom (opsm_delta set).
    register_mechanism("corrections", "opsm", lambda s: s.correction.opsm_delta is not None)
    # A custom function that a follow-up change registered under its own
    # mechanism name is declared under that name, not as generic 'custom'.
    register_mechanism(
        "corrections", "custom",
        lambda s: s.correction.method == "custom"
        and s.correction.function is not None
        and not (
            _named_correction_detected(s)
        ),
    )
    register_mechanism(
        "reward_postprocessors", "custom_reward_postprocess",
        lambda s: s.advantage.reward_postprocess is not None,
    )
    for flt in sorted(DYNAMIC_SAMPLING_FILTERS):
        register_mechanism(
            "dynamic_sampling_filters", flt, lambda s, f=flt: s.sampling.filter == f
        )
    features: dict[str, Callable[[AlgorithmSpec], bool]] = {
        "eps_clip": lambda s: s.loss.eps_clip is not None,
        "clip_higher": lambda s: s.loss.eps_clip_high is not None,
        "dual_clip": lambda s: s.loss.eps_clip_c is not None,
        "custom_pg_loss_reducer": lambda s: s.loss.reducer is not None
        and not _named_reducer_claimed(s),
        "no_grpo_std_normalization": lambda s: not s.advantage.std_normalization,
        "no_rewards_normalization": lambda s: not s.advantage.rewards_normalization,
        "whiten_advantages": lambda s: s.advantage.whiten,
        "kl_unbiased": lambda s: s.kl.unbiased,
        "entropy_bonus": lambda s: s.entropy_coef != 0.0,
        "rollout_logprobs_as_old": lambda s: s.correction.use_rollout_logprobs,
        "mismatch_metrics": lambda s: s.correction.mismatch_metrics,
        "over_sampling": lambda s: s.sampling.over_sampling_batch_size is not None,
        "overlong_filter": lambda s: s.sampling.overlong_filter,
        # Only plugins that no registered extension owns: the PluginRefs a
        # registered pipeline (reward shapers / advantage transforms) writes
        # into spec.plugins are identity records of mechanisms that are
        # capability-checked under their own names.
        "plugins": lambda s: any(not _owned_plugin(p.path) for p in s.plugins),
    }
    for name, detect in features.items():
        register_mechanism("features", name, detect)


_GENERIC_CORRECTIONS = frozenset(CORRECTION_METHODS) | {"opsm"}


def _named_correction_detected(spec: "AlgorithmSpec") -> bool:
    """One of the function path's own named mechanisms detects the spec."""

    owners = NAMED_CORRECTION_FUNCTIONS.get(spec.correction.function.path, frozenset())
    return any(
        m.dimension == "corrections" and m.name in owners
        and m.name not in _GENERIC_CORRECTIONS and m.detect(spec)
        for m in registered_mechanisms()
    )


def _reject_reward_kl(s: AlgorithmSpec) -> str | None:
    if (
        s.kl.placement == "reward"
        and (s.kl.coef or 0.0) > 0.0
        and s.advantage.estimator in REWARD_KL_DROPPING_ESTIMATORS
    ):
        return (
            f"advantage estimator {s.advantage.estimator!r} ignores a KL placed in the reward "
            f"(kl.placement='reward', kl.coef={s.kl.coef}); use kl.placement='loss' "
            "(--use-kl-loss) instead"
        )
    return None


def _reject_tis_with_rollout_logprobs(s: AlgorithmSpec) -> str | None:
    if s.correction.use_rollout_logprobs and s.correction.method in ("tis", "custom"):
        return (
            f"correction.method={s.correction.method!r} (importance weights rollout vs old "
            "logprobs) and correction.use_rollout_logprobs (rollout logprobs as old "
            "logprobs) are mutually exclusive; drop one of them"
        )
    return None


def _reject_sequence_ratio_without_clip(s: AlgorithmSpec) -> str | None:
    if s.advantage.estimator in SEQUENCE_RATIO_ESTIMATORS and (
        s.loss.eps_clip is None or s.loss.eps_clip_high is None
    ):
        return (
            f"advantage estimator {s.advantage.estimator!r} uses a sequence-level ratio; set "
            "loss.eps_clip and loss.eps_clip_high explicitly (token-level defaults do not apply)"
        )
    return None


def _reject_binary_reward(s: AlgorithmSpec) -> str | None:
    needing = sorted(m.name for m in s._mechanism_defs() if m.requires_binary_reward)
    if needing and not s.advantage.reward_binary:
        return (
            f"mechanisms {needing} require a binary {{0,1}} reward; declare "
            "advantage.reward_binary=true for a binary reward function or drop them"
        )
    return None


def _reject_critic(s: AlgorithmSpec) -> str | None:
    if s.advantage.estimator in CRITIC_ESTIMATORS and not s.execution.needs_critic:
        return (
            f"advantage estimator {s.advantage.estimator!r} needs a critic; set "
            "execution.needs_critic=true"
        )
    if s.execution.needs_critic and s.advantage.estimator not in CRITIC_ESTIMATORS:
        return (
            f"execution.needs_critic=true but advantage estimator {s.advantage.estimator!r} "
            f"trains no critic (critic estimators: {sorted(CRITIC_ESTIMATORS)})"
        )
    return None


_CRITIC_CORE_FIELDS = tuple(f.name for f in fields(CriticSpec) if f.name != "ext")


def _reject_critic_fields_without_critic(s: AlgorithmSpec) -> str | None:
    if s.execution.needs_critic:
        return None
    stray = [f"advantage.{n}" for n in CRITIC_ADVANTAGE_FIELDS
             if getattr(s.advantage, n) is not None]
    stray += [f"critic.{n}" for n in _CRITIC_CORE_FIELDS if getattr(s.critic, n) is not None]
    stray += [f"critic.{n}" for n, _ in s.critic.ext]
    if stray:
        return (
            f"{stray} only apply to critic algorithms (advantage.estimator in "
            f"{sorted(CRITIC_ESTIMATORS)} with execution.needs_critic=true); this spec uses "
            f"advantage.estimator={s.advantage.estimator!r} without a critic. Drop them"
        )
    return None


def _reject_critic_reward_kl(s: AlgorithmSpec) -> str | None:
    # Miles c35702e arguments.py:3598-3603 asserts kl_coef == 0 for shared PPO.
    if s.execution.needs_critic and s.kl.placement == "reward" and s.kl.coef:
        return (
            f"kl.placement='reward' with kl.coef={s.kl.coef} and a critic: Miles shared "
            "actor/critic PPO trains the critic before the actor without ref log probs, so "
            "its value targets would exclude the reward KL; use kl.coef=0 or "
            "kl.placement='loss' (--use-kl-loss)"
        )
    return None


def _reject_critic_lora(s: AlgorithmSpec) -> str | None:
    c = s.critic
    if c.param_mode == "lora":
        return (
            "critic.param_mode='lora' is planned (rl-algo-critic-family design D9) but not "
            "implemented yet: Miles trains the critic full-parameter (it skips the critic's "
            "LoRA setup). Use critic.param_mode='full'"
        )
    stray = [f"critic.{n}" for n in ("lora_rank", "lora_alpha", "lora_target_modules")
             if getattr(c, n) is not None]
    if stray:
        return f"{stray} only apply to critic.param_mode='lora'; drop them"
    return None


def _reject_critic_not_at_pin(s: AlgorithmSpec) -> str | None:
    if not s.execution.needs_critic:
        return None
    a, c = s.advantage, s.critic
    pending = []
    if a.gae_variant != "vanilla":
        pending.append(f"advantage.gae_variant={a.gae_variant!r}")
    if a.lambd_mode != "fixed":
        pending.append(f"advantage.lambd_mode={a.lambd_mode!r}")
    if c.value_loss != "mse":
        pending.append(f"critic.value_loss={c.value_loss!r}")
    if c.critic_updates_per_step != 1:
        pending.append(f"critic.critic_updates_per_step={c.critic_updates_per_step}")
    if pending:
        return (
            f"{pending} need the fork's GAE / value-loss extension point "
            "(rl-algo-critic-family design D6, group 6), which the pinned Miles does not "
            "have yet; use the vanilla PPO defaults"
        )
    stray = []
    if a.alpha is not None and a.lambd_mode != "length_adaptive":
        stray.append("advantage.alpha (only with lambd_mode='length_adaptive')")
    if c.hl_gauss_bins is not None and c.value_loss != "hl_gauss":
        stray.append("critic.hl_gauss_bins (only with value_loss='hl_gauss')")
    if stray:
        return f"{stray}: drop them"
    return None


def _reject_critic_init(s: AlgorithmSpec) -> str | None:
    c = s.critic
    if not s.execution.needs_critic:
        return None
    if c.init == "load" and c.load is None:
        return "critic.init='load' needs critic.load (the critic checkpoint)"
    if c.init == "copy_actor_backbone" and c.load is not None:
        return (
            "critic.load is set but critic.init='copy_actor_backbone' builds the critic "
            "from the initial actor; use critic.init='load' to load a critic checkpoint"
        )
    if c.init == "load" and c.warmup_steps:
        return (
            "critic.warmup_steps > 0 is the warm-up of a critic copied from the actor "
            "(critic.init='copy_actor_backbone', design D5); a loaded critic is not warmed up"
        )
    return None


def _reject_staleness(s: AlgorithmSpec) -> str | None:
    if s.execution.max_policy_staleness != 0:
        return (
            f"execution.max_policy_staleness={s.execution.max_policy_staleness}: no "
            "asynchronous (staleness>0) algorithm contract exists; it needs its own change "
            "(rl-infra-spec alignment A6). Use 0"
        )
    return None


def _reject_rpp_without_whiten(s: AlgorithmSpec) -> str | None:
    # Upstream miles_validate_args asserts it (found by 2.6's parse_args run).
    if s.advantage.estimator in WHITEN_REQUIRED_ESTIMATORS and not s.advantage.whiten:
        return (
            f"advantage estimator {s.advantage.estimator!r} requires advantage normalization; "
            "set advantage.whiten=true (--normalize-advantages)"
        )
    return None


def _reject_kl_loss_zero(s: AlgorithmSpec) -> str | None:
    if s.kl.placement == "loss" and s.kl.coef == 0.0:
        return "kl.placement='loss' with kl.coef=0 has no effect; use kl.placement='none'"
    return None


def _reject_constant_with_token(s: AlgorithmSpec) -> str | None:
    if s.loss.aggregation == "constant" and s.loss.reducer is None:
        return (
            "loss.aggregation='constant' needs its reducer plugin (loss.reducer); "
            "use aggregation 'default' or 'token'"
        )
    return None


def _builtin_rejections() -> None:
    register_rejection("reward_kl_ignored", _reject_reward_kl)
    register_rejection("tis_with_rollout_logprobs", _reject_tis_with_rollout_logprobs)
    register_rejection("sequence_ratio_without_clip", _reject_sequence_ratio_without_clip)
    register_rejection("binary_reward_required", _reject_binary_reward)
    register_rejection("critic_estimator", _reject_critic)
    register_rejection("critic_fields_without_critic", _reject_critic_fields_without_critic)
    register_rejection("critic_reward_kl", _reject_critic_reward_kl)
    register_rejection("critic_param_mode", _reject_critic_lora)
    register_rejection("critic_not_at_pin", _reject_critic_not_at_pin)
    register_rejection("critic_init", _reject_critic_init)
    register_rejection("policy_staleness", _reject_staleness)
    register_rejection("kl_loss_zero_coef", _reject_kl_loss_zero)
    register_rejection("rpp_requires_whiten", _reject_rpp_without_whiten)
    register_rejection("constant_aggregation_reducer", _reject_constant_with_token)


_builtin_mechanisms()
_builtin_rejections()


# --------------------------------------------------------------------------
# user input (design D8/D9/D11): shared by learner and launcher
# --------------------------------------------------------------------------


def resolve_ports_algorithm(args: Any, *, rl_engine: str) -> "AlgorithmSpec | None":
    """Base spec of a run (before extra-argv absorption).

    ``--rl-algorithm-spec PATH`` (v1 or v2 JSON) gives the base; without it
    the spec comes from the legacy CLI exactly as in R0. Legacy CLI algorithm
    options given together with the file must agree with it. The ports-only
    options are refused on the legacy engine (returns None there).
    """

    path = getattr(args, "rl_algorithm_spec", None)
    allow = list(getattr(args, "rl_allow_unverified_mechanism", None) or ())
    if rl_engine != "ports":
        used = [name for name, value in (
            ("--rl-algorithm-spec", path),
            ("--rl-allow-unverified-mechanism", allow),
            ("--rl-expected-algorithm-sha256",
             getattr(args, "rl_expected_algorithm_sha256", None)),
        ) if value]
        if used:
            raise AlgorithmSpecError(
                f"{', '.join(used)} only apply to --rl-engine ports (legacy is unchanged)"
            )
        return None
    legacy = AlgorithmSpec.from_legacy_args(args)
    if not path:
        return legacy
    spec = AlgorithmSpec.from_json_file(path)
    problems = []
    for flag, attr, given, in_file in (
        ("--kl-coef", "kl_coef", getattr(args, "kl_coef", None), spec.kl_coef),
        ("--dynamic-sampling-filter-path", "dynamic_sampling_filter_path",
         legacy.dynamic_sampling_filter, spec.dynamic_sampling_filter),
        ("--dynamic-sampling-max-replacements", "dynamic_sampling_max_replacements",
         legacy.dynamic_sampling_max_replacements, spec.dynamic_sampling_max_replacements),
    ):
        if given is not None and given != in_file:
            problems.append(f"{flag} {given!r} disagrees with --rl-algorithm-spec ({in_file!r})")
    if problems:
        raise AlgorithmSpecError("; ".join(problems))
    return spec


def check_unverified_allowance(
    names: Iterable[str], *, islands: int, outer_sync: bool
) -> tuple[str, ...]:
    """D11 as written: refused with multiple islands *or* any outer sync.

    Names are qualified ``dimension:name`` mechanisms.
    """

    names = tuple(sorted(set(names)))
    if not names:
        return ()
    unknown = sorted(set(names) - allowance_names())
    if unknown:
        raise AlgorithmSpecError(
            f"--rl-allow-unverified-mechanism: unknown mechanism(s) {unknown} "
            f"(known: {sorted(allowance_names())})"
        )
    if islands != 1 or outer_sync:
        raise AlgorithmSpecError(
            f"--rl-allow-unverified-mechanism {list(names)} is only allowed on a "
            f"single-island run without outer sync (this run: {islands} island(s), "
            f"outer sync {'on' if outer_sync else 'off'}); runs with outer sync use "
            "declared mechanisms only"
        )
    return names
