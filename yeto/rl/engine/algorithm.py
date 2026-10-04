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
SUPPORTED_ADVANTAGE_ESTIMATORS = frozenset({"grpo"})
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


def register_field(group: str, name: str, *, default: Any,
                   parse: Callable[[str, Any], Any],
                   to_json: Callable[[Any], Any] | None = None) -> FieldDef:
    if group not in _GROUPS:
        raise ValueError(f"unknown spec group {group!r}; one of {sorted(_GROUPS)}")
    core = {f.name for f in fields(_GROUPS[group])}
    if name in core or name in _FIELDS.get(group, {}):
        raise ValueError(f"field {group}.{name} already exists")
    try:
        hash(default)
    except TypeError as exc:
        raise ValueError(f"field {group}.{name}: default must be hashable") from exc
    definition = FieldDef(group, name, default, parse, to_json or (lambda value: value))
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
        for name in sorted(raw):
            value = definitions[name].parse(f"{self.GROUP}.{name}", raw[name])
            try:
                hash(value)
            except TypeError:
                raise AlgorithmSpecError(
                    f"{self.GROUP}.{name}: the registered parser returned an unhashable "
                    f"{type(value).__name__} (return tuples / frozen values)"
                ) from None
            if value != definitions[name].default:
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
        if self.method != "opsm" and self.opsm_delta is not None:
            raise AlgorithmSpecError("correction.opsm_delta requires correction.method='opsm'")
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


_GROUPS: dict[str, type] = {
    "advantage": AdvantageSpec,
    "loss": LossSpec,
    "kl": KlSpec,
    "correction": CorrectionSpec,
    "sampling": SamplingSpec,
    "execution": ExecutionSpec,
}
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
            ("sampling", sampling), ("execution", execution),
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
            **{name: getattr(self, name).to_dict() for name in _GROUPS},
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
        """(dimension, mechanism) pairs the engine must declare."""

        return frozenset(
            (m.dimension, m.name) for m in registered_mechanisms() if m.detect(self)
        )

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
        register_mechanism(
            "corrections", method, lambda s, m=method: s.correction.method == m
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
        "custom_pg_loss_reducer": lambda s: s.loss.reducer is not None,
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
            "execution.needs_critic=true (only the legacy engine drives a critic: "
            "--rl-engine legacy)"
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
    unknown = sorted(set(names) - mechanism_names())
    if unknown:
        raise AlgorithmSpecError(
            f"--rl-allow-unverified-mechanism: unknown mechanism(s) {unknown} "
            f"(known: {sorted(mechanism_names())})"
        )
    if islands != 1 or outer_sync:
        raise AlgorithmSpecError(
            f"--rl-allow-unverified-mechanism {list(names)} is only allowed on a "
            f"single-island run without outer sync (this run: {islands} island(s), "
            f"outer sync {'on' if outer_sync else 'off'}); runs with outer sync use "
            "declared mechanisms only"
        )
    return names
