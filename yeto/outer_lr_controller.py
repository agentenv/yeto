"""CPU reference for the syncer's age-aware outer-LR controller.

The public evaluator consumes ``(mu, t, T_planned)`` plus an optional named
spectral sketch or oracle scale and returns a multiplier for the configured
outer learning rate. JSON schemas and numerical conventions match
``syncer/src/outer_lr_controller.rs``.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


DRIFT_SURFACE_SCHEMA = "yeto.outer_lr_drift_surface.v1"
SPECTRAL_SKETCH_SCHEMA = "yeto.outer_lr_spectral_sketch.v1"
ORACLE_SCHEDULE_SCHEMA = "yeto.outer_lr_oracle_schedule.v1"
DRIFT_OUTPUT = "log2_lr_multiplier"
MAX_FACTOR_POWER = 16
_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_BUILTIN_SOURCES = frozenset(
    {
        "mu",
        "t",
        "T_planned",
        "age_fraction",
        "remaining_steps",
        "remaining_fraction",
    }
)


class ControllerMode(str, Enum):
    TRANSIENT = "transient"
    MEASURED_DRIFT = "measured-drift"
    ORACLE = "oracle"


@dataclass(frozen=True)
class ScaleOutput:
    scale: float
    transient_scale: float | None
    drift_scale: float | None


@dataclass(frozen=True)
class SurfaceFeature:
    name: str
    source: str
    transform: str
    center: float
    scale: float

    @classmethod
    def from_mapping(cls, value: object) -> SurfaceFeature:
        record = _record(value, "surface feature")
        _keys(
            record,
            required={"name", "source"},
            optional={"transform", "center", "scale"},
            context="surface feature",
        )
        name = _name(record["name"], "surface feature")
        source = _source(record["source"])
        transform = record.get("transform", "identity")
        if transform not in {"identity", "log2", "ln"}:
            raise ValueError(f"unknown surface feature transform {transform!r}")
        center = _number(record.get("center", 0.0), f"feature {name!r} center")
        scale = _number(record.get("scale", 1.0), f"feature {name!r} scale")
        if scale == 0.0:
            raise ValueError(f"feature {name!r} scale must be nonzero")
        return cls(name, source, transform, center, scale)

    def evaluate(
        self,
        *,
        mu: float,
        t: int,
        t_planned: int,
        spectral_sketch: Mapping[str, float] | None,
    ) -> float:
        if self.source == "mu":
            raw = mu
        elif self.source == "t":
            raw = float(t)
        elif self.source == "T_planned":
            raw = float(t_planned)
        elif self.source == "age_fraction":
            raw = t / t_planned
        elif self.source == "remaining_steps":
            raw = float(t_planned - t)
        elif self.source == "remaining_fraction":
            raw = (t_planned - t) / t_planned
        else:
            key = self.source.removeprefix("spectral.")
            if spectral_sketch is None or key not in spectral_sketch:
                raise ValueError(
                    f"outer-LR drift surface requires spectral feature {key!r}"
                )
            raw = _number(spectral_sketch[key], f"spectral feature {key!r}")

        if self.transform == "identity":
            transformed = raw
        elif raw <= 0.0:
            raise ValueError(
                f"outer-LR surface source {self.source!r} must be positive "
                f"for transform {self.transform!r}"
            )
        elif self.transform == "log2":
            transformed = math.log2(raw)
        else:
            transformed = math.log(raw)
        normalized = (transformed - self.center) / self.scale
        if not math.isfinite(normalized):
            raise ValueError(f"surface feature {self.name!r} is non-finite")
        return normalized


@dataclass(frozen=True)
class SurfaceTerm:
    coefficient: float
    powers: tuple[tuple[str, int], ...]

    @classmethod
    def from_mapping(cls, value: object) -> SurfaceTerm:
        record = _record(value, "surface term")
        _keys(
            record,
            required={"coefficient"},
            optional={"powers"},
            context="surface term",
        )
        coefficient = _number(record["coefficient"], "surface coefficient")
        raw_powers = _record(record.get("powers", {}), "surface term powers")
        powers: list[tuple[str, int]] = []
        for name, power in raw_powers.items():
            name = _name(name, "surface term feature")
            if isinstance(power, bool) or not isinstance(power, int):
                raise ValueError(f"surface power for {name!r} must be an integer")
            if not 1 <= power <= MAX_FACTOR_POWER:
                raise ValueError(
                    f"surface power for {name!r} must be in 1..={MAX_FACTOR_POWER}"
                )
            powers.append((name, power))
        return cls(coefficient, tuple(sorted(powers)))


@dataclass(frozen=True)
class FactorialSurface:
    features: tuple[SurfaceFeature, ...]
    terms: tuple[SurfaceTerm, ...]
    min_drift_scale: float
    max_drift_scale: float

    @classmethod
    def from_json(cls, path: str | Path) -> FactorialSurface:
        return cls.from_mapping(json.loads(Path(path).read_text()))

    @classmethod
    def from_mapping(cls, value: object) -> FactorialSurface:
        record = _record(value, "drift surface")
        _keys(
            record,
            required={
                "schema",
                "output",
                "features",
                "terms",
                "drift_scale_bounds",
            },
            optional=set(),
            context="drift surface",
        )
        if record["schema"] != DRIFT_SURFACE_SCHEMA:
            raise ValueError(
                f"drift surface schema must be {DRIFT_SURFACE_SCHEMA!r}"
            )
        if record["output"] != DRIFT_OUTPUT:
            raise ValueError(f"drift surface output must be {DRIFT_OUTPUT!r}")
        if not isinstance(record["features"], list):
            raise ValueError("drift surface features must be an array")
        if not isinstance(record["terms"], list):
            raise ValueError("drift surface terms must be an array")
        features = tuple(
            SurfaceFeature.from_mapping(item) for item in record["features"]
        )
        names = [feature.name for feature in features]
        if len(set(names)) != len(names):
            raise ValueError("drift surface feature names must be unique")
        terms = tuple(SurfaceTerm.from_mapping(item) for item in record["terms"])
        for term in terms:
            for name, _ in term.powers:
                if name not in names:
                    raise ValueError(
                        f"drift surface term references unknown feature {name!r}"
                    )

        bounds = _record(record["drift_scale_bounds"], "drift scale bounds")
        _keys(
            bounds,
            required={"min", "max"},
            optional=set(),
            context="drift scale bounds",
        )
        minimum = _number(bounds["min"], "minimum drift scale")
        maximum = _number(bounds["max"], "maximum drift scale")
        if minimum <= 0.0 or minimum > 1.0 or maximum < 1.0 or minimum > maximum:
            raise ValueError(
                "drift scale bounds must be positive, ordered, and contain 1"
            )
        return cls(features, terms, minimum, maximum)

    def drift_scale(
        self,
        *,
        mu: float,
        t: int,
        t_planned: int,
        spectral_sketch: Mapping[str, float] | None = None,
    ) -> float:
        feature_by_name = {feature.name: feature for feature in self.features}
        values: dict[str, float] = {}
        log2_multiplier = 0.0
        for term in self.terms:
            # This is also the exact-reduction rule: a zero coefficient never
            # demands a spectral feature and contributes no floating arithmetic.
            if term.coefficient == 0.0:
                continue
            product = 1.0
            for name, power in term.powers:
                if name not in values:
                    values[name] = feature_by_name[name].evaluate(
                        mu=mu,
                        t=t,
                        t_planned=t_planned,
                        spectral_sketch=spectral_sketch,
                    )
                product *= values[name] ** power
            log2_multiplier += term.coefficient * product
        if not math.isfinite(log2_multiplier):
            raise ValueError("drift surface produced a non-finite log2 multiplier")
        if log2_multiplier == 0.0:
            return 1.0
        min_log2 = math.log2(self.min_drift_scale)
        max_log2 = math.log2(self.max_drift_scale)
        if log2_multiplier <= min_log2:
            scale = self.min_drift_scale
        elif log2_multiplier >= max_log2:
            scale = self.max_drift_scale
        else:
            scale = 2.0**log2_multiplier
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("drift surface produced a non-finite multiplier")
        return scale


@dataclass(frozen=True)
class SpectralSketch:
    global_features: Mapping[str, float]
    fragment_features: Mapping[int, Mapping[str, float]]

    @classmethod
    def from_json(cls, path: str | Path) -> SpectralSketch:
        return cls.from_mapping(json.loads(Path(path).read_text()))

    @classmethod
    def from_mapping(cls, value: object) -> SpectralSketch:
        record = _record(value, "spectral sketch")
        _keys(
            record,
            required={"schema"},
            optional={"global_features", "fragment_features"},
            context="spectral sketch",
        )
        if record["schema"] != SPECTRAL_SKETCH_SCHEMA:
            raise ValueError(
                f"spectral sketch schema must be {SPECTRAL_SKETCH_SCHEMA!r}"
            )
        global_features = _feature_map(
            record.get("global_features", {}), "global spectral features"
        )
        raw_fragments = _record(
            record.get("fragment_features", {}), "fragment spectral features"
        )
        fragment_features: dict[int, Mapping[str, float]] = {}
        for raw_fid, features in raw_fragments.items():
            if not isinstance(raw_fid, str) or not raw_fid.isdecimal():
                raise ValueError("spectral fragment ids must be decimal strings")
            fid = int(raw_fid)
            fragment_features[fid] = _feature_map(
                features, f"fragment {fid} spectral features"
            )
        return cls(global_features, fragment_features)

    def for_fragment(self, fragment_id: int) -> dict[str, float]:
        features = dict(self.global_features)
        features.update(self.fragment_features.get(fragment_id, {}))
        return features


@dataclass(frozen=True)
class OracleSchedule:
    scales: tuple[float, ...] | None
    fragment_scales: Mapping[int, tuple[float, ...]]

    @classmethod
    def from_json(cls, path: str | Path) -> OracleSchedule:
        return cls.from_mapping(json.loads(Path(path).read_text()))

    @classmethod
    def from_mapping(cls, value: object) -> OracleSchedule:
        record = _record(value, "oracle schedule")
        _keys(
            record,
            required={"schema"},
            optional={"scales", "fragment_scales"},
            context="oracle schedule",
        )
        if record["schema"] != ORACLE_SCHEDULE_SCHEMA:
            raise ValueError(
                f"oracle schedule schema must be {ORACLE_SCHEDULE_SCHEMA!r}"
            )
        scales = (
            _scale_array(record["scales"], "default oracle scales")
            if "scales" in record
            else None
        )
        raw_fragments = _record(
            record.get("fragment_scales", {}), "fragment oracle scales"
        )
        fragment_scales: dict[int, tuple[float, ...]] = {}
        for raw_fid, raw_scales in raw_fragments.items():
            if not isinstance(raw_fid, str) or not raw_fid.isdecimal():
                raise ValueError("oracle fragment ids must be decimal strings")
            fid = int(raw_fid)
            fragment_scales[fid] = _scale_array(
                raw_scales, f"fragment {fid} oracle scales"
            )
        if scales is None and not fragment_scales:
            raise ValueError("oracle schedule must provide scales or fragment_scales")
        return cls(scales, fragment_scales)

    def scale_for_fragment(
        self, *, fragment_id: int, t: int, t_planned: int
    ) -> float:
        selected = self.fragment_scales.get(fragment_id, self.scales)
        if selected is None:
            raise ValueError(f"oracle schedule has no scales for fragment {fragment_id}")
        if len(selected) != t_planned:
            raise ValueError(
                f"oracle schedule has {len(selected)} scales for fragment "
                f"{fragment_id}, planned horizon is {t_planned}"
            )
        _validate_age(t, t_planned)
        return selected[t - 1]


def outer_lr_scale(
    mode: ControllerMode | str,
    *,
    mu: float,
    t: int,
    t_planned: int,
    surface: FactorialSurface | None = None,
    spectral_sketch: Mapping[str, float] | None = None,
    oracle_scale: float | None = None,
) -> ScaleOutput:
    """Return the multiplier for one outer update.

    ``t`` is one-indexed. Mode ``measured-drift`` computes
    ``transient_scale * drift_scale``. Mode ``oracle`` returns ``oracle_scale``
    without normalization or rounding.
    """

    mode = ControllerMode(mode)
    _validate_age(t, t_planned)
    if mode is ControllerMode.ORACLE:
        if oracle_scale is None:
            raise ValueError("oracle mode requires oracle_scale")
        scale = _number(oracle_scale, "oracle scale")
        if scale <= 0.0:
            raise ValueError("oracle scale must be positive")
        return ScaleOutput(scale, None, None)

    transient = transient_normalization_scale(mu=mu, t=t)
    if mode is ControllerMode.TRANSIENT:
        return ScaleOutput(transient, transient, None)
    if surface is None:
        raise ValueError("measured-drift mode requires a factorial surface")
    drift = surface.drift_scale(
        mu=mu,
        t=t,
        t_planned=t_planned,
        spectral_sketch=spectral_sketch,
    )
    combined = transient if drift == 1.0 else transient * drift
    if not math.isfinite(combined) or combined <= 0.0:
        raise ValueError("controller produced a non-finite or non-positive scale")
    return ScaleOutput(combined, transient, drift)


def transient_normalization_scale(*, mu: float, t: int) -> float:
    """The existing ``--outer-bias-correction`` direction-multiplier rule."""

    mu = _number(mu, "mu")
    if not 0.0 <= mu < 1.0:
        raise ValueError(f"transient normalization requires mu in [0, 1), got {mu}")
    if isinstance(t, bool) or not isinstance(t, int) or t < 1:
        raise ValueError("transient normalization requires one-indexed integer t >= 1")
    exponent = min(t + 1, 2_147_483_647)
    divisor = 1.0 - mu**exponent
    if not math.isfinite(divisor) or divisor <= 0.0:
        raise ValueError("transient normalization produced a non-positive divisor")
    return 1.0 / divisor


def planned_fragment_steps(*, total_steps: int, fragments: int, fragment_id: int) -> int:
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (total_steps, fragments, fragment_id)
    ):
        raise ValueError("total_steps, fragments, and fragment_id must be integers")
    if total_steps < 0 or fragments <= 0 or not 0 <= fragment_id < fragments:
        raise ValueError("invalid total-step or fragment coordinates")
    first_step = fragment_id + 1
    return 0 if first_step > total_steps else 1 + (total_steps - first_step) // fragments


def _validate_age(t: int, t_planned: int) -> None:
    if (
        isinstance(t, bool)
        or isinstance(t_planned, bool)
        or not isinstance(t, int)
        or not isinstance(t_planned, int)
        or t < 1
        or t_planned < 1
        or t > t_planned
    ):
        raise ValueError(
            f"outer-LR controller requires 1 <= t <= T_planned, "
            f"got t={t!r} T_planned={t_planned!r}"
        )


def _record(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(k, str) for k in value):
        raise ValueError(f"{context} must be a JSON object with string keys")
    return value


def _keys(
    record: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str],
    context: str,
) -> None:
    missing = required - record.keys()
    unknown = record.keys() - required - optional
    if missing:
        raise ValueError(f"{context} is missing fields {sorted(missing)!r}")
    if unknown:
        raise ValueError(f"{context} has unknown fields {sorted(unknown)!r}")


def _number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _name(value: object, context: str) -> str:
    if not isinstance(value, str) or _NAME.fullmatch(value) is None:
        raise ValueError(
            f"{context} name {value!r} must contain only ASCII letters, digits, "
            "'.', '-', or '_'"
        )
    return value


def _source(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("surface feature source must be a string")
    if value in _BUILTIN_SOURCES:
        return value
    if value.startswith("spectral."):
        _name(value.removeprefix("spectral."), "spectral source")
        return value
    raise ValueError(f"unknown surface feature source {value!r}")


def _feature_map(value: object, context: str) -> dict[str, float]:
    record = _record(value, context)
    return {_name(name, "spectral feature"): _number(item, name) for name, item in record.items()}


def _scale_array(value: object, context: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be an array")
    scales = tuple(_number(item, context) for item in value)
    if any(scale <= 0.0 for scale in scales):
        raise ValueError(f"{context} must contain only positive values")
    return scales
