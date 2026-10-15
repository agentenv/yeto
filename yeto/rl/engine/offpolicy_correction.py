"""Off-policy correction pass-through: the backend-neutral part
(change ``rl-offpolicy-correction-passthrough``).

yeto does not compute importance weights. A backend adapter maps the neutral
``CorrectionSpec`` (design D1) to its own settings and returns a
:class:`CorrectionMapping`. This module holds what every adapter shares:

* :class:`CorrectionMapping` / :class:`CorrectionWarning` -- the mapped
  values, the proximal-policy anchor (``prox_anchor``, design D2) and the
  fields the backend could not map (``correction_config_warning``, D3);
* :func:`refuse_correction_warnings` -- phase 2 (L6b.2): an unmappable field
  or an island difference refuses the launch;
* :class:`ForbiddenSetting` / :func:`refuse_forbidden` -- backend settings
  that break decoupled PPO (Miles ``--use-rollout-logprobs``, verl
  ``bypass_mode=True``, ...). They are refused at launch with the setting
  named, also in phase 1;
* :func:`correction_record` / :func:`compare_island_records` -- the manifest
  and tape record (D8) and the cross-island comparison (D2, D8);
* :func:`start_version` / :func:`start_version_gap` -- the version gap from
  the start version (D4);
* :func:`offpolicy_gate_event` -- the per-round drop and truncation event (D6);
* :func:`cross_version_bounds` / :func:`staleness_correction` -- the
  observe-only cross-version IS takes its bounds and name from the spec (D7).

Phase 1 warned on unmappable settings and on island differences. Phase 2
(L6b.2) refuses them: the launcher refuses the run manifest's warnings and each
island refuses the fields its backend cannot map.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from yeto.rl.engine.algorithm import AlgorithmSpec, CorrectionSpec

CORRECTION_RECORD_SCHEMA = "yeto-offpolicy-correction-v1"
# Tape / manifest event names (design D2, D3, D6, D8).
CORRECTION_RECORD_EVENT = "rl_offpolicy_correction"
CORRECTION_CONFIG_WARNING = "correction_config_warning"
OFFPOLICY_GATE_EVENT = "offpolicy_gate"

# Which weights the backend uses for pi_prox (design D2).
PROX_PER_STEP = "per_step"  # recomputed on the weights before each update (Miles)
PROX_FIRST_LOCAL_STEP = "first_local_step"  # first local step of the cycle (verl)
PROX_ANCHORS = (PROX_PER_STEP, PROX_FIRST_LOCAL_STEP)

# offpolicy_gate reasons (design D6). Backends may add their own admission drops.
GATE_KEPT = "kept"
GATE_GROUP_DROPPED = "group_dropped"  # gap over A_hard at consumption, whole group
GATE_UNKNOWN_VERSION_DROPPED = "unknown_version_dropped"
GATE_REASONS = (GATE_KEPT, GATE_GROUP_DROPPED, GATE_UNKNOWN_VERSION_DROPPED)
# Groups the backend dropped as stale inside its own buffer (Miles --fully-async
# ``stale_dropped_groups``), counted apart from the yeto gate drops. Present only
# when the backend reported the count; no lengths (the backend does not give them).
GATE_BACKEND_STALE_DROPPED = "backend_stale_dropped"
# Response-length bucket edges (tokens) of the dropped-trajectory histogram.
# Fixed in code so that rounds and runs stay comparable.
DROPPED_LENGTH_EDGES = (256, 1024, 4096, 16384)
GRANULARITY_TOKEN = "token"
GRANULARITY_TRAJECTORY = "trajectory"
GRANULARITIES = (GRANULARITY_TOKEN, GRANULARITY_TRAJECTORY)
# Round-metadata keys an adapter fills for the driver (removed from
# rl_rollout_carry_over): dropped lengths per reason, truncation per turn,
# and the granularity of the turn counts.
GATE_DROPPED_LENGTHS = "offpolicy_gate_dropped_lengths"
TRUNCATED_BY_TURN = "cross_version_truncated_by_turn"
GATE_GRANULARITY = "offpolicy_gate_granularity"


class CorrectionConfigError(ValueError):
    """A backend setting that breaks the decoupled PPO objective; launch refused."""


# --------------------------------------------------------------------------
# mapping, warnings, forbidden settings
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CorrectionWarning:
    """One field a backend cannot map as written (design D3 "warning" rows)."""

    field: str  # neutral field, e.g. "lower"
    backend: str
    reason: str
    ran_with: Any = None  # the value the backend actually runs with

    def to_dict(self) -> dict[str, Any]:
        return {"field": self.field, "backend": self.backend, "reason": self.reason,
                "ran_with": self.ran_with}

    def event(self) -> dict[str, Any]:
        return {"event": CORRECTION_CONFIG_WARNING, "level": "warning", **self.to_dict()}


@dataclass(frozen=True)
class CorrectionMapping:
    """What an adapter made of the correction group.

    ``values``: backend setting -> value (Miles: flag -> argv value or True;
    verl: Hydra key -> value). ``argv`` / ``runtime_attrs`` / ``overrides``
    are the forms the adapter passes on (empty when unused by the backend).
    """

    backend: str
    prox_anchor: str
    values: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[CorrectionWarning, ...] = ()
    argv: tuple[str, ...] = ()
    runtime_attrs: Mapping[str, Any] = field(default_factory=dict)
    overrides: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.prox_anchor not in PROX_ANCHORS:
            raise ValueError(f"prox_anchor must be one of {PROX_ANCHORS}, got {self.prox_anchor!r}")


@dataclass(frozen=True)
class ForbiddenSetting:
    """A backend setting refused at launch because it breaks decoupled PPO."""

    backend: str
    setting: str  # Miles flag, verl Hydra assignment ("key=value")
    reason: str

    def message(self) -> str:
        return (f"{self.setting} is refused on backend {self.backend!r}: {self.reason} "
                "(rl-offpolicy-correction-passthrough D2)")


def refuse_forbidden(present: Iterable[str], forbidden: Iterable[ForbiddenSetting]) -> None:
    """Raise :class:`CorrectionConfigError` naming the first forbidden setting in ``present``."""

    table = {f.setting: f for f in forbidden}
    for setting in present:
        if setting in table:
            raise CorrectionConfigError(table[setting].message())


def refuse_correction_warnings(warnings: Sequence[Mapping[str, Any]]) -> None:
    """Phase 2 (L6b.2): raise :class:`CorrectionConfigError` naming every
    ``correction_config_warning`` (unmappable field, backend without a
    mapping, difference between islands). No warnings: no-op."""

    if not warnings:
        return
    lines = []
    for warning in warnings:
        if warning.get("scope") == "cross_island":
            lines.append(f"islands differ in {sorted(warning.get('differences') or {})}: "
                         f"{warning.get('reason')}")
        else:
            lines.append(f"{warning.get('field')} on backend {warning.get('backend')!r}: "
                         f"{warning.get('reason')}")
    raise CorrectionConfigError(
        "correction settings refused (rl-offpolicy-correction-passthrough L6b.2): "
        + "; ".join(lines))


# --------------------------------------------------------------------------
# manifest / tape record and the cross-island comparison (D2, D8)
# --------------------------------------------------------------------------


def correction_record(correction: CorrectionSpec, mapping: CorrectionMapping | None,
                      *, backend: str) -> dict[str, Any]:
    """The full group, the backend values and ``prox_anchor`` (manifest and tape).

    ``mapping`` None: the backend has no correction mapping yet; the record
    says so (``mapped`` None) and the caller warns.
    """

    return {
        "schema": CORRECTION_RECORD_SCHEMA,
        "backend": backend,
        "group": correction.to_dict(),
        "mapped": None if mapping is None else dict(mapping.values),
        "prox_anchor": None if mapping is None else mapping.prox_anchor,
        "warnings": [] if mapping is None else [w.to_dict() for w in mapping.warnings],
    }


def compare_island_records(records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any] | None:
    """One ``correction_config_warning`` listing every field that differs
    between islands (group fields and ``prox_anchor``), or None.

    ``records``: island name -> :func:`correction_record`.
    """

    if len(records) < 2:
        return None
    islands = sorted(records)
    keys = sorted({k for r in records.values() for k in (r.get("group") or {})})
    differences: dict[str, dict[str, Any]] = {}
    for key in keys:
        values = {i: (records[i].get("group") or {}).get(key) for i in islands}
        if len({_comparable(v) for v in values.values()}) > 1:
            differences[key] = values
    anchors = {i: records[i].get("prox_anchor") for i in islands}
    if len(set(anchors.values())) > 1:
        differences["prox_anchor"] = anchors
    if not differences:
        return None
    return {"event": CORRECTION_CONFIG_WARNING, "level": "warning", "scope": "cross_island",
            "reason": "islands run different correction settings", "differences": differences}


def backend_mapping(spec: AlgorithmSpec, backend: str) -> CorrectionMapping | None:
    """The ``offpolicy_correction`` role of ``backend`` applied to ``spec``.

    Adapter interface (role ``offpolicy_correction`` in
    :mod:`yeto.rl.engine.backends`): a module with ``correction_mapping(spec)
    -> CorrectionMapping`` and ``FORBIDDEN`` (a tuple of
    :class:`ForbiddenSetting`). None when the backend has no such role yet.
    """

    from . import backends

    try:
        module = backends.module("offpolicy_correction", backend)
    except backends.UnknownBackend:
        return None
    return module.correction_mapping(spec)


def run_correction_manifest(spec: AlgorithmSpec,
                            island_backends: Mapping[str, str]) -> dict[str, Any]:
    """The run manifest's ``offpolicy_correction`` section (design D8, L6.5).

    ``island_backends``: island name -> backend. Returns the record of every
    island and the ``correction_config_warning`` events: one per field a
    backend cannot map, one per backend without a mapping, and at most one
    for differences between islands. The caller refuses them
    (:func:`refuse_correction_warnings`).
    """

    records: dict[str, dict[str, Any]] = {}
    warnings: list[dict[str, Any]] = []
    mappings: dict[str, CorrectionMapping | None] = {}
    for island, backend in sorted(island_backends.items()):
        if backend not in mappings:
            mapping = mappings[backend] = backend_mapping(spec, backend)
            if mapping is None and not spec.correction.is_default():
                warnings.append(CorrectionWarning(
                    "*", backend, "the backend adapter has no correction mapping yet; the "
                    "group is recorded but not passed on").event())
            elif mapping is not None:
                warnings.extend(w.event() for w in mapping.warnings)
        records[island] = correction_record(spec.correction, mappings[backend], backend=backend)
    cross = compare_island_records(records)
    if cross is not None:
        warnings.append(cross)
    return {"islands": records, "warnings": warnings}


def _comparable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted((k, _comparable(v)) for k, v in value.items()))
    if isinstance(value, list):
        return tuple(_comparable(v) for v in value)
    return value


# --------------------------------------------------------------------------
# version gap from the start version (D4)
# --------------------------------------------------------------------------


def start_version(token_versions: Sequence[int | None]) -> int | None:
    """The version of the first response token whose version is known."""

    return next((v for v in token_versions if v is not None), None)


def start_version_gap(trainer_version: int, token_versions: Sequence[int | None]) -> int | None:
    """Current trainer version minus the trajectory start version (design D4)."""

    start = start_version(token_versions)
    return None if start is None else trainer_version - start


def oldest_is_start(token_versions: Sequence[int | None]) -> bool | None:
    """L6.12: whether the oldest token version equals the start version.

    True for a continued trajectory (versions never decrease along the
    response). None when no version is known.
    """

    known = [v for v in token_versions if v is not None]
    return None if not known else min(known) == known[0]


def segments_start_check(records: Iterable[Mapping[str, Any]],
                         key: str = "policy_versions") -> dict[str, int]:
    """L6.12 on a tape: counts of records whose oldest segment version is (not)
    the first one. ``key`` holds ``[[version, start, end), ...]`` segments
    (``rl_trajectory_reward.policy_versions``)."""

    out = {"checked": 0, "oldest_is_start": 0, "oldest_before_start": 0}
    for record in records:
        segments = record.get(key)
        if not segments or not isinstance(segments[0], (list, tuple)):
            continue
        versions = [int(s[0]) for s in sorted(segments, key=lambda s: int(s[1]))]
        out["checked"] += 1
        out["oldest_is_start" if oldest_is_start(versions) else "oldest_before_start"] += 1
    return out


# --------------------------------------------------------------------------
# offpolicy_gate (D6)
# --------------------------------------------------------------------------


def length_bucket(length: int) -> str:
    low = 0
    for edge in DROPPED_LENGTH_EDGES:
        if length < edge:
            return f"{low}-{edge - 1}"
        low = edge
    return f"{low}+"


def length_histogram(lengths: Iterable[int]) -> dict[str, int]:
    """Fixed-bucket histogram; every bucket is present (zeros included)."""

    edges = (0, *DROPPED_LENGTH_EDGES)
    out = {f"{lo}-{hi - 1}": 0 for lo, hi in zip(edges, edges[1:])}
    out[f"{DROPPED_LENGTH_EDGES[-1]}+"] = 0
    for length in lengths:
        out[length_bucket(int(length))] += 1
    return out


def offpolicy_gate_event(*, backend: str, counts: Mapping[str, int],
                         dropped_lengths: Mapping[str, Sequence[int]],
                         truncation_by_turn: Mapping[int, tuple[int, int]] | None,
                         granularity: str, **extra: Any) -> dict[str, Any]:
    """The ``offpolicy_gate`` fields of one round (design D6).

    ``counts``: reason -> groups. ``dropped_lengths``: reason -> response
    lengths of the dropped trajectories. ``truncation_by_turn``: turn index
    -> (truncated, total); None when the backend reported nothing.
    ``granularity`` says whether the turn counts are tokens or trajectories.
    """

    if granularity not in GRANULARITIES:
        raise ValueError(f"granularity must be one of {GRANULARITIES}, got {granularity!r}")
    turns = None
    if truncation_by_turn is not None:
        turns = {str(turn): {"truncated": int(t), "total": int(n),
                             "ratio": (t / n) if n else None}
                 for turn, (t, n) in sorted(truncation_by_turn.items())}
    return {
        "backend": backend,
        "counts": {k: int(v) for k, v in sorted(counts.items())},
        "dropped_length_edges": list(DROPPED_LENGTH_EDGES),
        "dropped_length_histogram": {reason: length_histogram(lengths)
                                     for reason, lengths in sorted(dropped_lengths.items())},
        "truncation_by_turn": turns,
        "granularity": granularity,
        **extra,
    }


def gate_from_carry_over(fields: dict[str, Any], *, kept: int, backend: str
                         ) -> dict[str, Any] | None:
    """Pop the gate keys from a round's carry-over fields and build the
    ``offpolicy_gate`` event fields; None when the adapter gave none."""

    lengths = fields.pop(GATE_DROPPED_LENGTHS, None)
    turns = fields.pop(TRUNCATED_BY_TURN, None)
    granularity = fields.pop(GATE_GRANULARITY, GRANULARITY_TOKEN)
    if lengths is None:
        return None
    counts = {GATE_KEPT: kept,
              GATE_GROUP_DROPPED: int(fields.get("over_age_discarded_groups", 0) or 0),
              GATE_UNKNOWN_VERSION_DROPPED:
                  int(fields.get("unknown_version_discarded_groups", 0) or 0)}
    backend_stale = fields.get("stale_dropped_groups")
    if isinstance(backend_stale, int):  # None = the backend did not expose it
        counts[GATE_BACKEND_STALE_DROPPED] = backend_stale
    if turns is not None:
        turns = {int(k): tuple(v) for k, v in turns.items()}
    return offpolicy_gate_event(backend=backend, counts=counts, dropped_lengths=lengths,
                                truncation_by_turn=turns, granularity=granularity)


# --------------------------------------------------------------------------
# cross_version_is: observe-only, bounds and name from the spec (D7)
# --------------------------------------------------------------------------

# Bounds the observe-only estimate used before D7; kept when the spec has none.
OBSERVE_DEFAULT_BOUNDS = (0.0, 2.0)


def cross_version_bounds(correction: CorrectionSpec) -> tuple[float, float]:
    """``(clip_low, clip_high)`` for :func:`version_segments.cross_version_is`.

    The spec's ``lower`` / ``upper``; :data:`OBSERVE_DEFAULT_BOUNDS` for a
    missing bound. ``cross_version_is`` needs ``low <= 1 <= high``, so the
    bounds are clamped to that and the estimate stays observe-only.
    """

    low = OBSERVE_DEFAULT_BOUNDS[0] if correction.lower is None else float(correction.lower)
    high = OBSERVE_DEFAULT_BOUNDS[1] if correction.upper is None else float(correction.upper)
    return min(low, 1.0), max(high, 1.0)


def staleness_correction(spec: AlgorithmSpec, default: str = "tis") -> str:
    """The ``StalenessPolicy.correction`` name derived from the spec (D7)."""

    from yeto.rl.algos.mismatch_correction import selected_corrections

    named = [m for m in selected_corrections(spec) if m != "mismatch_observe"]
    return named[0] if named else default
