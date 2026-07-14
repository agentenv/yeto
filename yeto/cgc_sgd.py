"""Exact bounded direction kernel for Causal Geodesic-Continuation SGD.

CGC extrapolates the angular change between two consecutive stock outer
directions.  It preserves the current direction norm, caps the continuation
at ``atan(1/4)``, and returns the *identical* current ``bytes`` object whenever
the geometry cannot authorize a non-stock direction.

This module is deliberately state-free.  It implements no causal interlock,
optimizer update, loss gate, or empirical-performance claim.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from yeto.pti_sgd import (
    PTI_DEGENERATE_NORM_SQ,
    _divide_f32,
    _dot_f32,
    _f32,
    _multiply_f32,
    _norm_sq,
    _normalize_f32,
    _subtract_f32,
    decode_f32le,
    encode_f32le,
)


CGC_MAX_TANGENT = 0.25
"""Maximum transverse/radial ratio; the turn is at most ``atan(1/4)``."""


@dataclass(frozen=True)
class CGCDirection:
    """One deterministic CGC kernel decision.

    ``raw`` is newly materialized only for an authorized non-stock direction.
    Every fallback retains the exact ``current_raw`` object supplied by the
    caller, so callers can check the stock contract with ``is`` as well as
    byte equality.
    """

    raw: bytes
    used_nonstock: bool
    tangent: float
    capped: bool
    reason: str


def cgc_direction_f32le(
    current_raw: bytes,
    previous_raw: bytes,
    shape: tuple[int, ...],
) -> CGCDirection:
    """Return a bounded angular continuation of two exact f32 directions.

    For unit current and previous directions ``u`` and ``v``, let
    ``rho = dot(u, v)`` and let ``b`` be the unit component of ``v``
    orthogonal to ``u``.  When ``rho > 0``, CGC returns the norm-grafted

    ``normalize(u - lambda * b)``, where
    ``lambda = min(sqrt(1-rho^2)/rho, 1/4)``.

    Below the cap this is constant-angular-velocity continuation on the unit
    sphere.  Nonfinite, degenerate, stationary, nonacute, or f32-inert geometry
    returns the identical current bytes object.  Malformed byte/shape inputs
    are programmer/evidence errors and remain strict exceptions.
    """

    if type(current_raw) is not bytes or type(previous_raw) is not bytes:
        raise TypeError("CGC directions must have exact type bytes")
    if type(shape) is not tuple:
        raise TypeError("shape must have exact type tuple")

    current = decode_f32le(current_raw, shape)
    previous = decode_f32le(previous_raw, shape)
    if not all(math.isfinite(value) for value in (*current, *previous)):
        return _stock(current_raw, "nonfinite_direction")

    try:
        unit_current, current_norm = _normalize_f32(current)
        unit_previous, _previous_norm = _normalize_f32(previous)
    except ValueError:
        return _stock(current_raw, "degenerate_direction")

    rho = _dot_f32(unit_previous, unit_current)
    if not math.isfinite(rho):
        return _stock(current_raw, "nonfinite_geometry")
    # Extrapolation across a right angle is deliberately unsupported: the
    # tangent ratio is singular at rho=0 and changes orientation for rho<0.
    if rho <= 0.0:
        return _stock(current_raw, "nonacute_history")

    transverse = tuple(
        _subtract_f32(old, _multiply_f32(rho, new))
        for old, new in zip(unit_previous, unit_current, strict=True)
    )
    try:
        unit_backward, transverse_norm = _normalize_f32(transverse)
    except ValueError:
        return _stock(current_raw, "stationary_direction")

    try:
        tangent = _divide_f32(transverse_norm, rho)
    except ValueError:
        return _stock(current_raw, "nonfinite_geometry")
    if not math.isfinite(tangent) or tangent <= 0.0:
        return _stock(current_raw, "stationary_direction")
    max_tangent = _f32(CGC_MAX_TANGENT)
    coefficient = min(tangent, max_tangent)
    capped = tangent > max_tangent

    raw_direction = tuple(
        _subtract_f32(new, _multiply_f32(coefficient, backward))
        for new, backward in zip(unit_current, unit_backward, strict=True)
    )
    try:
        unit_candidate, _candidate_norm = _normalize_f32(raw_direction)
    except ValueError:
        return _stock(current_raw, "degenerate_candidate")
    candidate = tuple(
        _multiply_f32(current_norm, component) for component in unit_candidate
    )
    raw = encode_f32le(candidate)
    if raw == current_raw:
        return _stock(current_raw, "rounded_to_stock")
    if not all(math.isfinite(value) for value in candidate):
        return _stock(current_raw, "nonfinite_candidate")
    if _norm_sq(candidate) <= PTI_DEGENERATE_NORM_SQ:
        return _stock(current_raw, "degenerate_candidate")
    return CGCDirection(
        raw=raw,
        used_nonstock=True,
        tangent=tangent,
        capped=capped,
        reason="geodesic_continuation",
    )


def _stock(raw: bytes, reason: str) -> CGCDirection:
    return CGCDirection(
        raw=raw,
        used_nonstock=False,
        tangent=0.0,
        capped=False,
        reason=reason,
    )


__all__ = [
    "CGC_MAX_TANGENT",
    "CGCDirection",
    "cgc_direction_f32le",
]
