"""Focused exact-f32 geometry tests for the state-free CGC kernel."""

from __future__ import annotations

import math

import pytest

from yeto.cgc_sgd import CGC_MAX_TANGENT, cgc_direction_f32le
from yeto.pti_sgd import decode_f32le, encode_f32le


def _direction(degrees: float, norm: float = 1.0) -> bytes:
    radians = math.radians(degrees)
    return encode_f32le((norm * math.cos(radians), norm * math.sin(radians)))


def _angle(raw: bytes) -> float:
    x, y = decode_f32le(raw, (2,))
    return math.degrees(math.atan2(y, x))


def _norm(raw: bytes) -> float:
    values = decode_f32le(raw, (len(raw) // 4,))
    return math.sqrt(math.fsum(value * value for value in values))


def test_constant_angular_velocity_continues_small_turn() -> None:
    previous = _direction(0.0, norm=3.0)
    current = _direction(10.0, norm=7.0)

    decision = cgc_direction_f32le(current, previous, (2,))

    assert decision.used_nonstock is True
    assert decision.capped is False
    assert decision.reason == "geodesic_continuation"
    assert decision.tangent == pytest.approx(math.tan(math.radians(10.0)), abs=2e-7)
    assert _angle(decision.raw) == pytest.approx(20.0, abs=2e-5)
    assert _norm(decision.raw) == pytest.approx(_norm(current), abs=1e-6)


def test_large_turn_is_bounded_by_quarter_tangent() -> None:
    previous = _direction(0.0)
    current = _direction(30.0)

    decision = cgc_direction_f32le(current, previous, (2,))

    assert decision.used_nonstock is True
    assert decision.capped is True
    assert CGC_MAX_TANGENT == 0.25
    assert _angle(decision.raw) - _angle(current) == pytest.approx(
        math.degrees(math.atan(0.25)), abs=2e-5
    )


@pytest.mark.parametrize(
    ("previous", "reason"),
    [
        (_direction(10.0), "stationary_direction"),
        (_direction(100.0), "nonacute_history"),
        (encode_f32le((0.0, 0.0)), "degenerate_direction"),
        (encode_f32le((math.nan, 1.0)), "nonfinite_direction"),
    ],
)
def test_unsupported_geometry_returns_identical_stock_object(
    previous: bytes, reason: str
) -> None:
    current = _direction(10.0)

    decision = cgc_direction_f32le(current, previous, (2,))

    assert decision.used_nonstock is False
    assert decision.reason == reason
    assert decision.raw is current


def test_deterministic_exact_f32_vector_is_frozen() -> None:
    current = _direction(10.0)
    previous = _direction(0.0)

    first = cgc_direction_f32le(current, previous, (2,))
    second = cgc_direction_f32le(current, previous, (2,))

    assert first == second
    assert first.capped is False
    assert first.raw.hex() == "b18f703f421daf3e"


def test_malformed_evidence_is_strict_instead_of_relabelled_as_stock() -> None:
    current = _direction(10.0)
    previous = _direction(0.0)

    with pytest.raises(ValueError, match="requires 12 bytes"):
        cgc_direction_f32le(current, previous, (3,))
    with pytest.raises(TypeError, match="exact type bytes"):
        cgc_direction_f32le(bytearray(current), previous, (2,))  # type: ignore[arg-type]
