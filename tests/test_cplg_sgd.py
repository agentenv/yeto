"""Golden exact-f32 tests for the angle-based CPLG reference."""

from __future__ import annotations

import math
import struct

import pytest

from yeto.cplg_sgd import (
    CPLG_ANGLE_CAP_F32_BITS,
    CPLG_ANGLE_CAP_RADIANS,
    CPLG_CROSS_RUNTIME_BIT_PARITY,
    CPLG_DEGENERATE_NORM_SQ,
    CPLG_PLATFORM_CAP_ATAN2F_BITS,
    CPLG_PLATFORM_CAP_MATCHES_PINNED,
    CPLG_TRIG_PORTABILITY_LIMITATION,
    PLATFORM_F32_TRIG,
    cosine_gain_f32le,
    cplg_angle_based_direction_f32le,
    decode_f32le,
    encode_f32le,
)


def _direction(degrees: float, *, norm: float = 1.0) -> bytes:
    radians = math.radians(degrees)
    return encode_f32le((norm * math.cos(radians), norm * math.sin(radians)))


def _angle_degrees(raw: bytes) -> float:
    x, y = decode_f32le(raw, (2,))
    return math.degrees(math.atan2(y, x))


def _norm(raw: bytes) -> float:
    values = decode_f32le(raw, (len(raw) // 4,))
    return math.sqrt(math.fsum(value * value for value in values))


def test_pinned_angle_cap_and_portability_status_are_explicit() -> None:
    assert CPLG_ANGLE_CAP_F32_BITS == 0x3E7ADBB0
    assert struct.unpack("<I", struct.pack("<f", CPLG_ANGLE_CAP_RADIANS))[0] == (
        CPLG_ANGLE_CAP_F32_BITS
    )
    assert CPLG_PLATFORM_CAP_ATAN2F_BITS == CPLG_ANGLE_CAP_F32_BITS
    assert CPLG_PLATFORM_CAP_MATCHES_PINNED is True
    assert CPLG_DEGENERATE_NORM_SQ == 2.0**-40
    assert struct.unpack("<I", struct.pack("<f", CPLG_DEGENERATE_NORM_SQ))[0] == (
        0x2B800000
    )
    assert CPLG_CROSS_RUNTIME_BIT_PARITY == "UNSATISFIED_UNTIL_RUST_FIXTURE_MATCH"
    assert "host C libm" in CPLG_TRIG_PORTABILITY_LIMITATION


def test_forward_tangent_sign_continues_rotation_not_backtracks() -> None:
    previous_previous = _direction(0.0, norm=5.0)
    previous = _direction(10.0, norm=3.0)
    current = _direction(20.0, norm=7.0)

    decision = cplg_angle_based_direction_f32le(
        current,
        previous,
        previous_previous,
        (2,),
    )

    current_x, current_y = decode_f32le(current, (2,))
    candidate_x, candidate_y = decode_f32le(decision.raw, (2,))
    signed_forward_cross = current_x * candidate_y - current_y * candidate_x

    assert decision.used_nonstock is True
    assert decision.reason == "angle_based_phase_locked_geodesic"
    # rho*u-v is the forward tangent.  Using v-rho*u would rotate back to 10°.
    assert _angle_degrees(decision.raw) == pytest.approx(30.0, abs=2e-5)
    assert _angle_degrees(decision.raw) != pytest.approx(10.0, abs=1.0)
    assert signed_forward_cross > 0.0
    assert struct.pack("<f", decision.current_turn_angle_radians).hex() == "cfb8323e"
    assert struct.pack("<f", decision.previous_turn_angle_radians).hex() == ("cfb8323e")
    assert (
        struct.pack("<f", decision.transported_forward_tangent_coherence).hex()
        == "0000803f"
    )
    assert struct.pack("<f", decision.commanded_angle_radians).hex() == "cfb8323e"
    assert decision.raw.hex() == "5bfdc14005006040"
    assert _norm(decision.raw) == pytest.approx(_norm(current), abs=2e-6)


def test_command_is_angle_radians_not_normalized_linear_tangent_ratio() -> None:
    decision = cplg_angle_based_direction_f32le(
        _direction(20.0, norm=7.0),
        _direction(10.0, norm=3.0),
        _direction(0.0, norm=5.0),
        (2,),
    )

    realized_turn_degrees = _angle_degrees(decision.raw) - 20.0
    wrong_ratio_turn_degrees = math.degrees(math.atan(decision.commanded_angle_radians))

    assert realized_turn_degrees == pytest.approx(
        math.degrees(decision.commanded_angle_radians), abs=2e-5
    )
    # Passing the commanded angle to a CGC-style ratio kernel would realize
    # atan(phi), which is measurably smaller even for this ten-degree case.
    assert abs(realized_turn_degrees - wrong_ratio_turn_degrees) > 0.09


def test_cap_is_a_real_angle_not_the_tangent_ratio_of_another_angle() -> None:
    decision = cplg_angle_based_direction_f32le(
        _direction(40.0, norm=7.0),
        _direction(20.0, norm=3.0),
        _direction(0.0, norm=5.0),
        (2,),
    )

    assert decision.commanded_angle_radians == CPLG_ANGLE_CAP_RADIANS
    assert _angle_degrees(decision.raw) == pytest.approx(
        40.0 + math.degrees(CPLG_ANGLE_CAP_RADIANS),
        abs=2e-5,
    )
    assert _angle_degrees(decision.raw) != pytest.approx(
        40.0 + math.degrees(math.atan(CPLG_ANGLE_CAP_RADIANS)),
        abs=0.02,
    )
    assert decision.raw.hex() == "9a8c8340954db540"


def test_nonplanar_parallel_transport_fixture_is_frozen() -> None:
    inverse_sqrt_two = 1.0 / math.sqrt(2.0)
    inverse_sqrt_six = 1.0 / math.sqrt(6.0)
    plane_x = (inverse_sqrt_two, inverse_sqrt_two, 0.0)
    plane_y = (-inverse_sqrt_six, inverse_sqrt_six, 2.0 * inverse_sqrt_six)

    def plane_direction(radians: float) -> bytes:
        return encode_f32le(
            tuple(
                2.5 * (math.cos(radians) * x + math.sin(radians) * y)
                for x, y in zip(plane_x, plane_y, strict=True)
            )
        )

    decision = cplg_angle_based_direction_f32le(
        plane_direction(0.25),
        plane_direction(0.12),
        plane_direction(0.0),
        (3,),
    )

    assert decision.used_nonstock is True
    assert decision.transported_forward_tangent_coherence == 1.0
    assert decision.previous_turn_angle_radians < decision.current_turn_angle_radians
    assert decision.commanded_angle_radians == decision.previous_turn_angle_radians
    assert decision.raw.hex() == "64b8a33ff819014014f73c3f"


def test_reversal_gives_identical_stock_via_nonpositive_phase_coherence() -> None:
    current = _direction(0.0)
    decision = cplg_angle_based_direction_f32le(
        current,
        _direction(10.0),
        _direction(0.0),
        (2,),
    )

    assert decision.used_nonstock is False
    assert decision.reason == "nonpositive_transported_forward_tangent_coherence"
    assert decision.transported_forward_tangent_coherence == -1.0
    assert decision.commanded_angle_radians == 0.0
    assert decision.raw is current


class _RecordingTrig:
    def __init__(self) -> None:
        self.atan2_inputs: list[tuple[float, float]] = []
        self.sin_inputs: list[float] = []
        self.cos_inputs: list[float] = []

    def atan2f(self, y: float, x: float) -> float:
        self.atan2_inputs.append((y, x))
        return PLATFORM_F32_TRIG.atan2f(y, x)

    def sinf(self, value: float) -> float:
        self.sin_inputs.append(value)
        return PLATFORM_F32_TRIG.sinf(value)

    def cosf(self, value: float) -> float:
        self.cos_inputs.append(value)
        return PLATFORM_F32_TRIG.cosf(value)


def test_injectable_trig_sees_angles_not_tangent_ratios() -> None:
    trig = _RecordingTrig()
    decision = cplg_angle_based_direction_f32le(
        _direction(20.0),
        _direction(10.0),
        _direction(0.0),
        (2,),
        trig=trig,
    )

    assert len(trig.atan2_inputs) == 2
    assert trig.sin_inputs == [decision.commanded_angle_radians]
    assert trig.cos_inputs == [decision.commanded_angle_radians]
    assert trig.sin_inputs[0] != pytest.approx(
        math.tan(decision.commanded_angle_radians), abs=1e-3
    )


def test_shadow_cosine_gain_uses_full_vectors_and_sequential_f32() -> None:
    source_stock = _direction(20.0)
    candidate = cplg_angle_based_direction_f32le(
        source_stock,
        _direction(10.0),
        _direction(0.0),
        (2,),
    ).raw
    future = _direction(30.0)

    gain = cosine_gain_f32le(candidate, source_stock, future, (2,))

    assert gain > 0.0
    assert struct.pack("<f", gain).hex() == "00e9783c"


@pytest.mark.parametrize(
    ("previous", "previous_previous", "reason"),
    [
        (_direction(20.0), _direction(10.0), "current_pair_not_acute_nonstationary"),
        (_direction(110.0), _direction(0.0), "current_pair_not_acute_nonstationary"),
        (encode_f32le((0.0, 0.0)), _direction(0.0), "degenerate_direction"),
        (encode_f32le((math.nan, 1.0)), _direction(0.0), "nonfinite_direction"),
    ],
)
def test_unsupported_geometry_returns_identical_stock_object(
    previous: bytes,
    previous_previous: bytes,
    reason: str,
) -> None:
    current = _direction(20.0)

    decision = cplg_angle_based_direction_f32le(
        current,
        previous,
        previous_previous,
        (2,),
    )

    assert decision.used_nonstock is False
    assert decision.reason == reason
    assert decision.raw is current


def test_malformed_evidence_is_strict() -> None:
    current = _direction(20.0)
    previous = _direction(10.0)
    previous_previous = _direction(0.0)

    with pytest.raises(ValueError, match="requires 12 bytes"):
        cplg_angle_based_direction_f32le(
            current,
            previous,
            previous_previous,
            (3,),
        )
    with pytest.raises(TypeError, match="exact type bytes"):
        cplg_angle_based_direction_f32le(
            bytearray(current),  # type: ignore[arg-type]
            previous,
            previous_previous,
            (2,),
        )
