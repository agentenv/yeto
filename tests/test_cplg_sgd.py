"""Golden exact-f32 tests for the angle-based CPLG reference."""

from __future__ import annotations

import json
import math
from pathlib import Path
import struct
import subprocess

import pytest

from yeto.cplg_sgd import (
    CPLG_ANGLE_CAP_F32_BITS,
    CPLG_ANGLE_CAP_RADIANS,
    CPLG_CROSS_RUNTIME_BIT_PARITY,
    CPLG_DEGENERATE_NORM_SQ,
    CPLG_PLATFORM_CAP_ATAN2F_BITS,
    CPLG_PLATFORM_CAP_MATCHES_PINNED,
    CPLGReferenceMachine,
    CPLGReferenceState,
    CPLGRustLibmTrig,
    CPLG_RUST_LIBM_CROSS_RUNTIME_BIT_PARITY,
    CPLG_RUST_LIBM_TRIG_BACKEND,
    CPLG_TRIG_PORTABILITY_LIMITATION,
    PLATFORM_F32_TRIG,
    cosine_gain_f32le,
    cplg_angle_based_direction_f32le,
    decode_f32le,
    encode_f32le,
)


ROOT = Path(__file__).resolve().parents[1]
CAUSAL_TRACE = ROOT / "tests" / "fixtures" / "cplg_causal_trace_v1.json"


@pytest.fixture(scope="module")
def rust_libm_helper() -> Path:
    subprocess.run(
        ["cargo", "build", "--quiet", "--bin", "cplg_libm_oracle"],
        cwd=ROOT / "syncer",
        check=True,
    )
    helper = ROOT / "syncer" / "target" / "debug" / "cplg_libm_oracle"
    assert helper.is_file() and not helper.is_symlink()
    return helper


def _direction(degrees: float, *, norm: float = 1.0) -> bytes:
    radians = math.radians(degrees)
    return encode_f32le((norm * math.cos(radians), norm * math.sin(radians)))


def _angle_degrees(raw: bytes) -> float:
    x, y = decode_f32le(raw, (2,))
    return math.degrees(math.atan2(y, x))


def _norm(raw: bytes) -> float:
    values = decode_f32le(raw, (len(raw) // 4,))
    return math.sqrt(math.fsum(value * value for value in values))


def _f32_bits_hex(value: float | None) -> str | None:
    if value is None:
        return None
    return f"{struct.unpack('<I', struct.pack('<f', value))[0]:08x}"


def test_shared_raw_bit_causal_trace_matches_reference_and_resume(
    rust_libm_helper: Path,
) -> None:
    """Consume the same full causal fixture as the Rust production test."""

    fixture = json.loads(CAUSAL_TRACE.read_bytes())
    assert set(fixture) == {"schema", "shape", "resume_after_step", "steps"}
    assert fixture["schema"] == "cplg_causal_raw_bit_trace_v1"
    assert fixture["shape"] == [2]
    expected_step_fields = {
        "step",
        "input_f32le",
        "action_f32le",
        "candidate_f32le",
        "reason",
        "resolved_score_bits",
        "interlock_score_count",
        "interlock_open",
        "used_nonstock",
        "state_cleared",
        "next_state",
    }
    expected_state_fields = {
        "previous_stock_f32le",
        "previous_forward_tangent_f32le",
        "previous_theta_bits",
        "pending_candidate_f32le",
        "score_bits",
    }

    with CPLGRustLibmTrig(rust_libm_helper) as trig:
        machine = CPLGReferenceMachine((2,), trig=trig)
        resumed: CPLGReferenceMachine | None = None
        for expected_step, row in enumerate(fixture["steps"]):
            assert set(row) == expected_step_fields
            assert set(row["next_state"]) == expected_state_fields
            assert row["step"] == expected_step
            current = bytes.fromhex(row["input_f32le"])
            preview = machine.preview(current)
            assert machine.preview(current) == preview
            if resumed is not None:
                resumed_preview = resumed.preview(current)
                assert resumed_preview == preview
            else:
                resumed_preview = None

            next_state = preview.next_state
            assert preview.action_raw.hex() == row["action_f32le"]
            assert (
                None if preview.candidate_raw is None else preview.candidate_raw.hex()
            ) == row["candidate_f32le"]
            assert preview.reason == row["reason"]
            assert (
                _f32_bits_hex(preview.resolved_shadow_score)
                == row["resolved_score_bits"]
            )
            assert preview.interlock_score_count == row["interlock_score_count"]
            assert preview.interlock_open is row["interlock_open"]
            assert preview.used_nonstock is row["used_nonstock"]
            assert preview.state_cleared is row["state_cleared"]
            expected_state = row["next_state"]
            assert (
                None
                if next_state.previous_stock_raw is None
                else next_state.previous_stock_raw.hex()
            ) == expected_state["previous_stock_f32le"]
            assert (
                None
                if next_state.previous_forward_tangent_raw is None
                else next_state.previous_forward_tangent_raw.hex()
            ) == expected_state["previous_forward_tangent_f32le"]
            assert (
                _f32_bits_hex(next_state.previous_turn_angle_radians)
                == (expected_state["previous_theta_bits"])
            )
            assert (
                None
                if next_state.pending_candidate_raw is None
                else next_state.pending_candidate_raw.hex()
            ) == expected_state["pending_candidate_f32le"]
            assert [_f32_bits_hex(score) for score in next_state.scores] == (
                expected_state["score_bits"]
            )

            machine.commit_preview(preview)
            if resumed_preview is not None:
                resumed.commit_preview(resumed_preview)
                assert resumed.state == machine.state
            if expected_step == fixture["resume_after_step"]:
                resumed = CPLGReferenceMachine((2,), trig=trig, state=machine.state)

    signed_zero_step = fixture["steps"][9]
    assert signed_zero_step["input_f32le"].endswith("00000080")
    assert fixture["steps"][10]["resolved_score_bits"] == "00000000"


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


def test_rust_libm_helper_closes_cap_fixture_bit_gap(
    rust_libm_helper: Path,
) -> None:
    with CPLGRustLibmTrig(rust_libm_helper) as trig:
        decision = cplg_angle_based_direction_f32le(
            _direction(40.0, norm=7.0),
            _direction(20.0, norm=3.0),
            _direction(0.0, norm=5.0),
            (2,),
            trig=trig,
        )
        assert struct.pack("<f", trig.sinf(CPLG_ANGLE_CAP_RADIANS)).hex() == (
            "425b783e"
        )
        assert len(trig.executable_sha256) == 64

    assert decision.raw.hex() == "9b8c8340954db540"
    assert decision.raw.hex() != "9a8c8340954db540"
    assert CPLG_RUST_LIBM_TRIG_BACKEND == "pinned-rust-libm-0.2.15-subprocess"
    assert CPLG_RUST_LIBM_CROSS_RUNTIME_BIT_PARITY == (
        "SATISFIED_RUST_LIBM_AUTHORITATIVE"
    )


@pytest.mark.parametrize(
    (
        "current_bits",
        "previous_bits",
        "previous_previous_bits",
        "expected_raw_hex",
        "expected_scalar_hex",
    ),
    [
        (
            (0x40D27DBC, 0x4019399B),
            (0x403D1545, 0x3F055C9F),
            (0x40A00000, 0x00000000),
            "5bfdc14005006040",
            ("cfb8323e", "cfb8323e", "0000803f", "cfb8323e"),
        ),
        (
            (0x40AB980D, 0x408FFC03),
            (0x40346BC6, 0x3F8355F3),
            (0x40A00000, 0x00000000),
            "9b8c8340954db540",
            ("c7b8b23e", "c7b8b23e", "0000803f", "b0db7a3e"),
        ),
        (
            (0x3FBAEB4D, 0x3FFB8F82, 0x3F01486A),
            (0x3FD10200, 0x3FF0493D, 0x3E7A39E8),
            (0x3FE24630, 0x3FE24630, 0x00000000),
            "64b8a33ff819014014f73c3f",
            ("b41e053e", "c2c2f53d", "0000803f", "c2c2f53d"),
        ),
        (
            (0x3F800000, 0x00000000),
            (0x3F7C1C5C, 0x3E31D0D4),
            (0x3F800000, 0x00000000),
            "0000803f00000000",
            ("b7b8323e", "b7b8323e", "00000000", "00000000"),
        ),
    ],
    ids=("constant-phase", "cap", "nonplanar", "reversal"),
)
def test_rust_libm_authority_matches_production_raw_bit_fixtures(
    rust_libm_helper: Path,
    current_bits: tuple[int, ...],
    previous_bits: tuple[int, ...],
    previous_previous_bits: tuple[int, ...],
    expected_raw_hex: str,
    expected_scalar_hex: tuple[str, str, str, str],
) -> None:
    """Use the identical raw f32 inputs and Rust production expectations."""

    def raw(bits: tuple[int, ...]) -> bytes:
        return b"".join(struct.pack("<I", value) for value in bits)

    with CPLGRustLibmTrig(rust_libm_helper) as trig:
        decision = cplg_angle_based_direction_f32le(
            raw(current_bits),
            raw(previous_bits),
            raw(previous_previous_bits),
            (len(current_bits),),
            trig=trig,
        )

    actual_scalar_hex = tuple(
        struct.pack("<f", value).hex()
        for value in (
            decision.current_turn_angle_radians,
            decision.previous_turn_angle_radians,
            decision.transported_forward_tangent_coherence,
            decision.commanded_angle_radians,
        )
    )
    assert decision.raw.hex() == expected_raw_hex
    assert actual_scalar_hex == expected_scalar_hex


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
    assert decision.seal_shadow is True
    assert decision.reason == "nonpositive_transported_forward_tangent_coherence"
    assert decision.transported_forward_tangent_coherence == 0.0
    assert decision.commanded_angle_radians == 0.0
    assert decision.raw is current


def test_causal_reference_preview_commit_and_first_action(
    rust_libm_helper: Path,
) -> None:
    with CPLGRustLibmTrig(rust_libm_helper) as trig:
        machine = CPLGReferenceMachine((2,), trig=trig)
        previews = []
        for visit, degrees in enumerate((0.0, 10.0, 20.0, 30.0, 40.0, 50.0)):
            current = _direction(degrees)
            preview = machine.preview(current)
            assert machine.preview(current) == preview
            if visit < 5:
                assert preview.used_nonstock is False
                assert preview.action_raw is current
            else:
                assert preview.interlock_score_count == 3
                assert preview.interlock_open is True
                assert preview.used_nonstock is True
                assert preview.reason == "candidate_selected"
                assert preview.action_raw != current
            machine.commit_preview(preview)
            previews.append(preview)
        with pytest.raises(ValueError, match="stale or was already committed"):
            machine.commit_preview(previews[-1])


def test_causal_reference_seals_zero_phase_and_zero_closes_interlock(
    rust_libm_helper: Path,
) -> None:
    with CPLGRustLibmTrig(rust_libm_helper) as trig:
        machine = CPLGReferenceMachine((2,), trig=trig)
        for degrees in (0.0, 10.0):
            preview = machine.preview(_direction(degrees))
            machine.commit_preview(preview)
        reversal = _direction(0.0)
        preview = machine.preview(reversal)
        assert preview.reason == "zero_or_rounded_phase"
        assert preview.sealed_shadow is True
        assert preview.candidate_raw is reversal
        assert preview.used_nonstock is False
        machine.commit_preview(preview)

        following = machine.preview(_direction(10.0))
        assert following.resolved_shadow_score == 0.0
        assert following.interlock_score_count == 1
        assert following.interlock_open is False
        assert following.used_nonstock is False


def test_causal_reference_nonacute_clears_and_requires_fresh_warmup(
    rust_libm_helper: Path,
) -> None:
    with CPLGRustLibmTrig(rust_libm_helper) as trig:
        machine = CPLGReferenceMachine((2,), trig=trig)
        first = machine.preview(_direction(0.0))
        machine.commit_preview(first)
        discontinuity = machine.preview(_direction(100.0))
        assert discontinuity.reason == "nonacute_turn"
        assert discontinuity.state_cleared is True
        assert discontinuity.next_state == CPLGReferenceState()
        machine.commit_preview(discontinuity)
        recovered = machine.preview(_direction(10.0))
        assert recovered.reason == "stock_warmup"


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
