"""Independent exact-f32 reference for angle-based CPLG directions.

This module is a golden/reference implementation, not a live optimizer hook.
It deliberately names angles, tangent directions, and tangent *ratios*
differently because CPLG commands a real angle in radians.  In particular,
``commanded_angle_radians`` must never be passed to a normalized-linear
kernel as though it were a tangent ratio: that would realize
``atan(commanded_angle_radians)`` instead of the commanded angle.

For three chronological same-fragment stock directions ``w, v, u`` (oldest
to newest), CPLG computes the previous and current forward tangents, parallel
transports the previous forward tangent to ``u``, and phase-locks the smaller
of the two observed turn angles::

    theta_now  = atan2f(sqrt_f32(1 - rho_now**2), rho_now)
    theta_prev = atan2f(sqrt_f32(1 - rho_prev**2), rho_prev)
    phi = max(+0, coherence) * min(theta_now, theta_prev, atan2f(1/4, 1))
    q = normalize(cosf(phi) * u + sinf(phi) * current_forward_tangent)

The returned direction grafts ``q`` to the exact f32 norm of the current
stock direction.  Every arithmetic primitive rounds to f32, dot products are
coordinate-order product-then-add reductions, and the two products and add in
the great-circle expression are separate operations (no FMA).

The angle cap is pinned by IEEE-754 bits.  By default ``atan2f``, ``sinf``,
and ``cosf`` come from the host platform's C libm because Python provides no
pinned f32 transcendental implementation.  An injectable trig protocol lets
tests isolate the algebraic order from libm variation.  Golden byte fixtures
bind the tested host only: cross-runtime bit parity remains explicitly
unsatisfied until the Rust implementation produces matching scalar and vector
fixtures.  This module must not be described as a portable golden equivalent
before that comparison.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from fractions import Fraction
import math
import struct
from typing import Iterable, Protocol, Sequence


CPLG_ANGLE_CAP_F32_BITS = 0x3E7ADBB0
"""Pinned f32 bits of ``atan2f(0.25f, 1.0f)`` (about 14.036 degrees)."""

CPLG_RHO_OVERSHOOT_TOLERANCE = 2.0**-20
"""Maximum admitted f32 dot-product drift beyond ``[-1, 1]``."""

CPLG_DEGENERATE_NORM_SQ = 2.0**-40
"""Strict lower safety threshold, whose f32 bits are ``0x2b800000``."""

CPLG_TRIG_BACKEND = "host-c-libm-atan2f-sinf-cosf"
CPLG_TRIG_PORTABILITY_LIMITATION = (
    "f32 transcendental last bits are bound to the host C libm; only the "
    "angle-cap input/output bits and checked golden fixtures are pinned"
)
CPLG_CROSS_RUNTIME_BIT_PARITY = "UNSATISFIED_UNTIL_RUST_FIXTURE_MATCH"
"""Do not promote this diagnostic reference without Rust-produced fixtures."""


def _f32_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def _f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


CPLG_ANGLE_CAP_RADIANS = _f32_from_bits(CPLG_ANGLE_CAP_F32_BITS)


class CPLGF32Trig(Protocol):
    """Injectable f32 transcendental contract for parity isolation tests."""

    def atan2f(self, y: float, x: float) -> float: ...

    def sinf(self, value: float) -> float: ...

    def cosf(self, value: float) -> float: ...


class CPLGPlatformLibmTrig:
    """Thin typed binding to the host's actual f32 transcendental calls."""

    def __init__(self) -> None:
        library = ctypes.CDLL(None)
        self._atan2f = library.atan2f
        self._atan2f.argtypes = [ctypes.c_float, ctypes.c_float]
        self._atan2f.restype = ctypes.c_float
        self._sinf = library.sinf
        self._sinf.argtypes = [ctypes.c_float]
        self._sinf.restype = ctypes.c_float
        self._cosf = library.cosf
        self._cosf.argtypes = [ctypes.c_float]
        self._cosf.restype = ctypes.c_float

    def atan2f(self, y: float, x: float) -> float:
        return float(self._atan2f(ctypes.c_float(y), ctypes.c_float(x)))

    def sinf(self, value: float) -> float:
        return float(self._sinf(ctypes.c_float(value)))

    def cosf(self, value: float) -> float:
        return float(self._cosf(ctypes.c_float(value)))


PLATFORM_F32_TRIG = CPLGPlatformLibmTrig()
CPLG_PLATFORM_CAP_ATAN2F_BITS = _f32_bits(PLATFORM_F32_TRIG.atan2f(0.25, 1.0))
CPLG_PLATFORM_CAP_MATCHES_PINNED = (
    CPLG_PLATFORM_CAP_ATAN2F_BITS == CPLG_ANGLE_CAP_F32_BITS
)


@dataclass(frozen=True)
class CPLGDirection:
    """One deterministic angle-based CPLG reference decision.

    All angular fields are explicitly named ``*_angle_radians``.  None is a
    normalized-linear tangent ratio.  Fallbacks preserve the caller's exact
    ``current_stock_raw`` bytes object, including NaN payloads and signed zero.
    """

    raw: bytes
    used_nonstock: bool
    reason: str
    current_turn_angle_radians: float
    previous_turn_angle_radians: float
    transported_forward_tangent_coherence: float
    commanded_angle_radians: float


def encode_f32le(values: Iterable[float]) -> bytes:
    """Encode exact little-endian IEEE-754 f32 coordinates."""

    materialized = tuple(values)
    try:
        return struct.pack(f"<{len(materialized)}f", *materialized)
    except (OverflowError, struct.error) as error:
        raise ValueError("values cannot be represented as little-endian f32") from error


def decode_f32le(raw: bytes, shape: tuple[int, ...]) -> tuple[float, ...]:
    """Decode exact bytes after validating a nonempty positive shape."""

    if type(raw) is not bytes:
        raise TypeError("CPLG directions must have exact type bytes")
    count = _shape_numel(shape)
    required = count * 4
    if len(raw) != required:
        raise ValueError(
            f"shape requires {required} bytes of f32 data, received {len(raw)}"
        )
    return struct.unpack(f"<{count}f", raw)


def cplg_angle_based_direction_f32le(
    current_stock_raw: bytes,
    previous_stock_raw: bytes,
    previous_previous_stock_raw: bytes,
    shape: tuple[int, ...],
    *,
    trig: CPLGF32Trig | None = None,
) -> CPLGDirection:
    """Return the exact-f32, angle-based CPLG direction for ``w, v, u``.

    The byte arguments are chronological in the function name: ``current`` is
    ``u``, ``previous`` is ``v``, and ``previous_previous`` is ``w``.  The
    current forward continuation tangent is built as ``rho*u - v``.  This is
    the negative of the current *backward* tangent ``v - rho*u``; the two
    signs are intentionally never represented by one ambiguous name.

    V1 admits only acute, nonstationary pairs ``0 < rho < 1``.  Dot products
    outside ``[-1-2^-20, 1+2^-20]`` are rejected before clamping.  Invalid
    numerical geometry returns bit-identical stock, while malformed byte or
    shape evidence is a strict exception.
    """

    selected_trig = PLATFORM_F32_TRIG if trig is None else trig
    for method_name in ("atan2f", "sinf", "cosf"):
        if not callable(getattr(selected_trig, method_name, None)):
            raise TypeError("trig must provide callable atan2f, sinf, and cosf")

    for raw in (
        current_stock_raw,
        previous_stock_raw,
        previous_previous_stock_raw,
    ):
        if type(raw) is not bytes:
            raise TypeError("CPLG directions must have exact type bytes")
    if type(shape) is not tuple:
        raise TypeError("shape must have exact type tuple")

    current = decode_f32le(current_stock_raw, shape)
    previous = decode_f32le(previous_stock_raw, shape)
    previous_previous = decode_f32le(previous_previous_stock_raw, shape)
    if not all(
        math.isfinite(value) for value in (*current, *previous, *previous_previous)
    ):
        return _stock(current_stock_raw, "nonfinite_direction")

    try:
        unit_current, current_norm = _normalize_f32(current)
        unit_previous, _previous_norm = _normalize_f32(previous)
        unit_previous_previous, _previous_previous_norm = _normalize_f32(
            previous_previous
        )
    except ValueError:
        return _stock(current_stock_raw, "degenerate_direction")

    current_pair = _pair_geometry_f32(
        newer=unit_current,
        older=unit_previous,
        pair_name="current",
        trig=selected_trig,
    )
    if isinstance(current_pair, str):
        return _stock(current_stock_raw, current_pair)
    current_rho, current_turn_angle_radians, current_forward_tangent = current_pair

    previous_pair = _pair_geometry_f32(
        newer=unit_previous,
        older=unit_previous_previous,
        pair_name="previous",
        trig=selected_trig,
    )
    if isinstance(previous_pair, str):
        return _stock(current_stock_raw, previous_pair)
    previous_rho, previous_turn_angle_radians, previous_forward_tangent = previous_pair
    # Retain the explicitly named value so an accidental switch to current rho
    # in future edits is caught by static review and golden tests.
    del previous_rho

    try:
        transported_previous_forward_tangent = _parallel_transport_forward_f32(
            previous_unit_direction=unit_previous,
            current_unit_direction=unit_current,
            previous_forward_tangent=previous_forward_tangent,
            previous_current_rho=current_rho,
        )
    except ValueError:
        return _stock(
            current_stock_raw,
            "degenerate_transported_previous_forward_tangent",
            current_turn_angle_radians=current_turn_angle_radians,
            previous_turn_angle_radians=previous_turn_angle_radians,
        )

    coherence_raw = _dot_f32(
        current_forward_tangent,
        transported_previous_forward_tangent,
    )
    coherence = _clamp_unit_with_overshoot_guard(coherence_raw)
    if coherence is None:
        return _stock(
            current_stock_raw,
            "transported_forward_tangent_coherence_overshoot",
            current_turn_angle_radians=current_turn_angle_radians,
            previous_turn_angle_radians=previous_turn_angle_radians,
        )

    # Preserve +0 explicitly when coherence is -0 or negative.
    positive_phase_coherence = coherence if coherence > 0.0 else _f32(0.0)
    if positive_phase_coherence == 0.0:
        return _stock(
            current_stock_raw,
            "nonpositive_transported_forward_tangent_coherence",
            current_turn_angle_radians=current_turn_angle_radians,
            previous_turn_angle_radians=previous_turn_angle_radians,
            coherence=coherence,
        )

    uncoupled_turn_angle_radians = min(
        current_turn_angle_radians,
        previous_turn_angle_radians,
        CPLG_ANGLE_CAP_RADIANS,
    )
    commanded_angle_radians = _multiply_f32(
        positive_phase_coherence,
        uncoupled_turn_angle_radians,
    )
    if commanded_angle_radians <= 0.0:
        return _stock(
            current_stock_raw,
            "zero_commanded_angle",
            current_turn_angle_radians=current_turn_angle_radians,
            previous_turn_angle_radians=previous_turn_angle_radians,
            coherence=coherence,
        )

    cosine_angle = _f32(selected_trig.cosf(commanded_angle_radians))
    sine_angle = _f32(selected_trig.sinf(commanded_angle_radians))
    great_circle_raw = tuple(
        _add_f32(
            _multiply_f32(cosine_angle, stock_component),
            _multiply_f32(sine_angle, forward_component),
        )
        for stock_component, forward_component in zip(
            unit_current,
            current_forward_tangent,
            strict=True,
        )
    )
    try:
        unit_candidate, _candidate_raw_norm = _normalize_f32(great_circle_raw)
    except ValueError:
        return _stock(
            current_stock_raw,
            "degenerate_candidate",
            current_turn_angle_radians=current_turn_angle_radians,
            previous_turn_angle_radians=previous_turn_angle_radians,
            coherence=coherence,
            commanded_angle_radians=commanded_angle_radians,
        )
    candidate = tuple(
        _multiply_f32(current_norm, component) for component in unit_candidate
    )
    raw = encode_f32le(candidate)
    sealed_candidate = decode_f32le(raw, shape)
    if not all(math.isfinite(value) for value in sealed_candidate):
        return _stock(
            current_stock_raw,
            "nonfinite_candidate",
            current_turn_angle_radians=current_turn_angle_radians,
            previous_turn_angle_radians=previous_turn_angle_radians,
            coherence=coherence,
            commanded_angle_radians=commanded_angle_radians,
        )
    if _norm_sq_f32(sealed_candidate) <= CPLG_DEGENERATE_NORM_SQ:
        return _stock(
            current_stock_raw,
            "degenerate_candidate",
            current_turn_angle_radians=current_turn_angle_radians,
            previous_turn_angle_radians=previous_turn_angle_radians,
            coherence=coherence,
            commanded_angle_radians=commanded_angle_radians,
        )
    if raw == current_stock_raw:
        return _stock(
            current_stock_raw,
            "rounded_to_stock",
            current_turn_angle_radians=current_turn_angle_radians,
            previous_turn_angle_radians=previous_turn_angle_radians,
            coherence=coherence,
            commanded_angle_radians=commanded_angle_radians,
        )
    return CPLGDirection(
        raw=raw,
        used_nonstock=True,
        reason="angle_based_phase_locked_geodesic",
        current_turn_angle_radians=current_turn_angle_radians,
        previous_turn_angle_radians=previous_turn_angle_radians,
        transported_forward_tangent_coherence=coherence,
        commanded_angle_radians=commanded_angle_radians,
    )


def cosine_gain_f32le(
    candidate_raw: bytes,
    source_stock_raw: bytes,
    future_stock_raw: bytes,
    shape: tuple[int, ...],
) -> float:
    """Return f32 ``cos(candidate,future)-cos(source_stock,future)``."""

    candidate = decode_f32le(candidate_raw, shape)
    source_stock = decode_f32le(source_stock_raw, shape)
    future_stock = decode_f32le(future_stock_raw, shape)
    if not all(
        math.isfinite(value) for value in (*candidate, *source_stock, *future_stock)
    ):
        raise ValueError("shadow score requires finite full vectors")
    candidate_cosine = _cosine_f32(candidate, future_stock)
    stock_cosine = _cosine_f32(source_stock, future_stock)
    return _subtract_f32(candidate_cosine, stock_cosine)


def _pair_geometry_f32(
    *,
    newer: Sequence[float],
    older: Sequence[float],
    pair_name: str,
    trig: CPLGF32Trig,
) -> tuple[float, float, tuple[float, ...]] | str:
    rho_raw = _dot_f32(older, newer)
    rho = _clamp_unit_with_overshoot_guard(rho_raw)
    if rho is None:
        return f"{pair_name}_rho_overshoot"
    if not 0.0 < rho < 1.0:
        return f"{pair_name}_pair_not_acute_nonstationary"

    rho_squared = _multiply_f32(rho, rho)
    sine_squared = _subtract_f32(_f32(1.0), rho_squared)
    if sine_squared < 0.0:
        return f"{pair_name}_rho_overshoot"
    sine_component = _sqrt_f32(sine_squared)
    turn_angle_radians = _f32(trig.atan2f(sine_component, rho))
    if not 0.0 < turn_angle_radians <= CPLG_ANGLE_CAP_RADIANS * 8.0:
        # The upper bound is only a finite sanity check; the actual command is
        # capped later.  V1's rho contract already implies theta < pi/2.
        if not math.isfinite(turn_angle_radians) or turn_angle_radians <= 0.0:
            return f"{pair_name}_invalid_turn_angle"

    # The *forward* tangent points from older through newer and onward:
    # rho*newer - older.  The backward tangent has the opposite sign and is
    # intentionally not used or named ambiguously here.
    forward_residual = tuple(
        _subtract_f32(
            _multiply_f32(rho, newer_component),
            older_component,
        )
        for newer_component, older_component in zip(newer, older, strict=True)
    )
    try:
        forward_tangent = _reproject_and_normalize_tangent_f32(
            forward_residual,
            newer,
        )
    except ValueError:
        return f"degenerate_{pair_name}_forward_tangent"
    return rho, turn_angle_radians, forward_tangent


def _parallel_transport_forward_f32(
    *,
    previous_unit_direction: Sequence[float],
    current_unit_direction: Sequence[float],
    previous_forward_tangent: Sequence[float],
    previous_current_rho: float,
) -> tuple[float, ...]:
    """Transport the previous *forward* tangent from ``v`` to ``u``."""

    tangent_current_dot = _dot_f32(
        previous_forward_tangent,
        current_unit_direction,
    )
    denominator = _add_f32(_f32(1.0), previous_current_rho)
    scale = _divide_f32(tangent_current_dot, denominator)
    transported_raw = tuple(
        _subtract_f32(
            tangent_component,
            _multiply_f32(
                scale,
                _add_f32(previous_component, current_component),
            ),
        )
        for tangent_component, previous_component, current_component in zip(
            previous_forward_tangent,
            previous_unit_direction,
            current_unit_direction,
            strict=True,
        )
    )
    return _reproject_and_normalize_tangent_f32(
        transported_raw,
        current_unit_direction,
    )


def _reproject_and_normalize_tangent_f32(
    tangent: Sequence[float],
    base: Sequence[float],
) -> tuple[float, ...]:
    projection = _dot_f32(tangent, base)
    reprojected = tuple(
        _subtract_f32(
            tangent_component,
            _multiply_f32(projection, base_component),
        )
        for tangent_component, base_component in zip(tangent, base, strict=True)
    )
    normalized, _norm = _normalize_f32(reprojected)
    return normalized


def _clamp_unit_with_overshoot_guard(value: float) -> float | None:
    if not math.isfinite(value):
        return None
    tolerance = _f32(CPLG_RHO_OVERSHOOT_TOLERANCE)
    lower = _subtract_f32(_f32(-1.0), tolerance)
    upper = _add_f32(_f32(1.0), tolerance)
    if value < lower or value > upper:
        return None
    return min(_f32(1.0), max(_f32(-1.0), value))


def _cosine_f32(left: Sequence[float], right: Sequence[float]) -> float:
    unit_left, _left_norm = _normalize_f32(left)
    unit_right, _right_norm = _normalize_f32(right)
    return _dot_f32(unit_left, unit_right)


def _stock(
    raw: bytes,
    reason: str,
    *,
    current_turn_angle_radians: float = 0.0,
    previous_turn_angle_radians: float = 0.0,
    coherence: float = 0.0,
    commanded_angle_radians: float = 0.0,
) -> CPLGDirection:
    return CPLGDirection(
        raw=raw,
        used_nonstock=False,
        reason=reason,
        current_turn_angle_radians=_f32(current_turn_angle_radians),
        previous_turn_angle_radians=_f32(previous_turn_angle_radians),
        transported_forward_tangent_coherence=_f32(coherence),
        commanded_angle_radians=_f32(commanded_angle_radians),
    )


def _shape_numel(shape: tuple[int, ...]) -> int:
    if type(shape) is not tuple or not shape:
        raise ValueError("shape must be a nonempty tuple")
    count = 1
    for dimension in shape:
        if type(dimension) is not int or dimension <= 0:
            raise ValueError("shape dimensions must be positive exact integers")
        count *= dimension
    return count


def _f32(value: float) -> float:
    try:
        rounded = struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError("f32 operation overflowed") from error
    if not math.isfinite(rounded):
        raise ValueError("f32 operation produced a nonfinite value")
    return rounded


def _add_f32(left: float, right: float) -> float:
    return _f32(left + right)


def _subtract_f32(left: float, right: float) -> float:
    return _f32(left - right)


def _multiply_f32(left: float, right: float) -> float:
    return _f32(left * right)


def _divide_f32(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        raise ValueError("f32 division by zero")
    return _f32(numerator / denominator)


def _sqrt_f32(value: float) -> float:
    """Correctly round a nonnegative f32 square root, ties-to-even."""

    value = _f32(value)
    if value < 0.0:
        raise ValueError("f32 square root requires a nonnegative input")
    if value == 0.0:
        return _f32(0.0)

    exact_value = Fraction.from_float(value)
    candidate = _f32(math.sqrt(value))
    while True:
        bits = _f32_bits(candidate)
        if bits > 0:
            previous = _f32_from_bits(bits - 1)
            lower_midpoint = (
                Fraction.from_float(previous) + Fraction.from_float(candidate)
            ) / 2
            lower_square = lower_midpoint * lower_midpoint
            if exact_value < lower_square or (exact_value == lower_square and bits & 1):
                candidate = previous
                continue

        following = _f32_from_bits(bits + 1)
        upper_midpoint = (
            Fraction.from_float(candidate) + Fraction.from_float(following)
        ) / 2
        upper_square = upper_midpoint * upper_midpoint
        if exact_value > upper_square or (exact_value == upper_square and bits & 1):
            candidate = following
            continue
        return candidate


def _dot_f32(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("dot-product shapes differ")
    accumulator = _f32(0.0)
    for left_value, right_value in zip(left, right, strict=True):
        product = _multiply_f32(left_value, right_value)
        accumulator = _add_f32(accumulator, product)
    return accumulator


def _norm_sq_f32(vector: Sequence[float]) -> float:
    return _dot_f32(vector, vector)


def _normalize_f32(vector: Sequence[float]) -> tuple[tuple[float, ...], float]:
    norm_sq = _norm_sq_f32(vector)
    if norm_sq <= CPLG_DEGENERATE_NORM_SQ:
        raise ValueError("direction is degenerate")
    norm = _sqrt_f32(norm_sq)
    normalized = tuple(_divide_f32(value, norm) for value in vector)
    return normalized, norm


__all__ = [
    "CPLG_ANGLE_CAP_F32_BITS",
    "CPLG_ANGLE_CAP_RADIANS",
    "CPLG_CROSS_RUNTIME_BIT_PARITY",
    "CPLG_DEGENERATE_NORM_SQ",
    "CPLGDirection",
    "CPLGF32Trig",
    "CPLGPlatformLibmTrig",
    "CPLG_PLATFORM_CAP_ATAN2F_BITS",
    "CPLG_PLATFORM_CAP_MATCHES_PINNED",
    "CPLG_RHO_OVERSHOOT_TOLERANCE",
    "CPLG_TRIG_BACKEND",
    "CPLG_TRIG_PORTABILITY_LIMITATION",
    "PLATFORM_F32_TRIG",
    "cosine_gain_f32le",
    "cplg_angle_based_direction_f32le",
    "decode_f32le",
    "encode_f32le",
]
