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
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
from typing import Any, Iterable, Protocol, Sequence


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

CPLG_RUST_LIBM_ORACLE_SCHEMA = "cplg_rust_libm_oracle_v1"
CPLG_RUST_LIBM_VERSION = "0.2.15"
CPLG_RUST_LIBM_TRIG_BACKEND = "pinned-rust-libm-0.2.15-subprocess"
CPLG_RUST_LIBM_CROSS_RUNTIME_BIT_PARITY = "SATISFIED_RUST_LIBM_AUTHORITATIVE"


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


class CPLGRustLibmTrig:
    """Persistent raw-bit client for the pinned Rust-libm oracle.

    The helper supplies only three transcendental primitives.  All vector,
    reduction, transport, normalization, and state semantics remain in this
    independent Python reference.  The executable's SHA-256 is exposed for a
    run specification and report to pin before any outcome is opened.
    """

    def __init__(self, executable: Path) -> None:
        if not isinstance(executable, Path):
            raise TypeError("Rust libm helper path must be a pathlib Path")
        if executable.is_symlink() or not executable.is_file():
            raise ValueError("Rust libm helper must be a regular non-symlink file")
        self.executable = executable.resolve(strict=True)
        self.executable_sha256 = hashlib.sha256(
            self.executable.read_bytes()
        ).hexdigest()
        self._next_id = 1
        self._closed = False
        self._process = subprocess.Popen(
            [str(self.executable)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        if self._process.stdin is None or self._process.stdout is None:
            self.close()
            raise RuntimeError("Rust libm helper pipes were not created")
        ready_line = self._process.stdout.readline()
        if not ready_line:
            detail = self._finished_stderr()
            self.close()
            raise RuntimeError(f"Rust libm helper failed before handshake: {detail}")
        ready = self._strict_json_object(ready_line, context="Rust libm handshake")
        expected = {
            "schema": CPLG_RUST_LIBM_ORACLE_SCHEMA,
            "status": "ready",
            "libm_version": CPLG_RUST_LIBM_VERSION,
            "angle_cap_bits": f"{CPLG_ANGLE_CAP_F32_BITS:08x}",
            "cap_sinf_bits": "3e785b42",
            "cap_cosf_bits": "3f785b42",
        }
        if ready != expected:
            self.close()
            raise RuntimeError(
                "Rust libm helper handshake differs from pinned contract"
            )

    def __enter__(self) -> CPLGRustLibmTrig:
        if self._closed:
            raise RuntimeError("Rust libm helper is already closed")
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        process = self._process
        if process.stdin is not None:
            try:
                process.stdin.close()
            except BrokenPipeError:
                pass
        try:
            return_code = process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5.0)
            raise RuntimeError("Rust libm helper did not stop after stdin closed")
        if return_code != 0:
            detail = process.stderr.read().strip() if process.stderr is not None else ""
            raise RuntimeError(
                f"Rust libm helper exited with status {return_code}: {detail}"
            )

    def atan2f(self, y: float, x: float) -> float:
        return self._request("atan2f", x=x, y=y)

    def sinf(self, value: float) -> float:
        return self._request("sinf", x=value)

    def cosf(self, value: float) -> float:
        return self._request("cosf", x=value)

    def _request(self, operation: str, *, x: float, y: float | None = None) -> float:
        if self._closed:
            raise RuntimeError("Rust libm helper is closed")
        request_id = self._next_id
        self._next_id += 1
        request: dict[str, Any] = {
            "id": request_id,
            "op": operation,
            "x_bits": f"{_f32_bits(_f32(x)):08x}",
        }
        if y is not None:
            request["y_bits"] = f"{_f32_bits(_f32(y)):08x}"
        assert self._process.stdin is not None
        assert self._process.stdout is not None
        try:
            self._process.stdin.write(
                json.dumps(request, sort_keys=True, separators=(",", ":")) + "\n"
            )
            self._process.stdin.flush()
        except BrokenPipeError as error:
            detail = self._finished_stderr()
            raise RuntimeError(f"Rust libm helper pipe closed: {detail}") from error
        response_line = self._process.stdout.readline()
        if not response_line:
            detail = self._finished_stderr()
            raise RuntimeError(f"Rust libm helper returned no response: {detail}")
        response = self._strict_json_object(response_line, context="Rust libm response")
        if set(response) != {"schema", "id", "result_bits"}:
            raise RuntimeError("Rust libm helper response fields differ from contract")
        if response["schema"] != CPLG_RUST_LIBM_ORACLE_SCHEMA:
            raise RuntimeError("Rust libm helper response schema changed")
        if type(response["id"]) is not int or response["id"] != request_id:
            raise RuntimeError("Rust libm helper response ID mismatch")
        result_bits = response["result_bits"]
        if (
            type(result_bits) is not str
            or len(result_bits) != 8
            or any(character not in "0123456789abcdef" for character in result_bits)
        ):
            raise RuntimeError("Rust libm helper returned noncanonical result bits")
        result = _f32_from_bits(int(result_bits, 16))
        if not math.isfinite(result):
            raise RuntimeError("Rust libm helper returned nonfinite f32")
        return result

    def _finished_stderr(self) -> str:
        try:
            self._process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            return "process still running without protocol output"
        if self._process.stderr is None:
            return "no stderr"
        return self._process.stderr.read().strip() or "empty stderr"

    @staticmethod
    def _strict_json_object(raw: str, *, context: str) -> dict[str, Any]:
        def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise RuntimeError(f"{context} contains duplicate field {key!r}")
                result[key] = value
            return result

        try:
            parsed = json.loads(raw, object_pairs_hook=object_pairs)
        except (json.JSONDecodeError, RuntimeError) as error:
            raise RuntimeError(f"{context} is not canonical JSON") from error
        if type(parsed) is not dict:
            raise RuntimeError(f"{context} must be a JSON object")
        return parsed


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
    seal_shadow: bool
    reason: str
    current_turn_angle_radians: float
    previous_turn_angle_radians: float
    transported_forward_tangent_coherence: float
    commanded_angle_radians: float


@dataclass(frozen=True)
class CPLGReferenceState:
    """Independent causal state for one fragment of the CPLG reference."""

    previous_stock_raw: bytes | None = None
    previous_forward_tangent_raw: bytes | None = None
    previous_turn_angle_radians: float | None = None
    pending_candidate_raw: bytes | None = None
    scores: tuple[float, ...] = ()

    def validate(self, shape: tuple[int, ...]) -> None:
        numel = _shape_numel(shape)
        expected_bytes = numel * 4
        raw_fields = (
            self.previous_stock_raw,
            self.previous_forward_tangent_raw,
            self.pending_candidate_raw,
        )
        for raw in raw_fields:
            if raw is not None and (
                type(raw) is not bytes or len(raw) != expected_bytes
            ):
                raise ValueError("CPLG reference state byte field has invalid shape")
        tangent_present = self.previous_forward_tangent_raw is not None
        theta_present = self.previous_turn_angle_radians is not None
        if tangent_present != theta_present:
            raise ValueError("CPLG reference tangent and angle presence differ")
        if self.previous_stock_raw is None and (
            tangent_present or self.pending_candidate_raw is not None or self.scores
        ):
            raise ValueError("empty CPLG reference state has dependent fields")
        if self.pending_candidate_raw is not None and not tangent_present:
            raise ValueError("CPLG reference pending candidate lacks phase history")
        if len(self.scores) > 3 or any(
            not math.isfinite(score) for score in self.scores
        ):
            raise ValueError("CPLG reference score window is invalid")
        if self.scores and self.pending_candidate_raw is None:
            raise ValueError("CPLG reference scores require an active shadow stream")
        if self.previous_turn_angle_radians is not None and (
            not math.isfinite(self.previous_turn_angle_radians)
            or self.previous_turn_angle_radians <= 0.0
        ):
            raise ValueError("CPLG reference previous angle is invalid")
        for raw in raw_fields:
            if raw is not None and not all(
                math.isfinite(value) for value in decode_f32le(raw, shape)
            ):
                raise ValueError("CPLG reference state contains nonfinite bytes")


@dataclass(frozen=True)
class CPLGReferencePreview:
    """Pure causal preview whose state is installed only by commit."""

    prior_state: CPLGReferenceState
    next_state: CPLGReferenceState
    action_raw: bytes
    candidate_raw: bytes | None
    sealed_shadow: bool
    used_nonstock: bool
    interlock_open: bool
    resolved_shadow_score: float | None
    interlock_score_count: int
    state_cleared: bool
    reason: str
    current_turn_angle_radians: float = 0.0
    previous_turn_angle_radians: float = 0.0
    transported_forward_tangent_coherence: float = 0.0
    commanded_angle_radians: float = 0.0


class CPLGReferenceMachine:
    """Small commit-once wrapper around the pure fragment reference."""

    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        trig: CPLGF32Trig | None = None,
        state: CPLGReferenceState | None = None,
    ) -> None:
        _shape_numel(shape)
        self.shape = shape
        self.trig = trig
        self.state = CPLGReferenceState() if state is None else state
        self.state.validate(shape)

    def preview(self, current_stock_raw: bytes) -> CPLGReferencePreview:
        return preview_cplg_reference_f32le(
            current_stock_raw,
            self.shape,
            self.state,
            trig=self.trig,
        )

    def commit_preview(self, preview: CPLGReferencePreview) -> None:
        if type(preview) is not CPLGReferencePreview:
            raise TypeError("commit requires an exact CPLGReferencePreview")
        if preview.prior_state != self.state:
            raise ValueError("CPLG preview is stale or was already committed")
        preview.next_state.validate(self.shape)
        self.state = preview.next_state


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
            coherence=positive_phase_coherence,
            seal_shadow=True,
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
            seal_shadow=True,
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
            seal_shadow=True,
        )
    return CPLGDirection(
        raw=raw,
        used_nonstock=True,
        seal_shadow=True,
        reason="angle_based_phase_locked_geodesic",
        current_turn_angle_radians=current_turn_angle_radians,
        previous_turn_angle_radians=previous_turn_angle_radians,
        transported_forward_tangent_coherence=coherence,
        commanded_angle_radians=commanded_angle_radians,
    )


def preview_cplg_reference_f32le(
    current_stock_raw: bytes,
    shape: tuple[int, ...],
    prior: CPLGReferenceState,
    *,
    trig: CPLGF32Trig | None = None,
) -> CPLGReferencePreview:
    """Pure one-boundary CPLG preview matching the frozen production timing."""

    if type(current_stock_raw) is not bytes:
        raise TypeError("CPLG directions must have exact type bytes")
    if type(prior) is not CPLGReferenceState:
        raise TypeError("prior must have exact type CPLGReferenceState")
    prior.validate(shape)
    current = decode_f32le(current_stock_raw, shape)
    selected_trig = PLATFORM_F32_TRIG if trig is None else trig
    for method_name in ("atan2f", "sinf", "cosf"):
        if not callable(getattr(selected_trig, method_name, None)):
            raise TypeError("trig must provide callable atan2f, sinf, and cosf")

    def fallback(
        reason: str,
        *,
        resolved_score: float | None = None,
        score_count: int = 0,
    ) -> CPLGReferencePreview:
        return CPLGReferencePreview(
            prior_state=prior,
            next_state=CPLGReferenceState(),
            action_raw=current_stock_raw,
            candidate_raw=None,
            sealed_shadow=False,
            used_nonstock=False,
            interlock_open=False,
            resolved_shadow_score=resolved_score,
            interlock_score_count=score_count,
            state_cleared=True,
            reason=reason,
        )

    if not all(math.isfinite(value) for value in current):
        return fallback("degenerate_stock")
    try:
        unit_current, current_norm = _normalize_f32(current)
    except ValueError:
        return fallback("degenerate_stock")

    if prior.previous_stock_raw is None:
        return CPLGReferencePreview(
            prior_state=prior,
            next_state=CPLGReferenceState(previous_stock_raw=current_stock_raw),
            action_raw=current_stock_raw,
            candidate_raw=None,
            sealed_shadow=False,
            used_nonstock=False,
            interlock_open=False,
            resolved_shadow_score=None,
            interlock_score_count=0,
            state_cleared=False,
            reason="stock_warmup",
        )

    scores = list(prior.scores)
    resolved_score: float | None = None
    if prior.pending_candidate_raw is not None:
        try:
            resolved_score = cosine_gain_f32le(
                prior.pending_candidate_raw,
                prior.previous_stock_raw,
                current_stock_raw,
                shape,
            )
        except ValueError:
            return fallback("invalid_shadow_score", score_count=len(scores))
        if len(scores) == 3:
            del scores[0]
        scores.append(resolved_score)

    previous = decode_f32le(prior.previous_stock_raw, shape)
    try:
        unit_previous, _previous_norm = _normalize_f32(previous)
    except ValueError:
        return fallback(
            "invalid_geometry",
            resolved_score=resolved_score,
            score_count=len(scores),
        )

    current_pair = _pair_geometry_f32(
        newer=unit_current,
        older=unit_previous,
        pair_name="current",
        trig=selected_trig,
    )
    if isinstance(current_pair, str):
        reason = (
            "nonacute_turn"
            if current_pair == "current_pair_not_acute_nonstationary"
            else "invalid_geometry"
        )
        return fallback(
            reason,
            resolved_score=resolved_score,
            score_count=len(scores),
        )
    current_rho, current_turn_angle, current_forward_tangent = current_pair
    current_forward_tangent_raw = encode_f32le(current_forward_tangent)

    if prior.previous_forward_tangent_raw is None:
        next_state = CPLGReferenceState(
            previous_stock_raw=current_stock_raw,
            previous_forward_tangent_raw=current_forward_tangent_raw,
            previous_turn_angle_radians=current_turn_angle,
            scores=tuple(scores),
        )
        return CPLGReferencePreview(
            prior_state=prior,
            next_state=next_state,
            action_raw=current_stock_raw,
            candidate_raw=None,
            sealed_shadow=False,
            used_nonstock=False,
            interlock_open=False,
            resolved_shadow_score=resolved_score,
            interlock_score_count=len(scores),
            state_cleared=False,
            reason="phase_warmup",
            current_turn_angle_radians=current_turn_angle,
        )

    assert prior.previous_turn_angle_radians is not None
    previous_forward_tangent = decode_f32le(
        prior.previous_forward_tangent_raw,
        shape,
    )
    try:
        transported = _parallel_transport_forward_f32(
            previous_unit_direction=unit_previous,
            current_unit_direction=unit_current,
            previous_forward_tangent=previous_forward_tangent,
            previous_current_rho=current_rho,
        )
        coherence_raw = _dot_f32(current_forward_tangent, transported)
        checked_coherence = _clamp_unit_with_overshoot_guard(coherence_raw)
        if checked_coherence is None:
            raise ValueError("coherence overshoot")
        coherence = checked_coherence if checked_coherence > 0.0 else _f32(0.0)
        theta_star = min(
            current_turn_angle,
            prior.previous_turn_angle_radians,
            CPLG_ANGLE_CAP_RADIANS,
        )
        commanded_angle = _multiply_f32(coherence, theta_star)
        if not math.isfinite(theta_star) or theta_star <= 0.0:
            raise ValueError("invalid phase angle")

        if commanded_angle == 0.0:
            candidate_raw = current_stock_raw
        else:
            cosine_angle = _f32(selected_trig.cosf(commanded_angle))
            sine_angle = _f32(selected_trig.sinf(commanded_angle))
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
            unit_candidate, _candidate_norm = _normalize_f32(great_circle_raw)
            candidate_raw = encode_f32le(
                _multiply_f32(current_norm, component) for component in unit_candidate
            )
            sealed_candidate = decode_f32le(candidate_raw, shape)
            if not all(math.isfinite(value) for value in sealed_candidate):
                raise ValueError("nonfinite candidate")
            if _norm_sq_f32(sealed_candidate) <= CPLG_DEGENERATE_NORM_SQ:
                raise ValueError("degenerate candidate")
    except ValueError:
        return fallback(
            "invalid_geometry",
            resolved_score=resolved_score,
            score_count=len(scores),
        )

    candidate_is_stock = candidate_raw == current_stock_raw
    interlock_open = len(scores) == 3 and all(score > 0.0 for score in scores)
    used_nonstock = interlock_open and not candidate_is_stock
    action_raw = candidate_raw if used_nonstock else current_stock_raw
    reason = (
        "zero_or_rounded_phase"
        if candidate_is_stock
        else "candidate_selected"
        if used_nonstock
        else "interlock_closed"
    )
    next_state = CPLGReferenceState(
        previous_stock_raw=current_stock_raw,
        previous_forward_tangent_raw=current_forward_tangent_raw,
        previous_turn_angle_radians=current_turn_angle,
        pending_candidate_raw=candidate_raw,
        scores=tuple(scores),
    )
    return CPLGReferencePreview(
        prior_state=prior,
        next_state=next_state,
        action_raw=action_raw,
        candidate_raw=candidate_raw,
        sealed_shadow=True,
        used_nonstock=used_nonstock,
        interlock_open=interlock_open,
        resolved_shadow_score=resolved_score,
        interlock_score_count=len(scores),
        state_cleared=False,
        reason=reason,
        current_turn_angle_radians=current_turn_angle,
        previous_turn_angle_radians=prior.previous_turn_angle_radians,
        transported_forward_tangent_coherence=coherence,
        commanded_angle_radians=commanded_angle,
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
    seal_shadow: bool = False,
) -> CPLGDirection:
    return CPLGDirection(
        raw=raw,
        used_nonstock=False,
        seal_shadow=seal_shadow,
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
    "CPLGReferenceMachine",
    "CPLGReferencePreview",
    "CPLGReferenceState",
    "CPLGRustLibmTrig",
    "CPLG_PLATFORM_CAP_ATAN2F_BITS",
    "CPLG_PLATFORM_CAP_MATCHES_PINNED",
    "CPLG_RHO_OVERSHOOT_TOLERANCE",
    "CPLG_RUST_LIBM_CROSS_RUNTIME_BIT_PARITY",
    "CPLG_RUST_LIBM_ORACLE_SCHEMA",
    "CPLG_RUST_LIBM_TRIG_BACKEND",
    "CPLG_RUST_LIBM_VERSION",
    "CPLG_TRIG_BACKEND",
    "CPLG_TRIG_PORTABILITY_LIMITATION",
    "PLATFORM_F32_TRIG",
    "cosine_gain_f32le",
    "cplg_angle_based_direction_f32le",
    "decode_f32le",
    "encode_f32le",
    "preview_cplg_reference_f32le",
]
