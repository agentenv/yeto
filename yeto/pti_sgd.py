"""Causal PTI-SGD direction policy with a frozen ``c = -1/4`` candidate.

The policy consumes an exact chronological stream of stock directions.  A
fragment's candidate at boundary ``t`` is *shadow scored* against the factual
stock direction at boundary ``t + 1``.  Consequently, the action at ``t`` can
only use scores for candidates sealed before ``t``; the candidate constructed
at ``t`` never gets to look at the direction that will score it.

This module deliberately has no loss or objective-value input.  It is a
direction policy and its action records contain direction bytes and hashes,
not training outcomes.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import struct
from typing import Iterable, Sequence


PTI_COEFFICIENT = -0.25
"""The one frozen PTI transverse coefficient.  This is not a search grid."""

PTI_INTERLOCK_LENGTH = 3
"""Number of immediately preceding strictly-positive scores required."""

PTI_DEGENERATE_NORM_SQ = 2.0**-40
"""Fail-closed threshold for a direction or transverse component."""

_ZERO_HASH = "0" * 64
_POLICY_SCHEMA = "yeto.pti-sgd.direction-policy.v1"


def sha256_bytes(raw: bytes) -> str:
    """Return the lowercase SHA-256 digest of an exact bytes object."""

    if type(raw) is not bytes:
        raise TypeError("raw direction must have exact type bytes")
    return hashlib.sha256(raw).hexdigest()


def encode_f32le(values: Iterable[float]) -> bytes:
    """Encode values as an exact little-endian IEEE-754 f32 byte string."""

    materialized = tuple(values)
    try:
        return struct.pack(f"<{len(materialized)}f", *materialized)
    except (OverflowError, struct.error) as exc:
        raise ValueError("values cannot be represented as little-endian f32") from exc


def decode_f32le(raw: bytes, shape: tuple[int, ...]) -> tuple[float, ...]:
    """Decode an exactly shaped little-endian IEEE-754 f32 byte string.

    The helper is strict because implicit padding, truncation, native endian,
    and inferred shapes would weaken the evidence binding used by the policy.
    """

    if type(raw) is not bytes:
        raise TypeError("raw direction must have exact type bytes")
    count = _shape_numel(shape)
    if len(raw) != count * 4:
        raise ValueError(
            f"shape requires {count * 4} bytes of f32 data, received {len(raw)}"
        )
    return struct.unpack(f"<{count}f", raw)


@dataclass(frozen=True)
class PTIEvent:
    """One factual stock direction at a chronological fragment boundary.

    ``stock_sha256`` is the expected digest of ``stock_raw``.  ``stock_raw``
    must be the exact little-endian f32 bytes from the stock path.  Requiring
    the exact ``bytes`` type lets fail-closed actions preserve object identity,
    in addition to byte equality.
    """

    sequence: int
    fragment: int
    version: int
    shape: tuple[int, ...]
    stock_sha256: str
    stock_raw: bytes

    def __post_init__(self) -> None:
        for name in ("sequence", "fragment", "version"):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must have exact type int")
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
        if type(self.shape) is not tuple:
            raise TypeError("shape must have exact type tuple")
        if type(self.stock_sha256) is not str:
            raise TypeError("stock_sha256 must have exact type str")
        if type(self.stock_raw) is not bytes:
            raise TypeError("stock_raw must have exact type bytes")


@dataclass(frozen=True)
class PTIAction:
    """A sealed stock or PTI direction action.

    This type intentionally contains no loss, reward, objective, or outcome
    field.  ``raw`` is either the event's identical stock bytes object or a
    newly encoded fixed-coefficient PTI candidate.
    """

    raw: bytes
    action_sha256: str
    decision_sha256: str
    used_nonstock: bool
    coefficient: float
    reason: str


@dataclass(frozen=True)
class PTILedgerEntry:
    """Append-only audit record for one processed event."""

    sequence: int
    fragment: int
    version: int
    shape: tuple[int, ...]
    expected_stock_sha256: str
    actual_stock_sha256: str
    resolved_shadow_score: float | None
    interlock_scores: tuple[float, ...]
    interlock_open: bool
    sealed_shadow_sha256: str | None
    used_nonstock: bool
    reason: str
    fragment_state_cleared: bool
    action_sha256: str
    decision_sha256: str
    previous_ledger_sha256: str
    ledger_sha256: str


@dataclass(frozen=True)
class PTIClosureEntry:
    """Non-action audit record for one unresolved end-of-stream shadow."""

    last_sequence: int
    source_sequence: int
    fragment: int
    version: int
    shape: tuple[int, ...]
    unresolved_shadow_sha256: str
    source_used_nonstock: bool
    resolved_shadow_score: None
    direction_score_count_delta: int
    previous_ledger_sha256: str
    closure_sha256: str


@dataclass(frozen=True)
class PTIResult:
    """The action and audit entry emitted for a single event."""

    action: PTIAction
    ledger: PTILedgerEntry


@dataclass(frozen=True)
class _PendingShadow:
    source_sequence: int
    stock: tuple[float, ...]
    candidate: tuple[float, ...]
    candidate_sha256: str
    source_used_nonstock: bool


@dataclass
class _FragmentState:
    version: int
    shape: tuple[int, ...]
    stock: tuple[float, ...]
    pending: _PendingShadow | None
    scores: deque[float]


class PTISGDPolicy:
    """Executable causal direction-policy state machine.

    The instance owns a global chronological cursor and isolated state for
    every fragment.  Call :meth:`process` in event order, or :meth:`replay` for
    a finite iterable.  A sequence discontinuity clears every fragment because
    the missing event's fragment is unknowable; other evidence failures clear
    only the event's fragment.
    """

    coefficient = PTI_COEFFICIENT
    interlock_length = PTI_INTERLOCK_LENGTH

    def __init__(self) -> None:
        self._last_sequence: int | None = None
        self._fragments: dict[int, _FragmentState] = {}
        self._ledger: list[PTILedgerEntry] = []
        self._ledger_head = _ZERO_HASH
        self._closures: list[PTIClosureEntry] = []
        self._last_close_records: tuple[PTIClosureEntry, ...] = ()
        self._closed = False

    @property
    def ledger(self) -> tuple[PTILedgerEntry, ...]:
        """An immutable snapshot of the deterministic ledger."""

        return tuple(self._ledger)

    @property
    def ledger_head(self) -> str:
        """Hash of the most recently sealed action or closure audit record."""

        return self._ledger_head

    @property
    def closures(self) -> tuple[PTIClosureEntry, ...]:
        """All tail-closure records in their deterministic chain order."""

        return tuple(self._closures)

    @property
    def closed(self) -> bool:
        """Whether event admission is stopped pending an explicit reopen."""

        return self._closed

    @property
    def resolved_shadow_score_count(self) -> int:
        """Number of factual next-direction scores, excluding open tails."""

        return sum(entry.resolved_shadow_score is not None for entry in self._ledger)

    def reset(self) -> None:
        """Clear the chronological cursor, fragment states, and ledger."""

        self._last_sequence = None
        self._fragments.clear()
        self._ledger.clear()
        self._ledger_head = _ZERO_HASH
        self._closures.clear()
        self._last_close_records = ()
        self._closed = False

    def close(self) -> tuple[PTIClosureEntry, ...]:
        """Seal unresolved shadows as non-action records and clear live state.

        The records are ordered by fragment id and chained after the existing
        action ledger.  An unresolved shadow is explicitly unscored and never
        changes its already sealed source action.  Repeated close calls return
        the same records without extending or rewriting the audit chain.
        """

        if self._closed:
            return self._last_close_records

        records: list[PTIClosureEntry] = []
        last_sequence = -1 if self._last_sequence is None else self._last_sequence
        for fragment in sorted(self._fragments):
            state = self._fragments[fragment]
            pending = state.pending
            if pending is None:
                continue
            previous_hash = self._ledger_head
            closure_payload = {
                "schema": _POLICY_SCHEMA,
                "kind": "unresolved_shadow_tail_closure",
                "last_sequence": last_sequence,
                "source_sequence": pending.source_sequence,
                "fragment": fragment,
                "version": state.version,
                "shape": list(state.shape),
                "unresolved_shadow_sha256": pending.candidate_sha256,
                "source_used_nonstock": pending.source_used_nonstock,
                "resolved_shadow_score": None,
                "direction_score_count_delta": 0,
                "previous_ledger_sha256": previous_hash,
            }
            closure_hash = _canonical_hash(closure_payload)
            record = PTIClosureEntry(
                last_sequence=last_sequence,
                source_sequence=pending.source_sequence,
                fragment=fragment,
                version=state.version,
                shape=state.shape,
                unresolved_shadow_sha256=pending.candidate_sha256,
                source_used_nonstock=pending.source_used_nonstock,
                resolved_shadow_score=None,
                direction_score_count_delta=0,
                previous_ledger_sha256=previous_hash,
                closure_sha256=closure_hash,
            )
            records.append(record)
            self._closures.append(record)
            self._ledger_head = closure_hash

        self._fragments.clear()
        self._last_sequence = None
        self._last_close_records = tuple(records)
        self._closed = True
        return self._last_close_records

    def reopen(self) -> None:
        """Resume admission with empty fragment history and a stock warm-up."""

        if not self._closed:
            raise RuntimeError("PTI stream is already open")
        self._fragments.clear()
        self._last_sequence = None
        self._last_close_records = ()
        self._closed = False

    def replay(self, events: Iterable[PTIEvent]) -> tuple[PTIResult, ...]:
        """Process a finite event stream in its supplied chronological order."""

        return tuple(self.process(event) for event in events)

    def process(self, event: PTIEvent) -> PTIResult:
        """Consume one event and seal exactly one action and ledger entry."""

        if self._closed:
            raise RuntimeError("PTI stream is closed; call reopen() before process()")
        if not isinstance(event, PTIEvent):
            raise TypeError("event must be a PTIEvent")

        actual_hash = sha256_bytes(event.stock_raw)

        if (
            self._last_sequence is not None
            and event.sequence != self._last_sequence + 1
        ):
            self._fragments.clear()
            self._last_sequence = event.sequence
            return self._stock_result(
                event,
                actual_hash=actual_hash,
                reason="sequence_discontinuity",
                fragment_state_cleared=True,
            )
        self._last_sequence = event.sequence

        if not _is_sha256(event.stock_sha256) or event.stock_sha256 != actual_hash:
            self._fragments.pop(event.fragment, None)
            return self._stock_result(
                event,
                actual_hash=actual_hash,
                reason="stock_hash_mismatch",
                fragment_state_cleared=True,
            )

        try:
            stock = decode_f32le(event.stock_raw, event.shape)
        except ValueError:
            self._fragments.pop(event.fragment, None)
            return self._stock_result(
                event,
                actual_hash=actual_hash,
                reason="invalid_shape",
                fragment_state_cleared=True,
            )

        if not all(math.isfinite(value) for value in stock):
            self._fragments.pop(event.fragment, None)
            return self._stock_result(
                event,
                actual_hash=actual_hash,
                reason="nonfinite_stock",
                fragment_state_cleared=True,
            )
        try:
            stock_norm_sq = _norm_sq(stock)
        except ValueError:
            self._fragments.pop(event.fragment, None)
            return self._stock_result(
                event,
                actual_hash=actual_hash,
                reason="nonfinite_stock_geometry",
                fragment_state_cleared=True,
            )
        if stock_norm_sq <= PTI_DEGENERATE_NORM_SQ:
            self._fragments.pop(event.fragment, None)
            return self._stock_result(
                event,
                actual_hash=actual_hash,
                reason="degenerate_stock",
                fragment_state_cleared=True,
            )

        state = self._fragments.get(event.fragment)
        # Fragment versions are global commit versions.  Consecutive events
        # for one fragment can therefore skip values consumed by other
        # fragments; they must be strictly increasing, not adjacent.
        if state is not None and event.version <= state.version:
            self._fragments.pop(event.fragment, None)
            return self._stock_result(
                event,
                actual_hash=actual_hash,
                reason="version_discontinuity",
                fragment_state_cleared=True,
            )
        if state is not None and event.shape != state.shape:
            self._fragments.pop(event.fragment, None)
            return self._stock_result(
                event,
                actual_hash=actual_hash,
                reason="shape_changed",
                fragment_state_cleared=True,
            )

        if state is None:
            self._fragments[event.fragment] = _FragmentState(
                version=event.version,
                shape=event.shape,
                stock=stock,
                pending=None,
                scores=deque(maxlen=PTI_INTERLOCK_LENGTH),
            )
            return self._stock_result(
                event,
                actual_hash=actual_hash,
                reason="warmup",
                fragment_state_cleared=False,
            )

        # Causal ordering is deliberate:
        #   1. resolve only the shadow sealed at the preceding valid boundary;
        #   2. update the last-three strict-positive interlock;
        #   3. build/seal this boundary's shadow and choose the action using the
        #      already resolved scores.  This new shadow's score is unknowable.
        resolved_score: float | None = None
        if state.pending is not None:
            resolved_score = _shadow_score(state.pending, stock)
            if not math.isfinite(resolved_score):
                self._fragments.pop(event.fragment, None)
                return self._stock_result(
                    event,
                    actual_hash=actual_hash,
                    reason="nonfinite_shadow_score",
                    fragment_state_cleared=True,
                )
            state.scores.append(resolved_score)

        try:
            candidate_raw, candidate = _fixed_candidate(stock, state.stock)
        except ValueError:
            scores = tuple(state.scores)
            self._fragments.pop(event.fragment, None)
            return self._stock_result(
                event,
                actual_hash=actual_hash,
                reason="degenerate_transverse",
                fragment_state_cleared=True,
                resolved_shadow_score=resolved_score,
                interlock_scores=scores,
            )

        candidate_hash = sha256_bytes(candidate_raw)
        interlock_scores = tuple(state.scores)
        interlock_open = len(interlock_scores) == PTI_INTERLOCK_LENGTH and all(
            score > 0.0 for score in interlock_scores
        )

        state.version = event.version
        state.shape = event.shape
        state.stock = stock
        state.pending = _PendingShadow(
            source_sequence=event.sequence,
            stock=stock,
            candidate=candidate,
            candidate_sha256=candidate_hash,
            source_used_nonstock=interlock_open,
        )

        if interlock_open:
            return self._seal_result(
                event,
                actual_hash=actual_hash,
                raw=candidate_raw,
                used_nonstock=True,
                reason="interlock_open",
                resolved_shadow_score=resolved_score,
                interlock_scores=interlock_scores,
                interlock_open=True,
                sealed_shadow_sha256=candidate_hash,
                fragment_state_cleared=False,
            )
        return self._stock_result(
            event,
            actual_hash=actual_hash,
            reason="interlock_closed",
            fragment_state_cleared=False,
            resolved_shadow_score=resolved_score,
            interlock_scores=interlock_scores,
            sealed_shadow_sha256=candidate_hash,
        )

    def _stock_result(
        self,
        event: PTIEvent,
        *,
        actual_hash: str,
        reason: str,
        fragment_state_cleared: bool,
        resolved_shadow_score: float | None = None,
        interlock_scores: tuple[float, ...] = (),
        sealed_shadow_sha256: str | None = None,
    ) -> PTIResult:
        # Do not decode/re-encode this fallback.  Its identity is part of the
        # fail-closed contract and is tested with ``is``, not merely ``==``.
        return self._seal_result(
            event,
            actual_hash=actual_hash,
            raw=event.stock_raw,
            used_nonstock=False,
            reason=reason,
            resolved_shadow_score=resolved_shadow_score,
            interlock_scores=interlock_scores,
            interlock_open=False,
            sealed_shadow_sha256=sealed_shadow_sha256,
            fragment_state_cleared=fragment_state_cleared,
        )

    def _seal_result(
        self,
        event: PTIEvent,
        *,
        actual_hash: str,
        raw: bytes,
        used_nonstock: bool,
        reason: str,
        resolved_shadow_score: float | None,
        interlock_scores: tuple[float, ...],
        interlock_open: bool,
        sealed_shadow_sha256: str | None,
        fragment_state_cleared: bool,
    ) -> PTIResult:
        action_hash = sha256_bytes(raw)
        decision_payload = {
            "schema": _POLICY_SCHEMA,
            "coefficient": {"numerator": -1, "denominator": 4},
            "interlock_length": PTI_INTERLOCK_LENGTH,
            "event": {
                "sequence": event.sequence,
                "fragment": event.fragment,
                "version": event.version,
                "shape": list(event.shape),
                "expected_stock_sha256": event.stock_sha256,
                "actual_stock_sha256": actual_hash,
            },
            "resolved_shadow_score_f32le": _score_f32_bits(resolved_shadow_score),
            "interlock_scores_f32le": [
                _score_f32_bits(score) for score in interlock_scores
            ],
            "interlock_open": interlock_open,
            "sealed_shadow_sha256": sealed_shadow_sha256,
            "used_nonstock": used_nonstock,
            "reason": reason,
            "fragment_state_cleared": fragment_state_cleared,
            "action_sha256": action_hash,
        }
        decision_hash = _canonical_hash(decision_payload)
        previous_ledger_hash = self._ledger_head
        ledger_hash = _canonical_hash(
            {
                "schema": _POLICY_SCHEMA,
                "previous_ledger_sha256": previous_ledger_hash,
                "decision_sha256": decision_hash,
            }
        )
        action = PTIAction(
            raw=raw,
            action_sha256=action_hash,
            decision_sha256=decision_hash,
            used_nonstock=used_nonstock,
            coefficient=PTI_COEFFICIENT,
            reason=reason,
        )
        ledger = PTILedgerEntry(
            sequence=event.sequence,
            fragment=event.fragment,
            version=event.version,
            shape=event.shape,
            expected_stock_sha256=event.stock_sha256,
            actual_stock_sha256=actual_hash,
            resolved_shadow_score=resolved_shadow_score,
            interlock_scores=interlock_scores,
            interlock_open=interlock_open,
            sealed_shadow_sha256=sealed_shadow_sha256,
            used_nonstock=used_nonstock,
            reason=reason,
            fragment_state_cleared=fragment_state_cleared,
            action_sha256=action_hash,
            decision_sha256=decision_hash,
            previous_ledger_sha256=previous_ledger_hash,
            ledger_sha256=ledger_hash,
        )
        self._ledger.append(ledger)
        self._ledger_head = ledger_hash
        return PTIResult(action=action, ledger=ledger)


# Short aliases make the API comfortable for callers while retaining the
# descriptive primary class name in docs and tracebacks.
PTISGD = PTISGDPolicy
PTISGDStateMachine = PTISGDPolicy


def replay(events: Iterable[PTIEvent]) -> tuple[PTIResult, ...]:
    """Replay events with a new policy instance."""

    return PTISGDPolicy().replay(events)


def pti_candidate_f32le(
    current_raw: bytes, previous_raw: bytes, shape: tuple[int, ...]
) -> bytes:
    """Materialize the Amendment-1 candidate from exact stock f32 bytes."""

    current = decode_f32le(current_raw, shape)
    previous = decode_f32le(previous_raw, shape)
    if not all(math.isfinite(value) for value in (*current, *previous)):
        raise ValueError("candidate inputs must be finite f32 values")
    raw, _candidate = _fixed_candidate(current, previous)
    return raw


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
    """Round one real-valued Python operation result to IEEE-754 f32."""

    try:
        rounded = struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as exc:
        raise ValueError("f32 operation overflowed") from exc
    if not math.isfinite(rounded):
        raise ValueError("f32 operation produced a nonfinite value")
    return rounded


def _f32_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def _f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


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
    """Correctly round the real square root of a nonnegative f32 to f32.

    ``math.sqrt`` supplies only an initial candidate.  Exact rational midpoint
    comparisons then correct that candidate and implement ties-to-even, so the
    sealed result does not depend on the host libm's last binary64 bit.
    """

    value = _f32(value)
    if value < 0.0:
        raise ValueError("f32 square root requires a nonnegative input")
    if value == 0.0:
        return 0.0

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
    """Canonical coordinate-order f32 product-then-add reduction."""

    if len(left) != len(right):
        raise ValueError("dot-product shapes differ")
    accumulator = _f32(0.0)
    for left_value, right_value in zip(left, right, strict=True):
        product = _multiply_f32(left_value, right_value)
        accumulator = _add_f32(accumulator, product)
    return accumulator


def _norm_sq(vector: Sequence[float]) -> float:
    return _dot_f32(vector, vector)


def _normalize_f32(vector: Sequence[float]) -> tuple[tuple[float, ...], float]:
    norm_sq = _norm_sq(vector)
    if norm_sq <= PTI_DEGENERATE_NORM_SQ:
        raise ValueError("direction is degenerate")
    norm = _sqrt_f32(norm_sq)
    normalized = tuple(_divide_f32(value, norm) for value in vector)
    return normalized, norm


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    unit_left, _left_norm = _normalize_f32(left)
    unit_right, _right_norm = _normalize_f32(right)
    return _dot_f32(unit_left, unit_right)


def _shadow_score(pending: _PendingShadow, factual: Sequence[float]) -> float:
    candidate_cosine = _cosine(pending.candidate, factual)
    stock_cosine = _cosine(pending.stock, factual)
    return _subtract_f32(candidate_cosine, stock_cosine)


def _fixed_candidate(
    current: Sequence[float], previous: Sequence[float]
) -> tuple[bytes, tuple[float, ...]]:
    unit_current, current_norm = _normalize_f32(current)
    unit_previous, _previous_norm = _normalize_f32(previous)
    projection = _dot_f32(unit_previous, unit_current)
    transverse = tuple(
        _subtract_f32(old, _multiply_f32(projection, new))
        for old, new in zip(unit_previous, unit_current, strict=True)
    )
    try:
        unit_transverse, _transverse_norm = _normalize_f32(transverse)
    except ValueError:
        raise ValueError("previous direction has no stable transverse component")
    coefficient = _f32(PTI_COEFFICIENT)
    candidate_raw_direction = tuple(
        _add_f32(new, _multiply_f32(coefficient, orthogonal))
        for new, orthogonal in zip(unit_current, unit_transverse, strict=True)
    )
    candidate_raw_norm_sq = _norm_sq(candidate_raw_direction)
    if candidate_raw_norm_sq <= PTI_DEGENERATE_NORM_SQ:
        raise ValueError("candidate raw direction is degenerate")
    candidate_raw_norm = _sqrt_f32(candidate_raw_norm_sq)
    unit_candidate_raw = tuple(
        _divide_f32(component, candidate_raw_norm)
        for component in candidate_raw_direction
    )
    candidate = tuple(
        _multiply_f32(current_norm, component) for component in unit_candidate_raw
    )
    raw = encode_f32le(candidate)
    candidate = struct.unpack(f"<{len(candidate)}f", raw)
    if not all(math.isfinite(value) for value in candidate):
        raise ValueError("candidate is not finite after f32 sealing")
    if _norm_sq(candidate) <= PTI_DEGENERATE_NORM_SQ:
        raise ValueError("candidate is degenerate after f32 sealing")
    return raw, candidate


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _score_f32_bits(value: float | None) -> str | None:
    if value is None:
        return None
    if not math.isfinite(value):
        raise ValueError("nonfinite values cannot be sealed")
    return struct.pack("<f", _f32(value)).hex()


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "PTIAction",
    "PTIClosureEntry",
    "PTIEvent",
    "PTILedgerEntry",
    "PTIResult",
    "PTISGD",
    "PTISGDPolicy",
    "PTISGDStateMachine",
    "PTI_COEFFICIENT",
    "PTI_DEGENERATE_NORM_SQ",
    "PTI_INTERLOCK_LENGTH",
    "decode_f32le",
    "encode_f32le",
    "pti_candidate_f32le",
    "replay",
    "sha256_bytes",
]
