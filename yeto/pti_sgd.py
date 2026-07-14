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
class PTIResult:
    """The action and audit entry emitted for a single event."""

    action: PTIAction
    ledger: PTILedgerEntry


@dataclass(frozen=True)
class _PendingShadow:
    stock: tuple[float, ...]
    candidate: tuple[float, ...]
    candidate_sha256: str


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

    @property
    def ledger(self) -> tuple[PTILedgerEntry, ...]:
        """An immutable snapshot of the deterministic ledger."""

        return tuple(self._ledger)

    @property
    def ledger_head(self) -> str:
        """Hash of the most recently sealed ledger entry."""

        return self._ledger_head

    def reset(self) -> None:
        """Clear the chronological cursor, fragment states, and ledger."""

        self._last_sequence = None
        self._fragments.clear()
        self._ledger.clear()
        self._ledger_head = _ZERO_HASH

    def replay(self, events: Iterable[PTIEvent]) -> tuple[PTIResult, ...]:
        """Process a finite event stream in its supplied chronological order."""

        return tuple(self.process(event) for event in events)

    def process(self, event: PTIEvent) -> PTIResult:
        """Consume one event and seal exactly one action and ledger entry."""

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
        if _norm_sq(stock) <= PTI_DEGENERATE_NORM_SQ:
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
            stock=stock,
            candidate=candidate,
            candidate_sha256=candidate_hash,
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
            "resolved_shadow_score_f64le": _float_bits(resolved_shadow_score),
            "interlock_scores_f64le": [
                _float_bits(score) for score in interlock_scores
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


def _shape_numel(shape: tuple[int, ...]) -> int:
    if type(shape) is not tuple or not shape:
        raise ValueError("shape must be a nonempty tuple")
    count = 1
    for dimension in shape:
        if type(dimension) is not int or dimension <= 0:
            raise ValueError("shape dimensions must be positive exact integers")
        count *= dimension
    return count


def _norm_sq(vector: Sequence[float]) -> float:
    return math.fsum(value * value for value in vector)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    left_sq = _norm_sq(left)
    right_sq = _norm_sq(right)
    if left_sq <= PTI_DEGENERATE_NORM_SQ or right_sq <= PTI_DEGENERATE_NORM_SQ:
        raise ValueError("cosine direction is degenerate")
    dot = math.fsum(a * b for a, b in zip(left, right, strict=True))
    return dot / math.sqrt(left_sq * right_sq)


def _shadow_score(pending: _PendingShadow, factual: Sequence[float]) -> float:
    return _cosine(pending.candidate, factual) - _cosine(pending.stock, factual)


def _fixed_candidate(
    current: Sequence[float], previous: Sequence[float]
) -> tuple[bytes, tuple[float, ...]]:
    current_sq = _norm_sq(current)
    previous_sq = _norm_sq(previous)
    if current_sq <= PTI_DEGENERATE_NORM_SQ or previous_sq <= PTI_DEGENERATE_NORM_SQ:
        raise ValueError("stock direction is degenerate")

    current_norm = math.sqrt(current_sq)
    previous_norm = math.sqrt(previous_sq)
    unit_current = tuple(value / current_norm for value in current)
    unit_previous = tuple(value / previous_norm for value in previous)
    projection = math.fsum(
        a * b for a, b in zip(unit_previous, unit_current, strict=True)
    )
    transverse = tuple(
        old - projection * new
        for old, new in zip(unit_previous, unit_current, strict=True)
    )
    transverse_sq = _norm_sq(transverse)
    if transverse_sq <= PTI_DEGENERATE_NORM_SQ:
        raise ValueError("previous direction has no stable transverse component")
    transverse_norm = math.sqrt(transverse_sq)
    unit_transverse = tuple(value / transverse_norm for value in transverse)
    scale = current_norm / math.sqrt(1.0 + PTI_COEFFICIENT**2)
    ideal = tuple(
        scale * (new + PTI_COEFFICIENT * orthogonal)
        for new, orthogonal in zip(unit_current, unit_transverse, strict=True)
    )
    raw = encode_f32le(ideal)
    candidate = struct.unpack(f"<{len(ideal)}f", raw)
    if not all(math.isfinite(value) for value in candidate):
        raise ValueError("candidate is not finite after f32 sealing")
    if _norm_sq(candidate) <= PTI_DEGENERATE_NORM_SQ:
        raise ValueError("candidate is degenerate after f32 sealing")
    return raw, candidate


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _float_bits(value: float | None) -> str | None:
    if value is None:
        return None
    if not math.isfinite(value):
        raise ValueError("nonfinite values cannot be sealed")
    return struct.pack("<d", value).hex()


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
    "replay",
    "sha256_bytes",
]
