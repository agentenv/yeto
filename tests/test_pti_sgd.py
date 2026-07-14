"""Focused causal and exact-byte tests for the frozen PTI-SGD policy."""

from __future__ import annotations

from dataclasses import fields, replace
import math

import pytest

from yeto.pti_sgd import (
    PTIAction,
    PTIEvent,
    PTISGD,
    PTI_COEFFICIENT,
    PTI_INTERLOCK_LENGTH,
    decode_f32le,
    encode_f32le,
    replay,
    sha256_bytes,
)


def _event(
    sequence: int,
    fragment: int,
    version: int,
    values: tuple[float, ...],
    *,
    shape: tuple[int, ...] | None = None,
) -> PTIEvent:
    raw = encode_f32le(values)
    return PTIEvent(
        sequence=sequence,
        fragment=fragment,
        version=version,
        shape=shape or (len(values),),
        stock_sha256=sha256_bytes(raw),
        stock_raw=raw,
    )


def _angle_event(
    sequence: int, version: int, degrees: float, *, fragment: int = 0
) -> PTIEvent:
    radians = math.radians(degrees)
    return _event(
        sequence,
        fragment,
        version,
        (math.cos(radians), math.sin(radians)),
    )


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    dot = math.fsum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(math.fsum(value * value for value in left))
    right_norm = math.sqrt(math.fsum(value * value for value in right))
    return dot / (left_norm * right_norm)


def test_warmup_and_no_same_boundary_lookahead() -> None:
    policy = PTISGD()
    first = _angle_event(0, 0, 0.0)
    second = _angle_event(1, 1, 10.0)

    first_result = policy.process(first)
    second_result = policy.process(second)

    assert first_result.action.reason == "warmup"
    assert second_result.action.reason == "interlock_closed"
    assert first_result.action.raw is first.stock_raw
    assert second_result.action.raw is second.stock_raw
    assert second_result.ledger.resolved_shadow_score is None
    assert second_result.ledger.interlock_scores == ()
    assert second_result.ledger.sealed_shadow_sha256 is not None

    # The just-sealed candidate is only resolved by the next factual event.
    third_result = policy.process(_angle_event(2, 2, 20.0))
    assert third_result.ledger.resolved_shadow_score is not None
    assert third_result.ledger.resolved_shadow_score > 0.0
    assert len(third_result.ledger.interlock_scores) == 1
    assert third_result.action.used_nonstock is False


def test_three_prior_positive_scores_open_fixed_interlock() -> None:
    results = replay(
        _angle_event(index, index, degrees)
        for index, degrees in enumerate((0.0, 10.0, 20.0, 30.0, 40.0))
    )

    assert PTI_INTERLOCK_LENGTH == 3
    assert [result.action.used_nonstock for result in results] == [
        False,
        False,
        False,
        False,
        True,
    ]
    assert results[-1].ledger.interlock_open is True
    assert len(results[-1].ledger.interlock_scores) == 3
    assert all(score > 0.0 for score in results[-1].ledger.interlock_scores)


def test_negative_score_closes_then_three_positives_reopen() -> None:
    policy = PTISGD()
    results = [
        policy.process(_angle_event(index, index, degrees))
        for index, degrees in enumerate(
            (0.0, 10.0, 20.0, 30.0, 40.0, 30.0, 20.0, 10.0, 0.0)
        )
    ]

    assert results[4].action.used_nonstock is True
    assert results[5].ledger.resolved_shadow_score is not None
    assert results[5].ledger.resolved_shadow_score < 0.0
    assert results[5].action.used_nonstock is False
    assert results[6].action.used_nonstock is False
    assert results[7].action.used_nonstock is False
    assert results[8].action.used_nonstock is True
    assert all(score > 0.0 for score in results[8].ledger.interlock_scores)


def test_fragments_have_isolated_versions_warmup_and_interlocks() -> None:
    policy = PTISGD()
    events = (
        _angle_event(0, 0, 0.0, fragment=0),
        _angle_event(1, 0, 80.0, fragment=1),
        _angle_event(2, 1, 10.0, fragment=0),
        _angle_event(3, 1, 70.0, fragment=1),
        _angle_event(4, 2, 20.0, fragment=0),
        _angle_event(5, 3, 30.0, fragment=0),
        _angle_event(6, 4, 40.0, fragment=0),
        _angle_event(7, 2, 60.0, fragment=1),
    )
    results = policy.replay(events)

    assert results[1].action.reason == "warmup"
    assert results[6].action.used_nonstock is True
    assert results[7].action.used_nonstock is False
    assert len(results[7].ledger.interlock_scores) == 1


def test_fragment_versions_may_jump_but_must_strictly_increase() -> None:
    policy = PTISGD()
    first = policy.process(_angle_event(0, 3, 0.0))
    jumped = policy.process(_angle_event(1, 7, 10.0))
    jumped_again = policy.process(_angle_event(2, 11, 20.0))

    assert first.action.reason == "warmup"
    assert jumped.action.reason == "interlock_closed"
    assert jumped_again.action.reason == "interlock_closed"
    assert jumped_again.ledger.resolved_shadow_score is not None


@pytest.mark.parametrize("bad_version", [3, 2])
def test_equal_or_decreasing_fragment_version_clears_state(bad_version: int) -> None:
    policy = PTISGD()
    policy.process(_angle_event(0, 3, 0.0))
    bad = _angle_event(1, bad_version, 10.0)

    result = policy.process(bad)

    assert result.action.reason == "version_discontinuity"
    assert result.action.raw is bad.stock_raw
    assert result.ledger.fragment_state_cleared is True
    assert policy.process(_angle_event(2, 4, 20.0)).action.reason == "warmup"


def test_active_action_has_exact_negative_quarter_geometry() -> None:
    events = [
        _angle_event(index, index, degrees)
        for index, degrees in enumerate((0.0, 10.0, 20.0, 30.0, 40.0))
    ]
    result = replay(events)[-1]
    stock = decode_f32le(events[-1].stock_raw, events[-1].shape)
    candidate = decode_f32le(result.action.raw, events[-1].shape)

    stock_norm = math.sqrt(math.fsum(value * value for value in stock))
    candidate_norm = math.sqrt(math.fsum(value * value for value in candidate))
    angle = math.acos(max(-1.0, min(1.0, _cosine(stock, candidate))))
    signed_cross = stock[0] * candidate[1] - stock[1] * candidate[0]

    assert result.action.used_nonstock is True
    assert result.action.coefficient == PTI_COEFFICIENT == -0.25
    assert candidate_norm == pytest.approx(stock_norm, abs=1e-7)
    assert angle == pytest.approx(math.atan(0.25), abs=1e-7)
    assert signed_cross > 0.0
    assert result.action.action_sha256 == sha256_bytes(result.action.raw)


@pytest.mark.parametrize(
    ("make_bad", "reason"),
    [
        (
            lambda sequence, version: replace(
                _angle_event(sequence, version, 10.0),
                stock_sha256="f" * 64,
            ),
            "stock_hash_mismatch",
        ),
        (
            lambda sequence, version: _angle_event(sequence, version, 10.0),
            "version_discontinuity",
        ),
        (
            lambda sequence, version: replace(
                _angle_event(sequence, version, 10.0), shape=(1, 2)
            ),
            "shape_changed",
        ),
        (
            lambda sequence, version: _event(sequence, 0, version, (math.nan, 1.0)),
            "nonfinite_stock",
        ),
        (
            lambda sequence, version: _angle_event(sequence, version, 0.0),
            "degenerate_transverse",
        ),
    ],
)
def test_fragment_failures_are_bit_identical_and_clear_state(
    make_bad, reason: str
) -> None:
    policy = PTISGD()
    policy.process(_angle_event(0, 0, 0.0))
    # Only the version-discontinuity case needs a deliberate gap.  The other
    # factories use the expected version so their named fault is reached.
    bad_version = 0 if reason == "version_discontinuity" else 1
    bad = make_bad(1, bad_version)

    failed = policy.process(bad)

    assert failed.action.reason == reason
    assert failed.action.used_nonstock is False
    assert failed.action.raw is bad.stock_raw
    assert failed.action.action_sha256 == sha256_bytes(bad.stock_raw)
    assert failed.ledger.fragment_state_cleared is True

    recovered = _angle_event(2, bad_version + 1, 20.0)
    recovery_result = policy.process(recovered)
    assert recovery_result.action.reason == "warmup"
    assert recovery_result.action.raw is recovered.stock_raw


def test_bad_shape_length_and_zero_direction_fail_closed() -> None:
    policy = PTISGD()
    malformed = replace(_event(0, 0, 0, (1.0, 2.0)), shape=(3,))
    malformed_result = policy.process(malformed)
    assert malformed_result.action.reason == "invalid_shape"
    assert malformed_result.action.raw is malformed.stock_raw

    zero = _event(1, 0, 1, (0.0, 0.0))
    zero_result = policy.process(zero)
    assert zero_result.action.reason == "degenerate_stock"
    assert zero_result.action.raw is zero.stock_raw


def test_global_sequence_gap_clears_every_fragment() -> None:
    policy = PTISGD()
    policy.process(_angle_event(0, 0, 0.0, fragment=0))
    policy.process(_angle_event(1, 0, 80.0, fragment=1))
    gap = _angle_event(3, 1, 10.0, fragment=0)

    gap_result = policy.process(gap)

    assert gap_result.action.reason == "sequence_discontinuity"
    assert gap_result.action.raw is gap.stock_raw
    # Sequence 3 was consumed as a failed anchor.  Both fragments now warm up
    # independently even though their supplied versions continue increasing.
    fragment_zero = policy.process(_angle_event(4, 2, 20.0, fragment=0))
    fragment_one = policy.process(_angle_event(5, 1, 70.0, fragment=1))
    assert fragment_zero.action.reason == "warmup"
    assert fragment_one.action.reason == "warmup"


def test_deterministic_decision_and_append_only_ledger_hashes() -> None:
    events = tuple(
        _angle_event(index, index, degrees)
        for index, degrees in enumerate((0.0, 10.0, 20.0, 30.0, 40.0))
    )
    first_policy = PTISGD()
    second_policy = PTISGD()
    first = first_policy.replay(events)
    second = second_policy.replay(events)

    assert first == second
    assert first_policy.ledger == second_policy.ledger
    assert first_policy.ledger_head == second_policy.ledger_head
    assert all(
        entry.previous_ledger_sha256
        == ("0" * 64 if index == 0 else first[index - 1].ledger.ledger_sha256)
        for index, entry in enumerate(first_policy.ledger)
    )

    rebound = replace(events[0], fragment=7)
    rebound_result = PTISGD().process(rebound)
    assert rebound_result.action.action_sha256 == first[0].action.action_sha256
    assert rebound_result.action.decision_sha256 != first[0].action.decision_sha256
    assert rebound_result.ledger.ledger_sha256 != first[0].ledger.ledger_sha256


def test_action_schema_has_no_loss_or_outcome_fields() -> None:
    names = {field.name.lower() for field in fields(PTIAction)}
    forbidden = ("loss", "outcome", "reward", "objective")

    assert PTI_COEFFICIENT == -1 / 4
    assert all(not any(word in name for word in forbidden) for name in names)
