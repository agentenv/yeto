"""Cross-island policy-version ledger (rl-inter-island-scheduling stage 0). Pure CPU."""

from __future__ import annotations

import pytest

from yeto.rl.engine.island_ledger import (
    ACCEPT, ACCEPT_IS, REJECT, CrossIslandLedger, DeltaEntry, LedgerError, SampleGroup,
    IslandSchedulingMode, StalenessPolicy, check_same_mode, contract_fields, merge_weight, parse_mode,
)


def _ledger(**kw):
    kw.setdefault("mode", "elastic")
    led = CrossIslandLedger(**kw)
    for i in ("a", "b", "c"):
        led.join(i, now=0.0)
    return led


def _delta(island, led, tokens=100, steps=4):
    return DeltaEntry(island, led.outer_version, steps, led.published[led.outer_version], tokens, steps)


def _group(island, version, h, lp=True, inner=4):
    return SampleGroup(island, version, inner, h, n=8, behavior_logprob=(-0.1,) if lp else None)


def test_merge_weight_matches_syncer_formula():
    assert merge_weight(100, 4) == 2500.0
    assert merge_weight(0, 4) == 0.0 and merge_weight(10, 0) == 0.0


def test_three_verdicts():
    led = _ledger()
    for i in ("a", "b", "c"):
        led.submit(_delta(i, led))
    led.try_advance(timed_out=False)
    assert led.outer_version == 1
    v = led.judge(_group("b", 1, "h1"), consumer_island="a")
    assert v.verdict == ACCEPT and v.outer_lag == 0
    v = led.judge(_group("b", 0, "h0"), consumer_island="a")
    assert (v.verdict, v.correction, v.outer_lag) == (ACCEPT_IS, "tis", 1)
    assert led.judge(_group("b", 0, "h0", lp=False), consumer_island="a").reason == "missing_behavior_logprob"
    assert led.judge(_group("b", 1, "bogus"), consumer_island="a").reason == "policy_hash_mismatch"
    assert led.judge(_group("b", 7, "h7"), consumer_island="a").reason == "unknown_version"
    for _ in range(2):
        for i in ("a", "b", "c"):
            led.submit(_delta(i, led))
        led.try_advance(timed_out=False)
    assert led.judge(_group("b", 1, "h1"), consumer_island="a").verdict == ACCEPT_IS  # lag 2 ok
    v = led.judge(_group("b", 0, "h0"), consumer_island="a")
    assert (v.verdict, v.reason, v.outer_lag) == (REJECT, "outer_lag_exceeded", 3)


def test_inner_lag_cap_and_on_policy_only():
    led = _ledger(policy=StalenessPolicy(max_inner_lag=2))
    assert led.judge(_group("b", 0, "h0", inner=1), consumer_island="a",
                     consumer_inner_step=2).verdict == ACCEPT_IS
    assert led.judge(_group("b", 0, "h0", inner=1), consumer_island="a",
                     consumer_inner_step=9).reason == "inner_lag_exceeded"
    critic = _ledger(policy=StalenessPolicy(on_policy_only=True))
    critic.submit(_delta("a", critic)); critic.try_advance(timed_out=True)
    assert critic.judge(_group("b", 0, "h0"), consumer_island="a").reason == "on_policy_only"


def test_correction_name_must_be_mismatch_mechanism():
    with pytest.raises(LedgerError):
        StalenessPolicy(correction="nope")
    assert StalenessPolicy(correction="icepop").correction == "icepop"


def test_default_policy_is_user_ruling():
    p = StalenessPolicy()
    assert p.max_outer_lag == 2 and p.max_inner_lag is None and p.correction == "tis"


def test_quorum_timeout_and_idle():
    led = _ledger(theta=1.0, quorum_min=2)
    led.submit(_delta("a", led))
    assert led.try_advance(timed_out=False) is None
    assert led.try_advance(timed_out=True) is None  # 1 < q_min: idle, no new version
    assert led.outer_version == 0 and led.events[-1]["kind"] == "round_idle"
    led.submit(_delta("b", led))
    step = led.try_advance(timed_out=True)
    assert step["absent"] == ["c"] and led.outer_version == 1
    assert step["weights"] == {"a": 0.5, "b": 0.5}


def test_capacity_weighted_quorum():
    led = CrossIslandLedger(theta=0.75, mode="elastic")
    led.join("big", now=0.0, capacity=3.0); led.join("small", now=0.0, capacity=1.0)
    led.submit(_delta("small", led))
    assert led.try_advance(timed_out=False) is None  # 1/4 < 0.75
    led.submit(_delta("big", led))
    assert led.try_advance(timed_out=False)["cap_arrived"] == 4.0
    led.submit(_delta("big", led))
    step = led.try_advance(timed_out=False)  # 3/4 >= 0.75: no wait for small
    assert step is not None and step["absent"] == ["small"]


def test_late_delta_carried_with_discount():
    led = _ledger(theta=0.3, gamma=0.5)
    late = _delta("c", led)
    led.submit(_delta("a", led)); led.try_advance(timed_out=False)
    ev = led.submit(late)
    assert ev["kind"] == "delta_carried_over" and ev["lag"] == 1 and ev["discount"] == 0.5
    led.submit(_delta("a", led))
    step = led.try_advance(timed_out=False)
    assert step["carried_in"] == {"c@0": {"lag": 1, "discount": 0.5}}
    assert step["raw_weights"]["c@0"] == 0.5 * step["raw_weights"]["a"]
    for _ in range(2):
        led.submit(_delta("a", led)); led.try_advance(timed_out=False)
    assert led.submit(late)["reason"] == "carry_lag_exceeded"


def test_syncer_epoch_fence():
    led = CrossIslandLedger(syncer_epoch=3)
    led.check_fence(3)
    with pytest.raises(LedgerError):
        led.check_fence(2)


def test_leave_drops_uncommitted_and_bumps_epoch():
    led = _ledger()
    e0 = led.membership_epoch
    led.submit(_delta("c", led, tokens=50))
    ev = led.leave("c", reason="lease_expired")
    assert led.membership_epoch == e0 + 1
    assert ev["dropped_uncommitted"]["c_tokens"] == 50 and "c" not in led.pending
    assert led.submit(_delta("c", led))["reason"] == "not_member"


def test_lease_expiry():
    led = _ledger(lease_s=1.0)
    led.heartbeat("a", now=5.0); led.heartbeat("b", now=5.0)
    assert led.expire_leases(now=5.5) == ["c"]
    with pytest.raises(LedgerError):
        led.heartbeat("c", now=6.0)


def test_catch_up_member_has_zero_weight_first_round():
    led = _ledger(theta=1.0)
    for i in ("a", "b", "c"):
        led.submit(_delta(i, led))
    led.try_advance(timed_out=False)
    ev = led.join("d", now=1.0)
    assert ev["catch_up"] and ev["base_version"] == 1
    for i in ("a", "b", "c", "d"):
        led.submit(_delta(i, led))
    step = led.try_advance(timed_out=False)
    assert step["raw_weights"]["d"] == 0.0 and step["weights"]["a"] == pytest.approx(1 / 3)
    for i in ("a", "b", "c", "d"):
        led.submit(_delta(i, led))
    assert led.try_advance(timed_out=False)["raw_weights"]["d"] > 0


# ---------------------------------------------------------------- legacy mode
def _legacy():
    led = CrossIslandLedger()  # default mode is legacy
    for i in ("a", "b", "c"):
        led.join(i, now=0.0)
    return led


def test_default_mode_is_legacy_and_contract():
    assert CrossIslandLedger().mode is IslandSchedulingMode.LEGACY
    assert parse_mode(None) is IslandSchedulingMode.LEGACY
    assert contract_fields(None) == {"island_scheduling_mode": "legacy"}
    assert contract_fields("elastic", soft_deadline_s=900.0) == {
        "island_scheduling_mode": "elastic", "quorum_theta": 0.75, "carry_gamma": 0.5,
        "soft_deadline_s": 900.0}
    check_same_mode("legacy", None)
    with pytest.raises(ValueError):
        check_same_mode("legacy", "elastic")
    with pytest.raises(ValueError):
        parse_mode("auto")


def test_legacy_equals_syncer_strict():
    """quorum == learners, grace 0: step only when all arrived; timeout fails closed."""
    led = _legacy()
    led.submit(_delta("a", led)); led.submit(_delta("b", led))
    assert led.try_advance(timed_out=False) is None
    with pytest.raises(LedgerError):
        led.try_advance(timed_out=True)  # no partial step, no idle round
    led.submit(_delta("c", led, tokens=200))
    step = led.try_advance(timed_out=False)
    assert step["absent"] == [] and step["carried_in"] == {}
    assert step["raw_weights"] == {"a": merge_weight(100, 4), "b": merge_weight(100, 4),
                                   "c": merge_weight(200, 4)}


def test_legacy_fixed_members_no_discount_no_cross_island():
    led = _legacy()
    late = _delta("c", led)
    for i in ("a", "b", "c"):
        led.submit(_delta(i, led))
    led.try_advance(timed_out=False)
    assert led.submit(late)["reason"] == "stale_base"  # rejected, not carried over
    with pytest.raises(LedgerError):
        led.join("d", now=1.0)
    with pytest.raises(LedgerError):
        led.leave("a", reason="x")
    assert led.expire_leases(now=1e9) == []
    assert led.judge(_group("b", 1, "h1"), consumer_island="a").reason == "legacy_mode"
    assert led.judge(_group("a", 0, "h0"), consumer_island="a").reason == "legacy_mode"
    assert led.judge(_group("a", 1, "h1"), consumer_island="a").verdict == ACCEPT
