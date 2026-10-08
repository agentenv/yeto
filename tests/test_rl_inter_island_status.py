"""IslandStatus scheduling fields, journal pool_* events, PauseAdvice (stage 0). Pure CPU."""

from __future__ import annotations

import pytest

from yeto.rl.elastic_benchmark.capabilities import Attestation, ResourceConfig
from yeto.rl.engine.algorithm import AlgorithmSpec
from yeto.rl.engine.controller import SCHEDULING_FIELDS, IslandController
from yeto.rl.engine.execution_profile import ExecutionProfile
from yeto.rl.engine.journal import (
    EpochConflict, Journal, JournalError, append_pool_event, read_journal, replay_pool,
)
from yeto.rl.engine.pause_advice import PauseAdvice, lease_veto, merge_pause
from yeto.rl.engine.pause_audit import PauseDecision

FP = "sha256:" + "0" * 64
CONFIGS = {"A": ResourceConfig("A", 4, 2, 2), "B": ResourceConfig("B", 4, 4, 0)}


def _controller(tmp_path, **kw):
    att = Attestation(FP, frozenset({"partitioned-serial"}), frozenset(), frozenset(), False, True)
    prof = ExecutionProfile(name="t", execution_mode="partitioned-serial",
                            outer_protocol="none").bind_algorithm(AlgorithmSpec())
    return IslandController(state_dir=tmp_path / "state", configs=CONFIGS, attestation=att,
                            profile=prof, initial_config="A", runtime_fingerprint=FP, **kw)


def test_inspect_scheduling_fields_default_none(tmp_path):
    st = _controller(tmp_path).inspect()
    assert st.pool_epoch is None
    assert all(getattr(st, f) is None for f in SCHEDULING_FIELDS)


def test_inspect_fills_from_probe_and_journal(tmp_path):
    ctl = _controller(tmp_path, island_scheduling="elastic", scheduling_probe=lambda: {
        "round_wall_s": 12.5, "cloud": "modal", "price_per_hour": None, "bogus": 1})
    append_pool_event(ctl.journal, "pool_epoch", pool_epoch=3, mode="elastic")
    st = ctl.inspect()
    assert (st.pool_epoch, st.round_wall_s, st.cloud) == (3, 12.5, "modal")
    assert st.price_per_hour is None and st.tok_per_s is None


def test_broken_probe_does_not_break_inspect(tmp_path):
    def boom():
        raise RuntimeError("x")
    assert _controller(tmp_path, island_scheduling="elastic", scheduling_probe=boom).inspect().round_wall_s is None


def test_pool_events_replay_after_reopen(tmp_path):
    with Journal(tmp_path) as j:
        append_pool_event(j, mode="elastic", tx_kind="pool_join", pool_epoch=0, island_id="a", membership_epoch=1)
        append_pool_event(j, mode="elastic", tx_kind="pool_join", pool_epoch=0, island_id="b", membership_epoch=2)
        append_pool_event(j, mode="elastic", tx_kind="pool_leave", pool_epoch=0, island_id="a", membership_epoch=3,
                          reason="lease_expired")
        append_pool_event(j, mode="elastic", tx_kind="pool_epoch", pool_epoch=1)
        j.append("phase", phase="x")  # non-pool records are ignored by replay
    with Journal(tmp_path) as j:
        st = replay_pool(j.records, "elastic")
    assert (st.pool_epoch, st.members, st.membership_epoch) == (1, ("b",), 3)
    assert replay_pool(read_journal(tmp_path), "elastic") == st


def test_pool_epoch_monotonic_and_validation(tmp_path):
    with Journal(tmp_path) as j:
        append_pool_event(j, mode="elastic", tx_kind="pool_epoch", pool_epoch=2, membership_epoch=5)
        with pytest.raises(EpochConflict):
            append_pool_event(j, mode="elastic", tx_kind="pool_epoch", pool_epoch=1)
        with pytest.raises(EpochConflict):
            append_pool_event(j, mode="elastic", tx_kind="pool_epoch", pool_epoch=2, membership_epoch=4)
        with pytest.raises(JournalError):
            append_pool_event(j, mode="elastic", tx_kind="pool_grow_typo", pool_epoch=2)
        with pytest.raises(JournalError):
            append_pool_event(j, mode="elastic", tx_kind="pool_join", pool_epoch=2)


def test_pause_advice_only_tightens():
    ok = PauseDecision(True, "local ok", 100.0)
    no = PauseDecision(False, "local no", None)
    big = PauseAdvice("planner", 0.0, 10.0, pause_budget_s=1e6)
    assert merge_pause(no, [big], now=1.0) == no
    assert merge_pause(ok, [PauseAdvice("p", 0.0, 10.0, pause_budget_s=30.0)], now=1.0).budget_s == 30.0
    veto = PauseAdvice("p", 0.0, 10.0, veto=True, reason="strict wait")
    assert not merge_pause(ok, [veto], now=1.0).allowed
    assert merge_pause(ok, [veto], now=11.0).allowed  # expired advice does not exist
    assert not merge_pause(ok, [PauseAdvice("p", 0.0, 10.0, pause_budget_s=5.0)], now=1.0,
                           expected_pause_s=6.0).allowed
    intent = PauseAdvice("p", 0.0, 10.0, target_resource_intent={"island_id": "a", "action": "leave"})
    assert merge_pause(ok, [intent], now=1.0).allowed


def test_lease_veto():
    assert lease_veto("a", now=0.0, lease_remaining_s=5.0, margin_s=1.0).pause_budget_s == 4.0
    assert lease_veto("a", now=0.0, lease_remaining_s=0.5, margin_s=1.0).veto


def test_legacy_ignores_pool_records_and_sched_fields(tmp_path):
    ctl = _controller(tmp_path, scheduling_probe=lambda: {"round_wall_s": 1.0})  # default legacy
    append_pool_event(ctl.journal, "pool_epoch", pool_epoch=4, mode="elastic")  # written earlier
    st = ctl.inspect()
    assert st.pool_epoch is None and st.round_wall_s is None
    with pytest.raises(JournalError):
        append_pool_event(ctl.journal, "pool_epoch", pool_epoch=5)  # legacy: read-only
    assert replay_pool(ctl.journal.records) == replay_pool([])  # legacy replay ignores


# ---------------------------------------------------------------- 0.9 advice in controller
from yeto.rl.engine.pause_audit import PAUSABLE_PHASE, pause_decision  # noqa: E402


def _local(ctl, pause_s):
    return pause_decision(ctl.profile, outer_phase=PAUSABLE_PHASE, expected_pause_s=pause_s,
                          budget_mode=ctl.budget_mode, quorum_timeout_s=ctl.quorum_timeout_s,
                          margin=ctl.pause_margin, idle_flow_timeout_s=ctl.idle_flow_timeout_s)


def test_legacy_pause_decision_ignores_advice(tmp_path):
    veto = lambda: [PauseAdvice("lease:b", 0.0, 1e12, veto=True, reason="x")]  # noqa: E731
    ctl = _controller(tmp_path, pause_advice_source=veto)  # legacy default
    assert ctl._pause_decision(PAUSABLE_PHASE, 5.0) == _local(ctl, 5.0)


def test_elastic_pause_decision_merges_advice(tmp_path):
    advice = []
    ctl = _controller(tmp_path, island_scheduling="elastic", pause_advice_source=lambda: advice)
    local = _local(ctl, 5.0)
    assert local.allowed and ctl._pause_decision(PAUSABLE_PHASE, 5.0) == local
    advice[:] = [PauseAdvice("lease:b", 0.0, 1e12, pause_budget_s=3.0)]
    d = ctl._pause_decision(PAUSABLE_PHASE, 5.0)
    assert not d.allowed and "advised budget" in d.reason
    advice[:] = [PauseAdvice("lease:b", 0.0, 1e12, veto=True, reason="strict wait")]
    assert "vetoed by lease:b" in ctl._pause_decision(PAUSABLE_PHASE, 1.0).reason


def test_elastic_unreadable_advice_fails_closed(tmp_path):
    def boom():
        raise RuntimeError("coordinator down")
    ctl = _controller(tmp_path, island_scheduling="elastic", pause_advice_source=boom)
    assert not ctl._pause_decision(PAUSABLE_PHASE, 1.0).allowed


# ---------------------------------------------------------------- 0.11 slow-island advice
from yeto.rl.engine.pause_advice import ROLLOUT_ONLY, slow_island_advice  # noqa: E402


def test_slow_island_advice_is_suggestion_only():
    adv = slow_island_advice({"a": 1.0, "b": 1.2, "slow": 5.0}, now=0.0)
    assert len(adv) == 1
    a = adv[0]
    assert a.target_resource_intent["island_id"] == "slow"
    assert a.target_resource_intent["action"] == ROLLOUT_ONLY
    assert a.target_resource_intent["requires_human_confirmation"] is True
    assert not a.veto and a.pause_budget_s is None
    ok = PauseDecision(True, "local ok", 100.0)
    assert merge_pause(ok, adv, now=1.0) == ok  # does not change any pause decision
    assert slow_island_advice({"a": 1.0, "b": 1.5}, now=0.0) == []
    assert slow_island_advice({"only": 9.0}, now=0.0) == []
