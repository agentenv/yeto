"""rl-infra-spec task 1.5: per-profile pause audit (pure)."""

from __future__ import annotations

from yeto.rl.engine.execution_profile import ExecutionProfile
from yeto.rl.engine.pause_audit import PAUSABLE_PHASE, audit_for, pause_decision


def _p(mode="partitioned-serial", outer="strict-avg", **kw) -> ExecutionProfile:
    return ExecutionProfile(name="p", execution_mode=mode, outer_protocol=outer, **kw)


def test_unknown_or_unaudited_profiles_disable_reconfiguration():
    assert not pause_decision(None, outer_phase=PAUSABLE_PHASE, expected_pause_s=1).allowed
    overlap = _p("partitioned-overlap", allowed_overlap={("reward", "checkpoint")})
    assert audit_for(overlap) is None
    d = pause_decision(overlap, outer_phase=PAUSABLE_PHASE, expected_pause_s=1)
    assert not d.allowed and "not audited" in d.reason
    d = pause_decision(_p(outer="decoupled"), outer_phase=PAUSABLE_PHASE, expected_pause_s=1)
    assert not d.allowed and "X6-decoupled" in d.reason


def test_strict_pause_only_at_published_boundary_within_budget():
    p = _p()
    ok = pause_decision(p, outer_phase=PAUSABLE_PHASE, expected_pause_s=300)
    assert ok.allowed and ok.stalls_peers and ok.budget_s == 450
    for phase in ("in-boundary", "stop-round", "finalizing", "budget-consolidation",
                  "mid-collective"):
        assert not pause_decision(p, outer_phase=phase, expected_pause_s=1).allowed
    assert not pause_decision(p, outer_phase=PAUSABLE_PHASE, expected_pause_s=600).allowed
    capped = pause_decision(p, outer_phase=PAUSABLE_PHASE, expected_pause_s=300,
                            idle_flow_timeout_s=200)
    assert not capped.allowed and capped.budget_s == 200
    assert not pause_decision(p, outer_phase=PAUSABLE_PHASE, expected_pause_s=10,
                              budget_mode=True).allowed
    assert not pause_decision(p, outer_phase="bogus", expected_pause_s=10).allowed


def test_standalone_has_no_outer_limit():
    d = pause_decision(_p("colocated-serial", "none"), outer_phase=PAUSABLE_PHASE,
                       expected_pause_s=5000, idle_flow_timeout_s=100)
    assert d.allowed and d.budget_s is None and not d.stalls_peers


def test_strict_budget_follows_configured_quorum_timeout():
    d = pause_decision(_p(), outer_phase=PAUSABLE_PHASE, expected_pause_s=1000,
                       quorum_timeout_s=3600)
    assert d.allowed and d.budget_s == 1800
