"""agentic-rollout-utilization group 3: version segments, IS correction, cut, ledger, fallback."""

from __future__ import annotations

import json
import math

import pytest

from yeto.rl.engine.version_segments import (
    InFlightTrajectory,
    PolicyAgeGovernor,
    ProvenanceError,
    TokenProvenance,
    batch_truncated_fraction,
    cross_version_is,
    restore_in_flight,
)


# ----------------------------------------------------------------------- 3.1


def test_provenance_serialization_roundtrip_and_segments():
    p = TokenProvenance((4, 4, 5, 5, 5), (-0.1, -0.2, -0.3, -0.4, -0.5))
    raw = json.loads(json.dumps(p.to_dict()))
    assert raw["segments"] == [[4, 0, 2], [5, 2, 5]]
    assert TokenProvenance.from_dict(raw) == p and p.oldest == 4
    q = TokenProvenance((4,), (-0.1,)).extend(5, [-0.2, -0.3])
    assert q.segments() == [(4, 0, 1), (5, 1, 3)]
    with pytest.raises(ProvenanceError):
        TokenProvenance.from_dict({**raw, "segments": [[4, 0, 2], [5, 3, 5]]})
    with pytest.raises(ProvenanceError):
        TokenProvenance((1,), (0.5,))


# ----------------------------------------------------------------------- 3.2


def test_cross_version_is_with_known_ratios():
    gen = (math.log(0.5), math.log(0.5), math.log(0.2), math.log(0.4))
    p = TokenProvenance((4, 4, 4, 5), gen)
    # train probs: 0.5 (ratio 1), 0.25 (0.5), 0.6 (3.0 -> clipped 2.0), current token
    train = (math.log(0.5), math.log(0.25), math.log(0.6), math.log(0.9))
    c = cross_version_is(p, train, 5, clip_low=0.0, clip_high=2.0)
    assert c.weights == pytest.approx((1.0, 0.5, 2.0, 1.0))
    assert (c.cross_version_tokens, c.truncated_tokens) == (3, 1)
    assert c.truncated_fraction == pytest.approx(1 / 3)
    on_policy = cross_version_is(TokenProvenance((5,), (-1.0,)), (-3.0,), 5)
    assert on_policy.weights == (1.0,) and on_policy.truncated_fraction is None
    assert batch_truncated_fraction([c, on_policy]) == pytest.approx(1 / 3)
    with pytest.raises(ProvenanceError, match="future"):
        cross_version_is(TokenProvenance((6,), (-1.0,)), (-1.0,), 5)


# ----------------------------------------------------------------------- 3.3


def _traj(tid, group, versions):
    return InFlightTrajectory(tid, group, "task", TokenProvenance(tuple(versions),
                                                                  tuple(-0.1 for _ in versions)))


def test_cut_in_flight_section_empty_at_limit_zero(tmp_path):
    from tests.test_rl_cut import _expect, _manifest
    from yeto.rl.engine.cut import commit_manifest, context_problems, verify_cut

    m = _manifest(tmp_path)
    assert "in_flight" not in m.body()  # default manifests unchanged
    ctx = dict(cut_id="cut-1", progress=m.progress, algorithm=m.algorithm, data=m.data,
               outer=m.outer, runtime=m.runtime)
    entries = tuple(_traj(f"t{i}", f"g{i}", [1, 2]).to_dict() for i in range(5))
    assert context_problems(ledger=m.ledger, **ctx) == []
    probs = context_problems(ledger={**m.ledger, "carried_over": 5}, in_flight=entries, **ctx)
    assert any("max_policy_age is 0" in p for p in probs)
    assert context_problems(ledger={**m.ledger, "carried_over": 5}, in_flight=entries,
                            max_policy_age=1, **ctx) == []
    assert any("!= 5 in-flight" in p for p in context_problems(
        ledger=m.ledger, in_flight=entries, max_policy_age=1, **ctx))
    m2 = _manifest(tmp_path, cut_id="cut-2", in_flight=entries,
                   ledger={**m.ledger, "carried_over": 5, "max_policy_age": 1})
    commit_manifest(tmp_path, m2)
    assert verify_cut(tmp_path, "cut-2", _expect()).in_flight == entries


def test_restore_with_in_flight_trajectories():
    """5 unfinished trajectories at the cut: resumed within the limit, else
    discarded and reported; a group already trained is never used again."""
    entries = [_traj("t0", "g0", [2]), _traj("t1", "g1", [2, 3]), _traj("t2", "g2", [1, 3]),
               _traj("t3", "g3", [3]), _traj("t4", "g4", [3]).to_dict()]
    zero = restore_in_flight(entries, max_policy_age=0, current_version=3)
    assert zero.resumed == () and zero.report() == {
        "resumed": 0, "discarded": 5, "discarded_tokens": 7, "reasons": {"limit_zero": 5}}
    one = restore_in_flight(entries, max_policy_age=1, current_version=3, completed_group_ids={"g3"})
    assert [t.trajectory_id for t in one.resumed] == ["t0", "t1", "t4"]
    assert one.report()["reasons"] == {"policy_age_exceeded": 1, "group_already_completed": 1}


# ----------------------------------------------------------------------- 3.4


def _ledger(max_outer_lag):
    from yeto.rl.engine.island_ledger import CrossIslandLedger, IslandSchedulingMode, StalenessPolicy

    led = CrossIslandLedger(policy=StalenessPolicy(max_outer_lag=max_outer_lag),
                            mode=IslandSchedulingMode.ELASTIC)
    return led


def test_ledger_judges_carried_over_groups_by_their_oldest_segment():
    from yeto.rl.engine.island_ledger import ACCEPT, ACCEPT_IS, REJECT, SampleGroup

    led = _ledger(1)
    led.join("a", now=0.0)
    for v in range(4):
        led.published[v] = f"h{v}"
    led.outer_version = 3
    logp = (-0.1,)
    one = SampleGroup("a", 3, 0, "h3", behavior_logprob=logp, version_segments=((2, "h2"), (3, "h3")))
    two = SampleGroup("a", 3, 0, "h3", behavior_logprob=logp,
                      version_segments=((1, "h1"), (2, "h2"), (3, "h3")))
    forged = SampleGroup("a", 3, 0, "h3", behavior_logprob=logp, version_segments=((2, "hx"), (3, "h3")))
    plain = SampleGroup("a", 3, 0, "h3")
    assert led.judge(plain, consumer_island="a").verdict == ACCEPT
    v1 = led.judge(one, consumer_island="a")
    assert (v1.verdict, v1.reason, v1.outer_lag) == (ACCEPT_IS, "stale_within_bound", 1)
    v2 = led.judge(two, consumer_island="a")
    assert (v2.verdict, v2.reason, v2.outer_lag) == (REJECT, "outer_lag_exceeded", 2)
    assert led.judge(forged, consumer_island="a").reason == "segment_policy_hash_mismatch"
    assert _ledger_accepts_two_with_wider_bound()


def _ledger_accepts_two_with_wider_bound():
    from yeto.rl.engine.island_ledger import ACCEPT_IS, SampleGroup

    led = _ledger(2)
    led.join("a", now=0.0)
    for v in range(4):
        led.published[v] = f"h{v}"
    led.outer_version = 3
    g = SampleGroup("a", 3, 0, "h3", behavior_logprob=(-0.1,),
                    version_segments=((1, "h1"), (3, "h3")))
    return led.judge(g, consumer_island="a").verdict == ACCEPT_IS


# ----------------------------------------------------------------------- 3.5


def test_governor_warns_then_falls_back_and_never_goes_up():
    g = PolicyAgeGovernor(limit=1, warn_fraction=0.2, fallback_fraction=0.5)
    assert g.observe(1, 0.1) == 1 and g.events == []
    assert g.observe(2, 0.3) == 1 and g.events[-1]["event"] == "rl_policy_age_warning"
    assert g.observe(3, 0.9) == 0 and g.events[-1]["event"] == "rl_policy_age_fallback"
    assert g.observe(4, 0.0) == 0  # only down, never back up
    assert PolicyAgeGovernor(limit=0).observe(1, 0.99) == 0


def test_driver_injected_high_truncation_falls_back_to_zero(tmp_path):
    import dataclasses

    from tests.test_rl_policy_age import _driver

    driver = _driver(tmp_path, 1)
    calls = []
    driver.rollout.set_max_policy_age = calls.append
    trainer = driver.trainer
    original = trainer.step_metrics

    def high():
        return dataclasses.replace(original(), cross_version_truncated_fraction=0.9)

    trainer.step_metrics = high
    driver.run()
    events = [json.loads(l) for l in (tmp_path / "e.jsonl").read_text().splitlines()]
    fallback = [e for e in events if e["event"] == "rl_policy_age_fallback"]
    assert len(fallback) == 1 and fallback[0]["to_limit"] == 0 and calls == [0]
    assert driver._max_policy_age() == 0
    trained = [e for e in events if e["event"] == "rl_round_trained"]
    assert all(e["cross_version_truncated_fraction"] == 0.9 for e in trained)
